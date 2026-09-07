"""Keyword search across the house.

Home Assistant's Assist tools match whole sentences, not keywords. Measured on a
live install:

    "what is Rocket Battery level"      -> "Rocket battery level is 33 %"
    "how much is Rocket Battery level"  -> "Sorry, I couldn't understand that"
    "battery level of Rocket"           -> "not aware of any device called ..."

So the assistant can only read something it already knows the exact name of. Ask
it about "the Tesla" when the device is called "Rocket" and it reports, honestly
but uselessly, that it cannot find the car.

Writing every entity name into the system prompt fixes one case and grows the
prompt forever. This tool fixes the class: give the model a way to *look*.

The add-on runs inside Home Assistant and already holds `homeassistant_api`, so
this is a single local call -- no round trip through an external agent.

Read-only by design. Control still goes through the Hass* tools, which have the
safety and confirmation behaviour.
"""
import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

logger = logging.getLogger(__name__)

_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")
_STATES_URL = "http://supervisor/core/api/states"

# Enough for "every value the car has" without burying the model in a whole
# floor of lights.
DEFAULT_LIMIT = 25
MAX_LIMIT = 60

# States worth reporting as "no value" rather than hiding: the model should be
# able to say the sensor is offline instead of claiming it does not exist.
_EMPTY_STATES = {"unknown", "unavailable", "none", ""}


def get_search_home_tool_definition() -> Dict[str, Any]:
    """OpenAI Realtime function-tool definition for house search."""
    return {
        "type": "function",
        "name": "search_home",
        "description": (
            "Search the house by keyword and get back matching things with their "
            "current values. Use this whenever you do not know the exact name of "
            "something, or when a Hass* tool says it cannot find a device. "
            "Searches both technical ids and friendly names, so a car registered "
            "as 'Rocket' is found by 'rocket', and a washing machine by "
            "'washing'. Always try this before telling the user something does "
            "not exist. Read-only: to change something, use the Hass* tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "One or more keywords, e.g. 'rocket battery', 'washing "
                        "machine', 'bedroom temperature'. Every word must appear "
                        "in the id or the name, so fewer words find more."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum results (default {DEFAULT_LIMIT}).",
                },
            },
            "required": ["query"],
        },
    }


def _format(entity: Dict[str, Any]) -> str:
    attrs = entity.get("attributes") or {}
    name = attrs.get("friendly_name") or entity["entity_id"]
    state = str(entity.get("state", "")).strip()
    if state.lower() in _EMPTY_STATES:
        return f"{name}: no value right now"
    unit = attrs.get("unit_of_measurement")
    # Long floats read badly out loud: 147.68949888 km becomes "one hundred
    # forty seven point six eight nine..." unless it is rounded here.
    try:
        number = float(state)
        state = f"{number:.0f}" if number == int(number) else f"{number:.1f}"
    except ValueError:
        pass
    return f"{name}: {state} {unit}".rstrip() if unit else f"{name}: {state}"


def _matches(entity: Dict[str, Any], words: List[str]) -> bool:
    attrs = entity.get("attributes") or {}
    haystack = f"{entity['entity_id']} {attrs.get('friendly_name') or ''}".lower()
    return all(word in haystack for word in words)


def create_search_home_tool_handler() -> Callable[["FunctionCallParams"], Awaitable[None]]:
    """Create the search_home handler for pipecat's realtime service."""

    async def search_home_tool_handler(params: "FunctionCallParams") -> None:
        args = params.arguments or {}
        query = str(args.get("query", "")).strip()
        try:
            limit = min(int(args.get("limit") or DEFAULT_LIMIT), MAX_LIMIT)
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT

        logger.info(f"🔦 search_home called: {query!r} (limit={limit})")

        if not query:
            await params.result_callback("No search words given.")
            return
        if not _TOKEN:
            logger.error("❌ search_home: SUPERVISOR_TOKEN missing")
            await params.result_callback("I cannot reach the house right now.")
            return

        words = [w for w in query.lower().split() if w]
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(
                    _STATES_URL,
                    headers={"Authorization": f"Bearer {_TOKEN}"},
                )
                response.raise_for_status()
                states = response.json()
        except Exception as e:
            logger.error(f"❌ search_home failed: {e}", exc_info=True)
            await params.result_callback("I could not search the house just now.")
            return

        hits = [e for e in states if _matches(e, words)]
        if not hits:
            logger.info(f"🔦 search_home: no match for {query!r}")
            await params.result_callback(
                f"Nothing in the house matches '{query}'. Try fewer or different words."
            )
            return

        # Shortest names first: "Rocket Battery level" before
        # "Rocket Battery level at arrival estimate".
        hits.sort(key=lambda e: len((e.get("attributes") or {}).get("friendly_name") or e["entity_id"]))
        shown, total = hits[:limit], len(hits)
        lines = [_format(e) for e in shown]
        logger.info(f"🔦 search_home: {total} match(es), returning {len(shown)}")

        answer = "\n".join(lines)
        if total > len(shown):
            answer += f"\n({total} matches in total, showing the {len(shown)} shortest names.)"
        await params.result_callback(answer)

    return search_home_tool_handler

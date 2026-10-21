"""Björn must know when Home Assistant tools are missing at session build."""

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from tests.test_provider_selection import _bare_app


HA_NERE = (
    "Husets verktyg går inte att nå just nu. Säg det rakt om någon ber dig "
    "styra något, och påstå aldrig att något är gjort."
)


class _McpClient:
    def __init__(self, failures=0):
        self.failures = failures

    async def get_tools_schema(self):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("Home Assistant unavailable")
        return ToolsSchema(standard_tools=[
            FunctionSchema(
                name="HassTurnOn",
                description="Turn something on",
                properties={},
                required=[],
            )
        ])

    async def register_tools_schema(self, schema, service):
        pass


async def _build(app):
    from app.device_registry import DeviceConnection
    from app.providers import OPENAI

    connection = DeviceConnection(device_id="office", websocket=object())
    connection.provider = OPENAI
    service = await app.create_service(connection)
    return connection, service._session_properties.instructions


@pytest.mark.asyncio
async def test_instructions_say_when_ha_tools_could_not_be_fetched():
    from app.providers import OPENAI

    app = _bare_app(OPENAI)
    app.mcp_client = _McpClient(failures=1)

    connection, instructions = await _build(app)

    assert connection.ha_verktyg_saknas is True
    assert HA_NERE in instructions


@pytest.mark.asyncio
async def test_successful_fetch_leaves_instructions_byte_for_byte_unchanged():
    from app.providers import OPENAI

    app = _bare_app(OPENAI)
    baseline_app = _bare_app(OPENAI)
    app.mcp_client = _McpClient()
    _, expected = await _build(baseline_app)

    connection, instructions = await _build(app)

    assert connection.ha_verktyg_saknas is False
    assert instructions == expected
    assert HA_NERE not in instructions


@pytest.mark.asyncio
async def test_next_session_drops_warning_after_ha_recovers():
    from app.providers import OPENAI

    app = _bare_app(OPENAI)
    app.mcp_client = _McpClient(failures=1)

    first_connection, first_instructions = await _build(app)
    second_connection, second_instructions = await _build(app)

    assert first_connection.ha_verktyg_saknas is True
    assert HA_NERE in first_instructions
    assert second_connection.ha_verktyg_saknas is False
    assert HA_NERE not in second_instructions

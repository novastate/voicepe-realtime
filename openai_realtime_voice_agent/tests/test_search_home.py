"""search_home must find things by keyword and read them out cleanly."""

from app.search_home_tool import _format, _matches


def _entity(entity_id, name=None, state="on", **attrs):
    if name:
        attrs["friendly_name"] = name
    return {"entity_id": entity_id, "state": state, "attributes": attrs}


def test_finds_car_by_device_name_not_in_entity_name():
    # The entity is called "Battery level"; only the device says "Rocket".
    # Home Assistant composes the friendly name, which is what users say.
    car = _entity("sensor.rocket_battery_level", "Rocket Battery level", "33")
    assert _matches(car, ["rocket"])
    assert _matches(car, ["rocket", "battery"])
    assert not _matches(car, ["tesla"])


def test_matching_is_case_insensitive_and_needs_every_word():
    car = _entity("sensor.rocket_battery_range", "Rocket Battery range", "147.6")
    assert _matches(car, ["ROCKET".lower(), "RANGE".lower()])
    assert not _matches(car, ["rocket", "washing"])


def test_matches_technical_id_when_there_is_no_friendly_name():
    bare = _entity("switch.rocket_sentry_mode", state="off")
    assert _matches(bare, ["sentry"])


def test_rounds_long_floats_so_they_read_aloud_well():
    # 147.68949888 km would be read digit by digit otherwise.
    e = _entity("sensor.rocket_battery_range", "Rocket Battery range", "147.68949888",
                unit_of_measurement="km")
    assert _format(e) == "Rocket Battery range: 147.7 km"


def test_whole_numbers_lose_the_decimal():
    e = _entity("sensor.rocket_battery_level", "Rocket Battery level", "33",
                unit_of_measurement="%")
    assert _format(e) == "Rocket Battery level: 33 %"


def test_keeps_text_states_as_they_are():
    e = _entity("binary_sensor.rocket_plug", "Rocket Plug", "off")
    assert _format(e) == "Rocket Plug: off"


def test_offline_sensors_say_so_instead_of_vanishing():
    for missing in ("unknown", "unavailable", "none", ""):
        e = _entity("sensor.rocket_shift_state", "Rocket Shift state", missing)
        assert _format(e) == "Rocket Shift state: no value right now"


def test_falls_back_to_the_entity_id_when_unnamed():
    e = _entity("sensor.mystery", state="7")
    assert _format(e) == "sensor.mystery: 7"


def test_finds_swedish_names_written_without_umlauts():
    # Speech-to-text and the model both spell these inconsistently.
    from app.search_home_tool import _matches
    e = _entity("sensor.tvattmaskin_state", "Tvättmaskin", "running")
    assert _matches(e, ["tvattmaskin"])
    assert _matches(e, ["Tvättmaskin"])


def test_falls_back_to_partial_when_nothing_matches_every_word():
    # The house is named in English, the household speaks Swedish. Requiring
    # every word to hit made the assistant claim the car does not exist.
    from app.search_home_tool import _search
    states = [
        _entity("sensor.rocket_battery_level", "Rocket Battery level", "33"),
        _entity("sensor.rocket_battery_range", "Rocket Battery range", "152"),
        _entity("light.hall", "Hall", "off"),
    ]
    hits, partial = _search(states, ["rocket", "batteri"])
    assert partial is True
    assert {h["entity_id"] for h in hits} == {
        "sensor.rocket_battery_level",
        "sensor.rocket_battery_range",
    }


def test_exact_match_wins_and_is_not_flagged_as_partial():
    from app.search_home_tool import _search
    states = [
        _entity("sensor.rocket_battery_level", "Rocket Battery level", "33"),
        _entity("light.hall", "Hall", "off"),
    ]
    hits, partial = _search(states, ["rocket", "battery"])
    assert partial is False
    assert len(hits) == 1


def test_single_word_never_falls_back():
    # One word that matches nothing means nothing; guessing would be noise.
    from app.search_home_tool import _search
    hits, partial = _search([_entity("light.hall", "Hall", "off")], ["garage"])
    assert hits == [] and partial is False

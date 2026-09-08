"""play_media must pick the thing the user meant, on the speaker they meant."""

from app.play_media_tool import _pick, _score_name, _score_player

# Shapes copied from a live music_assistant.search response.
P3_TUNEIN = {"name": "P3", "uri": "tunein--x://radio/s25681", "favorite": False}
P3_LIBRARY = {"name": "Sveriges Radio P3", "uri": "library://radio/4", "favorite": True}
KENT_LIBRARY = {"name": "Kent", "uri": "library://artist/932", "favorite": False}
KENT_SPOTIFY = {"name": "kent", "uri": "spotify--x://artist/4KX", "favorite": False}
RHAPSODY = {"name": "Bohemian Rhapsody", "uri": "spotify--x://track/2Ji", "favorite": False}
BABY_STATION = {"name": "La Baby Station", "uri": "tunein--x://radio/s340179", "favorite": False}


def test_radio_wins_when_the_model_says_radio():
    # The whole point: HassMediaSearchAndPlay played a Spotify track for "P3".
    buckets = {"radio": [P3_TUNEIN, P3_LIBRARY], "tracks": [RHAPSODY]}
    assert _pick(buckets, "P3", "radio")["_kind"] == "radio"


def test_the_houses_own_favourite_beats_an_exact_foreign_name():
    # "P3" is also a Danish station on TuneIn. The library one is the one the
    # user put there and the one they mean.
    buckets = {"radio": [P3_TUNEIN, P3_LIBRARY]}
    assert _pick(buckets, "P3", "radio")["uri"] == "library://radio/4"


def test_library_beats_spotify_for_the_same_artist():
    buckets = {"artists": [KENT_SPOTIFY, KENT_LIBRARY]}
    assert _pick(buckets, "Kent", "artist")["uri"] == "library://artist/932"


def test_external_result_still_wins_when_the_library_has_nothing():
    buckets = {"tracks": [RHAPSODY]}
    assert _pick(buckets, "Bohemian Rhapsody", "track")["uri"].startswith("spotify")


def test_one_word_out_of_three_is_a_coincidence_not_a_match():
    # Music Assistant always returns something. Accepting it turned a nonsense
    # request into "La Baby Station" playing in the office.
    buckets = {"radio": [BABY_STATION]}
    assert _pick(buckets, "zzzz obefintlig station", "radio") is None


def test_nothing_is_picked_when_no_name_matches():
    assert _pick({"radio": [P3_TUNEIN]}, "kalle anka", "radio") is None


def test_items_without_a_uri_are_skipped():
    assert _pick({"radio": [{"name": "P3"}]}, "P3", "radio") is None


def test_without_a_media_type_the_earlier_kinds_win():
    # A bare "put on P3" is a radio station long before it is a Spotify artist
    # who is also called P3.
    buckets = {"artists": [{"name": "P3", "uri": "spotify--x://artist/4kE"}],
               "radio": [P3_LIBRARY]}
    assert _pick(buckets, "P3", "")["_kind"] == "radio"


def test_accents_do_not_block_a_match():
    swede = {"name": "Håkan Hellström", "uri": "library://artist/146"}
    assert _pick({"artists": [swede]}, "hakan hellstrom", "artist") is not None


def test_score_name_rewards_an_exact_match_most():
    assert _score_name("P3", "P3") > _score_name("Sveriges Radio P3", "P3")


# --- speakers ---------------------------------------------------------------

def test_swedish_definite_form_finds_the_speaker():
    # The speaker is "Kök"; people say "köket". Same for Kontor/kontoret.
    assert _score_player("Kök Sonos", "köket") > 0
    assert _score_player("Kontor", "kontoret") > 0
    assert _score_player("Uterum", "uterummet") > 0


def test_an_exact_speaker_name_outranks_a_partial_one():
    assert _score_player("Moas Rum", "moas rum") > _score_player("Sovrum", "moas rum")


def test_an_unknown_room_matches_nothing():
    assert _score_player("Kontor", "badrummet") == 0
    assert _score_player("Hela Huset", "garaget") == 0

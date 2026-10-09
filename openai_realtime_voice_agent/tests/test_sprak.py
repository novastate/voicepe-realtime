"""The answer language is pinned in the instruction (kitchen 2026-10-09)."""


def test_instruktionen_tvingar_svenska_svar_oavsett_vad_som_hors():
    """Köket 2026-10-09: brus före väckordet hördes som italienska och svaret började på italienska."""
    from types import SimpleNamespace

    import app.main as main

    app = SimpleNamespace(instructions="Du är Björn.", idag=SimpleNamespace(block=lambda: "IDAG"))
    text = main.Application._instructions(app)
    assert text.startswith("Du är Björn.") and "Svara ALLTID på svenska" in text and "ändrar aldrig svarsspråket" in text
    assert text.endswith("IDAG")  # the time block stays last

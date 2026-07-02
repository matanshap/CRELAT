from crelat.catalog import load_play_catalog


def test_canonical_catalog_has_37_unique_plays():
    plays = load_play_catalog(require_files=True)
    assert len(plays) == 37
    assert len({play.id for play in plays}) == 37
    assert {play.genre for play in plays} == {"comedy", "history", "tragedy"}

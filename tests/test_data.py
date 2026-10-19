import pandas as pd

from barca.data import add_missing_results, team_matches


def test_team_matches_perspective():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-23", "2026-08-27", "2026-09-01"]),
        "home": ["Elche", "Barcelona", "Barcelona"],
        "away": ["Barcelona", "Ath Bilbao", "Getafe"],
        "home_goals": [0, 2, 1],
        "away_goals": [5, 0, 1],
        "home_xg": [0.7, 3.9, 1.2],
        "away_xg": [4.4, 0.3, 0.8],
    })
    out = team_matches(df, "Barcelona")

    assert list(out["opponent"]) == ["Elche", "Ath Bilbao", "Getafe"]
    assert list(out["venue"]) == ["F", "C", "C"]
    assert list(out["goals_for"]) == [5, 2, 1]
    assert list(out["xg_for"]) == [4.4, 3.9, 1.2]
    assert list(out["result"]) == ["V", "V", "E"]


def test_add_missing_results_only_fills_gaps():
    ours = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-20"]), "home": ["Sevilla"], "away": ["Barcelona"],
        "home_goals": [1], "away_goals": [3], "home_xg": [0.45], "away_xg": [3.26],
        "season": ["2627"], "division": ["SP1"], "source": ["football-data"],
    })
    fallback = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-20", "2026-10-10", "2026-10-10", "2026-10-17"]),
        "home": ["Sevilla", "Barcelona", "Vallecano", "Betis"],
        "away": ["Barcelona", "Getafe", "Ath Bilbao", "Barcelona"],
        "home_goals": [9, 3, None, None],   # o 9 não pode substituir o resultado que já temos
        "away_goals": [9, 0, None, None],   # jogo adiado e jogo futuro ficam de fora
    })
    out = add_missing_results(ours, fallback)
    assert len(out) == 2
    sev = out[out["home"] == "Sevilla"].iloc[0]
    assert (sev["home_goals"], sev["source"]) == (1, "football-data")
    new = out[out["home"] == "Barcelona"].iloc[0]
    assert (new["home_goals"], new["away_goals"], new["source"]) == (3, 0, "fixturedownload")
    assert pd.isna(new["home_xg"])


def test_download_falls_back_to_saved_copy(tmp_path, monkeypatch):
    import requests

    from barca import data

    saved = tmp_path / "SP1_2627.csv"
    saved.write_text("copia de ontem")
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)

    def offline(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("site em baixo")

    monkeypatch.setattr(data.requests, "get", offline)
    assert data.download("2627", "SP1") == saved
    assert saved.read_text() == "copia de ontem"

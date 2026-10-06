import pandas as pd

from barca.data import team_matches


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

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from barca.model import DixonColes
from barca.data import FIXTUREDOWNLOAD_NAMES
from barca.predict import evaluate_logged, merge_fixtures, new_fixtures, predict_fixtures

MODEL = DixonColes(teams=["Barcelona", "Getafe"], attack=np.array([0.8, -0.8]),
                   defence=np.array([-0.5, 0.5]), intercept=0.0, home_adv=0.25, rho=-0.05)
FIXTURES = pd.DataFrame({"date": pd.to_datetime(["2026-10-10"]), "home": ["Barcelona"], "away": ["Getafe"],
                         "odds_home": [1.2], "odds_draw": [7.0], "odds_away": [15.0]})


def test_predict_fixtures():
    p = predict_fixtures(MODEL, FIXTURES).iloc[0]
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-9
    assert p["p_home"] > 0.8
    assert p["xg_home"] > p["xg_away"]


def test_unknown_team_raises():
    bad = FIXTURES.assign(away="Getafe CF")
    with pytest.raises(ValueError, match="Getafe CF"):
        predict_fixtures(MODEL, bad)


def test_evaluate_logged():
    preds = predict_fixtures(MODEL, FIXTURES)
    results = pd.DataFrame({"date": pd.to_datetime(["2026-10-10"]), "home": ["Barcelona"],
                            "away": ["Getafe"], "home_goals": [3], "away_goals": [0]})
    merged, metrics = evaluate_logged(preds, results)
    assert metrics["jogos"] == 1
    assert metrics["acerto_modelo"] == 1.0
    assert "rps_odds" in metrics


def test_new_fixtures_skips_past_and_already_predicted():
    fixtures = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-09", "2026-10-10", "2026-10-11"]),
        "home": ["Elche", "Getafe", "Real Madrid"],
        "away": ["Osasuna", "Barcelona", "Sevilla"],
    })
    done = pd.DataFrame({"date": pd.to_datetime(["2026-10-10"]), "home": ["Getafe"], "away": ["Barcelona"]})
    left = new_fixtures(fixtures, done, today=pd.Timestamp("2026-10-10"))
    assert list(left["home"]) == ["Real Madrid"]


def test_new_fixtures_without_previous_predictions():
    fixtures = FIXTURES[["date", "home", "away"]]
    empty = pd.DataFrame(columns=["date", "home", "away"])
    assert len(new_fixtures(fixtures, empty, today=pd.Timestamp("2026-10-01"))) == 1


def test_fixturedownload_names_cover_all_teams():
    # Os 20 clubes da época têm emblema em site/crests.json, com os nomes de football-data.co.uk.
    crests = json.loads((Path(__file__).resolve().parents[1] / "site" / "crests.json").read_text(encoding="utf-8"))
    assert set(FIXTUREDOWNLOAD_NAMES.values()) == set(crests)


def test_merge_fixtures_prefers_primary_and_fills_gaps():
    primary = pd.DataFrame({"date": pd.to_datetime(["2026-10-12"]), "home": ["Levante"], "away": ["Sevilla"],
                            "odds_home": [2.68], "odds_draw": [3.27], "odds_away": [2.56]})
    extra = pd.DataFrame({"date": pd.to_datetime(["2026-10-12", "2026-10-17"]),
                          "home": ["Levante", "Betis"], "away": ["Sevilla", "Barcelona"]})
    out = merge_fixtures(primary, extra)
    assert len(out) == 2
    assert out.loc[out["home"] == "Levante", "odds_home"].iloc[0] == 2.68
    assert "Betis" in set(out["home"])


def test_postponed_game_is_not_predicted_twice():
    done = pd.DataFrame({"date": pd.to_datetime(["2026-10-10"]), "home": ["Vallecano"], "away": ["Ath Bilbao"]})
    rescheduled = pd.DataFrame({"date": pd.to_datetime(["2026-11-04"]), "home": ["Vallecano"], "away": ["Ath Bilbao"]})
    assert new_fixtures(rescheduled, done, today=pd.Timestamp("2026-11-01")).empty

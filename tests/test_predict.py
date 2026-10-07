import numpy as np
import pandas as pd
import pytest

from barca.model import DixonColes
from barca.predict import evaluate_logged, new_fixtures, predict_fixtures

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

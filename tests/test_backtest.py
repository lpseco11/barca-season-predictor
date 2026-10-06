import numpy as np
import pandas as pd

from barca.backtest import implied_probs, outcome, rps


def test_rps_perfect_and_worst():
    y = np.array([0, 2])
    perfect = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    worst = np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
    assert rps(perfect, y) == 0
    assert rps(worst, y) == 1


def test_implied_probs_remove_margin():
    df = pd.DataFrame({"odds_home": [2.0], "odds_draw": [3.2], "odds_away": [3.8]})
    p = implied_probs(df)
    assert abs(p.sum() - 1) < 1e-12
    assert p[0, 0] > p[0, 2]


def test_outcome_codes():
    df = pd.DataFrame({"home_goals": [2, 1, 0], "away_goals": [0, 1, 3]})
    assert list(outcome(df)) == [0, 1, 2]

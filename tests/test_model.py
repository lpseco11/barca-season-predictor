import numpy as np
import pandas as pd
from scipy.optimize import check_grad

from barca.model import _neg_log_likelihood, fit


def _synthetic_league(seed=0, rounds=30):
    """Liga falsa com forças conhecidas, para ver se o modelo as recupera."""
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D", "E", "F"]
    attack = np.array([0.5, 0.3, 0.0, 0.0, -0.3, -0.5])
    defence = np.array([-0.4, -0.2, 0.0, 0.1, 0.2, 0.3])
    rows, day = [], pd.Timestamp("2025-01-01")
    for r in range(rounds):
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                lam = np.exp(0.1 + 0.25 + attack[i] + defence[j])
                mu = np.exp(0.1 + attack[j] + defence[i])
                rows.append((day + pd.Timedelta(days=r), teams[i], teams[j],
                             rng.poisson(lam), rng.poisson(mu)))
    df = pd.DataFrame(rows, columns=["date", "home", "away", "home_goals", "away_goals"])
    return df, attack, defence


def test_gradient_matches_numerical():
    rng = np.random.default_rng(1)
    n, m = 5, 200
    home, away = rng.integers(0, n, m), rng.integers(0, n, m)
    x, y = rng.poisson(1.5, m).astype(float), rng.poisson(1.1, m).astype(float)
    w = rng.uniform(0.3, 1.0, m)
    params = rng.normal(0, 0.2, 3 + 2 * n)
    params[2] = -0.05

    def f(p):
        return _neg_log_likelihood(p, home, away, x, y, w, n)[0]

    def g(p):
        return _neg_log_likelihood(p, home, away, x, y, w, n)[1]

    assert check_grad(f, g, params) < 1e-4


def test_recovers_known_strengths():
    df, attack, defence = _synthetic_league()
    model = fit(df, xi=0.0, stretch=1.0)
    order = [model.teams.index(t) for t in ["A", "B", "C", "D", "E", "F"]]
    assert np.allclose(model.attack[order], attack - attack.mean(), atol=0.12)
    assert np.allclose(model.defence[order], defence - defence.mean(), atol=0.12)
    assert abs(model.home_adv - 0.25) < 0.08


def test_probabilities_are_valid():
    df, _, _ = _synthetic_league(rounds=5)
    model = fit(df)
    probs = model.outcome_probs("A", "F")
    assert abs(sum(probs.values()) - 1) < 1e-9
    assert probs["home"] > probs["away"]
    assert abs(model.score_matrix("C", "D").sum() - 1) < 1e-9


def test_stretch_scales_strengths_only():
    df, _, _ = _synthetic_league(rounds=10)
    base = fit(df, stretch=1.0)
    wide = fit(df, stretch=1.2)
    assert np.allclose(wide.attack, base.attack * 1.2)
    assert np.allclose(wide.defence, base.defence * 1.2)
    assert wide.home_adv == base.home_adv and wide.intercept == base.intercept

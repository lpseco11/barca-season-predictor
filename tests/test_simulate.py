import numpy as np
import pandas as pd

from barca.model import DixonColes
from barca.simulate import remaining_fixtures, simulate, summary, table

PLAYED = pd.DataFrame({
    "home": ["A", "B", "C"],
    "away": ["B", "C", "A"],
    "home_goals": [2, 1, 0],
    "away_goals": [0, 1, 3],
})


def test_table():
    t = table(PLAYED)
    assert t.loc["A", "points"] == 6
    assert t.loc["B", "points"] == 1
    assert t.loc["A", "gd"] == 5
    assert list(t.index) == ["A", "B", "C"]


def test_remaining_fixtures_double_round_robin():
    left = remaining_fixtures(PLAYED, ["A", "B", "C"])
    assert sorted(left) == [("A", "C"), ("B", "A"), ("C", "B")]


def test_simulation_consistency():
    model = DixonColes(teams=["A", "B", "C"], attack=np.array([1.0, 0.0, -1.0]),
                       defence=np.array([-1.0, 0.0, 1.0]), intercept=0.0, home_adv=0.2, rho=0.0)
    sims = simulate(model, PLAYED, n_sims=2000, seed=0)
    s = summary(sims).set_index("team")

    # Cada simulação tem exatamente um campeão; A é muito mais forte.
    assert abs(s["p_titulo"].sum() - 1) < 1e-9
    assert s.loc["A", "p_titulo"] > 0.9
    # Os pontos finais nunca são menores do que os atuais.
    assert (sims["points"][:, 0] >= 6).all()
    # Cada equipa joga mais 2 jogos: no máximo +6 pontos.
    assert (sims["points"][:, 2] <= 3 + 6).all()


def _two_team_model(cov_scale):
    n = 2
    cov = np.eye(3 + 2 * n) * cov_scale if cov_scale else None
    return DixonColes(teams=["A", "B"], attack=np.array([0.3, -0.3]), defence=np.array([-0.3, 0.3]),
                      intercept=0.0, home_adv=0.2, rho=0.0, cov=cov)


def test_uncertainty_widens_distribution():
    played = pd.DataFrame({"home": ["A"], "away": ["B"], "home_goals": [1], "away_goals": [1]})
    fixed = simulate(_two_team_model(0), played, n_sims=20_000, seed=0)
    uncertain = simulate(_two_team_model(0.3), played, n_sims=20_000, seed=0)
    # Mesmo jogo (B vs A), mas com incerteza nas forças o nº de golos varia mais.
    assert uncertain["gf"][:, 0].var() > fixed["gf"][:, 0].var()


def test_matches_score_matrix_without_uncertainty():
    model = _two_team_model(0)
    played = pd.DataFrame({"home": ["A"], "away": ["B"], "home_goals": [0], "away_goals": [0]})
    sims = simulate(model, played, n_sims=40_000, seed=1)
    # O jogo em falta é B (casa) vs A. P(A ganha) na simulação ≈ P(vitória fora) do modelo.
    a_won = (sims["points"][:, 0] == 1 + 3).mean()
    assert abs(a_won - model.outcome_probs("B", "A")["away"]) < 0.01

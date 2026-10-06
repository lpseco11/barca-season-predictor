"""Simulação Monte Carlo do resto da época.

1. Parte da classificação atual (jogos já disputados).
2. Os jogos que faltam são todos os pares casa/fora que ainda não se jogaram
   (La Liga é uma volta dupla: cada equipa recebe cada adversário uma vez).
3. Para cada jogo, sorteia um resultado da matriz de Dixon-Coles.
4. Repete N vezes e conta: quantas vezes cada equipa foi campeã, etc.

Incerteza: as forças estimadas não são exatas. Se o modelo tiver covariância
(fit(..., with_cov=True)), cada simulação usa forças sorteadas à volta da
estimativa — um "Barça um pouco melhor" numa, "um pouco pior" noutra. Sem isto
as probabilidades ficam confiantes demais.

Simplificações:
- Desempate por pontos → diferença de golos → golos marcados → sorteio. La Liga
  usa primeiro o confronto direto; o efeito nas probabilidades é pequeno.
- Dentro de cada simulação, as forças ficam fixas durante a época.
"""

from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from barca.model import DixonColes


def table(played: pd.DataFrame) -> pd.DataFrame:
    """Classificação a partir dos jogos disputados."""
    rows = []
    for _, g in played.iterrows():
        hg, ag = g["home_goals"], g["away_goals"]
        hp, ap = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
        rows.append((g["home"], 1, hp, hg, ag))
        rows.append((g["away"], 1, ap, ag, hg))
    t = pd.DataFrame(rows, columns=["team", "played", "points", "gf", "ga"]).groupby("team").sum()
    t["gd"] = t["gf"] - t["ga"]
    return t.sort_values(["points", "gd", "gf"], ascending=False)


def remaining_fixtures(played: pd.DataFrame, teams: list[str]) -> list[tuple[str, str]]:
    done = set(zip(played["home"], played["away"]))
    return [(h, a) for h, a in permutations(teams, 2) if (h, a) not in done]


def _score_matrices(params: dict, i: int, j: int, max_goals: int) -> np.ndarray:
    """Matrizes de resultados (K, golos_casa, golos_fora) para K conjuntos de parâmetros."""
    lam = np.exp(params["intercept"] + params["home_adv"] + params["attack"][:, i] + params["defence"][:, j])
    mu = np.exp(params["intercept"] + params["attack"][:, j] + params["defence"][:, i])
    rho = params["rho"]
    goals = np.arange(max_goals + 1)
    m = poisson.pmf(goals, lam[:, None])[:, :, None] * poisson.pmf(goals, mu[:, None])[:, None, :]
    m[:, 0, 0] *= 1 - lam * mu * rho
    m[:, 0, 1] *= 1 + lam * rho
    m[:, 1, 0] *= 1 + mu * rho
    m[:, 1, 1] *= 1 - rho
    return m / m.sum(axis=(1, 2), keepdims=True)


def simulate(model: DixonColes, played: pd.DataFrame, n_sims: int = 10_000,
             seed: int | None = None, max_goals: int = 10, param_draws: int = 500) -> dict:
    """Devolve pontos/golos/posição finais de cada equipa em cada simulação.

    `param_draws`: quantos conjuntos de forças sortear (as simulações são
    repartidas entre eles). Ignorado se o modelo não tiver covariância.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(set(played["home"]) | set(played["away"]))
    idx = {t: k for k, t in enumerate(teams)}
    model_idx = np.array([model.teams.index(t) for t in teams])
    n = len(teams)

    params = model.sample_params(param_draws, rng)
    params["attack"] = params["attack"][:, model_idx]
    params["defence"] = params["defence"][:, model_idx]
    k = len(params["intercept"])
    block = np.arange(n_sims) % k  # que conjunto de parâmetros usa cada simulação

    current = table(played).reindex(teams).fillna(0)
    points = np.tile(current["points"].to_numpy(dtype=float), (n_sims, 1))
    gf = np.tile(current["gf"].to_numpy(dtype=float), (n_sims, 1))
    ga = np.tile(current["ga"].to_numpy(dtype=float), (n_sims, 1))

    size = max_goals + 1
    for home, away in remaining_fixtures(played, teams):
        i, j = idx[home], idx[away]
        cdf = _score_matrices(params, i, j, max_goals).reshape(k, -1).cumsum(axis=1)
        u = rng.random(n_sims)
        cells = np.minimum((u[:, None] > cdf[block]).sum(axis=1), size * size - 1)
        hg, ag = cells // size, cells % size
        points[:, i] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        points[:, j] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gf[:, i] += hg
        ga[:, i] += ag
        gf[:, j] += ag
        ga[:, j] += hg

    # Ordenação: pontos, depois diferença de golos, golos marcados e um sorteio.
    # Cada critério é escalado para não interferir com o anterior.
    key = points * 1e6 + (gf - ga + 500) * 1e3 + gf + rng.random(points.shape)
    order = np.argsort(-key, axis=1)
    position = np.empty_like(order)
    position[np.arange(n_sims)[:, None], order] = np.arange(1, n + 1)

    return {"teams": teams, "points": points, "gf": gf, "ga": ga, "position": position}


def summary(sims: dict) -> pd.DataFrame:
    """Probabilidades por equipa: título, top 4 (Champions), descida, etc."""
    pos, pts = sims["position"], sims["points"]
    df = pd.DataFrame({
        "team": sims["teams"],
        "pontos_esperados": pts.mean(axis=0),
        "posicao_media": pos.mean(axis=0),
        "p_titulo": (pos == 1).mean(axis=0),
        "p_top4": (pos <= 4).mean(axis=0),
        "p_descida": (pos >= 18).mean(axis=0),
    })
    return df.sort_values("pontos_esperados", ascending=False).reset_index(drop=True)

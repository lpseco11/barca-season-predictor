"""Backtest: prever épocas passadas como se estivéssemos lá.

Para cada semana de La Liga, o modelo é treinado só com jogos anteriores a
essa semana e prevê os jogos dessa semana. Assim não há "espreitar o futuro".

Métricas (quanto menor, melhor):
- log loss: penaliza muito estar confiante e errar.
- RPS (Ranked Probability Score): a métrica habitual no futebol; trata
  casa/empate/fora como ordenados (prever empate quando ganha a casa é
  "menos errado" do que prever vitória fora).
"""

import numpy as np
import pandas as pd

from barca.model import DEFAULT_XI, fit

OUTCOMES = ["home", "draw", "away"]


def outcome(df: pd.DataFrame) -> np.ndarray:
    """0 = vitória casa, 1 = empate, 2 = vitória fora."""
    return np.select([df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]], [0, 1], 2)


def implied_probs(df: pd.DataFrame) -> np.ndarray:
    """Probabilidades implícitas nas odds médias, sem a margem da casa de apostas."""
    inv = 1 / df[["odds_home", "odds_draw", "odds_away"]].to_numpy()
    return inv / inv.sum(axis=1, keepdims=True)


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)).mean())


def rps(probs: np.ndarray, y: np.ndarray) -> float:
    observed = np.eye(3)[y]
    diff = np.cumsum(probs, axis=1)[:, :2] - np.cumsum(observed, axis=1)[:, :2]
    return float((diff ** 2).sum(axis=1).mean() / 2)


def backtest(matches: pd.DataFrame, test_seasons: list[str], xi: float = DEFAULT_XI) -> pd.DataFrame:
    """Previsões semana a semana para os jogos de La Liga das épocas de teste."""
    test = matches[(matches["division"] == "SP1") & matches["season"].isin(test_seasons)].copy()
    test["week"] = test["date"].dt.to_period("W-SUN").dt.start_time

    rows = []
    for week, games in test.groupby("week"):
        model = fit(matches, as_of=week, xi=xi)
        for _, g in games.iterrows():
            if g["home"] not in model.teams or g["away"] not in model.teams:
                continue  # equipa sem histórico (não acontece com SP1+SP2)
            p = model.outcome_probs(g["home"], g["away"])
            rows.append({**g.to_dict(), **{f"p_{k}": p[k] for k in OUTCOMES}})
    return pd.DataFrame(rows)


def evaluate(preds: pd.DataFrame) -> pd.DataFrame:
    """Compara modelo, odds e um baseline ingénuo (frequências históricas), por época."""
    out = []
    for season, g in preds.groupby("season"):
        y = outcome(g)
        model = g[[f"p_{k}" for k in OUTCOMES]].to_numpy()
        market = implied_probs(g)
        naive = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
        for name, p in [("modelo", model), ("odds", market), ("ingénuo", naive)]:
            out.append({"season": season, "fonte": name, "jogos": len(y),
                        "log_loss": log_loss(p, y), "rps": rps(p, y),
                        "acerto": float((p.argmax(axis=1) == y).mean())})
    return pd.DataFrame(out)

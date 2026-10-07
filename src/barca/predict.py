"""Previsões jogo a jogo para a próxima jornada, e avaliação depois dos jogos."""

from pathlib import Path

import numpy as np
import pandas as pd

from barca.backtest import implied_probs, log_loss, outcome, rps
from barca.model import DixonColes

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def load_fixtures(division: str = "SP1", manual: Path | None = None) -> pd.DataFrame:
    """Próximos jogos: do ficheiro manual se existir, senão de football-data.co.uk.

    O ficheiro manual é um CSV com colunas date (dd/mm/aaaa), home, away e,
    opcionalmente, odds_home/odds_draw/odds_away.
    """
    if manual is not None and manual.exists():
        df = pd.read_csv(manual)
    else:
        df = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig", encoding_errors="replace")
        df = df[df["Div"] == division].rename(columns={
            "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
            "AvgH": "odds_home", "AvgD": "odds_draw", "AvgA": "odds_away"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    cols = [c for c in ["date", "home", "away", "odds_home", "odds_draw", "odds_away"] if c in df.columns]
    return df[cols].reset_index(drop=True)


def new_fixtures(fixtures: pd.DataFrame, done: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Jogos ainda por disputar (a partir de `today`) que não têm previsão registada."""
    key = ["date", "home", "away"]
    upcoming = fixtures[fixtures["date"] >= today].copy()
    upcoming["date"] = upcoming["date"].astype("datetime64[ns]")
    if done.empty:
        return upcoming.reset_index(drop=True)
    done = done[key].astype({"date": "datetime64[ns]"})
    merged = upcoming.merge(done, on=key, how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)


def predict_fixtures(model: DixonColes, fixtures: pd.DataFrame) -> pd.DataFrame:
    unknown = (set(fixtures["home"]) | set(fixtures["away"])) - set(model.teams)
    if unknown:
        raise ValueError(f"Equipas desconhecidas (nomes têm de ser os de football-data.co.uk): {sorted(unknown)}")
    rows = []
    for _, f in fixtures.iterrows():
        lam, mu = model.expected_goals(f["home"], f["away"])
        p = model.outcome_probs(f["home"], f["away"])
        m = model.score_matrix(f["home"], f["away"])
        hg, ag = np.unravel_index(m.argmax(), m.shape)
        rows.append({**f.to_dict(), "xg_home": lam, "xg_away": mu,
                     "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
                     "likely_score": f"{hg}-{ag}", "p_likely_score": m[hg, ag]})
    return pd.DataFrame(rows)


def evaluate_logged(predictions: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Junta previsões registadas com os resultados reais e calcula as métricas."""
    merged = predictions.merge(results[["date", "home", "away", "home_goals", "away_goals"]],
                               on=["date", "home", "away"], how="inner")
    if merged.empty:
        return merged, {}
    y = outcome(merged)
    model_p = merged[["p_home", "p_draw", "p_away"]].to_numpy()
    metrics = {"jogos": len(merged), "rps_modelo": rps(model_p, y), "log_loss_modelo": log_loss(model_p, y),
               "acerto_modelo": float((model_p.argmax(axis=1) == y).mean())}
    has_odds = merged[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1) if "odds_home" in merged else None
    if has_odds is not None and has_odds.all():
        market = implied_probs(merged)
        metrics.update({"rps_odds": rps(market, y), "log_loss_odds": log_loss(market, y)})
    return merged, metrics

"""Previsões jogo a jogo para a próxima jornada, e avaliação depois dos jogos."""

from pathlib import Path

import numpy as np
import pandas as pd

from barca.backtest import implied_probs, log_loss, outcome, rps
from barca.data import fetch_fixturedownload
from barca.model import DixonColes

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def load_fixtures(division: str = "SP1", manual: Path | None = None, days_ahead: int = 3) -> pd.DataFrame:
    """Jogos de La Liga dos próximos `days_ahead` dias (hoje incluído).

    Se existir o ficheiro manual (CSV com date dd/mm/aaaa, home, away e, opcionalmente,
    odds), usa só esse. Senão junta as duas fontes: football-data.co.uk, que traz as odds,
    e fixturedownload.com para os jogos que o football-data.co.uk ainda não publicou.

    A janela curta faz com que cada jogo seja previsto poucos dias antes, já com os
    resultados da jornada anterior, mas com várias corridas diárias até ao pontapé de saída.
    """
    today = pd.Timestamp.now().normalize()
    if manual is not None and manual.exists():
        df = pd.read_csv(manual)
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    else:
        fd = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig", encoding_errors="replace")
        fd = fd[fd["Div"] == division].rename(columns={
            "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
            "AvgH": "odds_home", "AvgD": "odds_draw", "AvgA": "odds_away"})
        fd["date"] = pd.to_datetime(fd["date"], dayfirst=True)
        df = merge_fixtures(fd, load_fixturedownload())
    df = df[(df["date"] >= today) & (df["date"] <= today + pd.Timedelta(days=days_ahead))]
    cols = [c for c in ["date", "home", "away", "odds_home", "odds_draw", "odds_away"] if c in df.columns]
    return df[cols].sort_values("date").reset_index(drop=True)


def merge_fixtures(primary: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Jogos de `primary` mais os de `extra` que lá não estão (o par casa/fora identifica o jogo)."""
    have = set(zip(primary["home"], primary["away"]))
    extra = extra[[(h, a) not in have for h, a in zip(extra["home"], extra["away"])]]
    return pd.concat([primary, extra], ignore_index=True)


def load_fixturedownload() -> pd.DataFrame:
    """Jogos ainda sem resultado no calendário de fixturedownload.com."""
    games = fetch_fixturedownload()
    return games[games["home_goals"].isna()][["date", "home", "away"]].reset_index(drop=True)


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
    # As odds vêm do ficheiro de resultados quando existem lá (a previsão pode ter sido
    # registada a partir de fixturedownload.com, que não tem odds).
    odds = [c for c in ["odds_home", "odds_draw", "odds_away"] if c in results.columns]
    if odds:
        predictions = predictions.drop(columns=[c for c in odds if c in predictions.columns])
    merged = predictions.merge(results[["date", "home", "away", "home_goals", "away_goals", *odds]],
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

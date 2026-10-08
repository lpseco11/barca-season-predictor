"""Previsões jogo a jogo para a próxima jornada, e avaliação depois dos jogos."""

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from barca.backtest import implied_probs, log_loss, outcome, rps
from barca.model import DixonColes

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FIXTUREDOWNLOAD_URL = "https://fixturedownload.com/feed/json/la-liga-2026"

# Nomes em fixturedownload.com → nomes em football-data.co.uk (os que o modelo usa).
FIXTUREDOWNLOAD_NAMES = {
    "Athletic Club": "Ath Bilbao", "Atlético de Madrid": "Ath Madrid", "CA Osasuna": "Osasuna",
    "Celta": "Celta", "Deportivo Alavés": "Alaves", "Elche CF": "Elche", "FC Barcelona": "Barcelona",
    "Getafe CF": "Getafe", "Levante UD": "Levante", "Málaga CF": "Malaga", "R. Racing Club": "Santander",
    "RC Deportivo": "La Coruna", "RCD Espanyol de Barcelona": "Espanol", "Rayo Vallecano": "Vallecano",
    "Real Betis": "Betis", "Real Madrid": "Real Madrid", "Real Sociedad": "Sociedad", "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia", "Villarreal CF": "Villarreal",
}


def load_fixtures(division: str = "SP1", manual: Path | None = None, days_ahead: int = 7) -> pd.DataFrame:
    """Próximos jogos de La Liga.

    Ordem de preferência:
    1. ficheiro manual, se existir (CSV com date dd/mm/aaaa, home, away e, opcionalmente, odds);
    2. football-data.co.uk, que também traz as odds;
    3. fixturedownload.com, que tem o calendário completo da época, para quando o
       football-data.co.uk ainda não publicou os jogos dos próximos dias.
    """
    if manual is not None and manual.exists():
        df = pd.read_csv(manual)
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    else:
        df = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig", encoding_errors="replace")
        df = df[df["Div"] == division].rename(columns={
            "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
            "AvgH": "odds_home", "AvgD": "odds_draw", "AvgA": "odds_away"})
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        today = pd.Timestamp.now().normalize()
        soon = (df["date"] >= today) & (df["date"] <= today + pd.Timedelta(days=days_ahead))
        if not soon.any():
            df = load_fixturedownload(days_ahead)
    cols = [c for c in ["date", "home", "away", "odds_home", "odds_draw", "odds_away"] if c in df.columns]
    return df[cols].reset_index(drop=True)


def load_fixturedownload(days_ahead: int = 7) -> pd.DataFrame:
    """Jogos ainda sem resultado nos próximos dias, do calendário de fixturedownload.com.

    As horas vêm em UTC; a data do jogo é a de Madrid, como em football-data.co.uk.
    """
    req = urllib.request.Request(FIXTUREDOWNLOAD_URL, headers={"User-Agent": "barca-season-predictor"})
    games = pd.DataFrame(json.load(urllib.request.urlopen(req, timeout=30)))
    games = games[games["HomeTeamScore"].isna()]
    games["date"] = (pd.to_datetime(games["DateUtc"], utc=True).dt.tz_convert("Europe/Madrid")
                     .dt.tz_localize(None).dt.normalize())
    today = pd.Timestamp.now().normalize()
    games = games[(games["date"] >= today) & (games["date"] <= today + pd.Timedelta(days=days_ahead))]
    unknown = (set(games["HomeTeam"]) | set(games["AwayTeam"])) - set(FIXTUREDOWNLOAD_NAMES)
    if unknown:
        raise ValueError(f"Nomes de equipas sem correspondência em FIXTUREDOWNLOAD_NAMES: {sorted(unknown)}")
    return pd.DataFrame({"date": games["date"], "home": games["HomeTeam"].map(FIXTUREDOWNLOAD_NAMES),
                         "away": games["AwayTeam"].map(FIXTUREDOWNLOAD_NAMES)}).reset_index(drop=True)


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

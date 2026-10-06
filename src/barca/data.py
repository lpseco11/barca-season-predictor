"""Download e limpeza dos dados de football-data.co.uk.

Cada época é um CSV por divisão: SP1 = La Liga, SP2 = Segunda División.
As épocas são códigos de 4 dígitos: "2627" = 2026/27.
"""

from pathlib import Path

import pandas as pd
import requests

from barca import CURRENT_SEASON

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Colunas que usamos, renomeadas para nomes mais legíveis.
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HxG": "home_xg",
    "AxG": "away_xg",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "AvgH": "odds_home",
    "AvgD": "odds_draw",
    "AvgA": "odds_away",
}


def download(season: str, division: str = "SP1", force: bool = False) -> Path:
    """Descarrega o CSV de uma época/divisão para data/raw/ e devolve o caminho.

    A época atual é sempre descarregada de novo (vai sendo atualizada);
    épocas passadas só são descarregadas se ainda não existirem.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{division}_{season}.csv"
    if path.exists() and not force and season != CURRENT_SEASON:
        return path
    response = requests.get(BASE_URL.format(season=season, division=division), timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_season(season: str, division: str = "SP1") -> pd.DataFrame:
    """Carrega uma época já limpa: só jogos disputados, colunas normalizadas."""
    path = download(season, division)
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = df[[c for c in COLUMNS if c in df.columns]].rename(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df[["home_goals", "away_goals"]] = df[["home_goals", "away_goals"]].astype(int)
    df["season"] = season
    df["division"] = division
    return df.sort_values("date").reset_index(drop=True)


def load_matches(seasons: list[str], divisions: tuple[str, ...] = ("SP1", "SP2")) -> pd.DataFrame:
    """Junta várias épocas e divisões num único DataFrame de jogos."""
    frames = [load_season(s, d) for s in seasons for d in divisions]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Vista dos jogos de uma equipa na sua perspetiva (golos/xG a favor e contra)."""
    home = df[df["home"] == team]
    away = df[df["away"] == team]
    as_home = pd.DataFrame({
        "date": home["date"], "opponent": home["away"], "venue": "C",
        "goals_for": home["home_goals"], "goals_against": home["away_goals"],
        "xg_for": home.get("home_xg"), "xg_against": home.get("away_xg"),
    })
    as_away = pd.DataFrame({
        "date": away["date"], "opponent": away["home"], "venue": "F",
        "goals_for": away["away_goals"], "goals_against": away["home_goals"],
        "xg_for": away.get("away_xg"), "xg_against": away.get("home_xg"),
    })
    out = pd.concat([as_home, as_away]).sort_values("date").reset_index(drop=True)
    out["result"] = (out["goals_for"] > out["goals_against"]).map({True: "V", False: "E"})
    out.loc[out["goals_for"] < out["goals_against"], "result"] = "D"
    return out

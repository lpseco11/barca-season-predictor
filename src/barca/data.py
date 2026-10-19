"""Download e limpeza dos dados de football-data.co.uk.

Cada época é um CSV por divisão: SP1 = La Liga, SP2 = Segunda División.
As épocas são códigos de 4 dígitos: "2627" = 2026/27.

Para La Liga da época atual, os jogos que o football-data.co.uk ainda não publicou
são completados com os resultados de fixturedownload.com (só golos, sem xG nem odds).
Quando o football-data.co.uk os publica, os dados dele substituem os da reserva.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import requests

from barca import CURRENT_SEASON

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{division}.csv"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
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
    url = BASE_URL.format(season=season, division=division)
    # O site às vezes demora a responder: tenta 3 vezes, com pausas crescentes.
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            return path
        except requests.RequestException as e:
            error = e
            time.sleep(10 * (attempt + 1))
    # Se continuar em baixo, usa a última cópia descarregada (a do dia anterior, no
    # GitHub Actions, graças à cache). Os dados ficam um dia atrasados em vez de falhar.
    if path.exists():
        print(f"Aviso: {url} não respondeu ({error}); a usar a cópia guardada.", file=sys.stderr)
        return path
    raise error


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
    df["source"] = "football-data"
    if season == CURRENT_SEASON and division == "SP1":
        try:
            df = add_missing_results(df, fetch_fixturedownload())
        except Exception as e:  # a reserva nunca deve impedir o resto de correr
            print(f"Aviso: sem resultados de reserva do fixturedownload.com ({e})", file=sys.stderr)
    return df.sort_values("date").reset_index(drop=True)


def fetch_fixturedownload() -> pd.DataFrame:
    """Calendário completo de La Liga em fixturedownload.com, com os nomes de football-data.co.uk.

    As horas vêm em UTC; a data de cada jogo é a de Madrid, como em football-data.co.uk.
    Jogos por disputar (ou adiados) têm os golos a NaN.
    """
    req = urllib.request.Request(FIXTUREDOWNLOAD_URL, headers={"User-Agent": "barca-season-predictor"})
    for attempt in range(3):
        try:
            games = pd.DataFrame(json.load(urllib.request.urlopen(req, timeout=60)))
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    unknown = (set(games["HomeTeam"]) | set(games["AwayTeam"])) - set(FIXTUREDOWNLOAD_NAMES)
    if unknown:
        raise ValueError(f"Nomes de equipas sem correspondência em FIXTUREDOWNLOAD_NAMES: {sorted(unknown)}")
    return pd.DataFrame({
        "date": pd.to_datetime(games["DateUtc"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None).dt.normalize(),
        "home": games["HomeTeam"].map(FIXTUREDOWNLOAD_NAMES),
        "away": games["AwayTeam"].map(FIXTUREDOWNLOAD_NAMES),
        "home_goals": games["HomeTeamScore"],
        "away_goals": games["AwayTeamScore"],
        "round": games["RoundNumber"],
    })


def add_missing_results(df: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a `df` os jogos já disputados em `fallback` que `df` ainda não tem.

    Em La Liga cada par casa/fora joga uma vez por época, por isso o par identifica o jogo.
    """
    have = set(zip(df["home"], df["away"]))
    extra = fallback.dropna(subset=["home_goals", "away_goals"])
    extra = extra[[(h, a) not in have for h, a in zip(extra["home"], extra["away"])]]
    extra = extra.drop(columns="round", errors="ignore").copy()
    if extra.empty:
        return df
    extra[["home_goals", "away_goals"]] = extra[["home_goals", "away_goals"]].astype(int)
    extra["season"] = df["season"].iloc[0] if len(df) else CURRENT_SEASON
    extra["division"] = "SP1"
    extra["source"] = "fixturedownload"
    return pd.concat([df, extra], ignore_index=True)


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

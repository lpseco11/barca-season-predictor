"""Regista as previsões dos próximos jogos de La Liga, uma única vez, ANTES dos jogos.

Pensado para correr todos os dias (GitHub Actions):
- vai buscar os próximos jogos a football-data.co.uk;
- ignora jogos já disputados e jogos que já têm previsão registada;
- se sobrar algum, treina o modelo e guarda as previsões;
- se não houver nada de novo, termina sem erro e sem alterar ficheiros.

Uso: .venv/bin/python scripts/predict_matchday.py

Se os jogos não forem publicados a tempo, cria predictions/fixtures_manual.csv:
  date,home,away
  10/10/2026,Barcelona,Getafe
(o ficheiro manual tem prioridade; apaga-o depois de usar.)
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from barca import CURRENT_SEASON
from barca.data import load_matches
from barca.model import fit
from barca.predict import load_fixtures, new_fixtures, predict_fixtures

SEASONS = ["2122", "2223", "2324", "2425", "2526", CURRENT_SEASON]
PRED_DIR = Path(__file__).resolve().parents[1] / "predictions"
MANUAL = PRED_DIR / "fixtures_manual.csv"
KEY = ["date", "home", "away"]


def already_predicted() -> pd.DataFrame:
    files = sorted(PRED_DIR.glob("matchday_*.csv"))
    if not files:
        return pd.DataFrame(columns=KEY)
    return pd.concat([pd.read_csv(f, parse_dates=["date"])[KEY] for f in files], ignore_index=True)


if __name__ == "__main__":
    fixtures = load_fixtures(manual=MANUAL)
    new = new_fixtures(fixtures, already_predicted(), today=pd.Timestamp(datetime.now().date()))
    if new.empty:
        print("Sem jogos novos de La Liga para prever.")
        raise SystemExit(0)

    matches = load_matches(SEASONS)
    model = fit(matches)
    preds = predict_fixtures(model, new.reset_index(drop=True))
    preds.insert(0, "created_at", datetime.now().isoformat(timespec="seconds"))

    # Um ficheiro por jornada, com o nome da data do primeiro jogo. Jogos publicados
    # mais tarde para a mesma jornada são acrescentados ao mesmo ficheiro.
    first = preds["date"].min()
    files = sorted(PRED_DIR.glob("matchday_*.csv"))
    out = PRED_DIR / f"matchday_{first:%Y-%m-%d}.csv"
    if files:
        last = files[-1]
        last_games = pd.read_csv(last, parse_dates=["date"])
        if (first - last_games["date"].max()).days <= 2:
            out = last
            preds = pd.concat([last_games, preds], ignore_index=True)
    PRED_DIR.mkdir(exist_ok=True)
    preds.to_csv(out, index=False)

    show = preds[["date", "home", "away", "p_home", "p_draw", "p_away", "likely_score"]].copy()
    show["date"] = show["date"].dt.strftime("%d/%m")
    for c in ["p_home", "p_draw", "p_away"]:
        show[c] = (show[c] * 100).round().astype(int).astype(str) + "%"
    print(show.rename(columns={"p_home": "1", "p_draw": "X", "p_away": "2"}).to_string(index=False))
    print(f"\n{len(new)} jogo(s) novo(s) guardado(s) em {out.relative_to(PRED_DIR.parent)}")

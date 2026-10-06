"""Regista as previsões da próxima jornada ANTES dos jogos.

Uso:
  .venv/bin/python scripts/predict_matchday.py            # jogos de football-data.co.uk
  .venv/bin/python scripts/predict_matchday.py --force    # reescrever depois do 1º jogo

Se os jogos ainda não estiverem publicados, cria predictions/fixtures_manual.csv:
  date,home,away
  10/10/2026,Barcelona,Getafe
(o ficheiro manual tem prioridade; apaga-o depois de usar.)
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from barca import CURRENT_SEASON
from barca.data import load_matches
from barca.model import fit
from barca.predict import load_fixtures, predict_fixtures

SEASONS = ["2122", "2223", "2324", "2425", "2526", CURRENT_SEASON]
PRED_DIR = Path(__file__).resolve().parents[1] / "predictions"
MANUAL = PRED_DIR / "fixtures_manual.csv"


if __name__ == "__main__":
    force = "--force" in sys.argv
    fixtures = load_fixtures(manual=MANUAL)
    if fixtures.empty:
        sys.exit("Sem jogos de La Liga publicados ainda. Tenta mais tarde ou cria "
                 f"{MANUAL.relative_to(PRED_DIR.parent)}.")

    first_game = fixtures["date"].min()
    out = PRED_DIR / f"matchday_{first_game:%Y-%m-%d}.csv"
    # Uma previsão só vale se for feita antes dos jogos: não reescrever depois do 1º jogo.
    if out.exists() and datetime.now() >= first_game and not force:
        sys.exit(f"{out.name} já existe e a jornada já começou. Usa --force se tiveres a certeza.")

    matches = load_matches(SEASONS)
    model = fit(matches)
    preds = predict_fixtures(model, fixtures)
    preds.insert(0, "created_at", datetime.now().isoformat(timespec="seconds"))
    PRED_DIR.mkdir(exist_ok=True)
    preds.to_csv(out, index=False)

    show = preds[["date", "home", "away", "xg_home", "xg_away", "p_home", "p_draw", "p_away", "likely_score"]].copy()
    show["date"] = show["date"].dt.strftime("%d/%m")
    for c in ["p_home", "p_draw", "p_away"]:
        show[c] = (show[c] * 100).round().astype(int).astype(str) + "%"
    pd.set_option("display.width", 120)
    print(show.round(2).rename(columns={"p_home": "1", "p_draw": "X", "p_away": "2"}).to_string(index=False))
    print(f"\nGuardado em {out.relative_to(PRED_DIR.parent)}")

"""Compara as previsões registadas com os resultados reais (e com as odds).

Uso: .venv/bin/python scripts/evaluate_predictions.py
"""

from pathlib import Path

import pandas as pd

from barca import CURRENT_SEASON
from barca.data import load_season
from barca.predict import evaluate_logged

PRED_DIR = Path(__file__).resolve().parents[1] / "predictions"


if __name__ == "__main__":
    files = sorted(PRED_DIR.glob("matchday_*.csv"))
    if not files:
        raise SystemExit("Ainda não há previsões registadas.")
    preds = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    merged, metrics = evaluate_logged(preds, load_season(CURRENT_SEASON))
    if not metrics:
        raise SystemExit("Nenhum dos jogos previstos tem resultado ainda.")

    show = merged[["date", "home", "away", "p_home", "p_draw", "p_away", "likely_score", "home_goals", "away_goals"]]
    print(show.round(2).to_string(index=False), "\n")
    for k, v in metrics.items():
        print(f"{k:16s} {v:.4f}" if isinstance(v, float) else f"{k:16s} {v}")
    print("\n(RPS e log loss: menor é melhor. No backtest: modelo 0.194 vs odds 0.189.)")

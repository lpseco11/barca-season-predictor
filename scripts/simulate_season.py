"""Simula o resto de La Liga 26/27 e regista a evolução das probabilidades.

Uso: .venv/bin/python scripts/simulate_season.py [n_sims]
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from barca import CURRENT_SEASON, TEAM
from barca.data import load_matches
from barca.model import fit
from barca.simulate import simulate, summary, table

SEASONS = ["2122", "2223", "2324", "2425", "2526", CURRENT_SEASON]
ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "predictions" / "season_history.csv"

# Recordes de La Liga (Real Madrid 2011/12 e Barça 2012/13: 100 pontos; Real Madrid 2011/12: 121 golos).
RECORD_POINTS, RECORD_GOALS = 100, 121


if __name__ == "__main__":
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    matches = load_matches(SEASONS)
    played = matches[(matches["season"] == CURRENT_SEASON) & (matches["division"] == "SP1")]

    model = fit(matches, with_cov=True)
    sims = simulate(model, played, n_sims=n_sims, seed=42)
    s = summary(sims)

    pd.set_option("display.width", 120)
    print(f"Jogos disputados: {len(played)} / 380 (até {played['date'].max():%d/%m/%Y})\n")
    print("Classificação atual (top 6):")
    print(table(played).head(6).to_string(), "\n")
    print(f"Projeção final ({n_sims:,} simulações):")
    out = s.copy()
    for c in ["p_titulo", "p_top4", "p_descida"]:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    print(out.round(1).to_string(index=False))

    k = sims["teams"].index(TEAM)
    pts, goals = sims["points"][:, k], sims["gf"][:, k]
    lo, hi = np.percentile(pts, [5, 95])
    print(f"\n{TEAM}:")
    print(f"  pontos: média {pts.mean():.1f} (90% entre {lo:.0f} e {hi:.0f})")
    print(f"  golos marcados: média {goals.mean():.0f}")
    print(f"  P(≥{RECORD_POINTS} pontos) = {(pts >= RECORD_POINTS).mean():.1%}")
    print(f"  P(≥{RECORD_GOALS} golos)  = {(goals >= RECORD_GOALS).mean():.1%}")

    row = s.set_index("team").loc[TEAM]
    entry = pd.DataFrame([{
        "run_date": date.today().isoformat(),
        "data_until": played["date"].max().date().isoformat(),
        "games_played": len(played),
        "p_title": row["p_titulo"],
        "expected_points": row["pontos_esperados"],
        "p_100_points": (pts >= RECORD_POINTS).mean(),
        "expected_goals": goals.mean(),
    }])
    HISTORY.parent.mkdir(exist_ok=True)
    history = pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame()
    # Uma linha por estado dos dados: correr de novo sem jogos novos substitui a linha.
    if not history.empty:
        history = history[history["data_until"] != entry["data_until"].iloc[0]]
    pd.concat([history, entry], ignore_index=True).to_csv(HISTORY, index=False)
    print(f"\nHistórico atualizado: {HISTORY.relative_to(ROOT)}")

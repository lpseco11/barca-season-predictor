"""Corre o backtest para vários valores de ξ e guarda os resultados.

Uso: .venv/bin/python scripts/run_backtest.py
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from barca.backtest import backtest, evaluate
from barca.data import load_matches

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
TEST_SEASONS = ["2324", "2425", "2526"]
XIS = [0.0005, 0.001, 0.0019, 0.003, 0.005]
OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def run(xi: float) -> pd.DataFrame:
    matches = load_matches(SEASONS)
    preds = backtest(matches, TEST_SEASONS, xi=xi)
    preds.to_csv(OUT / f"backtest_xi{xi}.csv", index=False)
    result = evaluate(preds)
    result["xi"] = xi
    return result


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    load_matches(SEASONS)  # descarrega antes de paralelizar
    with ProcessPoolExecutor() as pool:
        results = pd.concat(pool.map(run, XIS), ignore_index=True)
    results.to_csv(OUT / "backtest_summary.csv", index=False)

    pd.set_option("display.width", 120)
    print("\nPor época (ξ por defeito = 0.0019):")
    print(results[results["xi"] == 0.0019].drop(columns="xi").round(4).to_string(index=False))
    print("\nTotal das 3 épocas, por ξ:")
    total = (results.assign(w_ll=results.log_loss * results.jogos, w_rps=results.rps * results.jogos,
                            w_acc=results.acerto * results.jogos)
             .groupby(["xi", "fonte"])[["w_ll", "w_rps", "w_acc", "jogos"]].sum())
    total = pd.DataFrame({"log_loss": total.w_ll / total.jogos, "rps": total.w_rps / total.jogos,
                          "acerto": total.w_acc / total.jogos}).round(4)
    print(total.to_string())

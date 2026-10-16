"""Visita guiada ao projeto: mostra o resultado intermédio de cada parte.

Uso:
  .venv/bin/python scripts/explorar.py          # todos os passos
  .venv/bin/python scripts/explorar.py 3        # só o passo 3
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from barca import CURRENT_SEASON, TEAM
from barca.backtest import evaluate
from barca.data import load_matches, load_season, team_matches
from barca.model import fit
from barca.simulate import simulate, summary, table

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ["2122", "2223", "2324", "2425", "2526", CURRENT_SEASON]
pd.set_option("display.width", 140)


def titulo(n: int, texto: str) -> None:
    print(f"\n{'=' * 70}\nPASSO {n}: {texto}\n{'=' * 70}")


def passo_1():
    titulo(1, "os dados (data.py)")
    epoca = load_season(CURRENT_SEASON)
    print(f"Jogos de La Liga 26/27 com resultado: {len(epoca)}")
    print("Últimos 5 jogos, já com nomes de colunas legíveis:\n")
    print(epoca[["date", "home", "away", "home_goals", "away_goals", "home_xg", "away_xg"]].tail(5).to_string(index=False))
    todos = load_matches(SEASONS)
    print(f"\nPara treinar o modelo juntam-se {len(todos)} jogos (La Liga e Segunda, desde 2021/22).")
    print(f"\nOs jogos do {TEAM} na perspetiva do {TEAM} (team_matches):\n")
    print(team_matches(epoca, TEAM)[["date", "opponent", "venue", "goals_for", "goals_against", "result"]].tail(5).to_string(index=False))


def passo_2():
    titulo(2, "o modelo aprende as forças das equipas (model.fit)")
    modelo = fit(load_matches(SEASONS))
    print(f"Vantagem de jogar em casa: x{np.exp(modelo.home_adv):.2f} golos")
    print(f"Correção Dixon-Coles para resultados baixos (rho): {modelo.rho:.3f}")
    liga = set(load_season(CURRENT_SEASON)["home"])
    r = modelo.ratings()
    r = r[r["team"].isin(liga)].head(6)
    print("\nAs 6 equipas mais fortes (ataque alto é bom; defesa negativa é boa):\n")
    print(r.round(2).to_string(index=False))
    return modelo


def passo_3(modelo=None):
    titulo(3, "de forças a probabilidades de um jogo (score_matrix)")
    modelo = modelo or fit(load_matches(SEASONS))
    casa, fora = "Betis", "Barcelona"
    lam, mu = modelo.expected_goals(casa, fora)
    print(f"{casa} – {fora}: golos esperados {lam:.2f} – {mu:.2f}\n")
    m = modelo.score_matrix(casa, fora)
    tabela = pd.DataFrame(m[:5, :5], index=[f"{casa} {i}" for i in range(5)], columns=[f"{fora} {j}" for j in range(5)])
    print("Probabilidade de cada resultado (linhas: golos da casa; colunas: golos de fora):\n")
    print((tabela * 100).round(1).to_string())
    p = modelo.outcome_probs(casa, fora)
    print(f"\nSomando a matriz: vitória {casa} {p['home']:.0%}, empate {p['draw']:.0%}, vitória {fora} {p['away']:.0%}")


def passo_4():
    titulo(4, "simular o resto da época (simulate.py)")
    jogos = load_matches(SEASONS)
    disputados = jogos[(jogos["season"] == CURRENT_SEASON) & (jogos["division"] == "SP1")]
    print("Classificação atual (top 4):\n")
    print(table(disputados).head(4).to_string())
    modelo = fit(jogos, with_cov=True)
    sims = simulate(modelo, disputados, n_sims=2000, seed=1)
    k = sims["teams"].index(TEAM)
    print(f"\n2000 épocas simuladas. Pontos finais do {TEAM} nas primeiras 10: {sims['points'][:10, k].astype(int).tolist()}")
    print("\nResumo (contar em quantas simulações aconteceu cada coisa):\n")
    s = summary(sims).head(4)[["team", "pontos_esperados", "p_titulo", "p_top4"]]
    print(s.round(3).to_string(index=False))


def passo_5():
    titulo(5, "a validação nas épocas passadas (backtest.py)")
    preds = pd.read_csv(ROOT / "predictions" / "backtest_2324-2526.csv", parse_dates=["date"], dtype={"season": str})
    print(f"Previsões feitas semana a semana, sem ver o futuro: {len(preds)} jogos\n")
    amostra = preds[["date", "home", "away", "home_goals", "away_goals", "p_home", "p_draw", "p_away"]].head(3)
    print(amostra.assign(date=amostra["date"].dt.date).round(2).to_string(index=False))
    ev = evaluate(preds)
    tot = ev.assign(w=ev.rps * ev.jogos).groupby("fonte")[["w", "jogos"]].sum()
    print("\nRPS total (quanto mais baixo, melhor):")
    for fonte, row in tot.iterrows():
        print(f"  {fonte:10s} {row.w / row.jogos:.4f}")


def passo_6():
    titulo(6, "as previsões registadas antes dos jogos (predictions/)")
    for f in sorted((ROOT / "predictions").glob("matchday_*.csv")):
        p = pd.read_csv(f)
        print(f"{f.name}: {len(p)} jogos, registados entre {p.created_at.min()} e {p.created_at.max()}")
    ultimo = sorted((ROOT / "predictions").glob("matchday_*.csv"))[-1]
    p = pd.read_csv(ultimo)
    print(f"\nConteúdo de {ultimo.name}:\n")
    print(p[["created_at", "date", "home", "away", "p_home", "p_draw", "p_away", "likely_score"]].round(2).to_string(index=False))


PASSOS = [passo_1, passo_2, passo_3, passo_4, passo_5, passo_6]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        PASSOS[int(sys.argv[1]) - 1]()
    else:
        passo_1()
        modelo = passo_2()
        passo_3(modelo)
        passo_4()
        passo_5()
        passo_6()

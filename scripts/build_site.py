"""Gera o site estático (docs/index.html) a partir do modelo e das previsões.

Corre o modelo, simula a época e junta tudo num JSON que é embutido no
template site/template.html. O GitHub Pages serve a pasta docs/.

Uso: .venv/bin/python scripts/build_site.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from barca import CURRENT_SEASON, TEAM
from barca.backtest import backtest, evaluate, outcome, rps
from barca.data import fetch_fixturedownload, load_matches, team_matches
from barca.model import DEFAULT_XI, fit
from barca.simulate import simulate, summary, table

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "site" / "template.html"
CRESTS = json.loads((ROOT / "site" / "crests.json").read_text(encoding="utf-8"))
OUT = ROOT / "docs" / "index.html"
PRED_DIR = ROOT / "predictions"
# Previsões do backtest (épocas passadas, não mudam). Ficam no repositório para o site
# ser igual em qualquer máquina, sem diferenças de vírgula flutuante.
BACKTEST_CSV = PRED_DIR / "backtest_2324-2526.csv"

SEASONS = ["2122", "2223", "2324", "2425", "2526", CURRENT_SEASON]
BACKTEST_SEASONS = ["2324", "2425", "2526"]
RECORD_POINTS, RECORD_GOALS = 100, 121

# Nomes de football-data.co.uk → nomes para o site.
NAMES = {
    "Ath Madrid": "Atlético de Madrid", "Ath Bilbao": "Athletic Club", "Sociedad": "Real Sociedad",
    "Vallecano": "Rayo Vallecano", "Espanol": "Espanyol", "La Coruna": "Deportivo",
    "Santander": "Racing Santander", "Celta": "Celta de Vigo", "Alaves": "Alavés",
    "Malaga": "Málaga", "Betis": "Real Betis", "Barcelona": "Barcelona",
}


def name(team: str) -> str:
    return NAMES.get(team, team)


def crest(team: str) -> dict | None:
    """Emblema do clube (caminho relativo a docs/ e dimensões), se existir."""
    c = CRESTS.get(team)
    return {"src": c["src"], "w": c["w"], "h": c["h"]} if c else None


def backtest_section() -> dict:
    """Métricas do backtest e dados de calibração (corre o backtest se faltar o CSV)."""
    if BACKTEST_CSV.exists():
        preds = pd.read_csv(BACKTEST_CSV, parse_dates=["date"], dtype={"season": str})
    else:
        preds = backtest(load_matches(["2122", "2223", *BACKTEST_SEASONS]), BACKTEST_SEASONS)
        BACKTEST_CSV.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(BACKTEST_CSV, index=False)

    ev = evaluate(preds)
    totals = (ev.assign(w=ev.rps * ev.jogos, a=ev.acerto * ev.jogos)
              .groupby("fonte")[["w", "a", "jogos"]].sum())
    # Arredondado: diferenças mínimas de vírgula flutuante entre máquinas não devem mudar o site.
    metrics = {f: {"rps": round(totals.loc[f, "w"] / totals.loc[f, "jogos"], 4),
                   "acerto": round(totals.loc[f, "a"] / totals.loc[f, "jogos"], 4)} for f in totals.index}

    y = outcome(preds)
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy().ravel()
    hits = np.eye(3)[y].ravel()
    bins = pd.cut(probs, np.linspace(0, 1, 11))
    calib = (pd.DataFrame({"p": probs, "hit": hits}).groupby(bins, observed=True)
             .agg(n=("hit", "size"), previsto=("p", "mean"), real=("hit", "mean")))
    calib = calib[calib["n"] >= 20]
    return {"games": int(len(preds)), "metrics": metrics,
            "calibration": calib.reset_index(drop=True).round(4).to_dict("records")}


def load_predictions() -> pd.DataFrame:
    files = sorted(PRED_DIR.glob("matchday_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)


def game_card(g: pd.Series) -> dict:
    return {
        "date": g["date"].strftime("%Y-%m-%d"), "home": name(g["home"]), "away": name(g["away"]),
        "home_crest": crest(g["home"]), "away_crest": crest(g["away"]),
        "barca": TEAM in (g["home"], g["away"]),
        "p": [round(g["p_home"], 4), round(g["p_draw"], 4), round(g["p_away"], 4)],
        "xg": [round(g["xg_home"], 2), round(g["xg_away"], 2)], "score": g["likely_score"],
        "p_score": round(g["p_likely_score"], 4),
    }


def matchday_sections(played: pd.DataFrame, rounds: dict) -> tuple[dict | None, dict | None]:
    """(próximos jogos já previstos, avaliação da última jornada com previsões e resultados).

    Um jogo previsto conta como "próximo" se ainda não tem resultado e é posterior ao
    último resultado conhecido; um jogo sem resultado com data anterior foi adiado.
    """
    preds = load_predictions()
    if preds.empty:
        return None, None
    preds["round"] = [rounds.get((h, a)) for h, a in zip(preds["home"], preds["away"])]
    results = played[["home", "away", "home_goals", "away_goals"]]
    merged = preds.merge(results, on=["home", "away"], how="left")
    last_result = played["date"].max()

    upcoming = merged[merged["home_goals"].isna() & (merged["date"] > last_result)]
    nxt = None
    if not upcoming.empty:
        rnd = upcoming["round"].mode()
        nxt = {"round": int(rnd.iloc[0]) if len(rnd) else None, "created_at": upcoming["created_at"].min(),
               "games": [game_card(g) for _, g in upcoming.sort_values("date").iterrows()]}

    done = merged.dropna(subset=["home_goals"])
    last = None
    if not done.empty:
        rnd = done["round"].dropna().max() if done["round"].notna().any() else None
        games = done[done["round"] == rnd] if rnd is not None else done
        y = outcome(games)
        P = games[["p_home", "p_draw", "p_away"]].to_numpy()
        rows = []
        for (_, g), yi, pi in zip(games.sort_values("date").iterrows(), y, P):
            card = game_card(g)
            hg, ag = int(g["home_goals"]), int(g["away_goals"])
            card.update(result=[hg, ag], pick=int(pi.argmax()), actual=int(yi), hit=bool(pi.argmax() == yi),
                        exact=g["likely_score"] == f"{hg}-{ag}")
            rows.append(card)
        last = {"round": int(rnd) if rnd is not None else None, "games": rows,
                "hits": int(sum(r["hit"] for r in rows)), "exact": int(sum(r["exact"] for r in rows)),
                "rps": round(rps(P, y), 4)}
    return nxt, last


def barca_next(rounds_calendar: pd.DataFrame | None, played: pd.DataFrame) -> dict | None:
    """Próximo jogo do Barcelona no calendário, mesmo que ainda não tenha previsão."""
    if rounds_calendar is None:
        return None
    cal = rounds_calendar[rounds_calendar["home_goals"].isna() & (rounds_calendar["date"] > played["date"].max())]
    cal = cal[(cal["home"] == TEAM) | (cal["away"] == TEAM)].sort_values("date")
    if cal.empty:
        return None
    g = cal.iloc[0]
    return {"date": g["date"].strftime("%Y-%m-%d"), "home": name(g["home"]), "away": name(g["away"]),
            "home_crest": crest(g["home"]), "away_crest": crest(g["away"]), "round": int(g["round"])}


def form(played: pd.DataFrame, team: str, n: int = 5) -> list[str]:
    """Últimos n resultados da equipa (V/E/D), do mais antigo para o mais recente."""
    return team_matches(played, team)["result"].tail(n).tolist()


def last_update(played: pd.DataFrame) -> str:
    dates = [played["date"].max().date()]
    for f in PRED_DIR.glob("matchday_*.csv"):
        dates.append(pd.to_datetime(pd.read_csv(f)["created_at"]).max().date())
    return max(dates).isoformat()


if __name__ == "__main__":
    matches = load_matches(SEASONS)
    played = matches[(matches["season"] == CURRENT_SEASON) & (matches["division"] == "SP1")]
    model = fit(matches, with_cov=True)
    sims = simulate(model, played, n_sims=10_000, seed=42)
    s = summary(sims)

    k = sims["teams"].index(TEAM)
    pts, goals = sims["points"][:, k], sims["gf"][:, k]
    champions = pd.Series(np.array(sims["teams"])[np.argmax(sims["position"] == 1, axis=1)]).value_counts(normalize=True)

    # Classificação atual: pontos, diferença de golos, golos marcados (La Liga desempata
    # primeiro pelo confronto direto; com poucos jogos a diferença é rara).
    current = table(played)
    position = {t: i + 1 for i, t in enumerate(current.index)}
    rows = []
    for proj_rank, (_, r) in enumerate(s.iterrows(), start=1):
        t = r["team"]
        rows.append({
            "team": name(t), "barca": t == TEAM, "crest": crest(t), "form": form(played, t),
            "pos_now": position[t], "proj_rank": proj_rank,
            "points_now": int(current.loc[t, "points"]), "played": int(current.loc[t, "played"]),
            "gd": int(current.loc[t, "gd"]), "gf": int(current.loc[t, "gf"]),
            "exp_points": round(r["pontos_esperados"], 1),
            "p5": float(r["pontos_p5"]), "p95": float(r["pontos_p95"]),
            "p_title": round(r["p_titulo"], 4), "p_top4": round(r["p_top4"], 4), "p_releg": round(r["p_descida"], 4),
        })

    try:
        calendar = fetch_fixturedownload()
    except Exception as e:  # sem calendário, o site continua a funcionar sem estas partes
        print(f"Aviso: sem calendário do fixturedownload.com ({e})")
        calendar = None
    rounds = {} if calendar is None else dict(zip(zip(calendar["home"], calendar["away"]), calendar["round"]))
    next_md, last_md = matchday_sections(played, rounds)

    bm = team_matches(played, TEAM)
    history = pd.read_csv(PRED_DIR / "season_history.csv") if (PRED_DIR / "season_history.csv").exists() else pd.DataFrame()

    data = {
        # Data da última novidade (resultados ou previsões), e não do dia em que o script corre:
        # assim o site só muda quando há dados novos.
        "updated": last_update(played),
        "data_until": played["date"].max().strftime("%Y-%m-%d"),
        "games_played": int(len(played)),
        "matchday": math.ceil(len(played) / 10),  # jornada em curso ou acabada de jogar
        "title": [{"team": name(t), "barca": t == TEAM, "p": round(p, 4)} for t, p in champions.items()],
        "barca": {
            "exp_points": round(pts.mean(), 1), "p5": float(np.percentile(pts, 5)), "p95": float(np.percentile(pts, 95)),
            "p_100": round((pts >= RECORD_POINTS).mean(), 4), "exp_goals": round(goals.mean()),
            "p_goal_record": round((goals >= RECORD_GOALS).mean(), 4),
            "matches": [{"date": m["date"].strftime("%Y-%m-%d"), "opponent": name(m["opponent"]),
                         "crest": crest(m["opponent"]), "venue": m["venue"],
                         "gf": int(m["goals_for"]), "ga": int(m["goals_against"]),
                         # Jogos vindos da reserva (fixturedownload.com) ainda não têm xG.
                         "xgf": None if pd.isna(m["xg_for"]) else float(m["xg_for"]),
                         "xga": None if pd.isna(m["xg_against"]) else float(m["xg_against"])} for _, m in bm.iterrows()],
        },
        "table": rows,
        "history": history.to_dict("records"),
        "next_matchday": next_md,
        "last_matchday": last_md,
        "barca_next": barca_next(calendar, played),
        "backtest": backtest_section(),
    }

    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    (OUT.parent / ".nojekyll").touch()
    print(f"Site gerado: {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KB)")

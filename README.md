# Barça 26/27 — Season Tracker & Predictor

As goleadas do Barça são sustentáveis? Como vai acabar a época?

Projeto que acompanha o FC Barcelona em La Liga 2026/27 jornada a jornada:

1. **Desempenho vs xG** — o Barça marca o que "devia" marcar, ou está com sorte?
2. **Modelo de previsão** — Dixon-Coles (Poisson) com decaimento temporal, força ofensiva/defensiva de cada equipa.
3. **Simulação Monte Carlo** — 10 000 simulações do resto da época → probabilidade de título, pontos esperados.
4. **Validação** — previsões registadas *antes* de cada jornada e comparadas com as odds das casas de apostas.
5. **Dashboard** (Streamlit) e relatório semanal.

## Dados

[football-data.co.uk](https://www.football-data.co.uk/) — resultados, remates, xG (só 26/27) e odds.
Usamos La Liga (`SP1`) e Segunda (`SP2`), para os promovidos terem histórico.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/pytest
```

## Estrutura

```
src/barca/      código: data, model (Dixon-Coles), backtest, simulate, predict
scripts/        simulate_season, predict_matchday, evaluate_predictions, run_backtest, build_site
site/           template da página
docs/           site gerado (servido pelo GitHub Pages)
data/raw/       CSVs descarregados (não versionados)
predictions/    previsões registadas antes de cada jornada
tests/
```

Gerar o site: `.venv/bin/python scripts/build_site.py` → `docs/index.html`.

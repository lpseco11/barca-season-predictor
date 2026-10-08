"""Modelo Dixon-Coles (1997) com decaimento temporal.

Cada equipa tem uma força de ataque `a` e uma de defesa `d` (em escala log).
Os golos esperados num jogo casa vs fora são:

    λ (casa) = exp(c + h + a_casa + d_fora)
    μ (fora) = exp(c + a_fora + d_casa)

c = média da liga, h = vantagem de jogar em casa. Uma `d` negativa é boa
defesa (reduz os golos do adversário).

Os golos seguem distribuições de Poisson, com a correção τ de Dixon-Coles
para os resultados baixos (0-0, 1-0, 0-1, 1-1), que o Poisson puro estima mal.
O parâmetro ρ controla essa correção.

Os jogos mais antigos pesam menos: peso = exp(-ξ · dias_desde_o_jogo).
Os parâmetros são estimados por máxima verosimilhança (scipy L-BFGS-B).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Dixon & Coles estimaram ξ ≈ 0.0065 por meia-semana ≈ 0.0019/dia
# (um jogo de há 1 ano pesa ~50%). Afinamos isto depois no backtest.
DEFAULT_XI = 0.0019

# Ao fazer uma média das forças ao longo do tempo, o modelo comprime as diferenças
# entre equipas: no backtest, os favoritos ganhavam mais do que o previsto e os golos
# das equipas fortes ficavam abaixo do real. Esticar ataque e defesa por um fator
# comum corrige isso; 1,2 foi o melhor valor no backtest de 2023/24 a 2025/26
# (RPS 0,1942 → 0,1937 e favoritos calibrados).
DEFAULT_STRETCH = 1.2


@dataclass
class DixonColes:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    intercept: float
    home_adv: float
    rho: float
    # Covariância dos parâmetros (aproximação de Laplace: inversa da Hessiana).
    # Ordem: [c, h, ρ, ataques..., defesas...], igual ao vetor do otimizador.
    cov: np.ndarray | None = None

    def sample_params(self, k: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Sorteia k conjuntos de parâmetros plausíveis, dada a incerteza da estimativa.

        Sem covariância (ou k=0) devolve só a estimativa pontual (k=1).
        """
        n = len(self.teams)
        mean = np.concatenate([[self.intercept, self.home_adv, self.rho], self.attack, self.defence])
        if self.cov is None or k == 0:
            draws = mean[None, :]
        else:
            draws = rng.multivariate_normal(mean, self.cov, size=k, method="cholesky")
        return {
            "intercept": draws[:, 0],
            "home_adv": draws[:, 1],
            "rho": np.clip(draws[:, 2], -0.2, 0.2),
            "attack": draws[:, 3:3 + n],
            "defence": draws[:, 3 + n:],
        }

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        i, j = self.teams.index(home), self.teams.index(away)
        lam = np.exp(self.intercept + self.home_adv + self.attack[i] + self.defence[j])
        mu = np.exp(self.intercept + self.attack[j] + self.defence[i])
        return float(lam), float(mu)

    def score_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        """Matriz P[golos_casa, golos_fora]."""
        lam, mu = self.expected_goals(home, away)
        goals = np.arange(max_goals + 1)
        m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        m[0, 0] *= 1 - lam * mu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + mu * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum()

    def outcome_probs(self, home: str, away: str) -> dict[str, float]:
        m = self.score_matrix(home, away)
        return {
            "home": float(np.tril(m, -1).sum()),
            "draw": float(np.trace(m)),
            "away": float(np.triu(m, 1).sum()),
        }

    def ratings(self) -> pd.DataFrame:
        """Tabela de forças, ordenada pela diferença ataque - defesa."""
        df = pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence})
        df["strength"] = df["attack"] - df["defence"]
        return df.sort_values("strength", ascending=False).reset_index(drop=True)


def time_weights(dates: pd.Series, as_of: pd.Timestamp, xi: float) -> np.ndarray:
    days = (as_of - dates).dt.days.to_numpy()
    return np.exp(-xi * days)


def _neg_log_likelihood(params, home_idx, away_idx, x, y, w, n_teams):
    """-log L ponderada e o seu gradiente analítico (para o otimizador ser rápido)."""
    c, h, rho = params[0], params[1], params[2]
    att = params[3:3 + n_teams]
    dfn = params[3 + n_teams:]

    log_lam = c + h + att[home_idx] + dfn[away_idx]
    log_mu = c + att[away_idx] + dfn[home_idx]
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    # Correção τ: só afeta 0-0, 0-1, 1-0 e 1-1.
    tau = np.ones_like(lam)
    dtau_lam = np.zeros_like(lam)  # d log τ / d log λ
    dtau_mu = np.zeros_like(lam)   # d log τ / d log μ
    dtau_rho = np.zeros_like(lam)  # d log τ / d ρ

    s00 = (x == 0) & (y == 0)
    s01 = (x == 0) & (y == 1)
    s10 = (x == 1) & (y == 0)
    s11 = (x == 1) & (y == 1)

    lm = lam * mu
    tau[s00] = 1 - lm[s00] * rho
    tau[s01] = 1 + lam[s01] * rho
    tau[s10] = 1 + mu[s10] * rho
    tau[s11] = 1 - rho
    tau = np.maximum(tau, 1e-10)

    dtau_lam[s00] = -lm[s00] * rho / tau[s00]
    dtau_mu[s00] = -lm[s00] * rho / tau[s00]
    dtau_rho[s00] = -lm[s00] / tau[s00]
    dtau_lam[s01] = lam[s01] * rho / tau[s01]
    dtau_rho[s01] = lam[s01] / tau[s01]
    dtau_mu[s10] = mu[s10] * rho / tau[s10]
    dtau_rho[s10] = mu[s10] / tau[s10]
    dtau_rho[s11] = -1 / tau[s11]

    # log-verosimilhança (sem os termos log(x!), que não dependem dos parâmetros)
    ll = w * (np.log(tau) + x * log_lam - lam + y * log_mu - mu)

    # Identificabilidade: somar uma constante a todos os ataques e subtraí-la a `c`
    # dá o mesmo modelo. Penalizamos sum(att) e sum(def) para fixar a média em 0.
    penalty = att.sum() ** 2 + dfn.sum() ** 2
    nll = -ll.sum() + penalty

    g_lam = w * (x - lam + dtau_lam)
    g_mu = w * (y - mu + dtau_mu)
    grad = np.empty_like(params)
    grad[0] = -(g_lam.sum() + g_mu.sum())
    grad[1] = -g_lam.sum()
    grad[2] = -(w * dtau_rho).sum()
    grad[3:3 + n_teams] = -(np.bincount(home_idx, g_lam, n_teams)
                            + np.bincount(away_idx, g_mu, n_teams)) + 2 * att.sum()
    grad[3 + n_teams:] = -(np.bincount(away_idx, g_lam, n_teams)
                           + np.bincount(home_idx, g_mu, n_teams)) + 2 * dfn.sum()
    return nll, grad


def fit(matches: pd.DataFrame, as_of: pd.Timestamp | None = None, xi: float = DEFAULT_XI,
        with_cov: bool = False, season_decay: float = 1.0, stretch: float = DEFAULT_STRETCH) -> DixonColes:
    """Ajusta o modelo usando só jogos anteriores a `as_of` (por defeito, todos).

    `with_cov=True` calcula também a incerteza dos parâmetros (para a simulação).
    `season_decay` < 1 dá ainda menos peso às épocas anteriores (o plantel muda
    no verão): um jogo de há k épocas pesa season_decay**k, além do decaimento diário.
    `stretch` multiplica as forças de ataque e defesa depois do ajuste (ver DEFAULT_STRETCH).
    """
    if as_of is None:
        as_of = matches["date"].max() + pd.Timedelta(days=1)
    df = matches[matches["date"] < as_of]

    teams = sorted(set(df["home"]) | set(df["away"]))
    index = {t: k for k, t in enumerate(teams)}
    n = len(teams)
    home_idx = df["home"].map(index).to_numpy()
    away_idx = df["away"].map(index).to_numpy()
    x = df["home_goals"].to_numpy(dtype=float)
    y = df["away_goals"].to_numpy(dtype=float)
    w = time_weights(df["date"], as_of, xi)
    if season_decay != 1.0:
        seasons = sorted(df["season"].unique())
        back = (len(seasons) - 1) - df["season"].map({sn: i for i, sn in enumerate(seasons)}).to_numpy()
        w = w * season_decay ** back

    x0 = np.zeros(3 + 2 * n)
    x0[0] = np.log(max((x * w).sum() / w.sum(), 0.1))
    bounds = [(None, None), (None, None), (-0.2, 0.2)] + [(None, None)] * (2 * n)
    res = minimize(_neg_log_likelihood, x0, args=(home_idx, away_idx, x, y, w, n),
                   jac=True, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Otimização falhou: {res.message}")

    p = res.x
    args = (home_idx, away_idx, x, y, w, n)
    cov = np.linalg.inv(_hessian(p, args)) if with_cov else None
    if stretch != 1.0:
        scale = np.ones_like(p)
        scale[3:] = stretch
        p = p * scale
        if cov is not None:
            cov = cov * np.outer(scale, scale)
    return DixonColes(teams=teams, intercept=p[0], home_adv=p[1], rho=p[2],
                      attack=p[3:3 + n], defence=p[3 + n:], cov=cov)


def _hessian(p: np.ndarray, args: tuple, eps: float = 1e-5) -> np.ndarray:
    """Hessiana por diferenças finitas centrais do gradiente analítico."""
    hess = np.empty((p.size, p.size))
    for k in range(p.size):
        step = np.zeros_like(p)
        step[k] = eps
        hess[k] = (_neg_log_likelihood(p + step, *args)[1] - _neg_log_likelihood(p - step, *args)[1]) / (2 * eps)
    return (hess + hess.T) / 2

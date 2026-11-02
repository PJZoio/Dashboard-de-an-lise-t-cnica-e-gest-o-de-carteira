"""Regras de alocação. Cada regra recebe APENAS fechos até à véspera da execução.

Interface: target(closes_hist, ctx) -> pd.Series de pesos (soma <= 1; o resto
fica em liquidez). `closes_hist` termina no fecho anterior à abertura em que a
ordem é executada, por isso não há look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TD = 252


# ----------------------------------------------------------------- estimadores
def ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    """Covariância com encolhimento Ledoit-Wolf (2004) para identidade escalada.

    Reduz o erro de estimação, que é o principal problema de otimizadores de
    média-variância com janelas curtas e ativos muito correlacionados.
    """
    X = returns.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    t, n = X.shape
    S = X.T @ X / t
    mu = np.trace(S) / n
    F = mu * np.eye(n)
    d2 = np.sum((S - F) ** 2)
    Y = X ** 2
    b2 = np.sum(Y.T @ Y / t - S ** 2) / t
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * F + (1 - delta) * S


def _cap_feasible(n: int, cap: float) -> float:
    return max(cap, 1.0 / n + 1e-9)


def min_variance(cov: np.ndarray, cap: float) -> np.ndarray:
    n = cov.shape[0]
    cap = _cap_feasible(n, cap)
    w0 = np.full(n, 1 / n)
    res = minimize(lambda w: w @ cov @ w, w0, jac=lambda w: 2 * cov @ w, method='SLSQP',
                   bounds=[(0, cap)] * n, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
                   options={'maxiter': 500, 'ftol': 1e-12})
    w = np.clip(res.x if res.success else w0, 0, None)
    return w / w.sum()


def risk_parity(cov: np.ndarray, cap: float) -> np.ndarray:
    """Equal risk contribution via formulação convexa (log-barreira, Spinu 2013)."""
    n = cov.shape[0]
    b = np.full(n, 1 / n)
    x0 = 1 / np.sqrt(np.diag(cov))
    res = minimize(lambda x: 0.5 * x @ cov @ x - b @ np.log(x), x0,
                   jac=lambda x: cov @ x - b / x, method='L-BFGS-B',
                   bounds=[(1e-10, None)] * n)
    w = res.x / res.x.sum()
    cap = _cap_feasible(n, cap)
    if w.max() > cap + 1e-6:
        # Com limite ativo deixa de existir ERC exato: aproxima-se o mais possível.
        def obj(w):
            rc = w * (cov @ w)
            return float(((rc / rc.sum() - b) ** 2).sum())
        r2 = minimize(obj, np.clip(w, 0, cap), method='SLSQP', bounds=[(0, cap)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
                      options={'maxiter': 500, 'ftol': 1e-12})
        w = np.clip(r2.x, 0, None)
        w /= w.sum()
    return w


def project_capped(w: np.ndarray, cap: float, total: float = 1.0) -> np.ndarray:
    """Limita pesos a `cap` redistribuindo o excesso proporcionalmente."""
    w = w.copy().astype(float)
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w * (total / w.sum()) if w.sum() > 0 else w


def momentum_score(closes: pd.DataFrame, windows=(63, 126, 252), skip: int = 0) -> pd.Series:
    """Média dos retornos a 3, 6 e 12 meses (opcionalmente saltando o último mês)."""
    end = len(closes) - 1 - skip
    vals = []
    for n in windows:
        if end - n < 0:
            raise ValueError('Histórico insuficiente para o momentum.')
        vals.append(closes.iloc[end] / closes.iloc[end - n] - 1)
    return pd.concat(vals, axis=1).mean(axis=1)


# --------------------------------------------------------------------- regras
@dataclass
class Ctx:
    tickers: list[str]
    base_weights: pd.Series                   # carteira de referência (soma 1)
    lookback: int = 252                       # sessões para covariância
    cap: float = 0.40                         # peso máximo por ETF
    top_k: int = 3                            # rotação: nº de ETFs
    defensive: str | None = None              # ativo de refúgio (ou None = liquidez)
    abs_filter: bool = True                   # rotação: exigir momentum > 0
    signals: dict[str, pd.Series] = field(default_factory=dict)  # posição técnica 0/1 por ETF


def _risky(ctx: Ctx) -> list[str]:
    return [t for t in ctx.tickers if t != ctx.defensive]


def _fill_defensive(w: pd.Series, ctx: Ctx) -> pd.Series:
    w = w.reindex(ctx.tickers).fillna(0.0)
    rest = max(0.0, 1 - w.sum())
    if ctx.defensive and rest > 1e-12:
        w[ctx.defensive] += rest
    return w


def t_buy_hold(closes: pd.DataFrame, ctx: Ctx) -> pd.Series:
    return ctx.base_weights.reindex(ctx.tickers).fillna(0.0)


def _cov(closes: pd.DataFrame, cols: list[str], lookback: int) -> np.ndarray:
    r = closes[cols].iloc[-(lookback + 1):].pct_change().dropna()
    if len(r) < min(126, lookback):
        raise ValueError('Histórico insuficiente para a covariância.')
    return ledoit_wolf(r) * TD


def t_min_var(closes: pd.DataFrame, ctx: Ctx) -> pd.Series:
    cols = ctx.tickers
    return pd.Series(min_variance(_cov(closes, cols, ctx.lookback), ctx.cap), index=cols)


def t_risk_parity(closes: pd.DataFrame, ctx: Ctx) -> pd.Series:
    cols = ctx.tickers
    return pd.Series(risk_parity(_cov(closes, cols, ctx.lookback), ctx.cap), index=cols)


def t_momentum(closes: pd.DataFrame, ctx: Ctx) -> pd.Series:
    """Rotação: top-K por momentum 3/6/12m, pesos ∝ 1/volatilidade 63d.

    Filtro absoluto: um ETF só entra se o momentum for > 0; as vagas não
    preenchidas vão para o ativo defensivo (ou liquidez). É a regra que,
    historicamente, limita perdas em mercados em queda prolongada.
    """
    risky = _risky(ctx)
    k = max(1, min(ctx.top_k, len(risky)))
    mom = momentum_score(closes[risky])
    ranked = mom.sort_values(ascending=False)
    chosen = list(ranked.index[:k])
    if ctx.abs_filter:
        chosen = [t for t in chosen if ranked[t] > 0]
    w = pd.Series(0.0, index=ctx.tickers)
    if chosen:
        vol = closes[chosen].iloc[-64:].pct_change().dropna().std()
        iv = 1 / vol.clip(lower=1e-6)
        share = len(chosen) / k
        w[chosen] = project_capped((iv / iv.sum()).to_numpy() * share, ctx.cap, share)
    return _fill_defensive(w, ctx)


def t_tech_filter(closes: pd.DataFrame, ctx: Ctx) -> pd.Series:
    """Pesos de referência, mas só nos ETF com posição técnica = 1 na véspera."""
    day = closes.index[-1]
    w = ctx.base_weights.reindex(ctx.tickers).fillna(0.0).copy()
    for t in _risky(ctx):
        pos = ctx.signals.get(t)
        on = float(pos.asof(day)) if pos is not None and len(pos) else 0.0
        if not on >= 0.5:
            w[t] = 0.0
    if ctx.defensive:
        w[ctx.defensive] = 0.0
    return _fill_defensive(w, ctx)


STRATEGIES = {
    'bh': ('Buy & Hold (referência)', t_buy_hold, 'none'),
    'rebal': ('Rebalanceamento periódico', t_buy_hold, 'sched'),
    'minvar': ('Mínima variância', t_min_var, 'sched'),
    'erc': ('Paridade de risco', t_risk_parity, 'sched'),
    'mom': ('Rotação por momentum', t_momentum, 'sched'),
    'tech': ('Filtro técnico (score)', t_tech_filter, 'sched'),
}

"""Robust portfolio weight estimation from aligned price histories.

Research only; not an order execution system. All prices must be in the same
valuation currency and consistently total-return adjusted.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def returns_from_prices(prices: pd.DataFrame, lookback: int = 756) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2:
        raise ValueError('Escolha pelo menos dois ETFs.')
    p = prices.apply(pd.to_numeric, errors='coerce').sort_index()
    p = p.loc[:, ~p.columns.duplicated()]
    if p.isna().any().any():
        # No forward fill across missing trading sessions or listing history.
        p = p.dropna(how='any')
    p = p.tail(lookback + 1)
    if len(p) < 253 or (p <= 0).any().any():
        raise ValueError('Necessárias pelo menos 253 cotações comuns, positivas e válidas.')
    ret = p.pct_change(fill_method=None).dropna()
    if not np.isfinite(ret.values).all():
        raise ValueError('Retornos inválidos.')
    return ret


def optimize_weights(returns: pd.DataFrame, method: str = 'min_vol', max_weight: float = .40,
                     ridge: float = .03, turnover_penalty: float = 0.0,
                     current_weights: dict | None = None) -> pd.Series:
    """Long-only, fully invested, with a stabilised covariance estimate.

    method: min_vol | risk_parity | equal. The ridge is a convex blend toward
    diagonal covariance, to reduce sensitivity to historical correlations.
    """
    R = returns.dropna().astype(float)
    n = R.shape[1]
    if n < 2 or len(R) < 252:
        raise ValueError('São necessários dois ativos e pelo menos 252 retornos comuns.')
    if max_weight * n < 1 - 1e-8:
        raise ValueError(f'Limite {max_weight:.0%} inviável para {n} ETFs. Aumente o limite ou o cabaz.')
    if not 0 <= ridge <= 1:
        raise ValueError('ridge deve situar-se entre 0 e 1.')
    if method == 'equal':
        return pd.Series(np.ones(n)/n,index=R.columns,name='peso')
    cov = np.asarray(R.cov()) * TRADING_DAYS
    cov = (1-ridge)*cov + ridge*np.diag(np.diag(cov))
    cov += np.eye(n)*1e-10
    current = (pd.Series(current_weights or {},dtype=float).reindex(R.columns).fillna(0).to_numpy()
               if current_weights is not None else np.ones(n)/n)
    current = np.maximum(0,current)
    if current.sum() > 0: current /= current.sum()
    w0=np.ones(n)/n
    def objective(w):
        v = max(float(w@cov@w),1e-12)
        penalty = turnover_penalty*float(((w-current)**2).sum())
        if method=='min_vol': return v+penalty
        if method=='risk_parity':
            contribution = w*(cov@w)
            target = v/n
            return float(((contribution-target)**2).sum())/(v*v)+penalty
        raise ValueError('Método desconhecido.')
    res=minimize(objective,w0,method='SLSQP',bounds=[(0,max_weight)]*n,
                 constraints=[{'type':'eq','fun':lambda w: float(w.sum()-1)}],
                 options={'maxiter':1500,'ftol':1e-11})
    if not res.success:
        raise RuntimeError('Otimização não convergiu: '+res.message)
    w=np.maximum(0,res.x)
    w/=w.sum()
    return pd.Series(w,index=R.columns,name='peso')


def diagnostics(returns: pd.DataFrame, weights: pd.Series) -> dict:
    w=weights.reindex(returns.columns).fillna(0).to_numpy()
    r=returns.to_numpy()@w
    vol=float(np.std(r,ddof=1)*np.sqrt(TRADING_DAYS))
    corr=returns.corr()
    return {'volatilidade_anualizada':vol,
            'numero_efetivo_ativos':float(1/np.sum(w*w)),
            'maior_peso':float(np.max(w)),
            'retorno_historico_anualizado':float(np.expm1(np.log1p(r).mean()*TRADING_DAYS)),
            'correlacao_media':float(corr.where(~np.eye(len(corr),dtype=bool)).stack().mean())}

"""Simulador walk-forward único para todas as estratégias.

Relógio: decisão com o fecho de D-1, execução na abertura de D, avaliação ao
fecho. Unidades fracionárias. Custos (comissão + slippage) sobre o valor
negociado. Entre rebalanceamentos as unidades ficam fixas (os pesos derivam).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Panel
from .strategies import Ctx

TD = 252


@dataclass
class Result:
    name: str
    nav: pd.Series
    weights: pd.DataFrame        # pesos diários efetivos (inclui 'Liquidez')
    targets: pd.DataFrame        # pesos-alvo por data de decisão
    trades: pd.DataFrame
    costs: float                 # custos totais (moeda)
    turnover: pd.Series          # rotação unilateral por rebalanceamento executado


def rebalance_indices(index: pd.DatetimeIndex, start_i: int, freq: str) -> list[int]:
    if freq == 'none':
        return [start_i]
    per = index.to_period(freq)
    out = [start_i]
    for i in range(start_i + 1, len(index)):
        if per[i] != per[i - 1]:
            out.append(i)
    return out


def run(panel: Panel, name: str, target_fn, ctx: Ctx, start_i: int, freq: str = 'M',
        capital: float = 10_000.0, cost_bps: float = 15.0, band: float = 0.0,
        cash_rate: float = 0.0) -> Result:
    O = panel.open.to_numpy()
    C = panel.close.to_numpy()
    idx = panel.close.index
    cols = list(panel.close.columns)
    n_days, n = C.shape
    if start_i < 1 or start_i >= n_days - 5:
        raise ValueError('Data de início fora do histórico disponível.')
    c = cost_bps / 1e4
    g = (1 + cash_rate) ** (1 / TD)
    rebal = rebalance_indices(idx, start_i, freq)
    rebal_set = rebal + [n_days]

    units = np.zeros(n)
    cash = float(capital)
    nav = np.full(n_days, np.nan)
    vals = np.full((n_days, n), np.nan)
    cash_arr = np.full(n_days, np.nan)
    tgt_rows, trades, turn = [], [], {}
    total_cost = 0.0

    for k, i in enumerate(rebal):
        nav_open = cash + float(units @ O[i])
        cur_w = (units * O[i]) / nav_open if nav_open > 0 else np.zeros(n)
        w = target_fn(panel.close.iloc[:i], ctx).reindex(cols).fillna(0.0).to_numpy()
        if w.sum() > 1 + 1e-6:
            w = w / w.sum()
        execute = k == 0 or np.max(np.abs(cur_w - w)) >= band - 1e-12
        tgt_rows.append({'decisão': idx[i - 1], 'execução': idx[i], 'executado': bool(execute),
                         **dict(zip(cols, w))})
        if execute:
            cur_val = units * O[i]
            nav_net = nav_open
            for _ in range(6):  # ponto fixo: custo depende do NAV líquido
                nav_net = nav_open - c * np.abs(w * nav_net - cur_val).sum()
            delta = w * nav_net - cur_val
            cost = nav_open - nav_net
            total_cost += cost
            turn[idx[i]] = float(np.abs(delta).sum() / nav_open / 2)
            for j in np.nonzero(np.abs(delta) > 1e-6 * nav_open)[0]:
                trades.append({'data': idx[i].date(), 'ETF': cols[j],
                               'operação': 'COMPRA' if delta[j] > 0 else 'VENDA',
                               'unidades': abs(delta[j]) / O[i, j], 'preço': O[i, j],
                               'valor': abs(delta[j]), 'custo': c * abs(delta[j])})
            units = w * nav_net / O[i]
            cash = nav_net * (1 - w.sum())
        # segmento até ao próximo rebalanceamento (exclusivo)
        j1 = rebal_set[k + 1]
        seg_vals = C[i:j1] * units
        growth = g ** np.arange(1, j1 - i + 1)
        seg_cash = cash * growth
        vals[i:j1] = seg_vals
        cash_arr[i:j1] = seg_cash
        nav[i:j1] = seg_vals.sum(axis=1) + seg_cash
        cash = float(seg_cash[-1])
        # liquidez continua a render na abertura seguinte (aproximação ao dia)

    sl = slice(start_i, n_days)
    nav_s = pd.Series(nav[sl], index=idx[sl], name=name)
    wdf = pd.DataFrame(vals[sl], index=idx[sl], columns=cols).div(nav_s, axis=0)
    wdf['Liquidez'] = cash_arr[sl] / nav_s.to_numpy()
    return Result(name, nav_s, wdf, pd.DataFrame(tgt_rows), pd.DataFrame(trades),
                  total_cost, pd.Series(turn, dtype=float))


# ------------------------------------------------------------------- métricas
def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


def metrics(res: Result, ref: Result | None = None, rf: float = 0.0, capital: float = 10_000.0) -> dict:
    nav = res.nav
    r = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / capital) ** (1 / years) - 1
    vol = r.std() * np.sqrt(TD)
    ex = r - ((1 + rf) ** (1 / TD) - 1)
    down = ex[ex < 0].std() * np.sqrt(TD)
    mdd = drawdown(nav).min()
    out = {
        'Estratégia': res.name,
        'CAGR': cagr,
        'Volatilidade': vol,
        'Sharpe': ex.mean() / r.std() * np.sqrt(TD) if r.std() > 0 else np.nan,
        'Sortino': ex.mean() * TD / down if down > 0 else np.nan,
        'Máx. drawdown': mdd,
        'Calmar': cagr / abs(mdd) if mdd < 0 else np.nan,
        'Rotação anual': res.turnover.sum() / years if len(res.turnover) else 0.0,
        'Custos (% capital)': res.costs / capital,
        'Valor final': nav.iloc[-1],
    }
    if ref is not None and ref is not res:
        rr = ref.nav.pct_change().dropna().reindex(r.index)
        diff = r - rr
        te = diff.std() * np.sqrt(TD)
        ref_cagr = (ref.nav.iloc[-1] / capital) ** (1 / years) - 1
        m = nav.resample('ME').last().pct_change().dropna()
        mr = ref.nav.resample('ME').last().pct_change().dropna().reindex(m.index)
        out.update({'Δ CAGR vs B&H': cagr - ref_cagr,
                    'Tracking error': te,
                    'Information ratio': diff.mean() * TD / te if te > 0 else np.nan,
                    'Meses acima B&H': float((m > mr).mean())})
    else:
        out.update({'Δ CAGR vs B&H': 0.0, 'Tracking error': 0.0,
                    'Information ratio': np.nan, 'Meses acima B&H': np.nan})
    return out


def annual_returns(results: list[Result]) -> pd.DataFrame:
    rows = {}
    for res in results:
        y = res.nav.resample('YE').last()
        first = res.nav.iloc[0]
        prev = pd.concat([pd.Series([first], index=[res.nav.index[0]]), y.iloc[:-1]])
        rows[res.name] = pd.Series(y.to_numpy() / prev.to_numpy() - 1, index=y.index.year)
    return pd.DataFrame(rows)

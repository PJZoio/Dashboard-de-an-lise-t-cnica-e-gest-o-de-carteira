"""Indicadores técnicos (score v2 original) e métricas de rotação."""
from __future__ import annotations

import numpy as np
import pandas as pd


def macd(s: pd.Series):
    line = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 * up / (up + dn)).where((up + dn) > 0, 50)


def technical_score(px: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    """Score composto 0-100 do dashboard v2 (mesmos pesos e limiares).

    40% tendência semanal, 30% momentum diário, 20% força relativa vs benchmark,
    10% risco. Entrada: score ≥ 70 e filtro; saída: score < 40 ou filtro falso.
    A semana só conta depois do fecho de sexta-feira.
    """
    f = px[['Open', 'High', 'Low', 'Close']].copy()
    s = f.Close
    f['sma20'] = s.rolling(20).mean()
    f['sma50'] = s.rolling(50).mean()
    f['sma200'] = s.rolling(200).mean()
    f['rsi14'] = rsi(s)
    f['macd_hist'] = macd(s)[2]
    w = s.resample('W-FRI').last().dropna()
    wm10, wm30, wh = w.rolling(10).mean(), w.rolling(30).mean(), macd(w)[2]
    wscore = (w.gt(wm10).astype(int) + wm10.gt(wm30).astype(int) + w.gt(wm30).astype(int)
              + wh.gt(0).astype(int)) * 25
    wscore[wm30.isna()] = np.nan
    f['weekly'] = wscore.reindex(f.index, method='ffill')
    f['daily'] = (s.gt(f.sma20).astype(int) + f.sma20.gt(f.sma50).astype(int)
                  + f.macd_hist.gt(0).astype(int) + f.rsi14.between(45, 75).astype(int)) * 25
    b = bench_close.reindex(f.index).ffill(limit=3)
    ratio = s / b
    r20, r50 = ratio.rolling(20).mean(), ratio.rolling(50).mean()
    f['relative'] = (ratio.gt(r20).astype(int) * 34 + r20.gt(r50).astype(int) * 33
                     + macd(ratio)[2].gt(0).astype(int) * 33)
    f['vol20'] = s.pct_change().rolling(20).std() * np.sqrt(252)
    f['dd63'] = s / s.rolling(63).max() - 1
    f['risk'] = f.vol20.lt(.30).astype(int) * 50 + f.dd63.gt(-.12).astype(int) * 50
    f['score'] = .4 * f.weekly + .3 * f.daily + .2 * f.relative + .1 * f.risk
    f['gate'] = (s > f.sma200) & (b > b.rolling(200).mean()) & f.weekly.ge(50)
    f = f.dropna(subset=['weekly', 'sma200', 'score'])
    # máquina de estados vetorizável apenas em loop (histerese 70/40)
    state, pos = 0, np.zeros(len(f))
    sc, gt = f.score.to_numpy(), f.gate.to_numpy()
    for i in range(len(f)):
        if state == 0 and gt[i] and sc[i] >= 70:
            state = 1
        elif state == 1 and (sc[i] < 40 or not gt[i]):
            state = 0
        pos[i] = state
    f['position'] = pos
    f['leitura'] = np.select([f.score >= 70, f.score >= 55, f.score >= 40],
                             ['FORTE', 'MANTER/OBSERVAR', 'VIGIAR/REDUZIR'], default='FRACO')
    return f


def rotation_table(closes: pd.DataFrame, bench: str) -> pd.DataFrame:
    """Retornos 1/3/6/12m, força relativa vs benchmark, momentum composto e tendência."""
    last = closes.ffill(limit=3).iloc[-1]
    rows = {}
    for lbl, n in [('1M', 21), ('3M', 63), ('6M', 126), ('12M', 252)]:
        rows[lbl] = last / closes.shift(n).ffill(limit=3).iloc[-1] - 1
    t = pd.DataFrame(rows)
    t['Momentum'] = t[['3M', '6M', '12M']].mean(axis=1)
    if bench in t.index:
        for lbl in ['3M', '6M', '12M']:
            t[f'RS {lbl}'] = (1 + t[lbl]) / (1 + t.loc[bench, lbl]) - 1
    sma200 = closes.rolling(200).mean().iloc[-1]
    t['> MM200'] = last > sma200
    t['Dist. MM200'] = last / sma200 - 1
    t['Vol 3M'] = closes.pct_change().iloc[-63:].std() * np.sqrt(252)
    t['Ranking'] = t['Momentum'].rank(ascending=False).astype('Int64')
    return t.sort_values('Momentum', ascending=False)


def rrg(closes: pd.DataFrame, bench: pd.Series, window: int = 26, mom: int = 4,
        tail: int = 8) -> pd.DataFrame:
    """Gráfico de rotação relativa *inspirado* no RRG (fórmula JdK é proprietária).

    rácio = 100 × (RS / média móvel 26 semanas de RS)
    momentum = 100 × rácio / rácio de há 4 semanas
    """
    wk = closes.resample('W-FRI').last()
    b = bench.resample('W-FRI').last().reindex(wk.index)
    out = []
    for t in wk.columns:
        rs = (wk[t] / b).dropna()
        ratio = 100 * rs / rs.rolling(window).mean()
        momentum = 100 * ratio / ratio.shift(mom)
        d = pd.DataFrame({'ratio': ratio, 'momentum': momentum}).dropna().tail(tail)
        d['ticker'] = t
        out.append(d)
    if not out:
        return pd.DataFrame(columns=['ratio', 'momentum', 'ticker'])
    res = pd.concat(out)
    res.index.name = 'semana'
    return res.reset_index()


def quadrant(ratio: float, momentum: float) -> str:
    if ratio >= 100 and momentum >= 100:
        return 'Liderança'
    if ratio >= 100:
        return 'Enfraquecimento'
    if momentum >= 100:
        return 'Melhoria'
    return 'Atraso'

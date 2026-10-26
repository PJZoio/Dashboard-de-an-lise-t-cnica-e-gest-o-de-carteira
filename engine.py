"""Signal engine and next-open portfolio simulator. Research prototype, not advice."""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

CATALOG = {
    'SPY': ('EUA', 'S&P 500'), 'RSP': ('EUA', 'S&P 500 Equal Weight'),
    'DIA': ('EUA', 'Dow Jones'), 'QQQ': ('EUA', 'Nasdaq 100'),
    'IWM': ('EUA', 'Russell 2000'), 'VTV': ('Estilo', 'Value EUA'),
    'QUAL': ('Estilo', 'Quality EUA'), 'XLV': ('Setor', 'Saúde'),
    'XLP': ('Setor', 'Consumo básico'), 'XLE': ('Setor', 'Energia'),
    'XLI': ('Setor', 'Indústria'), 'XLF': ('Setor', 'Financeiro'),
    'XLU': ('Setor', 'Utilities'), 'FEZ': ('Europa', 'Euro Stoxx 50'),
    'EWG': ('Europa', 'Alemanha'), 'EWU': ('Europa', 'Reino Unido'),
    'EWJ': ('Ásia', 'Japão'), 'FXI': ('Ásia', 'China large cap'),
    'EWH': ('Ásia', 'Hong Kong'), 'EEM': ('Emergentes', 'Mercados emergentes')
}

def fetch(tickers, start, end=None):
    symbols=list(dict.fromkeys([*tickers,'SPY','QQQ','DIA','RSP','XLE']))
    raw=yf.download(symbols, start=str(start), end=str(end) if end else None,
                    auto_adjust=True, progress=False, threads=False, group_by='column')
    if raw.empty: raise ValueError('Sem dados disponíveis: verificar rede, datas ou Yahoo Finance.')
    out={}
    for t in symbols:
        try:
            frame=raw.xs(t,axis=1,level=1)[['Open','High','Low','Close','Volume']].copy()
            frame=frame.dropna(subset=['Open','Close'])
            frame=frame[(frame[['Open','Close']]>0).all(axis=1)]
            if len(frame)>=200: out[t]=frame
        except (KeyError, ValueError): pass
    if 'SPY' not in out: raise ValueError('SPY indisponível, necessário como benchmark.')
    return out

def macd(s):
    line=s.ewm(span=12,adjust=False).mean()-s.ewm(span=26,adjust=False).mean()
    signal=line.ewm(span=9,adjust=False).mean()
    return line,signal,line-signal

def rsi(s,n=14):
    diff=s.diff()
    up=diff.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    down=(-diff.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return (100*up/(up+down)).where((up+down)>0,50)

def calc_signal(ticker, data):
    f=data[ticker].copy(); s=f.Close
    f['sma20']=s.rolling(20).mean(); f['sma50']=s.rolling(50).mean(); f['sma200']=s.rolling(200).mean()
    f['rsi14']=rsi(s)
    f['macd'], f['macd_signal'],f['macd_hist']=macd(s)
    w=s.resample('W-FRI').last().dropna()
    wm10=w.rolling(10).mean(); wm30=w.rolling(30).mean(); wh=macd(w)[2]
    # Week is available on Friday AFTER CLOSE, not before; daily decisions are made after close.
    wscore=(w.gt(wm10).astype(int)+wm10.gt(wm30).astype(int)+w.gt(wm30).astype(int)+wh.gt(0).astype(int))*25
    # Friday score is never used on prior days; on holiday-shortened week, waits until next day.
    f['weekly']=wscore.reindex(f.index,method='ffill')
    f['daily']=(s.gt(f.sma20).astype(int)+f.sma20.gt(f.sma50).astype(int)+f.macd_hist.gt(0).astype(int)+f.rsi14.between(45,75).astype(int))*25
    # Relative strength versus SPY (price denominated in same USD currency).
    spy=data['SPY'].Close.reindex(f.index).ffill(limit=3)
    ratio=s/spy
    f['relative_ratio']=ratio
    r20=ratio.rolling(20).mean();r50=ratio.rolling(50).mean()
    f['relative']=ratio.gt(r20).astype(int)*34+r20.gt(r50).astype(int)*33+macd(ratio)[2].gt(0).astype(int)*33
    vol=s.pct_change().rolling(20).std()*np.sqrt(252)
    f['vol20']=vol; f['drawdown63']=s/s.rolling(63).max()-1
    f['risk']=vol.lt(.30).astype(int)*50+f.drawdown63.gt(-.12).astype(int)*50
    f['score']=.4*f.weekly+.3*f.daily+.2*f.relative+.1*f.risk
    spy200=spy.rolling(200).mean()
    f['gate']=(s>f.sma200)&(spy>spy200)&f.weekly.ge(50)
    # Explicit rotation ratios: exposed separately, not silently double counted in score.
    for code,num,den in [('dow_nasdaq','DIA','QQQ'),('equal_sp','RSP','SPY'),('energy_sp','XLE','SPY')]:
        if num in data and den in data:
            a=data[num].Close.reindex(f.index).ffill(limit=3)
            b=data[den].Close.reindex(f.index).ffill(limit=3)
            rr=a/b
            f[f'ratio_{code}']=rr
            f[f'ratio_{code}_bull']=(rr>rr.rolling(20).mean())&(rr.rolling(20).mean()>rr.rolling(50).mean())
    f=f.loc[f.index>=max(f.index.min(),spy.index.min())]
    f=f.dropna(subset=['weekly','sma200','score'])
    state=0;states=[];actions=[];reasons=[]
    for _,x in f.iterrows():
        action='SEM OPERAÇÃO';why=''
        if state==0 and bool(x.gate) and x.score>=70:
            state=1;action='COMPRA';why='Score ≥ 70 e filtro favorável'
        elif state==1 and (x.score<40 or not bool(x.gate)):
            state=0;action='VENDA';why='Score < 40 ou filtro desfavorável'
        else:
            why=('Manter posição' if state else 'Aguardar entrada')
        states.append(state);actions.append(action);reasons.append(why)
    f['position_target']=states;f['signal']=actions;f['reason']=reasons
    f['interpretation']=np.select([f.score>=70,f.score>=55,f.score>=40],['FORTE','MANTER/OBSERVAR','VIGIAR/REDUZIR'],default='FRACO')
    return f

def metrics(returns,nav,trades=0):
    if len(nav)<2: return {}
    years=(nav.index[-1]-nav.index[0]).days/365.25
    if years<=0:return {}
    dd=nav/nav.cummax()-1
    r=returns.dropna()
    std=r.std()
    return {'CAGR':float((nav.iloc[-1]/nav.iloc[0])**(1/years)-1),
      'MaxDD':float(dd.min()), 'Sharpe_0rf':float(np.sqrt(252)*r.mean()/std) if std>0 else np.nan,
      'Vol_anual':float(std*np.sqrt(252)), 'Operações':int(trades),
      'Início':str(nav.index[0].date()),'Fim':str(nav.index[-1].date())}

def portfolio(signals,data,weights,capital=10000,fee_bps=10,slippage_bps=5,start='2016-01-01'):
    """At each day's open implement prior close target, mark to close. Fractional shares; cash 0% yield."""
    tickers=list(signals)
    common=pd.DatetimeIndex(sorted(set.intersection(*(set(data[t].index) for t in tickers),set(data['SPY'].index))))
    common=common[common>=pd.Timestamp(start)]
    # Avoid stale asset dates: strict common calendar. Only act if previous common session had computed signals.
    common=pd.DatetimeIndex([dt for dt in common if all(dt in signals[t].index for t in tickers)])
    if len(common)<3: raise ValueError('Datas de negociação comuns insuficientes.')
    w=np.array([max(0,weights[t]) for t in tickers],dtype=float)
    if w.sum()>1.00001:raise ValueError('Soma dos pesos acima de 100%.')
    cash=float(capital); units={t:0.0 for t in tickers};rows=[];operations=[]
    last_target={t:0 for t in tickers}
    for i,dt in enumerate(common):
        opens={t:float(data[t].loc[dt,'Open']) for t in tickers}
        closes={t:float(data[t].loc[dt,'Close']) for t in tickers}
        nav_open=cash+sum(units[t]*opens[t] for t in tickers)
        if i>0:
            prev=common[i-1]
            targets={t:int(signals[t].loc[prev,'position_target']) for t in tickers}
            # Execute only transitions in target position, not daily weight rebalancing.
            # Sell positions whose prior-close target is now zero.
            for t in tickers:
                if targets[t]==0 and units[t]>1e-10:
                    qty=units[t]; gross=qty*opens[t]
                    cost=gross*(fee_bps+slippage_bps)/1e4
                    cash+=gross-cost; units[t]=0.0
                    operations.append({'signal_date':prev.date(),'execution_date':dt.date(),'ticker':t,'side':'VENDA','shares':qty,'price_open':opens[t],'gross_USD':gross,'cost_USD':cost})
            # When a new signal turns on, buy desired allocation of NAV at open.
            for t in tickers:
                if targets[t]==1 and units[t]<=1e-10 and weights[t]>0:
                    gross=min(nav_open*weights[t],max(cash,0)/(1+(fee_bps+slippage_bps)/1e4))
                    if gross>1e-6:
                        cost=gross*(fee_bps+slippage_bps)/1e4
                        units[t]=gross/opens[t];cash-=gross+cost
                        operations.append({'signal_date':prev.date(),'execution_date':dt.date(),'ticker':t,'side':'COMPRA','shares':units[t],'price_open':opens[t],'gross_USD':gross,'cost_USD':cost})
        nav_close=cash+sum(units[t]*closes[t] for t in tickers)
        row={'date':dt,'nav':nav_close,'cash':cash,'exposure':1-cash/nav_close if nav_close else 0}
        for t in tickers:row[f'value_{t}']=units[t]*closes[t]
        rows.append(row)
    hist=pd.DataFrame(rows).set_index('date')
    hist['ret']=hist.nav.pct_change().fillna(0)
    sp=data['SPY'].Close.reindex(hist.index)
    hist['bh_spy']=float(capital)*sp/sp.iloc[0]
    return hist,pd.DataFrame(operations),metrics(hist.ret,hist.nav,len(operations)),metrics(hist.bh_spy.pct_change().fillna(0),hist.bh_spy)

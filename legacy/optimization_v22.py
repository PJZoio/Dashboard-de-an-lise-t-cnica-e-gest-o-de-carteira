"""Walk-forward long-only allocation; research prototype, not investment advice.
All assets USD-priced; rebalances execute at next common-session open.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimal_weights(ret, method='min_vol', cap=0.45):
    ret=ret.dropna()
    n=ret.shape[1]
    if n<2 or len(ret)<252: raise ValueError('Precisamos de pelo menos 2 ETFs e 252 retornos comuns de calibração.')
    if cap*n < 1-1e-9: raise ValueError(f'Limite de {cap:.0%} não permite alocar 100% em {n} ETFs.')
    if method=='equal': return pd.Series(1/n,index=ret.columns)
    cov=np.asarray(ret.cov())*252
    cov=.95*cov+.05*np.diag(np.diag(cov))+np.eye(n)*1e-9
    def objective(w):
        variance=max(float(w@cov@w),1e-12)
        if method=='min_vol': return variance
        if method=='risk_parity':
            rc=w*(cov@w)
            return float(((rc/variance-1/n)**2).sum())
        raise ValueError('Método desconhecido.')
    result=minimize(objective,np.repeat(1/n,n),method='SLSQP',bounds=[(0,cap)]*n,
       constraints={'type':'eq','fun':lambda x: x.sum()-1},options={'ftol':1e-11,'maxiter':1200})
    if not result.success: raise ValueError('Otimização não convergiu: '+result.message)
    w=np.maximum(result.x,0);w/=w.sum()
    return pd.Series(w,index=ret.columns)


def prepare(data,selected,fx,currency,start):
    names=list(selected)
    common=pd.DatetimeIndex(sorted(set.intersection(*(set(data[t].index) for t in names))))
    if len(common)<300: raise ValueError('Histórico comum insuficiente para otimização.')
    opens=pd.DataFrame({t:data[t].Open.reindex(common) for t in names})
    closes=pd.DataFrame({t:data[t].Close.reindex(common) for t in names})
    valid=opens.notna().all(axis=1)&closes.notna().all(axis=1)&(opens>0).all(axis=1)&(closes>0).all(axis=1)
    common=common[valid];opens=opens.loc[common];closes=closes.loc[common]
    if currency=='EUR':
        if fx is None: raise ValueError('Falta histórico EURUSD=X para valorização em EUR.')
        fxopen=fx.Open.reindex(common,method='ffill',limit=5)
        fxclose=fx.Close.reindex(common,method='ffill',limit=5)
        if fxopen.isna().any() or fxclose.isna().any():
            common=common[fxopen.notna()&fxclose.notna()]
            opens=opens.loc[common];closes=closes.loc[common];fxopen=fxopen.loc[common];fxclose=fxclose.loc[common]
    else:
        fxopen=pd.Series(1.,index=common);fxclose=fxopen
    startidx=np.searchsorted(common.values,np.datetime64(pd.Timestamp(start)))
    if startidx<253: raise ValueError('Selecione uma data de simulação posterior para permitir 252 sessões anteriores de calibração.')
    if len(common)-startidx<10: raise ValueError('Período de simulação demasiado curto.')
    return common,opens,closes,fxopen,fxclose,startidx


def backtest(data,signals,tickers,capital=10000,currency='EUR',fx=None,start='2016-01-01',
             method='min_vol',cap=.45,lookback=504,fee_bps=10,slip_bps=5,
             mode='optimized',tilt=.25,rebalance='M',min_drift=.03):
    """Monthly/quarterly walk-forward at previous close, execution next open.
    Weights are recomputed only at rebalance. Fees/slippage charged on gross traded.
    Exact fractional units, cash held in USD, no interest/tax. Score tilt is cross-sectional,
    not a timing signal. Buy-hold uses equal weights with no rebalance.
    """
    names=list(tickers)
    common,opens,closes,fxo,fxc,startidx=prepare(data,names,fx,currency,start)
    returns=closes.pct_change(fill_method=None)
    fee=(fee_bps+slip_bps)/10000
    units=pd.Series(0.,index=names);cash=float(capital)*float(fxo.iloc[startidx]);operations=[];rows=[]
    last_month=None;last_target=pd.Series(0.,index=names);weights_log=[]
    for i in range(startidx,len(common)):
        dt=common[i];period=dt.to_period('M' if rebalance=='M' else 'Q')
        should_rebalance=(i==startidx) or (mode!='equal_bh' and period!=last_month)
        open_prices=opens.loc[dt]
        if should_rebalance:
            # All calibration data ends on the prior trading close (no look-ahead).
            r=returns.iloc[max(1,i-lookback):i].dropna()
            if len(r)<252: raise ValueError('Histórico de calibração insuficiente.')
            base=optimal_weights(r,'equal' if mode=='equal_bh' else method,cap if mode!='equal_bh' else 1.)
            if mode=='score_tilt':
                scores=pd.Series({t:float(signals[t].score.asof(common[i-1])) for t in names})
                if scores.isna().any(): raise ValueError('Score incompleto na data de rebalanceamento.')
                multipliers=np.clip(1+tilt*(scores-50)/50,0.5,1.5)
                raw=base*multipliers; raw=raw/raw.sum()
                # Strict cap via constrained projection closest to tilted weights.
                result=minimize(lambda w:float(((w-raw.to_numpy())**2).sum()),base.to_numpy(),
                   bounds=[(0,cap)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],
                   method='SLSQP',options={'ftol':1e-11})
                if not result.success: raise ValueError('Não foi possível projetar os pesos após ajuste pelo score.')
                base=pd.Series(result.x,index=names)
            last_target=base
            weights_log.append({'data_decisao':common[i-1].date(), 'data_execucao':dt.date(), **base.to_dict()})
        last_month=period
        current_nav=cash+float((units*open_prices).sum())
        if should_rebalance:
            target_values=last_target*current_nav
            desired=target_values/open_prices
            delta=desired-units
            drift=max(abs((units*open_prices/current_nav)-last_target)) if current_nav>0 else 1
            if i==startidx or drift>=min_drift:
                # Sell first; rebalance buys proportionally to cash available and costs.
                for t in names:
                    if delta[t]<-1e-10:
                        qty=min(-float(delta[t]),float(units[t]));gross=qty*open_prices[t]
                        cash+=gross*(1-fee);units[t]-=qty
                        operations.append({'data':dt.date(),'ETF':t,'operação':'VENDA','unidades':qty,'valor_USD':gross,'custo_USD':gross*fee})
                buys=pd.Series({t:max(float(desired[t]-units[t]),0.)*open_prices[t] for t in names})
                factor=min(1.,max(cash,0)/(buys.sum()*(1+fee))) if buys.sum()>0 else 0.
                for t in names:
                    gross=buys[t]*factor
                    if gross>1e-7:
                        units[t]+=gross/open_prices[t];cash-=gross*(1+fee)
                        operations.append({'data':dt.date(),'ETF':t,'operação':'COMPRA','unidades':gross/open_prices[t],'valor_USD':gross,'custo_USD':gross*fee})
        close_values=units*closes.loc[dt]
        nav_usd=float(cash+close_values.sum());factor=float(fxc.loc[dt])
        rows.append({'date':dt,'nav':nav_usd/factor,'cash':cash/factor,
                     **{'value_'+t:float(close_values[t])/factor for t in names}})
    hist=pd.DataFrame(rows).set_index('date')
    hist['ret']=hist.nav.pct_change(fill_method=None).fillna(0)
    years=(hist.index[-1]-hist.index[0]).days/365.25
    dd=hist.nav/hist.nav.cummax()-1
    std=hist.ret.std()
    stats={'CAGR':float((hist.nav.iloc[-1]/hist.nav.iloc[0])**(1/years)-1),
           'MaxDD':float(dd.min()),'Sharpe_0rf':float(np.sqrt(252)*hist.ret.mean()/std) if std>0 else np.nan,
           'Vol_anual':float(std*np.sqrt(252)),'Operações':len(operations)}
    return hist,pd.DataFrame(operations),stats,pd.DataFrame(weights_log)

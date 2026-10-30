import io
from datetime import date,timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from engine import CATALOG,fetch,read_snapshot,calc_signal,portfolio

st.set_page_config(page_title='ETF Signals 2.2',layout='wide',page_icon='📈')
st.title('ETF Signals · Dashboard v2.2')
st.caption('Modelo experimental | Sinais no fecho e ordens na abertura seguinte | USD ou EUR (conversão histórica)')
with st.sidebar:
    st.header('Universo e carteira')
    from pathlib import Path
    custom_file=Path('watchlist.txt')
    custom=[z.strip().upper() for z in custom_file.read_text().splitlines() if z.strip() and not z.strip().startswith('#')] if custom_file.exists() else []
    universe=dict(CATALOG)
    universe.update({t:('Personalizado','Ticker adicionado') for t in custom if t not in universe})
    typed=st.text_input('Adicionar ticker (ex.: VOO)',placeholder='Escrever símbolo e premir Enter').strip().upper()
    if typed and typed not in universe:
        import re
        if re.fullmatch(r'[A-Z0-9][A-Z0-9.\-^=]{0,19}', typed):
            universe[typed]=('Pesquisa','Ticker introduzido')
        else:st.warning('Símbolo inválido.')
    choices=st.multiselect('ETFs a acompanhar',list(universe),default=['DIA','RSP','QQQ'],format_func=lambda t:f'{t} — {universe[t][1]}')
    if typed and typed in universe and typed not in choices:
        st.info('Assinale o ticker introduzido na lista acima para o incluir na análise.')
    st.caption('Em modo local, adicione novos tickers a watchlist.txt e execute o workflow para criar os respetivos CSV.')
    source=st.radio('Origem dos preços', ['Histórico do repositório (recomendado)','Yahoo Finance (experimental)'], index=0)
    start=st.date_input('Início da simulação',date(2016,1,1),min_value=date(2005,1,1),max_value=date.today())
    currency=st.radio('Moeda da carteira', ['EUR','USD'],horizontal=True)
    capital=st.number_input(f'Capital inicial ({currency})',100.,10000000.,10000.,step=1000.)
    strategy=st.radio('Estratégia', ['Buy & Hold (referência)','Sinais v2 (experimental)'],index=0)
    st.caption('Buy & Hold: compras na primeira abertura disponível e manutenção das posições, sem rebalanceamento diário.')
    fee=st.number_input('Comissão por operação (p.b.)',0.,200.,10.,step=1.)
    slip=st.number_input('Slippage por operação (p.b.)',0.,200.,5.,step=1.)
    st.subheader('Pesos máximos')
    st.caption('O saldo não atribuído permanece em liquidez. Pesos não são redistribuídos automaticamente.')
    weights={}
    for t in choices:weights[t]=st.number_input(f'{t} · alocação (%)',0.,100.,round(100/max(len(choices),1),1),step=1.,key='w_'+t)/100
    resample=st.radio('Velas',('Diárias','Semanais'),horizontal=True)
    visible=st.slider('Sessões exibidas',60,1000,260,20)
    run=st.button('Atualizar dados e análise',type='primary')
if not choices:
    st.info('Selecione pelo menos um ETF.'); st.stop()
if sum(weights.values())>1.00001:
    st.error('A soma dos pesos não pode exceder 100%.');st.stop()
@st.cache_data(ttl=3600,show_spinner='A obter histórico ajustado...')
def get_prices(selection,start_date,source,version):
    lookback=str(pd.Timestamp(start_date)-pd.Timedelta(days=900))
    if source.startswith('Histórico'):
        return read_snapshot('data',list(selection),lookback)
    return fetch(list(selection),lookback)
try:
    from pathlib import Path
    version=max((p.stat().st_mtime_ns for p in Path('data').glob('*.csv')),default=0)
    data=get_prices(tuple(choices),start,source,version)
    fx=None
    if currency=='EUR':
        from engine import read_fx_snapshot
        fx=read_fx_snapshot('data',str(pd.Timestamp(start)-pd.Timedelta(days=900)))
except Exception as exc:
    st.error(f'Falha ao descarregar dados: {exc}');st.stop()
dates={t:d.index.max() for t,d in data.items() if len(d)}
if dates:
    newest=max(dates.values()); oldest=min(dates.values())
    st.caption(f'Fonte: {source} | Última sessão nos ficheiros: {oldest:%Y-%m-%d} a {newest:%Y-%m-%d}. Verifique se os dados estão atualizados antes de operar.')
missing=[t for t in choices if t not in data]
if missing:
    st.error('Sem histórico suficiente para: '+', '.join(missing));st.stop()
signals={t:calc_signal(t,data) for t in choices}
try:
    from engine import portfolio_v21
    hist,operations,stats,benchmark=portfolio_v21(signals,data,weights,capital,fee,slip,str(start),strategy=strategy,currency=currency,fx=fx)
except Exception as exc:
    st.error(f'Não foi possível simular a carteira: {exc}');st.stop()

st.subheader('Estado técnico mais recente')
summary=[]
for t,x in signals.items():
    z=x.iloc[-1]
    summary.append({'ETF':t,'Mercado':universe[t][0],'Última sessão':x.index[-1].strftime('%Y-%m-%d'),
                    'Fecho USD (origem)':round(float(z.Close),2),'Score':round(float(z.score),1),
                    'Semanal':int(z.weekly),'Diário':int(z.daily),'Relativo':int(z.relative),'Risco':int(z.risk),
                    'Filtro':bool(z.gate),'Posição pretendida': 'COMPRADO' if z.position_target else 'FORA',
                    'Sinal do fecho':str(z.signal),'Leitura':str(z.interpretation)})
summary=pd.DataFrame(summary)
st.dataframe(summary,use_container_width=True,hide_index=True)
st.caption('Sinal do fecho só é aplicável à sessão seguinte. “Posição pretendida” não significa ordem executada.')

chosen=st.selectbox('ETF para os gráficos',choices,format_func=lambda t:f'{t} · {universe[t][1]}')
x=signals[chosen].copy(); px=data[chosen][['Open','High','Low','Close']].reindex(x.index)
view=x.copy()
for col in ['Open','High','Low']:
    if col not in view.columns:
        view[col]=px[col].reindex(view.index)
view=view.tail(visible)
if currency=='EUR':
    # Display only: indicators/signals continue to use original USD series.
    chart_fx=fx.Close.reindex(view.index,method='ffill',limit=5)
    if chart_fx.isna().any():
        st.error('Histórico cambial indisponível para algumas datas do gráfico.');st.stop()
    for column in ['Open','High','Low','Close','sma20','sma50','sma200']:
        if column in view.columns:
            view[column]=view[column]/chart_fx
if resample=='Semanais':
    candle=view[['Open','High','Low','Close']].resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    score=view.score.resample('W-FRI').last().dropna()
else:
    candle=view[['Open','High','Low','Close']].dropna();score=view.score
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.7,.3],vertical_spacing=.06,
                  subplot_titles=[f'{chosen} · cotações ajustadas ({currency})','Score composto (0–100)'])
fig.add_trace(go.Candlestick(x=candle.index,open=candle.Open,high=candle.High,low=candle.Low,close=candle.Close,name='OHLC'),row=1,col=1)
if resample=='Diárias':
    for col,name in [('sma20','MM20'),('sma50','MM50'),('sma200','MM200')]:
        fig.add_trace(go.Scatter(x=view.index,y=view[col],name=name,mode='lines',line={'width':1.1}),row=1,col=1)
else:
    weekly_close=view.Close.resample('W-FRI').last()
    for n in [10,30]:
        sm=weekly_close.rolling(n).mean().reindex(candle.index)
        fig.add_trace(go.Scatter(x=sm.index,y=sm,name=f'MM{n} semanas',line={'width':1.3}),row=1,col=1)
# Display actual simulated buys and sells, rather than same-close theoretical signal prices.
executed=operations[operations.ticker.eq(chosen)].copy() if len(operations) else pd.DataFrame()
if len(executed):
    executed['execution_date']=pd.to_datetime(executed.execution_date)
    if currency=='EUR':
        executed['price_open']=executed['price_open_USD']/executed['fx_USD_por_EUR']
        executed['gross_USD']=executed['gross_EUR']
        executed['cost_USD']=executed['cost_USD']/executed['fx_USD_por_EUR']
    else:
        executed['price_open']=executed['price_open_USD']
    executed=executed[executed.execution_date.ge(candle.index.min())]
    for side,symbol,color in [('COMPRA','triangle-up','#159A7D'),('VENDA','triangle-down','#DE5856')]:
        sub=executed[executed.side.eq(side)]
        fig.add_trace(go.Scatter(x=sub.execution_date,y=sub.price_open,mode='markers',name=f'{side} na abertura',
          marker=dict(symbol=symbol,size=12,color=color,line=dict(width=1,color='white')),
          customdata=sub[['gross_USD','cost_USD']].to_numpy(),
          hovertemplate=f'%{{x}}<br>Abertura ({currency}): %{{y:.2f}}<br>Valor ({currency}): %{{customdata[0]:,.2f}}<br>Custos ({currency}): %{{customdata[1]:.2f}}<extra></extra>'),row=1,col=1)
fig.add_trace(go.Scatter(x=score.index,y=score,name='Score',line=dict(color='#6973D4',width=2)),row=2,col=1)
for threshold,color in [(70,'#159A7D'),(40,'#DE5856')]:
    fig.add_hline(y=threshold,line_dash='dash',line_color=color,opacity=.65,row=2,col=1)
fig.update_yaxes(range=[0,100],row=2,col=1)
fig.update_layout(height=690,hovermode='x unified',xaxis_rangeslider_visible=False,legend=dict(orientation='h',y=-.12),margin=dict(t=60,b=50))
st.plotly_chart(fig,use_container_width=True)
st.caption('Compras/vendas no gráfico são operações simuladas na abertura seguinte, calculadas a partir dos sinais do fecho anterior; a visualização semanal agrega as velas, não altera o relógio das ordens.')

st.subheader('Carteira conjunta — alocação e estratégia vs SPY Buy & Hold')
st.caption(f'Carteira em {currency} · modo: {strategy} · os pesos definem a percentagem do capital inicial destinada a cada ETF.')
cols=st.columns(4)
for col,key,fmt in zip(cols,['CAGR','MaxDD','Sharpe_0rf','Operações'],['.2%','.2%','.2f','d']):
    val=stats.get(key,float('nan'))
    col.metric(key,format(val,fmt) if np.isfinite(val) else '—')
st.caption(f"Período simulado: {stats.get('Início','—')} a {stats.get('Fim','—')} | CAGR S&P 500 Buy & Hold: {benchmark.get('CAGR',float('nan')):.2%} | Custos por operação: {fee+slip:.0f} p.b.")
curve=go.Figure()
curve.add_trace(go.Scatter(x=hist.index,y=hist.nav,name='Estratégia',mode='lines'))
curve.add_trace(go.Scatter(x=hist.index,y=hist.bh_spy,name='SPY Buy & Hold',mode='lines'))
curve.update_layout(yaxis_title=f'Valor teórico da carteira ({currency})',height=380,hovermode='x unified')
st.plotly_chart(curve,use_container_width=True)
st.subheader('Alocação da carteira')
last=hist.iloc[-1]
alloc=pd.DataFrame([{'Ativo':t,'Peso inicial %':round(weights[t]*100,2),'Valor atual '+currency:round(float(last.get('value_'+t,0)),2),'Peso atual %':round(100*float(last.get('value_'+t,0))/float(last.nav),2) if last.nav else 0} for t in choices]+[{'Ativo':'LIQUIDEZ','Peso inicial %':round(100*(1-sum(weights.values())),2),'Valor atual '+currency:round(float(last.cash),2),'Peso atual %':round(100*float(last.cash)/float(last.nav),2) if last.nav else 0}])
st.dataframe(alloc,use_container_width=True,hide_index=True)
st.subheader('Operações simuladas')
if len(operations): st.dataframe(operations.iloc[::-1],use_container_width=True,hide_index=True)
else:st.info('Sem operações no período de simulação.')

st.subheader('Rácios de rotação')
ratio_cols=[('ratio_dow_nasdaq','Dow / Nasdaq 100'),('ratio_equal_sp','Equal Weight / S&P 500'),('ratio_energy_sp','Energia / S&P 500')]
# These are shared market-wide ratios; obtain from chosen ticker signal dataframe.
ratfig=go.Figure()
for col,label in ratio_cols:
    if col in x:
        s=x[col].dropna().loc[lambda z:z.index>=pd.Timestamp(start)]
        if len(s):ratfig.add_trace(go.Scatter(x=s.index,y=100*s/s.iloc[0],name=label))
ratfig.update_layout(height=340,yaxis_title='Índice 100 no início',hovermode='x unified')
st.plotly_chart(ratfig,use_container_width=True)
st.caption('Rácios normalizados (base 100). Servem para contextualizar rotação, não fazem parte dos 20% de força relativa da versão atual (ETF/SPY).')

st.divider()
st.header('Otimização walk-forward · v2.2')
st.caption('Comparação de três carteiras: Buy & Hold com pesos iguais, mínima volatilidade/paridade de risco e otimização com ajuste moderado pelo score. Apenas dados disponíveis antes de cada operação; ordens na abertura seguinte.')
from optimization_v22 import backtest
oc1,oc2,oc3=st.columns(3)
with oc1:
    optimizer_method=st.selectbox('Otimização base', ['min_vol','risk_parity'],format_func=lambda q:{'min_vol':'Mínima volatilidade','risk_parity':'Paridade de risco'}[q])
    optimizer_lookback=st.selectbox('Histórico de calibração (sessões)',[252,504,756],index=1)
with oc2:
    optimizer_cap=st.slider('Peso máximo por ETF (%)',10,100,50,5)/100
    optimizer_frequency=st.radio('Rebalanceamento', ['Mensal','Trimestral'],horizontal=True)
with oc3:
    optimizer_tilt=st.slider('Intensidade do score (%)',0,50,25,5)/100
    optimizer_drift=st.slider('Desvio mínimo para negociar (p.p.)',0,15,3,1)/100
st.caption('O cabaz pode incluir ETFs dos EUA, Europa e Ásia, cotados em USD. Para outros mercados/moedas é necessário acrescentar conversão cambial por ativo.')
if st.button('Calcular carteiras otimizadas',type='primary'):
    if len(choices)<2:
        st.warning('Selecione pelo menos dois ETFs para a otimização.')
    elif optimizer_cap*len(choices)<.99999:
        st.error('O peso máximo selecionado é incompatível com o número de ETFs.')
    else:
        try:
            results={}
            for key,mode in [('Pesos iguais Buy & Hold','equal_bh'),('Otimizada','optimized'),('Otimizada + score','score_tilt')]:
                results[key]=backtest(data,signals,choices,capital=capital,currency=currency,fx=fx,start=str(start),
                    method=optimizer_method,cap=optimizer_cap,lookback=optimizer_lookback,
                    fee_bps=fee,slip_bps=slip,mode=mode,tilt=optimizer_tilt,
                    rebalance='M' if optimizer_frequency=='Mensal' else 'Q',min_drift=optimizer_drift)
            st.session_state['opt22_result']=results
            st.session_state['opt22_choices']=tuple(choices)
        except Exception as exc:
            st.error(f'Otimização não concluída: {exc}')
if st.session_state.get('opt22_result') and st.session_state.get('opt22_choices')==tuple(choices):
    results=st.session_state['opt22_result']
    compar=[];chart=go.Figure()
    for label,(curve,orders,mets,log) in results.items():
        compar.append({'Estratégia':label,'CAGR':mets.get('CAGR'), 'MaxDD':mets.get('MaxDD'),
              'Sharpe':mets.get('Sharpe_0rf'),'Volatilidade':mets.get('Vol_anual'),'Operações':mets.get('Operações')})
        chart.add_trace(go.Scatter(x=curve.index,y=curve.nav,name=label,mode='lines'))
    st.dataframe(pd.DataFrame(compar).style.format({'CAGR':'{:.2%}','MaxDD':'{:.2%}','Sharpe':'{:.2f}','Volatilidade':'{:.2%}'}),use_container_width=True,hide_index=True)
    chart.update_layout(height=390,yaxis_title=f'Valor da carteira ({currency})',hovermode='x unified')
    st.plotly_chart(chart,use_container_width=True)
    selected_strategy=st.selectbox('Mostrar pesos e transações',list(results))
    curve,orders,mets,log=results[selected_strategy]
    if not log.empty:
        st.subheader('Histórico de pesos-alvo por rebalanceamento')
        st.dataframe(log.iloc[::-1].head(24),use_container_width=True,hide_index=True)
    if not orders.empty:
        st.subheader('Operações simuladas')
        st.dataframe(orders.iloc[::-1].head(100),use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    c1.download_button('Descarregar pesos (CSV)',log.to_csv(index=False).encode('utf-8-sig'),'pesos_otimizados.csv','text/csv')
    c2.download_button('Descarregar carteira (CSV)',curve.to_csv().encode('utf-8-sig'),'carteira_otimizada.csv','text/csv')
st.caption('A otimização é histórica, não preditiva. A variante ajustada pelo score apenas altera pesos; não vende automaticamente em score baixo. A referência B&H nesta secção usa pesos iguais. Evitar interpretar a otimização in-sample como indicação de retornos futuros.')

st.subheader('Exportações')
co1,co2,co3=st.columns(3)
co1.download_button('Resumo CSV',summary.to_csv(index=False).encode('utf-8-sig'),'resumo_v2.csv','text/csv')
co2.download_button('Carteira CSV',hist.to_csv().encode('utf-8-sig'),'carteira_v2.csv','text/csv')
co3.download_button('Ordens CSV',operations.to_csv(index=False).encode('utf-8-sig'),'operacoes_v21.csv','text/csv')
st.download_button('Alocação CSV',alloc.to_csv(index=False).encode('utf-8-sig'),'alocacao_v21.csv','text/csv')
with st.expander('Dados e indicadores do ETF selecionado'):
    st.dataframe(x.tail(50),use_container_width=True)
    st.download_button('Descarregar série técnica CSV',x.to_csv().encode('utf-8-sig'),f'indicadores_{chosen}_v2.csv','text/csv')
st.warning('Protótipo de investigação: sem contas reais, execução de ordens, previsões garantidas, impostos, dividendos em numerário, remuneração de caixa ou tratamento completo de horas de bolsas diferentes. A conversão EUR/USD pressupõe ativos cotados em USD. Yahoo pode rever preços ajustados. Investimento em ETFs dos EUA pode não estar disponível a investidores não profissionais na UE. Resultados passados não são garantia de rendimentos futuros.')

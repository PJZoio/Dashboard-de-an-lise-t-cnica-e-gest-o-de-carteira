import io
from datetime import date,timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from engine import CATALOG,fetch,read_snapshot,calc_signal,portfolio

st.set_page_config(page_title='ETF Signals 2.0',layout='wide',page_icon='📈')
st.title('ETF Signals · Dashboard v2.0')
st.caption('Modelo experimental | Sinais no fecho; simulação de ordens na abertura da sessão seguinte | Valores em USD')
with st.sidebar:
    st.header('Universo e carteira')
    choices=st.multiselect('ETFs a acompanhar',list(CATALOG),default=['DIA','RSP','QQQ'],format_func=lambda t:f'{t} — {CATALOG[t][1]}')
    source=st.radio('Origem dos preços', ['Histórico do repositório (recomendado)','Yahoo Finance (experimental)'], index=0)
    start=st.date_input('Início da simulação',date(2016,1,1),min_value=date(2005,1,1),max_value=date.today())
    capital=st.number_input('Capital inicial (USD)',100.,10000000.,10000.,step=1000.)
    fee=st.number_input('Comissão por operação (p.b.)',0.,200.,10.,step=1.)
    slip=st.number_input('Slippage por operação (p.b.)',0.,200.,5.,step=1.)
    st.subheader('Pesos máximos')
    st.caption('O saldo não atribuído permanece em liquidez. Pesos não são redistribuídos automaticamente.')
    weights={}
    for t in choices:weights[t]=st.number_input(f'{t} · peso máximo (%)',0.,100.,round(100/max(len(choices),1),1),step=5.)/100
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
    hist,operations,stats,benchmark=portfolio(signals,data,weights,capital,fee,slip,str(start))
except Exception as exc:
    st.error(f'Não foi possível simular a carteira: {exc}');st.stop()

st.subheader('Estado técnico mais recente')
summary=[]
for t,x in signals.items():
    z=x.iloc[-1]
    summary.append({'ETF':t,'Mercado':CATALOG[t][0],'Última sessão':x.index[-1].strftime('%Y-%m-%d'),
                    'Fecho USD':round(float(z.Close),2),'Score':round(float(z.score),1),
                    'Semanal':int(z.weekly),'Diário':int(z.daily),'Relativo':int(z.relative),'Risco':int(z.risk),
                    'Filtro':bool(z.gate),'Posição pretendida': 'COMPRADO' if z.position_target else 'FORA',
                    'Sinal do fecho':str(z.signal),'Leitura':str(z.interpretation)})
summary=pd.DataFrame(summary)
st.dataframe(summary,use_container_width=True,hide_index=True)
st.caption('Sinal do fecho só é aplicável à sessão seguinte. “Posição pretendida” não significa ordem executada.')

chosen=st.selectbox('ETF para os gráficos',choices,format_func=lambda t:f'{t} · {CATALOG[t][1]}')
x=signals[chosen].copy(); px=data[chosen][['Open','High','Low','Close']].reindex(x.index)
view=x.join(px[['Open','High','Low']],how='left').tail(visible)
if resample=='Semanais':
    candle=view[['Open','High','Low','Close']].resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    score=view.score.resample('W-FRI').last().dropna()
else:
    candle=view[['Open','High','Low','Close']].dropna();score=view.score
fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.7,.3],vertical_spacing=.06,
                  subplot_titles=[f'{chosen} · cotações ajustadas','Score composto (0–100)'])
fig.add_trace(go.Candlestick(x=candle.index,open=candle.Open,high=candle.High,low=candle.Low,close=candle.Close,name='OHLC'),row=1,col=1)
if resample=='Diárias':
    for col,name in [('sma20','MM20'),('sma50','MM50'),('sma200','MM200')]:
        fig.add_trace(go.Scatter(x=view.index,y=view[col],name=name,mode='lines',line={'width':1.1}),row=1,col=1)
else:
    weekly_close=x.Close.resample('W-FRI').last()
    for n in [10,30]:
        sm=weekly_close.rolling(n).mean().reindex(candle.index)
        fig.add_trace(go.Scatter(x=sm.index,y=sm,name=f'MM{n} semanas',line={'width':1.3}),row=1,col=1)
# Display actual simulated buys and sells, rather than same-close theoretical signal prices.
executed=operations[operations.ticker.eq(chosen)].copy() if len(operations) else pd.DataFrame()
if len(executed):
    executed['execution_date']=pd.to_datetime(executed.execution_date)
    executed=executed[executed.execution_date.ge(candle.index.min())]
    for side,symbol,color in [('COMPRA','triangle-up','#159A7D'),('VENDA','triangle-down','#DE5856')]:
        sub=executed[executed.side.eq(side)]
        fig.add_trace(go.Scatter(x=sub.execution_date,y=sub.price_open,mode='markers',name=f'{side} na abertura',
          marker=dict(symbol=symbol,size=12,color=color,line=dict(width=1,color='white')),
          customdata=sub[['gross_USD','cost_USD']].to_numpy(),
          hovertemplate='%{x}<br>Abertura: $%{y:.2f}<br>Valor: $%{customdata[0]:,.2f}<br>Custos: $%{customdata[1]:.2f}<extra></extra>'),row=1,col=1)
fig.add_trace(go.Scatter(x=score.index,y=score,name='Score',line=dict(color='#6973D4',width=2)),row=2,col=1)
for threshold,color in [(70,'#159A7D'),(40,'#DE5856')]:
    fig.add_hline(y=threshold,line_dash='dash',line_color=color,opacity=.65,row=2,col=1)
fig.update_yaxes(range=[0,100],row=2,col=1)
fig.update_layout(height=690,hovermode='x unified',xaxis_rangeslider_visible=False,legend=dict(orientation='h',y=-.12),margin=dict(t=60,b=50))
st.plotly_chart(fig,use_container_width=True)
st.caption('Compras/vendas no gráfico são operações simuladas na abertura seguinte, calculadas a partir dos sinais do fecho anterior; a visualização semanal agrega as velas, não altera o relógio das ordens.')

st.subheader('Carteira conjunta — estratégia vs S&P 500 Buy & Hold')
cols=st.columns(4)
for col,key,fmt in zip(cols,['CAGR','MaxDD','Sharpe_0rf','Operações'],['.2%','.2%','.2f','d']):
    val=stats.get(key,float('nan'))
    col.metric(key,format(val,fmt) if np.isfinite(val) else '—')
st.caption(f"Período simulado: {stats.get('Início','—')} a {stats.get('Fim','—')} | CAGR S&P 500 Buy & Hold: {benchmark.get('CAGR',float('nan')):.2%} | Custos por operação: {fee+slip:.0f} p.b.")
curve=go.Figure()
curve.add_trace(go.Scatter(x=hist.index,y=hist.nav,name='Estratégia',mode='lines'))
curve.add_trace(go.Scatter(x=hist.index,y=hist.bh_spy,name='SPY Buy & Hold',mode='lines'))
curve.update_layout(yaxis_title='Valor teórico da carteira (USD)',height=380,hovermode='x unified')
st.plotly_chart(curve,use_container_width=True)
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

st.subheader('Exportações')
co1,co2,co3=st.columns(3)
co1.download_button('Resumo CSV',summary.to_csv(index=False).encode('utf-8-sig'),'resumo_v2.csv','text/csv')
co2.download_button('Carteira CSV',hist.to_csv().encode('utf-8-sig'),'carteira_v2.csv','text/csv')
co3.download_button('Ordens CSV',operations.to_csv(index=False).encode('utf-8-sig'),'operacoes_v2.csv','text/csv')
with st.expander('Dados e indicadores do ETF selecionado'):
    st.dataframe(x.tail(50),use_container_width=True)
    st.download_button('Descarregar série técnica CSV',x.to_csv().encode('utf-8-sig'),f'indicadores_{chosen}_v2.csv','text/csv')
st.warning('Protótipo de investigação: sem contas reais, execução de ordens, previsões garantidas, impostos, dividendos em numerário, remuneração de caixa, conversão EUR/USD ou tratamento de horas de bolsas diferentes. Yahoo pode rever preços ajustados. Investimento em ETFs dos EUA pode não estar disponível a investidores não profissionais na UE. Resultados passados não são garantia de rendimentos futuros.')

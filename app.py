"""ETF Rotação & Otimização · dashboard Streamlit v3.

Pergunta central: que pesos devo ter hoje no cabaz, e essa regra teria
batido um Buy & Hold simples, depois de custos, sem olhar para o futuro?
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import backtest as bt
from core.data import (CATALOG, FX_TICKER, GROUP_ORDER, build_panel, describe, last_dates,
                       load_folder, read_watchlist, to_currency_close)
from core.signals import quadrant, rotation_table, rrg, technical_score
from core.strategies import STRATEGIES, Ctx

st.set_page_config(page_title='ETF Rotação & Otimização', layout='wide', page_icon='📈')

COLORS = {'bh': '#374151', 'rebal': '#9CA3AF', 'minvar': '#2563EB', 'erc': '#059669',
          'mom': '#D97706', 'tech': '#7C3AED', 'bench': '#DC2626'}
DEFAULT_BASKET = ['SPY', 'QQQ', 'VGK', 'EWJ', 'EEM', 'XLV', 'XLE', 'XLU']
PCT = lambda c, d=1: st.column_config.NumberColumn(c, format=f'%.{d}f%%')


# ------------------------------------------------------------------ dados
@st.cache_data(show_spinner='A ler históricos…')
def get_data(version: int) -> dict[str, pd.DataFrame]:
    return load_folder('data')


def data_version() -> int:
    return max((p.stat().st_mtime_ns for p in Path('data').glob('*.csv')), default=0)


@st.cache_data(show_spinner=False)
def get_signals(version: int, tickers: tuple[str, ...], bench: str) -> dict[str, pd.DataFrame]:
    data = get_data(version)
    b = data[bench]['Close']
    return {t: technical_score(data[t], b) for t in tickers}


@st.cache_data(show_spinner='A simular estratégias (walk-forward)…')
def run_all(version, tickers, weights, currency, start, freq, cost, band, cap, lookback,
            top_k, defensive, cash_rate, abs_filter, bench, capital, keys):
    data = get_data(version)
    panel = build_panel(data, list(tickers), currency)
    idx = panel.close.index
    warm = max(260, lookback + 1)
    req = int(np.searchsorted(idx, pd.Timestamp(start)))
    start_i = max(req, warm)
    sig = get_signals(version, tickers, 'SPY' if 'SPY' in data else tickers[0])
    ctx = Ctx(list(tickers), pd.Series(dict(weights)), lookback=lookback, cap=cap, top_k=top_k,
              defensive=defensive, abs_filter=abs_filter,
              signals={t: s['position'] for t, s in sig.items()})
    results = {}
    for k in keys:
        name, fn, mode = STRATEGIES[k]
        results[k] = bt.run(panel, name, fn, ctx, start_i, 'none' if mode == 'none' else freq,
                            capital, cost, band, cash_rate)
    # benchmark: B&H de um único ETF, mesmo calendário e custos
    bp = build_panel(data, [bench], currency)
    bp = type(bp)(bp.open.reindex(idx).ffill(limit=5), bp.close.reindex(idx).ffill(limit=5), currency, 0)
    bctx = Ctx([bench], pd.Series({bench: 1.0}))
    results['bench'] = bt.run(bp, f'Benchmark · {bench}', STRATEGIES['bh'][1], bctx, start_i,
                              'none', capital, cost, 0, 0)
    # alvo de hoje: cada regra aplicada a todo o histórico disponível
    today = {k: STRATEGIES[k][1](panel.close, ctx).reindex(list(tickers)).fillna(0.0)
             for k in keys}
    clamped = req < warm
    return results, today, idx[start_i], clamped, panel.dropped_dates


@st.cache_data(show_spinner='A testar sensibilidade da rotação…')
def momentum_grid(version, tickers, currency, start_ts, freq, cost, band, cap, defensive,
                  cash_rate, capital):
    data = get_data(version)
    panel = build_panel(data, list(tickers), currency)
    start_i = max(int(np.searchsorted(panel.close.index, start_ts)), 260)
    rows = []
    n_risky = len([t for t in tickers if t != defensive])
    for f in ['M', 'Q']:
        for k in range(1, min(6, n_risky) + 1):
            for af in [True, False]:
                ctx = Ctx(list(tickers), pd.Series(1 / len(tickers), index=list(tickers)), cap=cap,
                          top_k=k, defensive=defensive, abs_filter=af)
                r = bt.run(panel, 'm', STRATEGIES['mom'][1], ctx, start_i, f, capital, cost, band, cash_rate)
                m = bt.metrics(r, capital=capital)
                rows.append({'Frequência': 'Mensal' if f == 'M' else 'Trimestral', 'Top-K': k,
                             'Filtro absoluto': 'Sim' if af else 'Não', 'CAGR': m['CAGR'] * 100,
                             'Sharpe': m['Sharpe'], 'Máx. drawdown': m['Máx. drawdown'] * 100})
    return pd.DataFrame(rows)


version = data_version()
data = get_data(version)
available = [t for t in data if t != FX_TICKER]
watch = read_watchlist()
pending = [t for t in [*CATALOG, *watch] if t not in data]


def label(t: str) -> str:
    g, d = describe(t)
    return f'{t} · {d}'


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header('Cabaz')
    ordered = sorted(available, key=lambda t: (GROUP_ORDER.index(describe(t)[0])
                                               if describe(t)[0] in GROUP_ORDER else 99, t))
    basket = st.multiselect('ETFs', ordered, default=[t for t in DEFAULT_BASKET if t in available],
                            format_func=label)
    c1, c2 = st.columns(2)
    currency = c1.radio('Moeda', ['EUR', 'USD'], horizontal=True)
    capital = c2.number_input(f'Capital ({currency})', 1_000.0, 1e8, 10_000.0, step=1_000.0)
    start = st.date_input('Início do backtest', date(2016, 1, 1), min_value=date(2010, 1, 1),
                          max_value=date.today())
    bench_opts = [t for t in ['URTH', 'ACWI', 'SPY'] if t in available] or available
    bench = st.selectbox('Benchmark', bench_opts, format_func=label,
                         help='URTH ≈ MSCI World (proxy do IWDA). Fica disponível após o workflow descarregar o CSV.')

    with st.expander('Carteira atual / referência', expanded=False):
        st.caption('Pesos do Buy & Hold e ponto de partida para as ordens no separador Decisão. '
                   'São normalizados para 100%.')
        if basket:
            key = 'w_' + '_'.join(basket)
            base = pd.DataFrame({'ETF': basket, 'Peso %': [round(100 / len(basket), 2)] * len(basket)})
            ed = st.data_editor(base, key=key, hide_index=True, disabled=['ETF'],
                                column_config={'Peso %': st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0)})
            raw = ed.set_index('ETF')['Peso %'].clip(lower=0)
            if raw.sum() <= 0:
                raw[:] = 1
            base_w = raw / raw.sum()
            if abs(raw.sum() - 100) > 0.5:
                st.caption(f'Soma introduzida: {raw.sum():.1f}% → normalizada para 100%.')
        else:
            base_w = pd.Series(dtype=float)

    with st.expander('Parâmetros das estratégias', expanded=False):
        freq = st.radio('Rebalanceamento', ['M', 'Q'], horizontal=True,
                        format_func=lambda f: {'M': 'Mensal', 'Q': 'Trimestral'}[f])
        cost = st.number_input('Custos por operação (p.b., comissão + slippage)', 0.0, 200.0, 15.0, 1.0)
        band = st.slider('Banda de tolerância (p.p.)', 0, 10, 2,
                         help='Só se negoceia se algum peso se afastar do alvo mais do que isto.') / 100
        cap = st.slider('Peso máximo por ETF', 10, 100, 40, 5, format='%d%%') / 100
        lookback = st.select_slider('Janela da covariância (sessões)', [126, 252, 504], 252)
        top_k = st.slider('Rotação: nº de ETFs (top-K)', 1, 8, 3)
        abs_filter = st.checkbox('Rotação: filtro de momentum absoluto', True,
                                 help='Só entra um ETF com momentum positivo; o resto vai para o defensivo/liquidez.')
        def_opts = ['(liquidez)'] + basket
        dsel = st.selectbox('Ativo defensivo', def_opts,
                            index=next((i for i, t in enumerate(def_opts) if t in ('IEF', 'SHY', 'AGG', 'GLD', 'TLT')), 0),
                            help='Recebe as vagas não preenchidas na rotação e no filtro técnico.')
        defensive = None if dsel == '(liquidez)' else dsel
        cash_rate = st.number_input('Remuneração da liquidez (%/ano)', 0.0, 10.0, 0.0, 0.25) / 100

    st.caption(f'{len(available)} ETF com histórico local.' +
               (f' Em falta (workflow): {", ".join(pending)}.' if pending else ''))

st.title('ETF · Rotação e otimização de carteira')
if len(basket) < 2:
    st.info('Escolha pelo menos dois ETF na barra lateral.')
    st.stop()

oldest, newest = last_dates(data, basket)
keys = list(STRATEGIES)
try:
    results, today, real_start, clamped, dropped = run_all(
        version, tuple(basket), tuple(base_w.items()), currency, pd.Timestamp(start), freq, cost,
        band, cap, lookback, top_k, defensive, cash_rate, abs_filter, bench, capital, tuple(keys))
except Exception as exc:  # noqa: BLE001
    st.error(f'Não foi possível simular: {exc}')
    st.stop()

ref = results['bh']
st.caption(f'Dados até {oldest:%d/%m/%Y}'
           + (f' (alguns até {newest:%d/%m/%Y})' if newest != oldest else '')
           + f' · backtest desde {real_start:%d/%m/%Y} · valores em {currency} · '
           + f'custos {cost:.0f} p.b. · rebalanceamento {"mensal" if freq == "M" else "trimestral"}')
if clamped:
    st.warning(f'O início foi adiado para {real_start:%d/%m/%Y}: são precisas ≥ {max(260, lookback + 1)} '
               'sessões comuns antes da primeira decisão (aquecimento). ETF com histórico curto '
               '(ex. XLC, desde 2018) encurtam o período de todo o cabaz.')
if (pd.Timestamp.today() - oldest).days > 7:
    st.warning('Os dados têm mais de uma semana. Corra o workflow “Atualizar OHLC ETFs”.')

tab_dec, tab_bt, tab_rot, tab_tec, tab_met = st.tabs(
    ['🎯 Decisão', '📊 Backtest vs Buy & Hold', '🔄 Rotação', '📈 Análise técnica', 'ℹ️ Metodologia'])

all_metrics = pd.DataFrame([bt.metrics(results[k], ref, cash_rate, capital) for k in keys]
                           + [bt.metrics(results['bench'], ref, cash_rate, capital)]).set_index('Estratégia')
all_metrics.loc[results['bench'].name, ['Rotação anual', 'Custos (% capital)']] = np.nan

# ===================================================================== DECISÃO
with tab_dec:
    usd = {t: data[t]['Close'] for t in basket}
    above = {t: float(s.iloc[-1] > s.rolling(200).mean().iloc[-1]) for t, s in usd.items()}
    bclose = data[bench]['Close']
    regime_on = bclose.iloc[-1] > bclose.rolling(200).mean().iloc[-1]
    nxt = (pd.Timestamp(oldest) + (pd.offsets.MonthBegin(1) if freq == 'M' else pd.offsets.QuarterBegin(1, startingMonth=1)))
    nxt = nxt + pd.offsets.BDay(0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Regime do benchmark', 'Tendência ↑' if regime_on else 'Tendência ↓',
              help=f'{bench} acima/abaixo da média móvel de 200 sessões.')
    m2.metric('ETF do cabaz acima da MM200', f'{sum(above.values()):.0f} / {len(above)}')
    m3.metric('Próximo rebalanceamento', f'{nxt:%d/%m/%Y}', help='1.ª sessão do próximo período; decisão no fecho anterior.')
    m4.metric('Última sessão', f'{oldest:%d/%m/%Y}')

    st.divider()
    left, right = st.columns([1, 1.3])
    with left:
        active_opts = [k for k in keys if k != 'bh']
        active = st.selectbox('Regra de decisão', active_opts, index=active_opts.index('mom'),
                              format_func=lambda k: STRATEGIES[k][0])
        pv = st.number_input(f'Valor atual da carteira ({currency})', 0.0, 1e9, float(capital), 1_000.0)
        mm = all_metrics.loc[STRATEGIES[active][0]]
        rm = all_metrics.loc[STRATEGIES['bh'][0]]
        st.markdown(
            f'**Histórico desta regra vs Buy & Hold** ({real_start:%Y}–{oldest:%Y}, após custos)  \n'
            f'CAGR {mm["CAGR"]:.1%} vs {rm["CAGR"]:.1%} · volatilidade {mm["Volatilidade"]:.1%} vs {rm["Volatilidade"]:.1%} · '
            f'máx. drawdown {mm["Máx. drawdown"]:.1%} vs {rm["Máx. drawdown"]:.1%} · '
            f'Sharpe {mm["Sharpe"]:.2f} vs {rm["Sharpe"]:.2f} · rotação {mm["Rotação anual"]:.0%}/ano')
        verdict = ('melhor' if mm['Sharpe'] > rm['Sharpe'] + 0.1 else
                   'pior' if mm['Sharpe'] < rm['Sharpe'] - 0.1 else 'semelhante')
        st.caption(f'Retorno ajustado ao risco historicamente **{verdict}** que o B&H (diferença de Sharpe '
                   f'{mm["Sharpe"] - rm["Sharpe"]:+.2f}). Diferenças de Sharpe abaixo de ~0,3 em 10 anos '
                   'raramente são estatisticamente distinguíveis de sorte.')

    tgt = today[active]
    cur = base_w.reindex(basket).fillna(0)
    dec = pd.DataFrame({'ETF': basket + ['Liquidez'],
                        'Descrição': [describe(t)[1] for t in basket] + ['—'],
                        'Atual %': list(cur * 100) + [0.0],
                        'Alvo %': list(tgt * 100) + [max(0.0, 1 - tgt.sum()) * 100]})
    dec['Δ p.p.'] = dec['Alvo %'] - dec['Atual %']
    dec['Ordem'] = np.where(dec['Δ p.p.'].abs() < band * 100, 'Manter',
                            np.where(dec['Δ p.p.'] > 0, 'Comprar', 'Vender'))
    dec[f'Montante {currency}'] = dec['Δ p.p.'] / 100 * pv
    dec = dec[(dec['Atual %'] > 0.01) | (dec['Alvo %'] > 0.01)]
    with right:
        fig = go.Figure()
        fig.add_bar(y=dec.ETF, x=dec['Atual %'], name='Atual', orientation='h', marker_color='#CBD5E1')
        fig.add_bar(y=dec.ETF, x=dec['Alvo %'], name='Alvo', orientation='h', marker_color=COLORS[active])
        fig.update_layout(barmode='group', height=80 + 34 * len(dec), margin=dict(t=40, b=10, l=10, r=10),
                          xaxis_title='%', legend=dict(orientation='h', y=1.0, yanchor='bottom'), yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, width='stretch')
    st.subheader('Ordens para atingir o alvo')
    st.dataframe(dec, hide_index=True, width='stretch',
                 column_config={'Atual %': PCT('Atual'), 'Alvo %': PCT('Alvo'), 'Δ p.p.': PCT('Δ', 1),
                                f'Montante {currency}': st.column_config.NumberColumn(format='%.0f')})
    st.caption('Alvo calculado com o fecho mais recente; executar só na abertura seguinte. '
               '“Manter” = desvio inferior à banda de tolerância. Não inclui impostos nem custos de câmbio da corretora.')

    with st.expander('Alvo de todas as regras (comparação)'):
        comp = pd.DataFrame({STRATEGIES[k][0]: today[k] * 100 for k in keys}).round(1)
        comp.loc['Liquidez'] = (100 - comp.sum()).clip(lower=0).round(1)
        st.dataframe(comp, width='stretch')

# ===================================================================== BACKTEST
with tab_bt:
    show = st.multiselect('Estratégias', keys + ['bench'],
                          default=['bh', 'rebal', 'minvar', 'erc', 'mom', 'bench'],
                          format_func=lambda k: results[k].name)
    tbl = all_metrics.loc[[results[k].name for k in show]].copy()
    for c in ['CAGR', 'Volatilidade', 'Máx. drawdown', 'Rotação anual', 'Custos (% capital)',
              'Δ CAGR vs B&H', 'Tracking error', 'Meses acima B&H']:
        tbl[c] = tbl[c] * 100
    st.dataframe(tbl, width='stretch', column_config={
        'CAGR': PCT('CAGR'), 'Volatilidade': PCT('Vol.'), 'Máx. drawdown': PCT('Máx. DD'),
        'Rotação anual': PCT('Rotação/ano', 0), 'Custos (% capital)': PCT('Custos', 2),
        'Δ CAGR vs B&H': PCT('Δ CAGR vs B&H', 2), 'Tracking error': PCT('Tracking err.'),
        'Meses acima B&H': PCT('Meses > B&H', 0),
        'Sharpe': st.column_config.NumberColumn(format='%.2f'),
        'Sortino': st.column_config.NumberColumn(format='%.2f'),
        'Calmar': st.column_config.NumberColumn(format='%.2f'),
        'Information ratio': st.column_config.NumberColumn('IR vs B&H', format='%.2f'),
        'Valor final': st.column_config.NumberColumn(f'Valor final {currency}', format='%.0f')})
    st.caption(f'Sharpe/Sortino com taxa sem risco = remuneração da liquidez ({cash_rate:.2%}). '
               'Rotação anual = soma do valor negociado (unilateral) / NAV. IR = excesso anualizado / tracking error face ao B&H.')

    logy = st.toggle('Escala logarítmica', value=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.04)
    for k in show:
        r = results[k]
        dash = 'dot' if k == 'bench' else None
        width = 2.6 if k in ('bh', active) else 1.5
        fig.add_scatter(x=r.nav.index, y=r.nav, name=r.name, line=dict(color=COLORS[k], width=width, dash=dash),
                        row=1, col=1, legendgroup=k)
        fig.add_scatter(x=r.nav.index, y=bt.drawdown(r.nav) * 100, name=r.name, showlegend=False,
                        line=dict(color=COLORS[k], width=1, dash=dash), row=2, col=1, legendgroup=k)
    fig.update_yaxes(type='log' if logy else 'linear', title=f'Valor ({currency})', row=1, col=1)
    fig.update_yaxes(title='Drawdown %', row=2, col=1)
    fig.update_layout(height=620, hovermode='x unified', legend=dict(orientation='h', y=1.06),
                      margin=dict(t=30, b=20))
    st.plotly_chart(fig, width='stretch')

    st.subheader('Retorno por ano civil')
    ann = bt.annual_returns([results[k] for k in show]) * 100
    st.dataframe(ann.style.format('{:.1f}%').background_gradient(cmap='RdYlGn', axis=None, vmin=-25, vmax=25),
                 width='stretch')
    st.caption('Útil para ver se a vantagem de uma regra vem de um ou dois anos isolados (ex. 2020, 2022).')

    st.subheader('Pesos ao longo do tempo')
    sel = st.selectbox('Estratégia', [k for k in keys if k != 'bh'] + ['bh'],
                       index=[k for k in keys if k != 'bh'].index(active), format_func=lambda k: STRATEGIES[k][0])
    w = results[sel].weights.resample('W-FRI').last() * 100
    w = w.loc[:, w.max() > 0.5]
    area = go.Figure()
    for c in w.columns:
        area.add_scatter(x=w.index, y=w[c], name=c, stackgroup='one', mode='lines', line=dict(width=0.5))
    area.update_layout(height=360, yaxis=dict(range=[0, 100], title='%'), hovermode='x unified',
                       margin=dict(t=10, b=10), legend=dict(orientation='h', y=1.1))
    st.plotly_chart(area, width='stretch')

    with st.expander('Operações e downloads'):
        tr = results[sel].trades
        st.dataframe(tr.iloc[::-1], hide_index=True, width='stretch',
                     column_config={'unidades': st.column_config.NumberColumn(format='%.4f'),
                                    'preço': st.column_config.NumberColumn(format='%.2f'),
                                    'valor': st.column_config.NumberColumn(format='%.2f'),
                                    'custo': st.column_config.NumberColumn(format='%.2f')})
        navs = pd.concat({results[k].name: results[k].nav for k in keys + ['bench']}, axis=1)
        d1, d2, d3 = st.columns(3)
        d1.download_button('Curvas de valor (CSV)', navs.to_csv().encode('utf-8-sig'), 'curvas.csv', 'text/csv')
        d2.download_button('Pesos-alvo (CSV)', results[sel].targets.to_csv(index=False).encode('utf-8-sig'),
                           f'pesos_{sel}.csv', 'text/csv')
        d3.download_button('Operações (CSV)', tr.to_csv(index=False).encode('utf-8-sig'), f'operacoes_{sel}.csv', 'text/csv')

    with st.expander('Robustez da rotação por momentum (sensibilidade a parâmetros)'):
        st.caption('Se só uma combinação de parâmetros bate o B&H, o resultado é provavelmente sobreajuste. '
                   'Procure zonas estáveis, não o máximo.')
        if st.button('Calcular grelha (≈ 20 simulações)'):
            g = momentum_grid(version, tuple(basket), currency, real_start, freq, cost, band, cap,
                              defensive, cash_rate, capital)
            st.session_state['grid'] = (tuple(basket), g)
        if st.session_state.get('grid') and st.session_state['grid'][0] == tuple(basket):
            g = st.session_state['grid'][1]
            piv = g.pivot_table(index=['Frequência', 'Filtro absoluto'], columns='Top-K', values='Sharpe')
            st.markdown(f'**Sharpe** (B&H = {all_metrics.loc[ref.name, "Sharpe"]:.2f})')
            st.dataframe(piv.style.format('{:.2f}').background_gradient(cmap='RdYlGn', axis=None),
                         width='stretch')
            piv2 = g.pivot_table(index=['Frequência', 'Filtro absoluto'], columns='Top-K', values='CAGR')
            st.markdown(f'**CAGR %** (B&H = {all_metrics.loc[ref.name, "CAGR"]:.1%})')
            st.dataframe(piv2.style.format('{:.1f}').background_gradient(cmap='RdYlGn', axis=None),
                         width='stretch')

# ===================================================================== ROTAÇÃO
with tab_rot:
    groups = {'Setores': 'Setor', 'Regiões': 'Região', 'Estilo / dimensão': 'Estilo',
              'Defensivos': 'Defensivo', 'Só o cabaz': None}
    gsel = st.radio('Universo a analisar', list(groups), horizontal=True)
    if groups[gsel] is None:
        uni = basket
    else:
        uni = [t for t in available if describe(t)[0] == groups[gsel]]
    rbench = st.selectbox('Comparar contra', [bench] + [t for t in ['SPY', 'URTH', 'ACWI'] if t in available and t != bench],
                          format_func=label, key='rbench')
    uni = [t for t in uni if t != rbench] or uni
    missing_g = [t for t, (g, _) in CATALOG.items() if groups[gsel] and g == groups[gsel] and t not in data]
    if missing_g:
        st.caption('Sem dados ainda (adicionados ao catálogo, aguardam o workflow): ' + ', '.join(missing_g))
    if len(uni) < 1:
        st.info('Sem ETF disponíveis neste grupo.')
    else:
        closes = pd.concat({t: to_currency_close(data, t, currency) for t in uni + [rbench]}, axis=1).ffill(limit=3)
        closes = closes.dropna(how='all')
        rt = rotation_table(closes, rbench)
        rr = rrg(closes[uni].dropna(how='all'), closes[rbench])
        cl, cr = st.columns([1.15, 1])
        with cl:
            fig = go.Figure()
            for q, (x0, x1, y0, y1, col) in {'Liderança': (100, 200, 100, 200, 'rgba(5,150,105,.07)'),
                                             'Enfraquecimento': (100, 200, 0, 100, 'rgba(217,119,6,.07)'),
                                             'Atraso': (0, 100, 0, 100, 'rgba(220,38,38,.07)'),
                                             'Melhoria': (0, 100, 100, 200, 'rgba(37,99,235,.07)')}.items():
                fig.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=col, line_width=0, layer='below')
            for t in uni:
                d = rr[rr.ticker == t]
                if d.empty:
                    continue
                inb = t in basket
                fig.add_scatter(x=d.ratio, y=d.momentum, mode='lines+markers', name=t,
                                line=dict(width=2.5 if inb else 1.2),
                                marker=dict(size=[4] * (len(d) - 1) + [12]),
                                hovertemplate=f'{t}<br>RS-rácio %{{x:.1f}}<br>RS-momentum %{{y:.1f}}<extra></extra>')
                fig.add_annotation(x=d.ratio.iloc[-1], y=d.momentum.iloc[-1], text=f'<b>{t}</b>' if inb else t,
                                   showarrow=False, yshift=12, font=dict(size=11))
            if len(rr):
                pad = 1.0
                fig.update_xaxes(range=[min(rr.ratio.min(), 98) - pad, max(rr.ratio.max(), 102) + pad])
                fig.update_yaxes(range=[min(rr.momentum.min(), 98) - pad, max(rr.momentum.max(), 102) + pad])
            for lbl, ax, ay in [('Liderança', 1, 1), ('Enfraquecimento', 1, 0), ('Atraso', 0, 0), ('Melhoria', 0, 1)]:
                fig.add_annotation(xref='paper', yref='paper', x=ax, y=ay, text=lbl, showarrow=False,
                                   xanchor='right' if ax else 'left', yanchor='top' if ay else 'bottom',
                                   font=dict(color='#6B7280', size=12))
            fig.update_layout(height=520, showlegend=False, margin=dict(t=10, b=10),
                              xaxis_title=f'Força relativa vs {rbench} (100 = média 26 sem.)',
                              yaxis_title='Momentum da força relativa (4 sem.)')
            st.plotly_chart(fig, width='stretch')
            st.caption('Rotação típica no sentido horário: Melhoria → Liderança → Enfraquecimento → Atraso. '
                       'Cauda = últimas 8 semanas; negrito = ETF no cabaz. Inspirado no RRG (não é a fórmula JdK).')
        with cr:
            last_q = rr.groupby('ticker').tail(1).set_index('ticker')
            rt['Quadrante'] = [quadrant(last_q.ratio.get(t, np.nan), last_q.momentum.get(t, np.nan))
                               if t in last_q.index else '—' for t in rt.index]
            rt['No cabaz'] = [t in basket for t in rt.index]
            view = rt.drop(index=[rbench], errors='ignore')
            show_cols = ['Ranking', 'Momentum', 'RS 3M', 'RS 6M', 'RS 12M', 'Quadrante', '> MM200', 'No cabaz']
            v = view[[c for c in show_cols if c in view]].copy()
            for c in ['Momentum', 'RS 3M', 'RS 6M', 'RS 12M']:
                if c in v:
                    v[c] = v[c] * 100
            v.insert(0, 'Descrição', [describe(t)[1] for t in v.index])
            st.dataframe(v, width='stretch', column_config={
                'Momentum': PCT('Mom. 3/6/12m'), 'RS 3M': PCT('RS 3M'), 'RS 6M': PCT('RS 6M'), 'RS 12M': PCT('RS 12M')})
        opp_in = view[(view['Quadrante'].isin(['Melhoria', 'Liderança'])) & (~view['No cabaz']) & view['> MM200']]
        opp_out = view[(view['Quadrante'] == 'Atraso') & (view['No cabaz'])]
        if len(opp_in) or len(opp_out):
            st.markdown('**Candidatos a rotação** (heurística, não recomendação): '
                        + (f'reforçar/estudar → {", ".join(opp_in.index[:5])}' if len(opp_in) else '')
                        + (' · ' if len(opp_in) and len(opp_out) else '')
                        + (f'rever no cabaz (em atraso) → {", ".join(opp_out.index)}' if len(opp_out) else ''))

    st.subheader('Rácios entre ativos')
    presets = [('DIA', 'QQQ'), ('RSP', 'SPY'), ('XLE', 'SPY'), ('VGK', 'SPY'), ('EEM', 'SPY'), ('VTV', 'QQQ'), ('IWM', 'SPY')]
    presets = [p for p in presets if p[0] in data and p[1] in data]
    pc = st.multiselect('Pares (numerador / denominador)', [f'{a}/{b}' for a, b in presets],
                        default=[f'{a}/{b}' for a, b in presets[:3]])
    rfig = go.Figure()
    for p in pc:
        a, b = p.split('/')
        s = (data[a]['Close'] / data[b]['Close'].reindex(data[a].index)).dropna()
        s = s[s.index >= pd.Timestamp(real_start)]
        rfig.add_scatter(x=s.index, y=100 * s / s.iloc[0], name=f'{a} / {b}')
    rfig.update_layout(height=320, yaxis_title='Base 100', hovermode='x unified', margin=dict(t=10, b=10),
                       legend=dict(orientation='h', y=1.1))
    st.plotly_chart(rfig, width='stretch')
    st.caption('Subida = numerador a ganhar ao denominador (ex. DIA/QQQ ↑ = value/industrial a ganhar à tecnologia).')

# ===================================================================== TÉCNICA
with tab_tec:
    sig = get_signals(version, tuple(basket), 'SPY' if 'SPY' in data else basket[0])
    rows = []
    for t, f in sig.items():
        z = f.iloc[-1]
        rows.append({'ETF': t, 'Descrição': describe(t)[1], 'Score': z.score, 'Semanal': z.weekly, 'Diário': z.daily,
                     'Relativo': z.relative, 'Risco': z.risk, 'Filtro': bool(z.gate),
                     'Posição técnica': 'Comprado' if z.position else 'Fora', 'Leitura': z.leitura})
    st.dataframe(pd.DataFrame(rows).sort_values('Score', ascending=False), hide_index=True, width='stretch',
                 column_config={'Score': st.column_config.ProgressColumn(min_value=0, max_value=100, format='%.0f')})
    st.caption('Score v2: 40% tendência semanal, 30% momentum diário, 20% força relativa vs SPY, 10% risco. '
               'Entrada ≥ 70 com filtro; saída < 40 ou filtro falso. Calculado em USD. '
               'No backtest, o “Filtro técnico” só é avaliado nas datas de rebalanceamento.')
    c1, c2 = st.columns([1, 3])
    chosen = c1.selectbox('ETF', basket, format_func=label)
    nview = c2.select_slider('Período', [126, 252, 504, 1008], 252, format_func=lambda n: f'{n // 252 or 0.5} ano(s)' if n >= 252 else '6 meses')
    v = sig[chosen].tail(nview)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.7, .3], vertical_spacing=.05)
    fig.add_candlestick(x=v.index, open=v.Open, high=v.High, low=v.Low, close=v.Close, name='OHLC (USD)', row=1, col=1)
    for c, n, col in [('sma50', 'MM50', '#2563EB'), ('sma200', 'MM200', '#DC2626')]:
        fig.add_scatter(x=v.index, y=v[c], name=n, line=dict(width=1.3, color=col), row=1, col=1)
    on = v.position.astype(bool)
    fig.add_scatter(x=v.index, y=v.score, name='Score', line=dict(color='#7C3AED', width=1.8), row=2, col=1)
    fig.add_scatter(x=v.index[on], y=v.score[on], mode='markers', marker=dict(size=3, color='#059669'),
                    name='Posição = comprado', row=2, col=1)
    for th, col in [(70, '#059669'), (40, '#DC2626')]:
        fig.add_hline(y=th, line_dash='dash', line_color=col, opacity=.6, row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, hovermode='x unified',
                      legend=dict(orientation='h', y=1.05), margin=dict(t=30, b=10))
    st.plotly_chart(fig, width='stretch')

# ===================================================================== METODOLOGIA
with tab_met:
    st.markdown(f"""
**Relógio e ausência de look-ahead.** Em cada data de rebalanceamento a regra só vê fechos até à sessão
anterior; a ordem é executada na abertura seguinte e avaliada ao fecho. O alvo “de hoje” no separador
Decisão usa o último fecho disponível.

**Estratégias**
- **Buy & Hold** — compra os pesos da carteira de referência na primeira abertura e nunca mais negoceia (os pesos derivam).
- **Rebalanceamento periódico** — mesmos pesos, repostos em cada período. Isola o efeito “rebalancear” do efeito “otimizar”.
- **Mínima variância** — minimiza a variância estimada com covariância Ledoit-Wolf (janela de {lookback} sessões), long-only, peso máximo {cap:.0%}.
- **Paridade de risco** — cada ETF contribui igualmente para o risco total (formulação convexa de Spinu); com limite ativo é aproximada.
- **Rotação por momentum** — escolhe os {top_k} ETF com maior média de retornos a 3/6/12 meses; pesos ∝ 1/volatilidade 63 sessões;
  {'só entram ETF com momentum > 0 e ' if abs_filter else ''}as vagas vazias vão para {defensive or 'liquidez'}.
- **Filtro técnico** — pesos de referência só nos ETF com posição técnica “comprado” (score v2); o resto vai para {defensive or 'liquidez'}.

**Porque não há “máximo Sharpe”?** Otimizar com retornos médios históricos é a forma mais conhecida de
sobreajuste em carteiras: pequenos erros nos retornos esperados produzem pesos extremos e instáveis
(Michaud, 1989; DeMiguel et al., 2009 mostram que 1/N bate a maioria destes modelos fora da amostra).
As regras incluídas usam apenas risco (variância/covariância) ou ordenação de momentum, que são mais estáveis.

**Moeda.** Em EUR, preço_EUR = preço_USD ÷ EURUSD (abertura com taxa de abertura, fecho com taxa de fecho);
covariâncias e momentum são calculados já em EUR. Não simula spread cambial da corretora.

**Pressupostos e limitações**
- ETF cotados nos EUA como *proxies* de exposições; na UE a compra efetiva é via UCITS (custos, replicação e fiscalidade diferem).
- Preços ajustados (dividendos reinvestidos implicitamente), unidades fracionárias, custos {cost:.0f} p.b. sobre o valor negociado,
  sem impostos. Liquidez remunerada a {cash_rate:.2%}/ano.
- O universo e os parâmetros foram escolhidos hoje, com conhecimento do passado (viés de seleção). Use a grelha de robustez.
- Calendário = interseção das sessões dos ETF do cabaz{f'; {dropped} sessões sem câmbio foram excluídas' if dropped else ''}.
- Investigação, não aconselhamento financeiro.
""")

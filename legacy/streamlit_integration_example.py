"""Example to insert in app.py after loading data and before portfolio_v21."""
from optimizer import returns_from_prices, optimize_weights, diagnostics

# Data 'data' is the existing dict ticker -> OHLC DataFrame; fx is EURUSD=X.
# Prices use the same valuation currency as the simulated portfolio.
if len(choices) >= 2:
    st.sidebar.subheader('Otimização de pesos (experimental)')
    optimize_on = st.sidebar.checkbox('Calcular pesos automaticamente', value=False)
    if optimize_on:
        method = st.sidebar.selectbox('Critério', ['min_vol','risk_parity','equal'],
                                      format_func=lambda x: {'min_vol':'Mínima volatilidade','risk_parity':'Paridade de risco','equal':'Pesos iguais'}[x])
        lookback = st.sidebar.selectbox('Janela histórica', [252,504,756], index=1)
        maximum = st.sidebar.slider('Peso máximo por ETF', 0.10, 1.0, 0.50, 0.05)
        closes = pd.concat({t: data[t]['Close'] for t in choices}, axis=1).dropna()
        if currency == 'EUR':
            fxrate = fx['Close'].reindex(closes.index).ffill(limit=5)
            closes = closes.div(fxrate, axis=0).dropna()
        try:
            ret = returns_from_prices(closes, lookback)
            proposal = optimize_weights(ret, method=method, max_weight=maximum)
            weights = proposal.to_dict()  # Supply to existing portfolio_v21 call.
            st.subheader('Proposta de alocação')
            st.dataframe((proposal*100).round(2).rename('Peso (%)'))
            st.json(diagnostics(ret,proposal))
        except (ValueError, RuntimeError) as exc:
            st.warning(str(exc))

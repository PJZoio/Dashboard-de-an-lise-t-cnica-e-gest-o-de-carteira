# ETF Signals v2.0 — Streamlit

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run from Google Colab

In Colab upload/unzip the package, then run:

```python
!pip -q install -r etf_sinais_v2/requirements.txt
!streamlit run etf_sinais_v2/app.py --server.port 8501 &
```

Colab doesn't expose Streamlit directly to the internet: use a **trusted, authorized tunnel** or run the app locally. Do not publish the dashboard with sensitive portfolio settings publicly. Alternative: use the Colab notebook that accompanies the archive for charts and CSV exports without any tunnel.

## Important design choices

- SPY is the US market benchmark. For each selected ETF, component **relative** compares ETF/SPY; Dow/Nasdaq, equal weight/SPY and energy/SPY are visual contextual ratios, not yet incorporated into the composite score.
- 40% weekly trend, 30% daily momentum, 20% ETF/SPY relative momentum, 10% risk.
- Entry at close: score >=70 and gate true. Exit at close: score <40 or gate false. Orders are **simulated at the following regular session opening**, with open-price fractional share targets and bps costs.
- Portfolio assets are quoted in USD; US ETFs may not be directly available as UCITS to retail EU investors. No automatic currency conversion.
- Adjusted historical OHLC obtained from Yahoo Finance, which may vary over time. No broker execution.
- Backtest is preliminary and in-sample, with fixed parameters, no causal claim of future returns.
- Simulations use the **intersection** of daily calendars for selected ETFs. For multi-geography funds listed in the US this is generally consistent; true non-US listed UCITS require a more sophisticated calendar and FX engine.
- The weekly trend becomes available after Friday's weekly close and enters same-Friday AFTER-CLOSE signals; orders only next session opening.
- Each selected ETF has a maximum weight; the remaining capital stays in cash, and weights are not dynamically redistributed.

## Outputs

Download from the application: summary, historical NAV, order ledger, historical component scores + indicators for selected ETF.

## Colab sem túnel (recomendado)

Abra `ETF_Signals_v2_Colab.ipynb` no Colab, execute a instalação, carregue o ficheiro `etf_sinais_v2.zip` quando solicitado, selecione os ETFs e clique **Analisar**. Produz gráficos e CSV diretamente no Colab. Na versão notebook, a distribuição inicial é equitativa entre os ETFs selecionados; o dashboard Streamlit permite alterar os pesos.

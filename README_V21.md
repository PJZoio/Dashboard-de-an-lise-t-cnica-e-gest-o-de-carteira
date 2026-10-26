# ETF Signals 2.1 — atualização do dashboard

Substitua app.py, engine.py, scripts/refresh_prices.py e adicione watchlist.txt. Preserve data/ com todos os CSV existentes e .github/workflows/prices.yml.

## Novidades
- Pesquisa de ticker por símbolo: introduza ticker na barra lateral, marque-o em ETFs a acompanhar.
- Para histórico no repositório, adicione-o em watchlist.txt e execute GitHub Actions (workflow Atualizar OHLC ETFs), que produzirá data/TICKER.csv. Até lá poderá experimentar fonte Yahoo diretamente, sujeito a bloqueio 429.
- Moeda da carteira EUR/USD: conversão histórica usando data/EURUSD=X.csv, que deverá ser gerado com o workflow atualizado. A aplicação assume que os ativos analisados são originalmente cotados em USD. Não introduza ETF negociado em EUR sem adaptar a lógica de câmbio ao ticker.
- Carteira conjunta: alocação percentual, capital inicial, Buy & Hold por defeito, sinal técnico experimental opcional e tabela de alocação real após ganhos/perdas.
- Custos simplificados por operação, unidades fracionárias, não inclui impostos, câmbio de corretora, dividendos em numerário, nem remuneração de liquidez.

## Atenção metodológica
A estratégia B&H investe na primeira abertura comum; para a série de referência SPY considera-se investimento no primeiro fecho para normalização. Não equivale a uma comparação executável exata das mesmas condições de compra/custos. O modo EUR usa preços ajustados e taxas EURUSD para valorização diária e não simula conversões cambiais separadas, spread cambial ou retenção tributária.

## Início
`pip install -r requirements.txt` e `streamlit run app.py`

Não substitua os ficheiros históricos CSV já gerados no GitHub. Execute o workflow após adicionar os novos ficheiros.

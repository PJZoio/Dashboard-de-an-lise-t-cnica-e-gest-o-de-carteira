# ETF · Rotação e otimização de carteira (v3)

Dashboard Streamlit que responde a duas perguntas:
1. **Que pesos devo ter hoje no cabaz?** (separador *Decisão*: alvo + ordens)
2. **Essa regra teria batido um Buy & Hold**, depois de custos e sem olhar para o futuro? (separador *Backtest*)

## Correr
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estrutura
| Ficheiro | Função |
|---|---|
| `app.py` | Interface (5 separadores: Decisão, Backtest, Rotação, Análise técnica, Metodologia) |
| `core/data.py` | Catálogo de ETF, leitura dos CSV, painel na moeda da carteira (EUR/USD) |
| `core/strategies.py` | Regras de alocação: B&H, rebalanceamento, mínima variância, paridade de risco, rotação por momentum, filtro técnico |
| `core/backtest.py` | Simulador walk-forward único + métricas |
| `core/ucits.py` | Carteira UCITS: ISIN → listagem EUR, pesos da folha, proxy EUA |
| `core/signals.py` | Score técnico v2, tabela de rotação, gráfico de rotação relativa |
| `scripts/refresh_prices.py` | Descarrega OHLC ajustado (Yahoo) para `data/` — usado pelo GitHub Action |
| `legacy/` | Código v2.x substituído (engine, optimizer, optimization_v22, notebook) |

## Relógio da simulação
Decisão com o fecho de D-1 → execução na abertura de D → avaliação ao fecho. Custos sobre o valor negociado; banda de tolerância evita micro-negociação.

## Carteira UCITS (folha Carteira_ETF2.xlsx)
`core/ucits.py` mapeia cada ISIN (coluna O) para uma listagem em EUR, com símbolos alternativos.
O workflow tenta-os por ordem, só aceita séries em EUR e regista a fonte usada em `data/_fontes_ucits.csv`.
Na barra lateral, “Ponto de partida → Carteira UCITS” carrega estes ETF com os pesos da coluna H.

## Novos ETF no catálogo
IWDA.AS (UCITS, EUR), URTH (proxy IWDA), ACWI, setores XLK/XLC/XLY/XLB/XLRE, MTUM, USMV e defensivos SHY/IEF/TLT/AGG/GLD.
Só aparecem depois de correr o workflow **Actions → Atualizar OHLC ETFs → Run workflow**.

Investigação, não aconselhamento financeiro.

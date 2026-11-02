"""Catálogo, leitura de CSV e construção do painel de preços na moeda da carteira.

Todos os tickers do catálogo são ETF cotados em USD (EUA). São proxies de
investigação para exposições que, na prática, um investidor da UE compra via
UCITS. Os sinais e a carteira são calculados na moeda escolhida (EUR ou USD).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ticker: (grupo, descrição). Grupos usados nas vistas de rotação.
CATALOG: dict[str, tuple[str, str]] = {
    # Global / benchmark
    'URTH': ('Global', 'MSCI World (proxy IWDA)'),
    'ACWI': ('Global', 'MSCI ACWI'),
    # Regiões
    'SPY': ('Região', 'EUA · S&P 500'),
    'VGK': ('Região', 'Europa'),
    'FEZ': ('Região', 'Zona Euro · Euro Stoxx 50'),
    'EWG': ('Região', 'Alemanha'),
    'EWU': ('Região', 'Reino Unido'),
    'EWJ': ('Região', 'Japão'),
    'VPL': ('Região', 'Ásia-Pacífico desenvolvido'),
    'FXI': ('Região', 'China large cap'),
    'EWH': ('Região', 'Hong Kong'),
    'INDA': ('Região', 'Índia'),
    'EWY': ('Região', 'Coreia do Sul'),
    'EEM': ('Região', 'Mercados emergentes'),
    # Setores EUA (SPDR)
    'XLK': ('Setor', 'Tecnologia'),
    'XLC': ('Setor', 'Comunicações'),
    'XLY': ('Setor', 'Consumo discricionário'),
    'XLP': ('Setor', 'Consumo básico'),
    'XLV': ('Setor', 'Saúde'),
    'XLF': ('Setor', 'Financeiro'),
    'XLI': ('Setor', 'Indústria'),
    'XLE': ('Setor', 'Energia'),
    'XLU': ('Setor', 'Utilities'),
    'XLB': ('Setor', 'Materiais'),
    'XLRE': ('Setor', 'Imobiliário'),
    # Estilo / dimensão (EUA)
    'QQQ': ('Estilo', 'Nasdaq 100'),
    'DIA': ('Estilo', 'Dow Jones'),
    'RSP': ('Estilo', 'S&P 500 Equal Weight'),
    'IWM': ('Estilo', 'Russell 2000 (small caps)'),
    'VTV': ('Estilo', 'Value'),
    'QUAL': ('Estilo', 'Quality'),
    'SCHD': ('Estilo', 'Dividendos'),
    'MTUM': ('Estilo', 'Momentum'),
    'USMV': ('Estilo', 'Mínima volatilidade'),
    'VOO': ('Estilo', 'S&P 500 (Vanguard)'),
    # Defensivos
    'SHY': ('Defensivo', 'Obrigações EUA 1-3 anos'),
    'IEF': ('Defensivo', 'Obrigações EUA 7-10 anos'),
    'TLT': ('Defensivo', 'Obrigações EUA 20+ anos'),
    'AGG': ('Defensivo', 'Obrigações agregadas EUA'),
    'GLD': ('Defensivo', 'Ouro'),
}
FX_TICKER = 'EURUSD=X'
GROUP_ORDER = ['Global', 'Região', 'Setor', 'Estilo', 'Defensivo', 'Personalizado']


def describe(ticker: str) -> tuple[str, str]:
    return CATALOG.get(ticker, ('Personalizado', 'Ticker da watchlist'))


def _normalise_yahoo(raw: pd.DataFrame | None, ticker: str) -> pd.DataFrame | None:
    """Aceita saída plana ou MultiIndex do yfinance e devolve OHLCV limpo."""
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if ticker in raw.columns.get_level_values(level):
                candidate = raw.xs(ticker, axis=1, level=level).copy()
                break
        else:
            return None
    else:
        candidate = raw.copy()
    required = ['Open', 'High', 'Low', 'Close']
    if any(c not in candidate.columns for c in required):
        return None
    if 'Volume' not in candidate.columns:
        candidate['Volume'] = 0
    candidate = candidate[required + ['Volume']].apply(pd.to_numeric, errors='coerce')
    candidate = candidate.dropna(subset=required)
    candidate = candidate[(candidate[required] > 0).all(axis=1)]
    candidate.index = pd.to_datetime(candidate.index).tz_localize(None)
    candidate = candidate[~candidate.index.duplicated(keep='last')]
    return candidate.sort_index() if not candidate.empty else None


def read_watchlist(path: str | Path = 'watchlist.txt') -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [z.strip().upper() for z in p.read_text().splitlines()
            if z.strip() and not z.strip().startswith('#')]


def load_folder(folder: str | Path = 'data', min_rows: int = 260) -> dict[str, pd.DataFrame]:
    """Lê todos os CSV disponíveis. Ficheiros curtos ou inválidos são ignorados."""
    out: dict[str, pd.DataFrame] = {}
    for fp in sorted(Path(folder).glob('*.csv')):
        sym = fp.stem
        try:
            frame = _normalise_yahoo(pd.read_csv(fp, index_col='Date', parse_dates=['Date']), sym)
        except Exception:
            continue
        if frame is not None and len(frame) >= min_rows:
            out[sym] = frame
    return out


@dataclass
class Panel:
    """Preços alinhados num calendário comum, já na moeda da carteira."""
    open: pd.DataFrame
    close: pd.DataFrame
    currency: str
    dropped_dates: int

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)


def build_panel(data: dict[str, pd.DataFrame], tickers: list[str], currency: str = 'EUR') -> Panel:
    """Calendário = interseção das sessões; sem forward-fill de preços de ativos.

    EUR: preço_EUR = preço_USD / EURUSD (USD por 1 EUR); abertura com a taxa de
    abertura e fecho com a taxa de fecho. O câmbio é preenchido no máximo 5 dias
    (feriados cambiais vs. bolsa dos EUA).
    """
    missing = [t for t in tickers if t not in data]
    if missing:
        raise ValueError('Sem histórico local para: ' + ', '.join(missing))
    opens = pd.concat({t: data[t]['Open'] for t in tickers}, axis=1)
    closes = pd.concat({t: data[t]['Close'] for t in tickers}, axis=1)
    valid = opens.notna().all(axis=1) & closes.notna().all(axis=1)
    n_before = len(opens)
    opens, closes = opens[valid], closes[valid]
    if currency == 'EUR':
        if FX_TICKER not in data:
            raise ValueError('Modo EUR requer data/EURUSD=X.csv.')
        fx = data[FX_TICKER]
        fxo = fx['Open'].reindex(opens.index, method='ffill', limit=5)
        fxc = fx['Close'].reindex(closes.index, method='ffill', limit=5)
        ok = fxo.notna() & fxc.notna()
        opens = opens[ok].div(fxo[ok], axis=0)
        closes = closes[ok].div(fxc[ok], axis=0)
    dropped = n_before - len(opens)
    return Panel(opens.astype(float), closes.astype(float), currency, int(dropped))


def common_start(data: dict[str, pd.DataFrame], tickers: list[str]) -> pd.Timestamp:
    return max(data[t].index.min() for t in tickers)


def last_dates(data: dict[str, pd.DataFrame], tickers: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = [data[t].index.max() for t in tickers if t in data]
    return min(d), max(d)


def to_currency_close(data: dict[str, pd.DataFrame], ticker: str, currency: str) -> pd.Series:
    s = data[ticker]['Close']
    if currency == 'EUR' and FX_TICKER in data:
        fx = data[FX_TICKER]['Close'].reindex(s.index, method='ffill', limit=5)
        s = (s / fx).dropna()
    return s

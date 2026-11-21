"""Descarrega OHLC ajustado do Yahoo para data/*.csv. Correr a partir da raiz do repo.

- ETF dos EUA, câmbio e watchlist: um símbolo cada.
- Carteira UCITS (core/ucits.py): tenta os candidatos por ordem e só aceita
  séries cotadas em EUR; grava com o id da linha e regista a fonte usada em
  data/_fontes_ucits.csv.
Escrita atómica: um histórico válido nunca é substituído por dados vazios.
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data import CATALOG, FX_TICKER, _normalise_yahoo  # noqa: E402
from core.ucits import UCITS  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / 'data'
out.mkdir(exist_ok=True)
custom_file = root / 'watchlist.txt'
custom = ([z.strip().upper() for z in custom_file.read_text().splitlines()
           if z.strip() and not z.strip().startswith('#')] if custom_file.exists() else [])
ucits_ids = {u.id for u in UCITS}
symbols = [s for s in dict.fromkeys([*CATALOG, *custom, FX_TICKER]) if s not in ucits_ids]


def fetch(sym: str) -> pd.DataFrame:
    raw = yf.download(sym, start='2013-01-01', auto_adjust=True, progress=False, threads=False, timeout=25)
    frame = _normalise_yahoo(raw, sym)
    if frame is None or len(frame) < 200:
        raise ValueError('sem pelo menos 200 sessões válidas')
    frame.index.name = 'Date'
    return frame


def currency_of(sym: str) -> str | None:
    t = yf.Ticker(sym)
    try:
        c = t.fast_info.get('currency')
        if c:
            return str(c)
    except Exception:  # noqa: BLE001
        pass
    try:
        return t.info.get('currency')
    except Exception:  # noqa: BLE001
        return None


def save(frame: pd.DataFrame, name: str) -> None:
    tmp = out / f'{name}.tmp'
    frame.to_csv(tmp, index_label='Date', float_format='%.8f')
    tmp.replace(out / f'{name}.csv')


fail, ok = [], 0
for ticker in symbols:
    try:
        f = fetch(ticker)
        save(f, ticker)
        print(f'{ticker}: {len(f)} sessões, último {f.index[-1]:%Y-%m-%d}')
        ok += 1
    except Exception as exc:  # noqa: BLE001
        fail.append(f'{ticker}: {type(exc).__name__}: {exc}')

sources = []
for u in UCITS:
    tried = []
    for sym in u.candidates:
        try:
            ccy = currency_of(sym)
            if ccy is not None and ccy.upper() != 'EUR':
                tried.append(f'{sym} ({ccy}, rejeitado)')
                continue
            f = fetch(sym)
            save(f, u.id)
            sources.append({'id': u.id, 'isin': u.isin, 'simbolo_usado': sym, 'moeda': ccy or '?',
                            'inicio': f.index[0].date(), 'fim': f.index[-1].date(), 'sessoes': len(f)})
            print(f'{u.id} ← {sym} [{ccy or "moeda ?"}]: {len(f)} sessões desde {f.index[0]:%Y-%m-%d}')
            ok += 1
            break
        except Exception as exc:  # noqa: BLE001
            tried.append(f'{sym} ({type(exc).__name__})')
    else:
        fail.append(f'{u.id} / {u.isin}: nenhum candidato válido: ' + '; '.join(tried))
        sources.append({'id': u.id, 'isin': u.isin, 'simbolo_usado': '', 'moeda': '',
                        'inicio': '', 'fim': '', 'sessoes': 0})
pd.DataFrame(sources).to_csv(out / '_fontes_ucits.csv', index=False)

if fail:
    print('FALHAS\n' + '\n'.join(fail), file=sys.stderr)
if ok == 0:
    sys.exit('Nenhum ficheiro atualizado. Teste localmente/Colab ou use uma fonte licenciada.')
print(f'Atualizados: {ok}/{len(symbols) + len(UCITS)}')

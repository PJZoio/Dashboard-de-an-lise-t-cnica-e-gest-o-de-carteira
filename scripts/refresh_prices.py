"""Download Yahoo adjusted OHLC to static snapshot CSV files. Run from repo root."""
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import CATALOG, _normalise_yahoo

root=Path(__file__).resolve().parents[1]
out=root/'data';out.mkdir(exist_ok=True)
fail=[]; successful=0
for ticker in CATALOG:
    try:
        raw=yf.download(ticker,start='2013-01-01',auto_adjust=True,progress=False,threads=False,timeout=25)
        frame=_normalise_yahoo(raw,ticker)
        if frame is None or len(frame)<200:
            raise ValueError('Sem pelo menos 200 sessões válidas')
        frame.index.name='Date'
        # Write atomically: do not overwrite valid history with corrupt/empty data
        tmp=out/f'{ticker}.tmp'
        frame.to_csv(tmp,index_label='Date',float_format='%.8f')
        tmp.replace(out/f'{ticker}.csv')
        print(f'{ticker}: {len(frame)} sessões, último {frame.index[-1]:%Y-%m-%d}')
        successful+=1
    except Exception as exc:
        fail.append(f'{ticker}: {type(exc).__name__}: {exc}')
if fail:
    print('FALHAS\n'+'\n'.join(fail),file=sys.stderr)
if successful == 0:
    sys.exit('Nenhum ficheiro atualizado. Teste localmente/Colab ou use uma fonte licenciada.')
print(f'Atualizados: {successful}/{len(CATALOG)}')

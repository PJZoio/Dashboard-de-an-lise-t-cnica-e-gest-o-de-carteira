"""Carteira UCITS real (folha Carteira_ETF2.xlsx, ISIN da coluna O).

Cada linha usa uma listagem em EUR (Xetra .DE, Euronext .PA/.AS, Milão .MI).
Classes Acc indisponíveis no Big (confirmado 21/11/2026): IE00BFZXGZ54, IE000CK5G8J7,
IE000U58J0M1, IE00BF3N7094 — infraestruturas, energias limpas e high yield ficam em Dist.

O id é o símbolo Yahoo preferido e é também o nome do CSV em data/. O script
de atualização tenta os candidatos por ordem e só aceita séries cotadas em EUR;
se usar um candidato alternativo, grava-o com o id da linha.

Os nomes são os do fundo correspondente ao ISIN (não os da coluna C da folha,
que em alguns casos descrevem outro fundo). Preços Yahoo com auto_adjust:
classes de distribuição ficam ajustadas a dividendos (retorno total aproximado).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ucits:
    id: str                  # símbolo Yahoo preferido = nome do CSV
    sheet: str               # ticker usado na folha
    isin: str
    name: str                # fundo a que o ISIN corresponde
    policy: str              # Acc / Dist
    sleeve: str              # agrupamento para a vista de rotação
    weight: float | None     # peso na folha (coluna H)
    proxy: str | None        # ETF dos EUA com histórico longo
    candidates: tuple[str, ...]
    series_note: str = ''    # quando a série de análise não é a do ISIN comprado


UCITS: list[Ucits] = [
    Ucits('VGWL.DE', 'VAGF.L', 'IE00BK5BQT80', 'Vanguard FTSE All-World', 'Acc', 'Global', .10, 'ACWI',
          ('VGWL.DE', 'VWRL.AS', 'VWRL.MI'),
          'Série da classe Dist IE00B3RBWM25 (mesmo fundo, preço ajustado; histórico desde 2012)'),
    Ucits('SXR8.DE', 'CSPX.L', 'IE00B5BMR087', 'iShares Core S&P 500', 'Acc', 'EUA', .18, 'SPY',
          ('SXR8.DE', 'CSSPX.MI')),
    Ucits('SXRV.DE', 'EQQQ.L', 'IE00B53SZB19', 'iShares Nasdaq-100', 'Acc', 'EUA', .14, 'QQQ',
          ('SXRV.DE', 'SXRV.F')),
    Ucits('RS2K.PA', 'RS2K.PA', 'LU1681038672', 'Amundi Russell 2000', 'Acc', 'EUA', .05, 'IWM',
          ('RS2K.PA', 'RS2K.MI')),
    Ucits('XDEW.DE', 'XDEW.L', 'IE00BLNMYC90', 'Xtrackers S&P 500 Equal Weight', 'Acc', 'EUA', .05, 'RSP',
          ('XDEW.DE', 'XDEW.MI')),
    Ucits('MEUD.PA', 'MEUD.PA', 'LU0908500753', 'Amundi Core Stoxx Europe 600', 'Acc', 'Europa', .07, 'VGK',
          ('MEUD.PA', 'MEUD.MI')),
    Ucits('APEX.MI', 'PAAS.PA', 'LU1900068161', 'Amundi MSCI AC Asia Ex Japan', 'Acc', 'Ásia/EM', .08, 'AAXJ',
          ('APEX.MI', 'APX.MI', 'AASI.PA', 'LASI.DE')),
    Ucits('IQQI.DE', 'INFR.L', 'IE00B1FZS467', 'iShares Global Infrastructure', 'Dist', 'Temático', .04, 'IGF',
          ('IQQI.DE', 'INFR.MI')),
    Ucits('IQQH.DE', 'IQQH.DE', 'IE00B1XNHC34', 'iShares Global Clean Energy', 'Dist', 'Temático', .05, 'ICLN',
          ('IQQH.DE', 'INRG.MI')),
    Ucits('HLTW.PA', 'HEAL.L', 'LU0533033238', 'Amundi MSCI World Health Care', 'Acc', 'Temático', .06, 'XLV',
          ('HLTW.PA', 'LYPE.DE', 'HLTW.MI')),
    Ucits('EUNM.DE', 'ISAC.L', 'IE00B4L5YC18', 'iShares MSCI EM', 'Acc', 'Ásia/EM', .08, 'EEM',
          ('EUNM.DE', 'IEMA.AS', 'SEMA.MI')),
    Ucits('GOLD.PA', 'SGLD.L', 'FR0013416716', 'Amundi Physical Gold ETC', 'Acc', 'Defensivo', .05, 'GLD',
          ('GOLD.PA', 'GOLD.MI')),
    Ucits('XGLE.DE', 'IEGA.L', 'LU0290355717', 'Xtrackers II Eurozone Government Bond 1C', 'Acc', 'Defensivo', .03,
          None, ('XGLE.DE', 'XGLE.MI')),
    Ucits('EUNW.DE', 'IHYG.L', 'IE00B66F4759', 'iShares € High Yield Corp Bond', 'Dist', 'Defensivo', .02, 'HYG',
          ('EUNW.DE', 'IHYG.MI')),
    Ucits('IWDA.AS', 'IWDA', 'IE00B4L5Y983', 'iShares Core MSCI World', 'Acc', 'Global', None, 'URTH',
          ('IWDA.AS', 'EUNL.DE', 'SWDA.MI')),
]
BY_ID = {u.id: u for u in UCITS}
SHEET_WEIGHTS = {u.id: u.weight for u in UCITS if u.weight}
UCITS_DEFENSIVE = 'XGLE.DE'

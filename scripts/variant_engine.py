"""
variant_engine.py — 買う順番の並走記録（2026年11月10日。計画はプロジェクトの variant_forward_plan_2026-11-10）

本の検証（21〜34段目）で、両方の期間ででたらめより良かった形は、どれも「利益を出していて、現金が多く借金が少ない割安株」
（ヘムの H0・山口揚平の保守的な安全域）だった。本番の「低PBR順」と「安全域順」を、2026年12月1日から同じ1,000万円で並べて記録する。

  PBR   … 2026年11月12日までの本番（8条件・利回りの足切り・PBR が全銘柄の安い側20％・Tier 順・同じ Tier の中は PBR の低い順）
  SAFE  … 足切りと順番だけを安全域に変える（安全域が全銘柄の大きい側20％・Tier 順・同じ Tier の中は安全域の大きい順）
  YIELD … 2026年11月13日からの本番（8条件・利回りの足切り・Tier 順・同じ Tier の中は利回りの高い順。PBR の足切りなし）
          11月13日に足した。記録を始める12月1日より前なので、variant_forward_plan_2026-11-10 の決まりに書き足してある

  D1    … YIELD ＋増配の発表で買い増す（予想配当が前の月より5％以上増えた持ち株を目標の1.5倍まで。足りなければ
          目標をいちばん上回る持ち株を目標まで売って回す）。20年分の long_trade で平均は両方の期間でプラス、悪い条件で
          基準にわずかに届かなかった形。2026年11月15日に足した（計画は rebal_live_plan_2026-11-15）
  YH    … YIELD ＋大きく上がったあと（前の月末までの12か月の TOPIX〈1306〉が +20％以上）だけ、利回り2.0％割れで全部売る。
          20年分の long_yregime で平均は両方の期間でプラス、悪い条件で基準にわずかに届かなかった形。11月15日に足した
  D1・YH は YIELD と並べて比べる（同じ日に同じ1,000万円から始まり、違いはこの決まりだけ）。

どれも本番の portfolio_engine の決まり（予算・売り・配当・税・買い増し・損出し・待機資金の1306）をそのまま使う。
本番の検証運用（portfolio_state.json）とは別のファイルに書き、本番の売買には影響しない。LINE には送らない。
"""
from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import portfolio_engine as pe

log = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / 'data'
START = '2026-12-01'
TOP_PCT = 20.0
VARIANTS = ('PBR', 'SAFE', 'YIELD', 'D1', 'YH')
# 並走記録の D1・YH に足す決まり（portfolio_engine の RAISE_BUY・YIELD_EXIT。本番では None）
EXTRA = {
    'D1': {'RAISE_BUY': {'min': 0.05, 'cap': 1.5, 'trim': True}},
    'YH': {'YIELD_EXIT': {'below': 2.0, 'hot': 0.20}},
}
PBR_TOP_PCT = 20.0   # PBR：全銘柄の中で PBR が安い側20％（generate.py の PBR_TOP_PCT=80 と同じ意味）


def _prepare(stocks: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    out = copy.deepcopy(stocks)
    if kind in ('YIELD', 'D1', 'YH'):      # 利回りの高い順・PBR の足切りなし（本番の BUY_MODE が何であっても、この形で作る）
        for x in out:
            x['buy_sort'] = -(x.get('current_yield') or 0)
            x['signal'] = x.get('signal_pre_rank', x.get('signal'))
        return out
    if kind == 'PBR':        # 低PBR順・PBR が全銘柄の安い側20％（本番の BUY_MODE が何であっても、この形で作る）
        pb = sorted([(x['code'], x['pbr']) for x in out if x.get('pbr') is not None and x['pbr'] > 0], key=lambda t: -t[1])
        n = len(pb)
        rank = {c: (i + 1) / n * 100 for i, (c, _v) in enumerate(pb)} if n else {}
        for x in out:
            x['buy_sort'] = x['pbr'] if x.get('pbr') else 99.0
            pre = x.get('signal_pre_rank', x.get('signal'))
            if pre == 'BUY':
                x['signal'] = 'BUY' if rank.get(x['code'], 0.0) >= 100 - PBR_TOP_PCT else 'NEUTRAL'
            elif x.get('signal') == 'BUY':
                x['signal'] = 'NEUTRAL'
        return out
    sm = sorted([(x['code'], x['safety_y1']) for x in out if x.get('safety_y1') is not None], key=lambda t: t[1])
    n = len(sm)
    rank = {c: (i + 1) / n * 100 for i, (c, _v) in enumerate(sm)} if n else {}
    for x in out:
        v = x.get('safety_y1')
        x['buy_sort'] = -v if v is not None else 99.0
        pre = x.get('signal_pre_rank', x.get('signal'))
        if pre == 'BUY':
            x['signal'] = 'BUY' if rank.get(x['code'], 0.0) >= 100 - TOP_PCT else 'NEUTRAL'
        elif x.get('signal') == 'BUY':
            x['signal'] = 'NEUTRAL'
    return out


def update_variants(stocks: list[dict[str, Any]], etf_px: float | None, etf_ret12: float | None = None) -> None:
    today = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    if today < START:
        log.info('Variant tracks: 記録は %s から', START)
        return
    keep = (pe.STATE_PATH, pe.HISTORY_PATH, pe.send_line_notification, pe.RAISE_BUY, pe.YIELD_EXIT)
    try:
        pe.send_line_notification = lambda m: True
        for k in VARIANTS:
            pe.RAISE_BUY = EXTRA.get(k, {}).get('RAISE_BUY')
            pe.YIELD_EXIT = EXTRA.get(k, {}).get('YIELD_EXIT')
            pe.STATE_PATH = DATA_DIR / f'variant_{k.lower()}_state.json'
            pe.HISTORY_PATH = DATA_DIR / f'variant_{k.lower()}_history.json'
            if not pe.STATE_PATH.exists():
                st = pe.init_state()
                st['started_at'] = today
                st['engine_version'] = pe.ENGINE_VERSION
                st['variant'] = k
                pe.save_json(pe.STATE_PATH, st)
            st = pe.update_portfolio(_prepare(stocks, k), etf_px, etf_ret12)
            log.info('Variant %s: total=¥%s holdings=%d', k, f"{st['total_value']:,.0f}", len(st['holdings']))
    finally:
        pe.STATE_PATH, pe.HISTORY_PATH, pe.send_line_notification, pe.RAISE_BUY, pe.YIELD_EXIT = keep

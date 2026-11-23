"""並走記録 v1（2026年12月1日から）の実行。計画は variant_forward_plan_2026-11-10・rebal_live_plan_2026-11-15、固定の決まりは audit_fix_plan_2026-11-22。

この記録は、固定したコード（scripts/frozen_v1_engine.py と、この scripts/frozen_v1_run.py の VARIANTS・EXTRA。
2026年11月23日の固定のし直しから、権利付き最終日の暦 scripts/market_calendar.py も）だけで動く。
本番の検証運用（scripts/portfolio_engine.py）や並走記録の入口（scripts/variant_engine.py）を変えても、この記録の動きは変わらない。

毎日の流れ（株価の日付 D の実行）：
  1. 本番の判定コード（scripts/generate.py）が固定したときと同じか、指紋（sha256）で確かめる。違えば記録を止め、
     data/frozen/v1/status.json に理由を書く（既存の記録は上書きしない。仕様を変えて続けるときは別の系列 v2 を別のファイルで作る）
  2. 前の営業日（D−1）の判定の写し（data/frozen/v1/inputs/）に、今日（D）の株価を重ねて、5つの形を動かす。
     始値で約定するので「D−1 の夜に決めた売買を、D の始値で約定」になる。評価は D の終値
  3. 今日（D）の判定の写しを残す（翌営業日に使う。日ごとの入力の記録も兼ねる）
  4. 日ごとの記録（data/frozen/v1/log.jsonl）に、入力の写しのファイル・判定の要点・各形の売買と評価を1行ずつ足す
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import frozen_v1_engine as pe

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
REPO = HERE.parent
OUT = REPO / 'data' / 'frozen' / 'v1'
SERIES = 'v1_20261201'
START = '2026-12-01'
TOP_PCT = 20.0
PBR_TOP_PCT = 20.0
VARIANTS = ('PBR', 'SAFE', 'YIELD', 'D1', 'YH')
EXTRA = {
    'D1': {'RAISE_BUY': {'min': 0.05, 'cap': 1.5, 'trim': True}},
    'YH': {'YIELD_EXIT': {'below': 2.0, 'hot': 0.20}},
}
# 株価の側（今日の値を使う）。これ以外は前の営業日の判定の値を使う
PRICE_KEYS = ('price', 'price_history', 'yield_history', 'valuation_history', 'financials_history', 'last_split')
DROP_KEYS = ('links',)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads((HERE / 'frozen_v1_manifest.json').read_text(encoding='utf-8'))


def check_frozen() -> tuple[bool, str]:
    """固定したときと同じコードか。本番の判定（generate.py）と、固定版の2つのファイル（frozen_v1_engine.py・frozen_v1_run.py）。"""
    m = _manifest()
    for rel, h in m.get('sha256', {}).items():
        p = REPO / rel
        if not p.exists():
            return False, f'{rel} がない'
        if _sha(p) != h:
            return False, f'{rel} が固定したときと違う（固定 {h[:12]}・いま {_sha(p)[:12]}）'
    return True, ''


def prepare(stocks: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """買う順番の形ごとに、合図と並べ方を作る（scripts/variant_engine.py の _prepare と同じ決まり。2026年11月21日の版）。"""
    out = copy.deepcopy(stocks)
    if kind in ('YIELD', 'D1', 'YH'):
        for x in out:
            x['buy_sort'] = -(x.get('current_yield') or 0)
            x['signal'] = x.get('signal_pre_rank', x.get('signal'))
        return out
    if kind == 'PBR':
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


def price_date_of(stocks: list[dict[str, Any]]) -> str | None:
    ds = [s['price_history'][-1].get('d') for s in stocks if s.get('price_history')]
    ds = [d for d in ds if d]
    return max(ds) if ds else None


def _judge_part(s: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in s.items() if k not in PRICE_KEYS and k not in DROP_KEYS}


def _write_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))


def _read_gz(path: Path) -> Any:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def _status(obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'status.json').write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')


def _now() -> datetime:
    """いまの日本時間（テストで差し替える）。"""
    return datetime.now(timezone(timedelta(hours=9)))


def update(stocks: list[dict[str, Any]], etf_px: float | None, etf_ret12: float | None = None) -> dict | None:
    pdate = price_date_of(stocks)
    run_at = _now().isoformat(timespec='seconds')
    if not pdate or pdate < START:
        log.info('Frozen %s: 記録は %s から（株価の日付 %s）', SERIES, START, pdate)
        return None
    ok, why = check_frozen()
    if not ok:
        log.warning('Frozen %s: 止めました（%s）。既存の記録は変えていません', SERIES, why)
        _status({'series': SERIES, 'state': 'paused', 'reason': why, 'price_date': pdate, 'run_at': run_at})
        return None
    inputs = OUT / 'inputs'
    latest = inputs / 'latest.json.gz'
    prev = _read_gz(latest) if latest.exists() else None
    today_snap = {'price_date': pdate, 'saved_at': run_at, 'stocks': {str(s['code']): _judge_part(s) for s in stocks}}
    if prev is not None and prev.get('price_date', '') >= pdate:
        log.info('Frozen %s: 株価の日付 %s は処理済み（前回 %s）', SERIES, pdate, prev.get('price_date'))
        return None
    _write_gz(inputs / f'{pdate}.json.gz', today_snap)
    rec = {'price_date': pdate, 'run_at': run_at, 'input_file': f'inputs/{pdate}.json.gz',
           'decided_from': prev.get('price_date') if prev else None,
           'candidates': sorted(c for c, s in today_snap['stocks'].items()
                                if s.get('screening_pass') and s.get('signal') == 'BUY' and not s.get('emergency_exit')),
           'variants': {}}
    # 11月23日：先読みの防止。前の判定を保存した時刻が今日の始値（9:00）より後なら、今日の約定に使わない（合図なしで評価だけ）
    late = prev is not None and (not prev.get('saved_at') or
                                 datetime.fromisoformat(prev['saved_at']) >= datetime.fromisoformat(f'{pdate}T09:00:00+09:00'))
    if late:
        rec['late_judgment'] = prev.get('saved_at')
        rec['decided_from'] = None
        log.warning('Frozen %s: 前の判定（%s）の保存が %s の始値より後なので、今日は売買しない', SERIES, prev.get('price_date'), pdate)
    if prev is None:
        log.info('Frozen %s: 最初の日（%s）。判定を残しただけで、売買は翌営業日の始値から', SERIES, pdate)
    else:
        merged = []
        for s in stocks:
            j = None if late else prev['stocks'].get(str(s['code']))
            if j is None:
                # 前の営業日に判定がない銘柄は、合図なし・印なしとして値段だけ使う（持ち株の評価のため）
                j = {k: v for k, v in _judge_part(s).items()}
                j.update(signal='NEUTRAL', signal_pre_rank='NEUTRAL', emergency_exit=False, screening_pass=False)
            m = dict(j)
            m.update({k: s[k] for k in PRICE_KEYS if k in s})
            merged.append(m)
        keep = (pe.STATE_PATH, pe.HISTORY_PATH, pe.send_line_notification, pe.RAISE_BUY, pe.YIELD_EXIT, pe.TODAY_OVERRIDE)
        try:
            pe.send_line_notification = lambda m_: True
            pe.TODAY_OVERRIDE = date.fromisoformat(pdate)
            for k in VARIANTS:
                pe.RAISE_BUY = EXTRA.get(k, {}).get('RAISE_BUY')
                pe.YIELD_EXIT = EXTRA.get(k, {}).get('YIELD_EXIT')
                pe.STATE_PATH = OUT / f'variant_{k.lower()}_state.json'
                pe.HISTORY_PATH = OUT / f'variant_{k.lower()}_history.json'
                if not pe.STATE_PATH.exists():
                    OUT.mkdir(parents=True, exist_ok=True)
                    st0 = pe.init_state()
                    st0.update(started_at=pdate, engine_version=pe.ENGINE_VERSION, variant=k, series=SERIES)
                    pe.save_json(pe.STATE_PATH, st0)
                st = pe.update_portfolio(prepare(merged, k), etf_px, etf_ret12)
                hist = pe.load_json(pe.HISTORY_PATH, pe.init_history())
                rec['variants'][k] = {
                    'total_value': st.get('total_value'), 'cash': st.get('cash'), 'holdings': len(st.get('holdings', [])),
                    'unpaid_dividends_gross': pe.unpaid_dividends(st),
                    'trades': [t for t in hist.get('trades', []) if t.get('date') == pdate],
                }
                log.info('Frozen %s %s: total=¥%s holdings=%d', SERIES, k, f"{st['total_value']:,.0f}", len(st['holdings']))
        finally:
            (pe.STATE_PATH, pe.HISTORY_PATH, pe.send_line_notification, pe.RAISE_BUY, pe.YIELD_EXIT, pe.TODAY_OVERRIDE) = keep
    _write_gz(latest, today_snap)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'log.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':')) + '\n')
    _status({'series': SERIES, 'state': 'running', 'price_date': pdate, 'run_at': run_at})
    return rec

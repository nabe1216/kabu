"""本番の検証運用（portfolio_engine）に、前の営業日の判定を今日の株価で渡す（2026年11月 engine v5）。

これまでは、その日の終値で判定した合図を、同じ日の始値で約定していた（判定より前の値段で売買していた）。
ここで判定を1日ずらし、「D−1 の夜に決めた売買を、D の始値で約定」にする。並走記録 v1 と同じ決まり。

保存の決まり（v5.1：GPT のレビュー 2-2・2-3 を受けて直した）
- 判定の保存（入力）と、運用の更新が終わったこと（処理済み）を分けて記録する
  - judged/<株価の日付>.json.gz … その日の判定（何度保存しても同じ日付のファイルを上書きするだけ）
  - processed.json … 運用の更新が最後まで終わった株価の日付。generate.py が update_portfolio の成功後に mark_processed で書く
- 同じ株価の日付を2回処理しない：processed.json の日付 >= 今日の株価の日付なら、lagged_stocks は None を返す。
  呼び出し側は None のとき運用の更新を呼ばない
- 判定の保存のあとで運用の更新が失敗したら、processed.json は進まないので、再実行で同じ日をやり直せる
  （取り落としなし）。更新が成功して processed.json を書く前に止まった場合も、再実行では engine の
  「同じ日付は処理済み」の確認（state['last_updated']）で売買が二重にならない
- 前の判定は「今日より前で、いちばん新しい judged/ のファイル」。ただし、その判定の約定予定日（次の営業日）が今日でなければ
  失効させる（2026年11月24日。実行が抜けた日の判定を、後日の寄付きへ持ち越さない）
- 先読みの防止（2026年11月23日・GPT のレビュー）：判定には保存した時刻（saved_at・日本時間）を残す。
  今日の始値（今日 9:00）より後に保存した判定は、今日の約定に使わない（合図なしにする）。
  また、株価の日付 D の判定は、翌営業日の 9:00 を過ぎたら上書きしない（取引が始まったあとの開示を混ぜない）
"""
from __future__ import annotations

import gzip
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))
OPEN = time(9, 0)


def _now() -> datetime:
    """いまの日本時間（テストで差し替える）。"""
    return datetime.now(JST)


def open_of(d: str) -> datetime:
    """その日の取引の始まり（日本時間 9:00）。"""
    return datetime.combine(date.fromisoformat(d), OPEN, tzinfo=JST)


def next_open_after(d: str) -> datetime:
    """株価の日付 d の次の営業日の 9:00（d の判定で約定する時刻）。"""
    from market_calendar import is_business_day
    x = date.fromisoformat(d) + timedelta(days=1)
    while not is_business_day(x):
        x += timedelta(days=1)
    return datetime.combine(x, OPEN, tzinfo=JST)

PRICE_KEYS = ('price', 'price_history', 'yield_history', 'valuation_history', 'financials_history', 'last_split')
DROP_KEYS = ('links',)
KEEP_JUDGED = 15          # judged/ に残す日数（古いものから消す）


def _judge_part(s: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in s.items() if k not in PRICE_KEYS and k not in DROP_KEYS}


def price_date_of(stocks: list[dict[str, Any]]) -> str | None:
    ds = [s['price_history'][-1].get('d') for s in stocks if s.get('price_history')]
    ds = [d for d in ds if d]
    return max(ds) if ds else None


def _read_json(p: Path) -> Any:
    if p.suffix == '.gz':
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _write_atomic(p: Path, data: Any, gz: bool) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    if gz:
        with gzip.open(tmp, 'wt', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    else:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def last_processed(folder: Path) -> str | None:
    p = folder / 'processed.json'
    return _read_json(p).get('price_date') if p.exists() else None


def mark_processed(folder: Path, info: dict) -> None:
    """運用の更新が最後まで終わったあとで呼ぶ。"""
    _write_atomic(folder / 'processed.json',
                  {'price_date': info.get('price_date'), 'decided_from': info.get('decided_from')}, gz=False)


def lagged_stocks(stocks: list[dict[str, Any]], folder: Path) -> tuple[list[dict[str, Any]] | None, dict]:
    """今日の株価に、前の営業日の判定を重ねた stocks を返す。
    - 今日の判定は judged/<株価の日付>.json.gz に保存する（処理済みかどうかに関係なく、毎回）
    - 処理済みの株価の日付なら None（呼び出し側は運用の更新を呼ばない）
    - 前の判定がない日（最初の日）は、合図なし（売買しない）で返す"""
    pdate = price_date_of(stocks)
    jdir = folder / 'judged'
    info: dict[str, Any] = {'price_date': pdate, 'decided_from': None}
    if pdate is None:
        info['skip'] = '株価の日付が分からない'
        return None, info
    now = _now()
    jf = jdir / f'{pdate}.json.gz'
    if jf.exists() and now >= next_open_after(pdate):
        info['kept'] = f'{pdate} の判定は翌営業日の取引が始まったあとなので上書きしない'
    else:
        _write_atomic(jf, {'price_date': pdate, 'saved_at': now.isoformat(timespec='seconds'),
                           'stocks': {str(s['code']): _judge_part(s) for s in stocks}}, gz=True)
    files = sorted(jdir.glob('*.json.gz'))
    for old in files[:-KEEP_JUDGED]:
        old.unlink()
    done = last_processed(folder)
    if done is not None and done >= pdate:
        info['skip'] = f'処理済みの株価の日付（{done}）'
        return None, info
    prevs = [f for f in sorted(jdir.glob('*.json.gz')) if f.name[:10] < pdate]
    prev = _read_json(prevs[-1]) if prevs else None
    if prev is not None:
        sa = prev.get('saved_at')
        if not sa or datetime.fromisoformat(sa) >= open_of(pdate):
            info['late'] = f"{prev.get('price_date')} の判定の保存（{sa}）が今日の始値より後なので、今日は売買しない"
            prev = None
    # 2026年11月24日（GPT のレビュー）：判定は「次の営業日の寄付き」の1回だけに使う（約定予定日）。
    # 実行が抜けて約定予定日を過ぎた判定は失効させ、後日の寄付きへ持ち越さない（その日は売買しない）
    if prev is not None and next_open_after(prev['price_date']).date().isoformat() != pdate:
        info['expired'] = (f"{prev.get('price_date')} の判定の約定予定日は "
                           f"{next_open_after(prev['price_date']).date().isoformat()}。過ぎたので今日は売買しない")
        prev = None
    info['decided_from'] = prev.get('price_date') if prev else None
    merged = []
    for s in stocks:
        j = (prev or {}).get('stocks', {}).get(str(s['code']))
        if j is None:
            j = _judge_part(s)
            j.update(signal='NEUTRAL', signal_pre_rank='NEUTRAL', emergency_exit=False, screening_pass=False)
        m = dict(j)
        m.update({k: s[k] for k in PRICE_KEYS if k in s})
        merged.append(m)
    return merged, info

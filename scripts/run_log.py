"""毎日の実行の起動のしかたと時刻を data/run_log.json に残す（2026年11月24日・GPT のレビュー）。

本番の起動は火〜土の 1:05（日本時間）の API 起動（workflow_dispatch）。J-Quants の財務の確報（24時30分ごろ）のあとを狙った取得にするため。
ここでは、起動の種類・開始・判定の保存の時刻・株価の日付・処理済みの日付だけを残す（値段や配当の値は書かない。公開リポジトリのため）。
"""
from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
DATA = Path(__file__).parent.parent / 'data'
KEEP = 90


def basis(started: datetime) -> str:
    """開始の時刻から、財務の確報のあとの取得かどうかを書く（確報は24時30分ごろ。1:00〜9:00 の開始を確報後とする）。"""
    hm = started.hour * 100 + started.minute
    return '確報後' if 100 <= hm < 900 else '確報前か取引時間中'


def main() -> None:
    p = DATA / 'run_log.json'
    log = json.loads(p.read_text(encoding='utf-8')) if p.exists() else []
    r = json.loads((DATA / 'results.json').read_text(encoding='utf-8'))
    pi = DATA / 'paper_inputs'
    proc = json.loads((pi / 'processed.json').read_text(encoding='utf-8')) if (pi / 'processed.json').exists() else {}
    pdate = r.get('latest_price_date')
    saved = None
    jf = pi / 'judged' / f'{pdate}.json.gz'
    if pdate and jf.exists():
        with gzip.open(jf, 'rt', encoding='utf-8') as f:
            saved = json.load(f).get('saved_at')
    st = os.environ.get('RUN_STARTED_AT')
    started = datetime.fromtimestamp(int(st), JST) if st else datetime.now(JST)
    log.append({
        'run_id': os.environ.get('RUN_ID'),
        'event': os.environ.get('EVENT'),
        'started_at': started.isoformat(timespec='seconds'),
        'basis': basis(started),
        'generated_at': r.get('generated_at'),
        'price_date': pdate,
        'judged_saved_at': saved,
        'processed': proc,
    })
    p.write_text(json.dumps(log[-KEEP:], ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(log[-1], ensure_ascii=False))


if __name__ == '__main__':
    main()

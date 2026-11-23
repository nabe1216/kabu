"""固定記録 v1 の指紋（frozen_v1_manifest.json の sha256）を、いまのファイルで書き直す。

記録の開始（2026-12-01 の株価の日付）より前にだけ使う。開始後に仕様を変えるときは、v1 を上書きせず、
別の系列（v2・frozen_v2_*）を作る（manifest の note のとおり）。
使い方：python scripts/refreeze_v1.py "理由"
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
MANIFEST = HERE / 'frozen_v1_manifest.json'
FILES = ('scripts/generate.py', 'scripts/frozen_v1_engine.py', 'scripts/frozen_v1_run.py', 'scripts/market_calendar.py')


def main() -> int:
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    today = datetime.now(timezone(timedelta(hours=9))).date()
    if today >= date.fromisoformat(m['start']):
        print(f"記録はすでに始まっている（開始 {m['start']}）。v1 は固定し直さず、v2 を作る")
        return 1
    if (REPO / 'data' / 'frozen' / 'v1' / 'log.jsonl').exists():
        print('記録（data/frozen/v1/log.jsonl）がすでにある。v1 は固定し直さない')
        return 1
    reason = sys.argv[1] if len(sys.argv) > 1 else ''
    new = {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in FILES}
    changed = [rel for rel in FILES if m.get('sha256', {}).get(rel) != new[rel]]
    m.setdefault('refrozen', []).append({'on': today.isoformat(), 'reason': reason, 'changed': changed,
                                         'previous_sha256': m.get('sha256', {})})
    m['frozen_on'] = today.isoformat()
    m['sha256'] = new
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('固定し直した：', ', '.join(changed) or '（変更なし）')
    return 0


if __name__ == '__main__':
    sys.exit(main())

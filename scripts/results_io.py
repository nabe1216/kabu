"""results.json と、銘柄ごとの履歴を分けたファイル（data/history/h*.json）の読み書き（2026年11月8日）。

results.json が 67MB まで大きくなった（GitHub は1ファイル100MBを超えると保存を受け付けず、
そうなると毎日の判定が止まる）。中身の98%は銘柄ごとの履歴だったので、
株価・利回り・PER と PBR・財務の履歴を、銘柄コードの先頭の文字ごとのファイルに分ける。

- data/results.json            … 判定に使う値だけ（履歴なし）。history_shards に分けたファイルの一覧
- data/history/h<先頭の文字>.json … {"generated_at": ..., "stocks": {コード: {履歴の4つ}}}

ダッシュボードは銘柄の詳細を開いたときに、その銘柄のファイルだけを読む。
Python から全部読みたいときは load_results() を使う（履歴を元の形に戻して返す）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HISTORY_KEYS = ('price_history', 'yield_history', 'valuation_history', 'financials_history')
HISTORY_DIRNAME = 'history'


def shard_of(code: Any) -> str:
    """銘柄コードの先頭の文字（数字か英字）。それ以外は '0'。"""
    c = str(code or '')[:1]
    return c if c.isalnum() and c.isascii() else '0'


def split_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """payload を、判定の値だけの本体と、履歴のファイルごとの中身に分ける（payload そのものは変えない）。"""
    core_stocks: list[dict[str, Any]] = []
    shards: dict[str, dict[str, Any]] = {}
    for s in payload.get('stocks', []):
        core_stocks.append({k: v for k, v in s.items() if k not in HISTORY_KEYS})
        h = {k: s[k] for k in HISTORY_KEYS if k in s}
        if h:
            shards.setdefault(shard_of(s.get('code')), {})[str(s.get('code'))] = h
    core = {k: v for k, v in payload.items() if k != 'stocks'}
    core['history_shards'] = sorted(shards)
    core['history_keys'] = list(HISTORY_KEYS)
    core['stocks'] = core_stocks
    return core, shards


def _dump(obj: Any, path: Path) -> int:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'), default=str)
    tmp.replace(path)
    return path.stat().st_size


def write_results(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """本体を path に、履歴を path の隣の history/ に書く。書いたファイルの大きさを返す。"""
    core, shards = split_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    hist_dir = path.parent / HISTORY_DIRNAME
    hist_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name, data in shards.items():
        p = hist_dir / f'h{name}.json'
        sizes[p.name] = _dump({'generated_at': payload.get('generated_at'), 'stocks': data}, p)
    for p in hist_dir.glob('h*.json'):
        if p.stem[1:] not in shards:
            p.unlink()
    sizes[path.name] = _dump(core, path)
    return sizes


def load_results(path: Path, with_history: bool = True) -> dict[str, Any]:
    """results.json を読む。with_history なら、分けた履歴を各銘柄に戻して返す
    （履歴が results.json の中にある古い形でも、そのまま読める）。"""
    with Path(path).open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not with_history or not data.get('history_shards'):
        return data
    by_code = {str(s.get('code')): s for s in data.get('stocks', [])}
    hist_dir = Path(path).parent / HISTORY_DIRNAME
    for name in data['history_shards']:
        p = hist_dir / f'h{name}.json'
        if not p.exists():
            continue
        with p.open('r', encoding='utf-8') as f:
            hs = json.load(f).get('stocks', {})
        for code, h in hs.items():
            s = by_code.get(code)
            if s is not None:
                s.update(h)
    return data

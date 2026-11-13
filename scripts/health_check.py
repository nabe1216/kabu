"""毎日の自動点検（2026年11月8日）。

毎日の判定（generate.py）のあとに動き、結果を data/health.json に書く。ダッシュボードは
警告があれば画面の上に出す。この3か月で見つかった大きな誤り（分割を減配と読む・配当を2回数える・
1銘柄に4割が偏る）は、どれも黙って起きていたので、決まった点を毎日機械的に確かめる。

点検は判定や売買を一切変えない。失敗しても毎日の判定の保存は止めない（daily.yml で continue-on-error）。

使い方：python scripts/health_check.py [--prev 前回の results.json]
  --prev を省くと、git の HEAD にある data/results.json（前回の結果）と比べる。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RESULTS = DATA / 'results.json'
STATE = DATA / 'portfolio_state.json'
OUT = DATA / 'health.json'
JST = timezone(timedelta(hours=9))

# 点検の基準（変えるときは、ここだけ）
OVERWEIGHT_X = 2.0            # 持ち株が目標の何倍を超えたら警告するか
CUT_JUMP_ABS = 10             # 減配の緊急が前回から何件増えたら警告するか（かつ CUT_JUMP_REL）
CUT_JUMP_REL = 0.30           #   …前回の何割増えたら
SPLIT_NEAR_CUT_DAYS = 400     # 減配の緊急の銘柄に、この日数以内の分割・併合があれば確かめる
RUN_WARN_MIN = 100            # 実行時間がこの分を超えたら警告（上限は120分）
FILE_WARN_MB = 50             # ファイルがこの大きさを超えたら警告（GitHub の上限は100MB）
MIN_PROCESSED_RATIO = 0.95    # 対象のうち、この割合より処理できなかったら警告
STALE_DAYS = 4                # 株価の最後の日が、今日よりこの日数より古ければ警告
TIER_WEIGHT = {'S': 2.0, 'A': 1.5, 'B': 1.0}
TARGET_HOLDINGS = 15



def _split_in_risky_window(sd: date, fiscal_month) -> bool:
    """分割・併合の日が、期末の7日前〜期末の120日あと（決算短信の開示まで）に入るか（2026年11月10日）。
    この窓の外の分割は、比べる2つの期の配当を同じ係数で直すので、減配の読み違いにならない。期末の月が分からなければ True。"""
    if not fiscal_month:
        return True
    import calendar
    for y in (sd.year + 1, sd.year, sd.year - 1):
        fe = date(y, int(fiscal_month), calendar.monthrange(y, int(fiscal_month))[1])
        if fe - timedelta(days=7) <= sd <= fe + timedelta(days=120):
            return True
    return False

def _load(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _prev_results(arg: str | None) -> dict | None:
    try:
        if arg:
            return _load(Path(arg))
        out = subprocess.run(['git', 'show', 'HEAD:data/results.json'], cwd=ROOT,
                             capture_output=True, timeout=120)
        if out.returncode != 0:
            return None
        return json.loads(out.stdout.decode('utf-8'))
    except Exception:
        return None


def _is_cut(s: dict) -> bool:
    return any('減配' in str(r) for r in s.get('emergency_reasons') or [])


def _counts(res: dict) -> dict[str, int]:
    st = res.get('stocks', [])
    return {
        'pass': sum(1 for s in st if s.get('screening_pass')),
        'buy': sum(1 for s in st if s.get('signal') == 'BUY'),
        'emergency': sum(1 for s in st if s.get('emergency_exit')),
        'cut': sum(1 for s in st if _is_cut(s)),
    }


def run(prev_path: str | None = None, today: date | None = None) -> dict[str, Any]:
    today = today or datetime.now(JST).date()
    warns: list[dict[str, str]] = []
    checks: dict[str, Any] = {}

    def warn(code: str, msg: str) -> None:
        warns.append({'code': code, 'message': msg})

    # 1. 結果のファイル
    try:
        res = _load(RESULTS)
    except Exception as e:
        warn('results_missing', f'results.json を読めない（{e}）')
        res = {'stocks': []}
    cur = _counts(res)
    checks['counts'] = cur

    uni = res.get('universe_count') or 0
    proc = res.get('processed_count') or 0
    checks['processed'] = {'universe': uni, 'processed': proc, 'failures': res.get('failure_count')}
    if uni and proc < uni * MIN_PROCESSED_RATIO:
        warn('processed_low', f'処理できた銘柄が少ない（{proc} / {uni}）')

    lpd = res.get('latest_price_date')
    checks['latest_price_date'] = lpd
    if lpd:
        try:
            age = (today - date.fromisoformat(lpd[:10])).days
            if age > STALE_DAYS:
                warn('stale_prices', f'株価の最後の日が {lpd}（{age}日前）。データが更新されていない可能性')
        except ValueError:
            pass

    # 1b. 「配当を守る会社」のリストの見直し時期（2026年11月13日に追加）。日経累進高配当株指数は毎年6月末に入れ替わるので、
    #     その年の7月15日を過ぎても、確認日がその年の7月1日より前なら知らせる
    pc_raw = (res.get('thresholds') or {}).get('dividend_policy_checked')
    checks['dividend_policy_checked'] = pc_raw
    if pc_raw:
        try:
            pcd = date.fromisoformat(pc_raw[:10])
            due = date(today.year, 7, 1) if today >= date(today.year, 7, 15) else date(today.year - 1, 7, 1)
            if pcd < due:
                warn('div_policy_stale', f'「配当を守る会社」のリスト（Tier と減配の条件の免除に使う）の確認日が {pc_raw}。'
                     f'日経累進高配当株指数の6月の入れ替えと、各社の配当方針を確かめ直す時期')
        except ValueError:
            pass

    # 2. 前回との比べ（減配の緊急の急な増え方）
    prev = _prev_results(prev_path)
    if prev:
        pc = _counts(prev)
        checks['prev_counts'] = pc
        d = cur['cut'] - pc['cut']
        if d >= CUT_JUMP_ABS and d >= pc['cut'] * CUT_JUMP_REL:
            warn('cut_jump', f'減配の緊急が前回の {pc["cut"]} から {cur["cut"]} に増えた。読み違いがないか確かめる')

    # 3. 分割・併合と重なる「減配」
    near = []
    for s in res.get('stocks', []):
        if not _is_cut(s):
            continue
        ls = s.get('last_split') or {}
        try:
            if ls.get('date') and (today - date.fromisoformat(ls['date'])).days <= SPLIT_NEAR_CUT_DAYS \
                    and _split_in_risky_window(date.fromisoformat(ls['date']), s.get('fiscal_month')):
                near.append(f"{s.get('code')} {s.get('name', '')}（分割・併合 {ls['date']}）")
        except ValueError:
            continue
    checks['cut_near_split'] = near
    if near:
        warn('cut_near_split', f'分割・併合の近くで減配と判定した銘柄が {len(near)} 件：' + '、'.join(near[:5])
             + ('…' if len(near) > 5 else '') + '。決算短信で実際の減配か確かめる')

    # 4. 検証運用
    try:
        st = _load(STATE)
        hv = sum(float(h.get('value') or 0) for h in st.get('holdings', []))
        hv += float((st.get('etf') or {}).get('value') or 0)        # 待機資金の 1306（2026年11月10日）
        tv = float(st.get('total_value') or 0)
        cash = float(st.get('cash') or 0)
        checks['portfolio'] = {'total_value': tv, 'cash': cash, 'holdings': len(st.get('holdings', [])),
                               'last_updated': st.get('last_updated'), 'engine_version': st.get('engine_version')}
        if abs(tv - (cash + hv)) > 1:
            warn('pf_total_mismatch', f'検証運用の評価総額（{tv:,.0f}円）が、現金＋持ち株（{cash + hv:,.0f}円）と合わない')
        if st.get('last_updated') != today.isoformat():
            warn('pf_not_updated', f'検証運用が今日更新されていない（最後は {st.get("last_updated")}）')
        if tv > 0:
            avg_w = sum(TIER_WEIGHT.values()) / len(TIER_WEIGHT)
            over = []
            for h in st.get('holdings', []):
                tgt = tv / TARGET_HOLDINGS * TIER_WEIGHT.get(h.get('tier', 'B'), 1.0) / avg_w
                v = float(h.get('value') or 0)
                if tgt > 0 and v > tgt * OVERWEIGHT_X:
                    over.append(f"{h.get('code')} {h.get('name', '')}（資産の{v / tv * 100:.0f}%・目標の{v / tgt:.1f}倍）")
            checks['overweight'] = over
            if over:
                warn('pf_overweight', '目標の2倍を超える持ち株：' + '、'.join(over)
                     + '。値上がりで大きくなったのなら決まりどおり（売らない）だが、ほかの理由なら確かめる')
    except FileNotFoundError:
        pass
    except Exception as e:
        warn('pf_read_error', f'検証運用の記録を読めない（{e}）')

    # 5. 実行時間
    started = os.environ.get('RUN_STARTED_AT')
    if started:
        try:
            mins = (time.time() - float(started)) / 60
            checks['run_minutes'] = round(mins, 1)
            if mins > RUN_WARN_MIN:
                warn('run_long', f'毎日の判定に {mins:.0f}分かかった（上限120分）。上限に近い')
        except ValueError:
            pass

    # 6. ファイルの大きさ
    big = []
    for p in [RESULTS, *sorted((DATA / 'history').glob('h*.json')), DATA / 'stmts_cache.json.gz', DATA / 'stmts_cache.json']:
        if p.exists() and p.stat().st_size > FILE_WARN_MB * 1e6:
            big.append(f'{p.relative_to(ROOT)}（{p.stat().st_size / 1e6:.0f}MB）')
    checks['big_files'] = big
    if big:
        warn('file_big', 'GitHub の上限（100MB）に近いファイル：' + '、'.join(big))

    out = {
        'generated_at': datetime.now(JST).isoformat(timespec='seconds'),
        'date': today.isoformat(),
        'ok': not warns,
        'warnings': warns,
        'checks': checks,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prev', default=None, help='前回の results.json（省くと git の HEAD のもの）')
    a = ap.parse_args()
    out = run(a.prev)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if out['ok']:
        print('自動点検：問題なし')
    else:
        print(f'自動点検：警告 {len(out["warnings"])} 件')
        for w in out['warnings']:
            print(f'  - [{w["code"]}] {w["message"]}')
    summ = os.environ.get('GITHUB_STEP_SUMMARY')
    if summ:
        with open(summ, 'a', encoding='utf-8') as f:
            f.write('## 自動点検\n\n')
            f.write('問題なし\n' if out['ok'] else '\n'.join(f'- {w["message"]}' for w in out['warnings']) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())

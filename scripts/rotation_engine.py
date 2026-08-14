"""
rotation_engine.py — 入れ替えルールの並走フォワードテスト

既存の portfolio_engine.py（固定Q75ルール）とは別に、
「枠が埋まったら弱い保有を良い候補と入れ替える」ルールで
同じ相場・同じ条件の仮想売買を記録する。

■ なぜ作るか
  バックテストでは、入れ替えルールが固定ルールを4設定すべてで上回った。
  ただし検証できたのは上昇相場の2〜3年だけで、判断材料としては弱い。
  そこで、これから先を両方走らせて実データで比べる。
  過去と違って後から条件をいじれないので、自分を騙しようがない。

■ 既存との違いは売買ルールだけ
  初期資金・Tier別予算・保有上限・配当処理・約定価格は portfolio_engine と同一。
  比較を成立させるため、意図的に揃えている。

    売り  既存 … 緊急撤退 または SELLシグナル
          本件 … 上記に加えて「入れ替え」
    買い  両方とも同じ（BUYシグナル・スクリーニング通過）

■ 入れ替えの判定
  保有が上限に達しているとき、
  「候補の利回り分位」が「最も弱い保有の利回り分位」を
  ROTATE_MARGIN ポイント以上上回っていたら交換する。
  分位は results.json の yield_history から、その時点で計算する。

■ 出力（既存とは別ファイル。既存のダッシュボードには影響しない）
  data/rotation_state.json
  data/rotation_history.json

■ 単体実行
  python scripts/rotation_engine.py
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# 既存エンジンの共通部品をそのまま使う。
# 予算や約定の扱いを揃えることが、比較を成立させる条件になる。
from portfolio_engine import (  # noqa: E402
    INITIAL_CASH,
    MAX_HOLDINGS,
    TIER_BUDGET,
    buy_stock,
    get_today_open_price,
    load_json,
    process_dividends,
    save_json,
    sell_holding,
    update_valuations,
)

DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_PATH = DATA_DIR / 'results.json'
STATE_PATH = DATA_DIR / 'rotation_state.json'
HISTORY_PATH = DATA_DIR / 'rotation_history.json'
PORTFOLIO_STATE_PATH = DATA_DIR / 'portfolio_state.json'   # 比較表示用

# 入れ替えの判定基準（利回り分位のポイント差）。
# バックテストでは 8 / 15 / 25 を試し、25 がどの設定でも劣らなかった。
ROTATE_MARGIN = 25.0

# 1日に入れ替える上限。多すぎる売買を防ぐ。
MAX_ROTATIONS_PER_DAY = 2

# 分位を計算するのに最低限必要な履歴の本数
MIN_YIELD_SAMPLES = 12


# ────────────────────────────────── 分位の計算
def yield_percentile(stock: dict[str, Any]) -> float | None:
    """その銘柄の過去の利回り分布のなかで、いまが何％の位置にいるか。

    100 に近いほど「過去と比べて利回りが高い＝割安」。
    results.json の yield_history（月末ごとの利回り）から計算する。
    バックテストで使った pct_own と同じ考え方。
    """
    cur = stock.get('current_yield')
    if cur is None or cur <= 0:
        return None
    hist = stock.get('yield_history') or []
    vals = [h.get('yield') for h in hist if h.get('yield') is not None]
    if len(vals) < MIN_YIELD_SAMPLES:
        return None
    return sum(1 for v in vals if cur >= v) / len(vals) * 100.0


def init_state() -> dict[str, Any]:
    today_jst = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    return {
        'rule': f'入れ替え（差{ROTATE_MARGIN:.0f}pt）',
        'started_at': today_jst,
        'initial_cash': INITIAL_CASH,
        'cash': INITIAL_CASH,
        'holdings': [],
        'total_value': INITIAL_CASH,
        'total_invested': 0,
        'realized_pl': 0,
        'dividend_received': 0,
        'rotation_count': 0,
        'last_updated': '',
    }


def init_history() -> dict[str, Any]:
    return {'trades': [], 'daily_snapshots': [], 'dividends': []}


# ────────────────────────────────── 入れ替え
def do_rotations(
    state: dict[str, Any],
    history: dict[str, Any],
    stocks_by_code: dict[str, dict],
    candidates: list[dict[str, Any]],
    today: date,
) -> list[dict[str, Any]]:
    """枠が埋まっているとき、弱い保有を良い候補と入れ替える。

    「条件を満たしたから売る」ではなく「もっと良いものが現れたから売る」。
    相対的な判断なので、上げ相場でも下げ相場でも同じように機能する。
    """
    rotated: list[dict[str, Any]] = []
    if len(state['holdings']) < MAX_HOLDINGS or not candidates:
        return rotated

    for _ in range(MAX_ROTATIONS_PER_DAY):
        if len(state['holdings']) < MAX_HOLDINGS:
            break

        # 保有のなかで最も分位が低いもの（＝割安さが薄れたもの）
        held_scored = []
        for h in state['holdings']:
            st = stocks_by_code.get(h['code'])
            if not st:
                continue
            p = yield_percentile(st)
            if p is not None:
                held_scored.append((p, h))
        if not held_scored:
            break
        held_scored.sort(key=lambda x: x[0])
        worst_pct, worst = held_scored[0]

        # 候補のなかで最も分位が高いもの
        held_codes = {h['code'] for h in state['holdings']}
        cand_scored = []
        for s in candidates:
            if s['code'] in held_codes:
                continue
            p = yield_percentile(s)
            if p is not None:
                cand_scored.append((p, s))
        if not cand_scored:
            break
        cand_scored.sort(key=lambda x: -x[0])
        best_pct, best = cand_scored[0]

        if best_pct - worst_pct < ROTATE_MARGIN:
            break

        # 交換を実行（売って、次の買いステップで拾われる）
        sell_price = get_today_open_price(stocks_by_code[worst['code']]) \
            or worst.get('current_price') or worst['buy_price']
        reason = (f'ROTATION: {best["code"]} と入れ替え'
                  f'（分位 {worst_pct:.0f} → {best_pct:.0f}）')
        trades_before = len(history['trades'])
        sell_holding(state, history, worst, sell_price, today, reason)
        if len(history['trades']) > trades_before:
            rotated.append(history['trades'][-1])
        state['holdings'] = [h for h in state['holdings'] if h['code'] != worst['code']]
        state['rotation_count'] = state.get('rotation_count', 0) + 1
        log.info('ROTATE out %s (%.0f) → in %s (%.0f)',
                 worst['code'], worst_pct, best['code'], best_pct)

    return rotated


# ────────────────────────────────── メイン
def update_rotation_portfolio(stocks: list[dict[str, Any]]) -> dict[str, Any]:
    today_jst = datetime.now(timezone(timedelta(hours=9))).date()

    state = load_json(STATE_PATH, init_state())
    history = load_json(HISTORY_PATH, init_history())
    stocks_by_code = {s['code']: s for s in stocks}

    if state.get('last_updated') == today_jst.isoformat():
        update_valuations(state, stocks_by_code)
        save_json(STATE_PATH, state)
        log.info('本日は処理済みのため、評価額だけ更新しました（%s）', today_jst)
        return state

    # 1. 配当
    div_count = process_dividends(state, history, today_jst, stocks_by_code)

    # 2. 評価額
    update_valuations(state, stocks_by_code)

    # 3. 売り（緊急撤退・SELLシグナル）※既存と同じ判定
    removed = []
    for i, h in enumerate(state['holdings']):
        st = stocks_by_code.get(h['code'])
        if not st:
            continue
        reason = None
        if st.get('emergency_exit'):
            rs = st.get('emergency_reasons', [])
            reason = 'EMERGENCY_EXIT: ' + (' / '.join(rs) if rs else 'unspecified')
        elif st.get('signal') == 'SELL':
            reason = 'SELL_SIGNAL'
        if reason:
            price = get_today_open_price(st) or h.get('current_price') or h['buy_price']
            sell_holding(state, history, h, price, today_jst, reason)
            removed.append(i)
            log.info('SELL %s - %s', h['code'], reason)
    for i in reversed(removed):
        del state['holdings'][i]

    # 4. 買い候補（既存と同じ条件）
    held = {h['code'] for h in state['holdings']}
    candidates = [
        s for s in stocks
        if s.get('signal') == 'BUY'
        and s.get('screening_pass')
        and not s.get('emergency_exit')
        and s['code'] not in held
    ]

    # 5. 入れ替え（これが既存との唯一の違い）
    rotated = do_rotations(state, history, stocks_by_code, candidates, today_jst)

    # 6. 買い（既存と同じ優先順位）
    held = {h['code'] for h in state['holdings']}
    candidates = [s for s in candidates if s['code'] not in held]
    tier_order = {'S': 0, 'A': 1, 'B': 2}
    candidates.sort(key=lambda s: (
        tier_order.get(s.get('tier', 'B'), 99),
        -(yield_percentile(s) or 0),
        -(s.get('current_yield') or 0),
    ))

    buy_count = 0
    for s in candidates:
        if len(state['holdings']) >= MAX_HOLDINGS:
            break
        if buy_stock(state, history, s, today_jst):
            buy_count += 1
            log.info('BUY %s (%s)', s['code'], s.get('name', ''))

    update_valuations(state, stocks_by_code)

    # 7. 日次スナップショット
    snap = {
        'date': today_jst.isoformat(),
        'cash': state['cash'],
        'holdings_value': state.get('holdings_value', 0),
        'total_value': state['total_value'],
        'unrealized_pl': sum(h.get('unrealized_pl', 0) for h in state['holdings']),
        'realized_pl': state['realized_pl'],
        'dividend_total': state['dividend_received'],
        'holdings_count': len(state['holdings']),
        'rotation_count': state.get('rotation_count', 0),
    }
    history['daily_snapshots'] = [
        x for x in history['daily_snapshots'] if x['date'] != today_jst.isoformat()
    ]
    history['daily_snapshots'].append(snap)

    state['last_updated'] = today_jst.isoformat()
    save_json(STATE_PATH, state)
    save_json(HISTORY_PATH, history)

    log.info('入れ替えトラック更新: 評価額 ¥%s / 保有%d / 買い%d / 売り%d / 入替%d / 配当%d',
             f"{state['total_value']:,}", len(state['holdings']),
             buy_count, len(removed), len(rotated), div_count)
    return state


def compare_line(label: str, st: dict[str, Any]) -> str:
    init = st.get('initial_cash', INITIAL_CASH)
    total = st.get('total_value', init)
    pl = total - init
    pct = pl / init * 100 if init else 0
    return (f"{label:<16}評価額 ¥{int(total):>12,}  "
            f"損益 {pl:+12,.0f}円 ({pct:+6.2f}%)  "
            f"保有 {len(st.get('holdings', [])):>2}銘柄  "
            f"配当累計 ¥{int(st.get('dividend_received', 0)):>9,}")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    if not RESULTS_PATH.exists():
        log.error('%s がありません。先に generate.py を実行してください。', RESULTS_PATH)
        return 1

    with RESULTS_PATH.open('r', encoding='utf-8') as f:
        stocks = json.load(f).get('stocks', [])
    log.info('%d銘柄を読み込みました', len(stocks))

    rot = update_rotation_portfolio(stocks)

    print('\n=== 2つのルールの比較 ===')
    fixed = load_json(PORTFOLIO_STATE_PATH, {})
    if fixed:
        print(compare_line('固定Q75（既存）', fixed))
    print(compare_line(f'入れ替え{ROTATE_MARGIN:.0f}pt', rot))
    print(f"\n入れ替え回数（累計）: {rot.get('rotation_count', 0)}回")
    if fixed and fixed.get('started_at') != rot.get('started_at'):
        print('※ 開始日が違うため、単純比較はできません。'
              '両方の記録が揃った日以降で比べてください。')
    print('※ 手数料・税金は含めていません。')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

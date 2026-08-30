"""
portfolio_engine.py — DIVIDEND HEIST フォワードテストエンジン

シグナルに忠実に従って仮想売買を行い、リアルタイムで運用記録を生成する。
generate.py の実行後に呼び出される。

仕様:
- 初期資金: 1,000万円
- 売買タイミング: ロジック発動時（毎日チェック）
- 約定価格: 当日始値
- 手数料: なし
- 最大保有: 20銘柄
- Tier別予算: S=400万、A=200万、B=100万
- 配当: 権利月該当銘柄に DPS × 保有株数 × 0.5（中間・期末の半額ずつ）
- 利益計算: 含み損益 + 実現損益 + 配当（全部合計）
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# === 設定 ===
DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_PATH = DATA_DIR / 'results.json'
STATE_PATH = DATA_DIR / 'portfolio_state.json'
HISTORY_PATH = DATA_DIR / 'portfolio_history.json'

INITIAL_CASH = 10_000_000  # 1,000万円
MAX_HOLDINGS = 20

# 売却の方針。
#   none      … 売らない。含み益に課税されず、複利で回り続ける
#   emergency … 減配や業績急変のときだけ売る
#   all       … 緊急撤退＋SELLシグナル（もとの動き）
#
# 条件を総当たりで100回検証したところ、
# 「売らない」が現行を95％の条件で上回った。
# 売る判断そのものが税引後では価値を生んでいなかったため none にする。
SELL_MODE = 'emergency'

# Tier別の1銘柄あたりの予算（円）
TIER_BUDGET = {
    'S': 4_000_000,  # 400万円
    'A': 2_000_000,  # 200万円
    'B': 1_000_000,  # 100万円
}

# 目標保有銘柄数。1銘柄あたりの予算はここから逆算する。
# 固定額（S400万＝資産の40%）だと3銘柄で資金が尽き、
# 上限20銘柄が機能していなかったため、総資産に対する比率に変えた。
TARGET_HOLDINGS = 15

# Tierごとの重み。質が高いほど厚く持つ。
TIER_WEIGHT = {'S': 2.0, 'A': 1.5, 'B': 1.0}


def tier_budget(state: dict, tier: str) -> int:
    """1銘柄あたりの予算を、総資産に対する比率で決める。

        予算 = 総資産 ÷ 目標保有数 × （Tierの重み ÷ 重みの平均）

    総資産に連動するので、資産が増えれば1銘柄あたりも自動で増える。
    目標15銘柄なら S約8.9% / A約6.7% / B約4.4% の配分になる。
    """
    total = state.get('total_value') or \
        state.get('initial_cash', INITIAL_CASH)
    avg_w = sum(TIER_WEIGHT.values()) / len(TIER_WEIGHT)
    w = TIER_WEIGHT.get(tier, 1.0)
    return int(total / TARGET_HOLDINGS * (w / avg_w))

# === LINE 通知設定 (GitHub Secretsから読み込み) ===
LINE_CHANNEL_TOKEN = os.environ.get('LINE_CHANNEL_TOKEN', '')


def send_line_notification(message: str) -> bool:
    """LINE Messaging API (Broadcast) で通知を送る

    Bot を友達追加した全ユーザーに送信される。
    LINE_CHANNEL_TOKEN が未設定なら何もしない。
    """
    if not LINE_CHANNEL_TOKEN:
        log.info('LINE notification skipped (LINE_CHANNEL_TOKEN not set)')
        return False

    url = 'https://api.line.me/v2/bot/message/broadcast'
    # メッセージは5000文字まで。長すぎる場合は分割せず切り詰める。
    if len(message) > 4900:
        message = message[:4900] + '\n...(以下省略)'

    payload = json.dumps({
        'messages': [{'type': 'text', 'text': message}]
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {LINE_CHANNEL_TOKEN}',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log.info('LINE notification sent successfully')
                return True
            log.warning('LINE notification: HTTP %s', resp.status)
            return False
    except Exception as e:
        log.error('LINE notification failed: %s', e)
        return False


def build_trade_notification(
    state: dict[str, Any],
    new_buys: list[dict[str, Any]],
    new_sells: list[dict[str, Any]],
    today: date,
) -> str:
    """売買通知メッセージを組み立てる"""
    tier_emoji = {'S': '🥇', 'A': '🥈', 'B': '🥉'}
    lines = [
        '🎯 DIVIDEND HEIST',
        f'📅 {today.strftime("%Y/%m/%d")} 売買通知',
        '',
    ]

    # BUY セクション
    if new_buys:
        lines.append(f'[BUY] {len(new_buys)}件')
        for b in new_buys:
            emoji = tier_emoji.get(b.get('tier', 'B'), '')
            lines.append(
                f"{emoji} {b['code']} {b.get('name', '')}\n"
                f"  {b['qty']:,}株 @¥{int(b['price']):,} = ¥{int(b['amount']):,}"
            )
        lines.append('')

    # SELL セクション
    if new_sells:
        lines.append(f'[SELL] {len(new_sells)}件')
        for s in new_sells:
            emoji = '⚠️' if 'EMERGENCY' in s.get('reason', '') else '💸'
            pl = s.get('realized_pl', 0)
            pl_pct = s.get('realized_pl_pct', 0)
            sign = '+' if pl >= 0 else ''
            days = s.get('holding_days', 0)
            reason_short = s.get('reason', '').split(':')[0] if ':' in s.get('reason', '') else s.get('reason', '')
            lines.append(
                f"{emoji} {s['code']} {s.get('name', '')}\n"
                f"  {s['qty']:,}株 @¥{int(s['price']):,} → {sign}¥{int(pl):,} ({sign}{pl_pct:.2f}%, 保有{days}日)\n"
                f"  理由: {reason_short}"
            )
        lines.append('')

    # ポートフォリオサマリー
    total_value = state.get('total_value', 0)
    initial = state.get('initial_cash', INITIAL_CASH)
    ret = total_value - initial
    ret_pct = (ret / initial * 100) if initial else 0
    sign = '+' if ret >= 0 else ''
    lines.append('📊 PORTFOLIO')
    lines.append(f"評価額: ¥{int(total_value):,} ({sign}{ret_pct:.2f}%)")
    lines.append(f"損益: {sign}¥{int(ret):,}")
    lines.append(f"保有: {len(state.get('holdings', []))} / {MAX_HOLDINGS} 銘柄")
    lines.append(f"現金: ¥{int(state.get('cash', 0)):,}")
    lines.append(f"配当累計: ¥{int(state.get('dividend_received', 0)):,}")

    return '\n'.join(lines)


def load_json(path: Path, default: dict) -> dict:
    """JSON 読み込み（なければ default）"""
    if not path.exists():
        return default
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log.warning('load failed %s: %s', path, e)
        return default


def save_json(path: Path, data: dict) -> None:
    """JSON 保存"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def init_state() -> dict[str, Any]:
    """ポートフォリオ初期状態"""
    today_jst = datetime.now(timezone(timedelta(hours=9))).date().isoformat()
    return {
        'started_at': today_jst,
        'initial_cash': INITIAL_CASH,
        'cash': INITIAL_CASH,
        'holdings': [],
        'total_value': INITIAL_CASH,
        'total_invested': 0,
        'realized_pl': 0,
        'dividend_received': 0,
        'last_updated': '',  # 空文字 → 初回起動時に必ず処理が走る
    }


def init_history() -> dict[str, Any]:
    """履歴の初期状態"""
    return {
        'trades': [],
        'daily_snapshots': [],
        'dividends': [],
    }


def get_today_open_price(stock: dict[str, Any]) -> float | None:
    """価格履歴から直近の始値を取得"""
    history = stock.get('price_history', [])
    if not history:
        # 当日始値がない場合は終値（current price）でフォールバック
        return stock.get('price')
    last = history[-1]
    open_price = last.get('o') or last.get('c') or stock.get('price')
    return open_price


def update_valuations(state: dict[str, Any], stocks_by_code: dict[str, dict]) -> None:
    """保有銘柄の評価額を更新"""
    holdings_value = 0
    for holding in state['holdings']:
        stock = stocks_by_code.get(holding['code'])
        if stock:
            current_price = stock.get('price') or holding['buy_price']
            holding['current_price'] = current_price
            holding['value'] = current_price * holding['qty']
            holding['unrealized_pl'] = (current_price - holding['buy_price']) * holding['qty']
            cost = holding['buy_price'] * holding['qty']
            holding['unrealized_pl_pct'] = round(
                (holding['unrealized_pl'] / cost * 100) if cost > 0 else 0, 2
            )
            holdings_value += holding['value']
        else:
            # 銘柄データが取れない場合は buy_price で評価
            holdings_value += holding.get('value', holding['buy_price'] * holding['qty'])

    state['holdings_value'] = holdings_value
    state['total_value'] = state['cash'] + holdings_value + state['dividend_received']


def process_dividends(
    state: dict[str, Any],
    history: dict[str, Any],
    today: date,
    stocks_by_code: dict[str, dict],
) -> int:
    """配当受領処理（権利月の該当銘柄）

    各銘柄の fiscal_month / interim_month に該当する月の月初に1回処理する。
    1回の受領額 = DPS × 0.5 × 保有株数（年間DPSの半分が期末・中間でそれぞれ受領される）
    """
    received_codes = {d['code'] + '_' + d['date'][:7] for d in history.get('dividends', [])}
    total_received = 0
    received_count = 0

    for holding in state['holdings']:
        code = holding['code']
        stock = stocks_by_code.get(code)
        if not stock:
            continue

        # 権利月チェック
        fiscal_month = stock.get('fiscal_month')
        interim_month = stock.get('interim_month')
        annual_dps = stock.get('forecast_dps') or stock.get('dps')
        if not annual_dps or annual_dps <= 0:
            continue

        # 半額（中間/期末ごと）
        half_dps = annual_dps * 0.5

        # 権利月該当チェック（その月の中で初回のみ処理）
        for div_month, label in [(fiscal_month, 'final'), (interim_month, 'interim')]:
            if not div_month or div_month != today.month:
                continue
            key = f"{code}_{today.strftime('%Y-%m')}"
            if key in received_codes:
                continue

            amount = round(half_dps * holding['qty'])
            state['cash'] += amount
            state['dividend_received'] += amount
            received_codes.add(key)
            total_received += amount
            received_count += 1

            history['dividends'].append({
                'date': today.isoformat(),
                'code': code,
                'name': stock.get('name', ''),
                'type': label,
                'dps': half_dps,
                'shares': holding['qty'],
                'amount': amount,
            })

    return received_count


def sell_holding(
    state: dict[str, Any],
    history: dict[str, Any],
    holding: dict[str, Any],
    sell_price: float,
    today: date,
    reason: str,
) -> None:
    """保有銘柄の売却"""
    qty = holding['qty']
    amount = sell_price * qty
    realized_pl = (sell_price - holding['buy_price']) * qty

    state['cash'] += amount
    state['realized_pl'] += realized_pl

    holding_days = (today - date.fromisoformat(holding['buy_date'])).days

    history['trades'].append({
        'date': today.isoformat(),
        'action': 'SELL',
        'code': holding['code'],
        'name': holding.get('name', ''),
        'tier': holding.get('tier', 'B'),
        'qty': qty,
        'price': sell_price,
        'amount': amount,
        'buy_price': holding['buy_price'],
        'buy_date': holding['buy_date'],
        'holding_days': holding_days,
        'realized_pl': realized_pl,
        'realized_pl_pct': round((sell_price - holding['buy_price']) / holding['buy_price'] * 100, 2),
        'reason': reason,
    })


def buy_stock(
    state: dict[str, Any],
    history: dict[str, Any],
    stock: dict[str, Any],
    today: date,
) -> bool:
    """新規購入"""
    code = stock['code']
    tier = stock.get('tier', 'B')
    budget = tier_budget(state, tier)

    # 利用可能キャッシュが予算未満なら見送り
    if state['cash'] < budget:
        return False

    # 始値取得
    buy_price = get_today_open_price(stock)
    if not buy_price or buy_price <= 0:
        return False

    # 100株単位で計算
    qty = int(budget // buy_price // 100) * 100
    if qty < 100:
        return False

    amount = buy_price * qty
    if amount > state['cash']:
        return False

    # 購入実行
    state['cash'] -= amount
    state['total_invested'] += amount

    holding = {
        'code': code,
        'name': stock.get('name', ''),
        'tier': tier,
        'qty': qty,
        'buy_date': today.isoformat(),
        'buy_price': buy_price,
        'current_price': buy_price,
        'value': amount,
        'unrealized_pl': 0,
        'unrealized_pl_pct': 0,
    }
    state['holdings'].append(holding)

    history['trades'].append({
        'date': today.isoformat(),
        'action': 'BUY',
        'code': code,
        'name': stock.get('name', ''),
        'tier': tier,
        'qty': qty,
        'price': buy_price,
        'amount': amount,
        'reason': 'BUY_SIGNAL',
    })

    return True


def update_portfolio(stocks: list[dict[str, Any]]) -> dict[str, Any]:
    """メイン処理: 毎日のポートフォリオ更新"""
    today_jst = datetime.now(timezone(timedelta(hours=9))).date()

    # 既存データ読み込み
    state = load_json(STATE_PATH, init_state())
    history = load_json(HISTORY_PATH, init_history())

    # 同じ日に既に処理済みならスキップ
    if state.get('last_updated') == today_jst.isoformat():
        # スナップショットだけ更新
        stocks_by_code = {s['code']: s for s in stocks}
        update_valuations(state, stocks_by_code)
        log.info('Already processed today (%s), updated valuations only', today_jst)
        save_json(STATE_PATH, state)
        return state

    stocks_by_code = {s['code']: s for s in stocks}

    # ステップ1: 配当受領処理
    div_count = process_dividends(state, history, today_jst, stocks_by_code)
    if div_count > 0:
        log.info('Received dividends: %d stocks', div_count)

    # ステップ2: 評価額更新
    update_valuations(state, stocks_by_code)

    # ステップ3: 売却判定（緊急撤退、SELLシグナル）
    holdings_to_remove = []
    new_sells: list[dict[str, Any]] = []  # 通知用
    for i, holding in enumerate(state['holdings']):
        code = holding['code']
        stock = stocks_by_code.get(code)
        if not stock:
            continue

        sell_reason = None
        if SELL_MODE != 'none':
            if stock.get('emergency_exit'):
                reasons = stock.get('emergency_reasons', [])
                sell_reason = 'EMERGENCY_EXIT: ' + \
                    (' / '.join(reasons) if reasons else 'unspecified')
            elif SELL_MODE == 'all' and stock.get('signal') == 'SELL':
                sell_reason = 'SELL_SIGNAL'

        if sell_reason:
            sell_price = get_today_open_price(stock) or holding['current_price']
            # 売却前にホールディング情報をコピー（通知用）
            trades_before = len(history['trades'])
            sell_holding(state, history, holding, sell_price, today_jst, sell_reason)
            # 直前に追加された trade レコードを取得
            if len(history['trades']) > trades_before:
                new_sells.append(history['trades'][-1])
            holdings_to_remove.append(i)
            log.info('SELL %s (%s) - %s', code, holding['name'], sell_reason)

    # 売却した銘柄を削除（後ろから削除）
    for i in reversed(holdings_to_remove):
        del state['holdings'][i]

    # ステップ4: 購入判定
    existing_codes = {h['code'] for h in state['holdings']}
    buy_candidates = [
        s for s in stocks
        if s.get('signal') == 'BUY'
        and s.get('screening_pass')
        and not s.get('emergency_exit')
        and s['code'] not in existing_codes
    ]

    # Tier 順、利回り順でソート
    tier_order = {'S': 0, 'A': 1, 'B': 2}
    buy_candidates.sort(
        key=lambda s: (
            tier_order.get(s.get('tier', 'B'), 99),
            -(s.get('current_yield') or 0),
        )
    )

    buy_count = 0
    new_buys: list[dict[str, Any]] = []  # 通知用
    for stock in buy_candidates:
        if len(state['holdings']) >= MAX_HOLDINGS:
            break
        trades_before = len(history['trades'])
        if buy_stock(state, history, stock, today_jst):
            buy_count += 1
            if len(history['trades']) > trades_before:
                new_buys.append(history['trades'][-1])
            log.info('BUY %s (%s)', stock['code'], stock.get('name', ''))

    # 再度評価額更新（購入分を反映）
    update_valuations(state, stocks_by_code)

    # ステップ5: 日次スナップショット
    snapshot = {
        'date': today_jst.isoformat(),
        'cash': state['cash'],
        'holdings_value': state['holdings_value'],
        'total_value': state['total_value'],
        'unrealized_pl': sum(h.get('unrealized_pl', 0) for h in state['holdings']),
        'realized_pl': state['realized_pl'],
        'dividend_total': state['dividend_received'],
        'holdings_count': len(state['holdings']),
    }
    # 同日のスナップショットがあれば置き換え
    history['daily_snapshots'] = [
        s for s in history['daily_snapshots'] if s['date'] != today_jst.isoformat()
    ]
    history['daily_snapshots'].append(snapshot)

    # ステップ6: 完了
    state['last_updated'] = today_jst.isoformat()

    save_json(STATE_PATH, state)
    save_json(HISTORY_PATH, history)

    log.info(
        'Portfolio updated: total_value=¥%s, holdings=%d, buy=%d, sell=%d, div=%d',
        f"{state['total_value']:,}", len(state['holdings']),
        buy_count, len(holdings_to_remove), div_count,
    )

    # ステップ7: LINE通知 (売買が発生した場合のみ)
    if new_buys or new_sells:
        try:
            message = build_trade_notification(state, new_buys, new_sells, today_jst)
            send_line_notification(message)
        except Exception as e:
            log.error('LINE notification build failed: %s', e, exc_info=True)

    return state


def main():
    """単体実行用"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # results.json を読み込む
    if not RESULTS_PATH.exists():
        log.error('%s not found. Run generate.py first.', RESULTS_PATH)
        return 1

    with RESULTS_PATH.open('r', encoding='utf-8') as f:
        results = json.load(f)

    stocks = results.get('stocks', [])
    log.info('Loaded %d stocks from %s', len(stocks), RESULTS_PATH)

    state = update_portfolio(stocks)

    print('\n=== Portfolio Status ===')
    print(f"Started:          {state['started_at']}")
    print(f"Initial cash:     ¥{state['initial_cash']:,}")
    print(f"Current cash:     ¥{state['cash']:,}")
    print(f"Holdings value:   ¥{state.get('holdings_value', 0):,}")
    print(f"Dividend total:   ¥{state['dividend_received']:,}")
    print(f"Realized P/L:     ¥{state['realized_pl']:,}")
    print(f"Total value:      ¥{state['total_value']:,}")
    print(f"Holdings count:   {len(state['holdings'])} / {MAX_HOLDINGS}")

    total_return = state['total_value'] - state['initial_cash']
    return_pct = total_return / state['initial_cash'] * 100
    print(f"Total return:     ¥{total_return:+,} ({return_pct:+.2f}%)")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

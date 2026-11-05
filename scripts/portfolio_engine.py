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
- 評価総額: 現金 + 持ち株の評価額（配当は現金に入るので、別には足さない）

2026年11月6日の変更（20年分の検証に合わせる。計画はプロジェクトの tool_update_2026-11-06）:
- 税: 売却益と配当に 20.315％。売却の損は、その年の配当（源泉された税を戻す）と、のちの売却益・配当と相殺する
  （特定口座で配当を口座で受け取る前提。検証と同じ）
- 配当で買い増し（検証の X8）: 月に1回、新しい買いのあとに残った現金で、目標の大きさ（新しく買うときの予算）を
  いちばん下回っている持ち株を、100株ずつ目標まで買い増す。8条件を通り、減配・業績急変でない銘柄だけ
- 12月の損出し（検証の R6）: 12月15日以降の最初の実行日に、取得単価より2％以上下がっている持ち株を売り、
  同じ株数をすぐ買い戻す（持ち株は変わらず、損が確定して税が戻る）
- 株式分割を反映する（これまでは分割のあと株数が増えず、損に見えていた）
- 配当の二重計上を直す（これまでは評価総額に配当を2回足していた）
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

# 2026年11月6日：20年分の検証に合わせた売買の決まり（tool_update_2026-11-06）
ENGINE_VERSION = 2
TAX_RATE = 0.20315          # 売却益と配当の税（特定口座）
TOPUP = True                # 配当で比率の下がった持ち株を買い増す（検証の X8。「おおむね効く」）
HARVEST = True              # 12月の損出し（検証の R6。161通りで「いつ始めても効く」）
HARVEST_MIN_LOSS = 0.02     # 取得単価より2％以上下がっている持ち株だけ（検証と同じ）
HARVEST_FROM_DAY = 15       # 12月15日以降の最初の実行日（年内に受け渡しが終わるように）
SPLIT_TOL = 0.03            # 株価の記録が3％以上ずれたら、株式分割・併合とみなす


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
    lines.append(f"配当累計（税引き後）: ¥{int(state.get('dividend_received', 0)):,}")

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
        'dividend_gross': 0,
        'tax_paid': 0,
        'loss_pool': 0,
        'year_div_taxed': {},
        'engine_version': ENGINE_VERSION,
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
    # 配当は受け取ったときに現金へ入れているので、ここで足すと2回数えることになる（2026年11月6日に直した）
    state['total_value'] = state['cash'] + holdings_value


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

            gross = round(half_dps * holding['qty'])
            tax = dividend_tax(state, gross, today)
            amount = gross - tax
            state['cash'] += amount
            state['dividend_received'] += amount
            state['dividend_gross'] = state.get('dividend_gross', 0) + gross
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
                'gross': gross,
                'tax': tax,
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
    tax = gain_tax(state, realized_pl, today)

    state['cash'] += amount - tax
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
        'tax': tax,
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
    set_split_ref(holding, stock)
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


# ============================================================
# 2026年11月6日：税・株式分割・配当で買い増し・12月の損出し（tool_update_2026-11-06）
# ============================================================

def _year_taxed(state: dict[str, Any], today: date) -> dict[str, float]:
    yt = state.setdefault('year_div_taxed', {})
    yt.setdefault(str(today.year), 0)
    return yt


def dividend_tax(state: dict[str, Any], gross: float, today: date) -> int:
    """配当の税。たまっている売却の損があれば先に相殺する。戻り値は源泉する税（円）。"""
    if gross <= 0:
        return 0
    pool = float(state.get('loss_pool', 0) or 0)
    use = min(pool, gross)
    state['loss_pool'] = pool - use
    taxable = gross - use
    yt = _year_taxed(state, today)
    yt[str(today.year)] = yt[str(today.year)] + taxable
    tax = round(taxable * TAX_RATE)
    state['tax_paid'] = state.get('tax_paid', 0) + tax
    return tax


def gain_tax(state: dict[str, Any], pl: float, today: date) -> int:
    """売却の税。益なら、たまった損を引いた残りに税（正の値）。損なら、その年に税を払った配当の分の税を戻し
    （負の値）、残りは次の益・配当と相殺するために持っておく。"""
    pool = float(state.get('loss_pool', 0) or 0)
    if pl >= 0:
        use = min(pool, pl)
        state['loss_pool'] = pool - use
        tax = round((pl - use) * TAX_RATE)
        state['tax_paid'] = state.get('tax_paid', 0) + tax
        return tax
    loss = -pl
    yt = _year_taxed(state, today)
    back = min(loss, yt[str(today.year)])
    yt[str(today.year)] -= back
    state['loss_pool'] = pool + (loss - back)
    refund = round(back * TAX_RATE)
    state['tax_paid'] = state.get('tax_paid', 0) - refund
    return -refund


def set_split_ref(holding: dict[str, Any], stock: dict[str, Any]) -> None:
    """株式分割を見つけるための目印（株価の記録の最後の日と、その日の終値）を持っておく。"""
    ph = stock.get('price_history') or []
    if ph and ph[-1].get('c'):
        holding['ref_d'] = ph[-1]['d']
        holding['ref_c'] = ph[-1]['c']


def apply_splits(state: dict[str, Any], history: dict[str, Any], stocks_by_code: dict[str, dict], today: date) -> int:
    """株価の記録は分割のあと過去の値も直して届くので、目印の日の終値がずれていれば分割・併合とみなす。
    株数を増やし（減らし）、取得単価を同じ割合で直す（取得の総額は変えない）。"""
    n = 0
    for h in state['holdings']:
        stock = stocks_by_code.get(h['code'])
        if not stock:
            continue
        ph = stock.get('price_history') or []
        if h.get('ref_d') and h.get('ref_c'):
            now_c = next((p.get('c') for p in ph if p.get('d') == h['ref_d']), None)
            if now_c and now_c > 0:
                f = now_c / float(h['ref_c'])
                if abs(f - 1.0) > SPLIT_TOL:
                    old_qty = h['qty']
                    h['qty'] = int(round(old_qty / f))
                    h['buy_price'] = h['buy_price'] * old_qty / h['qty']
                    history.setdefault('notes', []).append({
                        'date': today.isoformat(), 'code': h['code'], 'name': h.get('name', ''),
                        'note': f'株式分割・併合を反映（{old_qty:,}株 → {h["qty"]:,}株）'})
                    log.info('SPLIT %s %s: %d -> %d', h['code'], h.get('name', ''), old_qty, h['qty'])
                    n += 1
        set_split_ref(h, stock)
    return n


def _target_value(state: dict[str, Any], tier: str) -> float:
    return float(tier_budget(state, tier))


def topup_holdings(state: dict[str, Any], history: dict[str, Any], stocks_by_code: dict[str, dict], today: date) -> list:
    """配当で比率の下がった持ち株を買い増す（検証の X8）。目標の大きさをいちばん下回っている持ち株から、
    100株ずつ目標を超えない範囲で。買えるのは、8条件を通り、減配・業績急変でない銘柄だけ。"""
    done = []
    while True:
        best = None
        for h in state['holdings']:
            st = stocks_by_code.get(h['code'])
            if not st or not st.get('screening_pass') or st.get('emergency_exit'):
                continue
            px = get_today_open_price(st)
            if not px or px <= 0:
                continue
            tgt = _target_value(state, st.get('tier') or h.get('tier', 'B'))
            val = h['qty'] * (st.get('price') or px)
            if tgt - val < 100 * px or state['cash'] < 100 * px:
                continue
            ratio = val / tgt if tgt > 0 else 9.9
            if best is None or ratio < best[0]:
                best = (ratio, h, st, px, tgt - val)
        if best is None:
            break
        _r, h, st, px, room = best
        lots = int(min(room // (100 * px), state['cash'] // (100 * px)))
        if lots <= 0:
            break
        qty = lots * 100
        amount = qty * px
        state['cash'] -= amount
        state['total_invested'] += amount
        h['buy_price'] = (h['buy_price'] * h['qty'] + amount) / (h['qty'] + qty)   # 取得単価は平均でまとめる
        h['qty'] += qty
        rec = {'date': today.isoformat(), 'action': 'BUY', 'code': h['code'], 'name': h.get('name', ''),
               'tier': h.get('tier', 'B'), 'qty': qty, 'price': px, 'amount': amount,
               'reason': '配当で買い増し（目標の大きさまで）'}
        history['trades'].append(rec)
        done.append(rec)
        log.info('TOPUP %s (%s) %d shares', h['code'], h.get('name', ''), qty)
    return done


def harvest_losses(state: dict[str, Any], history: dict[str, Any], stocks_by_code: dict[str, dict], today: date) -> list:
    """12月の損出し（検証の R6）。取得単価より2％以上下がっている持ち株を売って損を確定させ、同じ株数をすぐ買い戻す。"""
    done = []
    for h in state['holdings']:
        st = stocks_by_code.get(h['code'])
        if not st:
            continue
        px = get_today_open_price(st)
        if not px or px <= 0 or px >= h['buy_price'] * (1 - HARVEST_MIN_LOSS):
            continue
        qty = h['qty']
        pl = (px - h['buy_price']) * qty
        tax = gain_tax(state, pl, today)            # 損なので、税が戻る（負の値）
        state['cash'] += px * qty - tax
        state['realized_pl'] += pl
        state['cash'] -= px * qty                    # 同じ株数をすぐ買い戻す
        sell = {'date': today.isoformat(), 'action': 'SELL', 'code': h['code'], 'name': h.get('name', ''),
                'tier': h.get('tier', 'B'), 'qty': qty, 'price': px, 'amount': px * qty,
                'buy_price': h['buy_price'], 'buy_date': h['buy_date'],
                'holding_days': (today - date.fromisoformat(h['buy_date'])).days,
                'realized_pl': pl, 'realized_pl_pct': round((px - h['buy_price']) / h['buy_price'] * 100, 2),
                'tax': tax, 'reason': '12月の損出し（売り。同じ株数をすぐ買い戻す）'}
        buy = {'date': today.isoformat(), 'action': 'BUY', 'code': h['code'], 'name': h.get('name', ''),
               'tier': h.get('tier', 'B'), 'qty': qty, 'price': px, 'amount': px * qty,
               'reason': '12月の損出し（買い戻し）'}
        history['trades'] += [sell, buy]
        done += [sell, buy]
        h['buy_price'] = px
        h['buy_date'] = today.isoformat()
        log.info('HARVEST %s (%s) loss=%s tax_back=%s', h['code'], h.get('name', ''), round(pl), -tax)
    return done


def _close_on(stock: dict[str, Any] | None, d: str, key: str = 'c') -> float | None:
    """株価の記録（分割を直した値）から、d 以前でいちばん新しい日の値。"""
    if not stock:
        return None
    best = None
    for p in stock.get('price_history') or []:
        if p.get('d') and p['d'] <= d and p.get(key):
            best = p[key]
    return best


def migrate_v2(state: dict[str, Any], history: dict[str, Any], stocks_by_code: dict[str, dict]) -> None:
    """2026年11月6日：これまでの記録を、直した決まりで数え直す（1回だけ）。
    ① 持ち株の株式分割（買った日の始値と、いまの株価の記録の同じ日の始値の比で見つける）
    ② 配当と売却益の税（検証と同じ条件。その年の損と配当は相殺）
    ③ 毎日の評価総額（配当の二重計上をやめ、分割と税を入れて、売買の履歴と株価の記録から数え直す）"""
    if state.get('engine_version', 1) >= ENGINE_VERSION:
        return
    notes = history.setdefault('notes', [])
    trades = history.get('trades', [])
    # ① 分割：買った日の始値の比（いまの記録 ÷ 買ったときの値）
    fac = {}
    for t in trades:
        if t.get('action') != 'BUY':
            continue
        st = stocks_by_code.get(t['code'])
        o = None
        if st:
            o = next((p.get('o') for p in st.get('price_history') or [] if p.get('d') == t['date']), None) \
                or _close_on(st, t['date'], 'o')
        f = (o / t['price']) if (o and t.get('price')) else 1.0
        fac[(t['code'], t['date'])] = f if abs(f - 1.0) > SPLIT_TOL else 1.0
    for h in state['holdings']:
        f = fac.get((h['code'], h['buy_date']), 1.0)
        if f != 1.0:
            old_qty = h['qty']
            h['qty'] = int(round(old_qty / f))
            h['buy_price'] = h['buy_price'] * old_qty / h['qty']
            notes.append({'date': datetime.now(timezone(timedelta(hours=9))).date().isoformat(), 'code': h['code'],
                          'name': h.get('name', ''), 'note': f'株式分割を反映（{old_qty:,}株 → {h["qty"]:,}株。記録を数え直した）'})
        st = stocks_by_code.get(h['code'])
        if st:
            set_split_ref(h, st)
    # ② 税：日付の順に、配当と売却益にかける
    events = [(d['date'], 0, d) for d in history.get('dividends', [])] + \
             [(t['date'], 1, t) for t in trades if t.get('action') == 'SELL']
    events.sort(key=lambda x: (x[0], x[1]))
    tmp = {'loss_pool': 0, 'tax_paid': 0, 'year_div_taxed': {}}
    tax_by_date = []
    for d, kind, rec in events:
        dd = date.fromisoformat(d[:10])
        if kind == 0:
            g = rec.get('gross', rec.get('amount', 0))
            tx = dividend_tax(tmp, g, dd)
            rec['gross'] = g
            rec['tax'] = tx
            rec['amount'] = g - tx
        else:
            tx = gain_tax(tmp, rec.get('realized_pl', 0), dd)
            rec['tax'] = tx
        tax_by_date.append((d[:10], tx))
    total_tax = sum(x for _d, x in tax_by_date)
    gross_div = sum(d.get('gross', 0) for d in history.get('dividends', []))
    state['cash'] = state['cash'] - total_tax
    state['tax_paid'] = tmp['tax_paid']
    state['loss_pool'] = tmp['loss_pool']
    state['year_div_taxed'] = tmp['year_div_taxed']
    state['dividend_gross'] = gross_div
    state['dividend_received'] = gross_div - sum(d.get('tax', 0) for d in history.get('dividends', []))
    # ③ 毎日の評価総額を数え直す
    for snap in history.get('daily_snapshots', []):
        d = snap['date']
        held = {}
        for t in trades:
            if t['date'] > d:
                continue
            f = fac.get((t['code'], t['date']), 1.0) if t.get('action') == 'BUY' else None
            q = held.setdefault(t['code'], 0.0)
            if t.get('action') == 'BUY':
                held[t['code']] = q + t['qty'] / f
            else:
                held[t['code']] = 0.0
        hv, ok = 0.0, True
        for code, q in held.items():
            if q <= 0:
                continue
            c = _close_on(stocks_by_code.get(code), d)
            if c is None:
                ok = False
                break
            hv += q * c
        tx = sum(x for dd, x in tax_by_date if dd <= d)
        if ok:
            snap['holdings_value'] = round(hv, 1)
        snap['cash'] = snap['cash'] - tx
        snap['total_value'] = round(snap['cash'] + snap['holdings_value'], 1)
        snap['dividend_total'] = sum(x.get('amount', 0) for x in history.get('dividends', []) if x['date'][:10] <= d)
    notes.append({'date': datetime.now(timezone(timedelta(hours=9))).date().isoformat(), 'code': '', 'name': '',
                  'note': '記録を数え直した（配当の二重計上をやめた・株式分割を反映した・税 20.315％をかけた）'})
    state['engine_version'] = ENGINE_VERSION
    log.info('Migrated portfolio to engine v%d (tax=%s)', ENGINE_VERSION, total_tax)


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
        migrate_v2(state, history, stocks_by_code)
        apply_splits(state, history, stocks_by_code, today_jst)
        update_valuations(state, stocks_by_code)
        save_json(HISTORY_PATH, history)
        log.info('Already processed today (%s), updated valuations only', today_jst)
        save_json(STATE_PATH, state)
        return state

    stocks_by_code = {s['code']: s for s in stocks}

    # ステップ0: これまでの記録を直した決まりで数え直す（1回だけ）・株式分割を反映する
    migrate_v2(state, history, stocks_by_code)
    split_n = apply_splits(state, history, stocks_by_code, today_jst)
    if split_n:
        log.info('Applied splits: %d holdings', split_n)

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
            s.get('buy_sort', -(s.get('current_yield') or 0)),
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

    # ステップ4b: 配当で比率の下がった持ち株を買い増す（月に1回。検証の X8）
    ym = today_jst.strftime('%Y-%m')
    if TOPUP and state.get('last_topup_month') != ym:
        update_valuations(state, stocks_by_code)
        new_buys += topup_holdings(state, history, stocks_by_code, today_jst)
        state['last_topup_month'] = ym

    # ステップ4c: 12月の損出し（12月15日以降の最初の実行日に1回。検証の R6）
    if HARVEST and today_jst.month == 12 and today_jst.day >= HARVEST_FROM_DAY \
            and state.get('last_harvest_year') != today_jst.year:
        hv = harvest_losses(state, history, stocks_by_code, today_jst)
        new_sells += [t for t in hv if t['action'] == 'SELL']
        state['last_harvest_year'] = today_jst.year

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
        'tax_paid': state.get('tax_paid', 0),
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

#!/usr/bin/env python3
"""
DIVIDEND HEIST バックテスト (簡易版)
====================================

高配当スイング戦略を過去の市場データで検証する。

戦略:
  - 各月初にユニバース全銘柄をスクリーニング (look-ahead bias 回避)
  - BUY シグナル: スクリーニング PASS + 利回り ≥ 過去5Y Q75
  - SELL シグナル: 利回り ≤ 過去5Y Q25 (or 緊急撤退条件)
  - 等金額、最大20ポジション、月初リバランス
  - 配当再投資あり

実行方法:
  cd dividend-heist/scripts
  J_QUANTS_API_KEY=xxxxx python3 backtest.py

オプション:
  --start 2021-04-01     開始日 (デフォルト: 2021-04-01)
  --end   2026-04-30     終了日 (デフォルト: 2026-04-30)
  --capital 10000000     初期資金 (円, デフォルト: 1000万円)
  --positions 20         最大保有銘柄数 (デフォルト: 20)
  --no-cache             キャッシュを使わず再取得

出力 (./data/backtest/):
  summary.csv             サマリー指標 (CAGR, シャープ, 最大DDなど)
  equity_curve.csv        日次評価額推移 (戦略 vs TOPIX)
  monthly_returns.csv     月次リターン
  trades.csv              全取引履歴
  positions_history.csv   月初時点のポジション一覧

注意点 (v1):
  - 生存者バイアス: 現在のユニバースのみ使用 (上場廃止銘柄なし)
  - 取引コスト: 0.05%（買・売各々）+ 配当税20.315%
  - ベンチマーク: TOPIX 連動 ETF (1306) との比較
  - look-ahead bias 対策: 開示日 (DiscDate) より前のデータのみ使用
"""

from __future__ import annotations

import os
import sys
import json
import time
import csv
import math
import logging
import argparse
from pathlib import Path
from datetime import date, timedelta
from typing import Any
from collections import defaultdict

# 同ディレクトリの generate.py から関数を借用
sys.path.insert(0, str(Path(__file__).parent.absolute()))
from generate import (  # noqa: E402
    JQuantsClient,
    parse_date,
    safe_float,
    quantile,
    cumulative_adj_factor_after,
    extract_annual_dividends,
    extract_annual_metric,
    check_dividend_history,
    check_stability,
    check_payout_ratio,
    check_equity_ratio,
    check_valuation,
    check_min_yield,
    evaluate_box,
    EQUITY_RATIO_EXEMPT_SECTORS,
    DIVIDEND_CHECK_EXEMPT,
    PROGRESSIVE_DIVIDEND_LIST,
    DOE_LIST,
    MARKET_CODE_PRIME,
    API_SLEEP_SEC,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('backtest')

# ============================================================================
# Config
# ============================================================================


class Config:
    # キャッシュ
    CACHE_DIR = Path(__file__).parent.parent / 'data' / 'backtest_cache'

    # 出力
    OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'backtest'

    # スクリーニング 8 条件
    YIELD_MIN = 3.0
    GRAHAM_MAX = 22.5
    PAYOUT_MAX = 50.0
    EQUITY_MIN = 0.5

    # トレーディングコスト
    BUY_COMMISSION = 0.0005   # 0.05%
    SELL_COMMISSION = 0.0005  # 0.05%
    DIVIDEND_TAX = 0.20315
    CAPITAL_GAINS_TAX = 0.20315

    # ベンチマーク
    BENCHMARK_CODE = '1306'   # TOPIX連動型上場投資信託
    # フォールバック: 直接取れなかったら TOPIX 配当再投資指数を諦め単純な株価
    # にする


# ============================================================================
# Utility: 日付ヘルパー
# ============================================================================


def get_monthly_first_dates(start: date, end: date) -> list[date]:
    """月初日のリストを返す（start月から含む）"""
    dates = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        dates.append(cur)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return dates


def get_first_trading_close_after(quotes_sorted: list[dict], target: date) -> tuple[date, float] | None:
    """target以降で最初の取引日の調整済終値"""
    for q in quotes_sorted:
        d = parse_date(q.get('Date'))
        if d is None or d < target:
            continue
        c = safe_float(q.get('AdjustmentClose')) or safe_float(q.get('Close'))
        if c is not None and c > 0:
            return (d, c)
    return None


def get_last_trading_close_at_or_before(quotes_sorted: list[dict], target: date) -> tuple[date, float] | None:
    """target以前の最後の取引日の調整済終値"""
    last = None
    for q in quotes_sorted:
        d = parse_date(q.get('Date'))
        if d is None:
            continue
        if d > target:
            break
        c = safe_float(q.get('AdjustmentClose')) or safe_float(q.get('Close'))
        if c is not None and c > 0:
            last = (d, c)
    return last


# ============================================================================
# データ取得 (キャッシュあり)
# ============================================================================


def fetch_universe(client: JQuantsClient, force_refresh: bool = False) -> list[dict]:
    """
    ユニバース取得。

    対象: 東証プライム × TOPIX Core30 + Large70 + Mid400 (約470銘柄)
    キャッシュ: data/backtest_cache/universe.json
    """
    cache_file = Config.CACHE_DIR / 'universe.json'
    if cache_file.exists() and not force_refresh:
        log.info('Using cached universe: %s', cache_file)
        return json.loads(cache_file.read_text())

    log.info('Fetching universe from J-Quants...')
    info = client.get_listed_info()
    SCALE_TARGETS = {'TOPIX Core30', 'TOPIX Large70', 'TOPIX Mid400'}

    universe = []
    for row in info:
        if row.get('Mkt') != MARKET_CODE_PRIME:
            continue
        if row.get('ScaleCat') not in SCALE_TARGETS:
            continue
        code = row.get('Code', '')
        if len(code) == 5 and code.endswith('0'):
            code = code[:4]
        universe.append({
            'code': code,
            'name': row.get('CoName', ''),
            'sector33': row.get('S33', ''),
            'sector33_name': row.get('S33Nm', ''),
        })
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(universe, ensure_ascii=False, indent=2))
    log.info('Universe: %d stocks (cached to %s)', len(universe), cache_file)
    return universe


def fetch_stock_history(
    client: JQuantsClient,
    code: str,
    backtest_start: date,
    backtest_end: date,
    force_refresh: bool = False,
) -> tuple[list, list]:
    """
    銘柄1つの全データ (株価+財務) を取得しキャッシュ。

    必要範囲:
      - 株価: backtest_start - 5年 (Q25/Q75計算用) ~ backtest_end
      - 財務: 全期間
    """
    cache_file = Config.CACHE_DIR / f'{code}.json'
    if cache_file.exists() and not force_refresh:
        try:
            d = json.loads(cache_file.read_text())
            return d['quotes'], d['statements']
        except Exception:
            pass

    # J-Quants Standard プランは過去10年分のデータが取得可能。
    # 5年遡って Q25/Q75 を計算したいが、10年制限を超えないよう安全マージン込みで調整。
    today = date.today()
    safe_earliest = today - timedelta(days=int(9.5 * 365))  # 9.5年前 (6ヶ月の安全マージン)
    desired_from = backtest_start - timedelta(days=5 * 365 + 30)
    from_date_obj = max(desired_from, safe_earliest)
    from_date = from_date_obj.isoformat()
    to_date = backtest_end.isoformat()

    try:
        quotes = client.get_daily_quotes(code, from_date, to_date)
        time.sleep(API_SLEEP_SEC)
        statements = client.get_statements(code)
        time.sleep(API_SLEEP_SEC)
    except Exception as e:
        log.warning('Fetch fail %s: %s', code, e)
        return [], []

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(
        {'quotes': quotes, 'statements': statements},
        ensure_ascii=False,
    ))
    return quotes, statements


# ============================================================================
# As-of-date analysis (look-ahead bias 回避)
# ============================================================================


def filter_statements_disclosed_before(statements: list[dict], target: date) -> list[dict]:
    return [s for s in statements
            if (parse_date(s.get('DiscDate')) or date.min) <= target]


def filter_quotes_at_or_before(quotes_sorted: list[dict], target: date) -> list[dict]:
    return [q for q in quotes_sorted
            if (parse_date(q.get('Date')) or date.min) <= target]


def compute_forecast_dps_at(
    statements: list[dict],
    quotes_sorted: list[dict],
) -> tuple[float, date | None] | None:
    """
    その時点の予想DPS (分割調整済) を返す。
    """
    sorted_stmts = sorted(
        statements,
        key=lambda s: (
            parse_date(s.get('CurPerEn'))
            or parse_date(s.get('DiscDate'))
            or date.min
        ),
        reverse=True,
    )
    for s in sorted_stmts:
        per_type = s.get('CurPerType', '')
        if per_type in ('FY', '4Q'):
            f = safe_float(s.get('NxFDivAnn'))
        else:
            f = safe_float(s.get('FDivAnn'))
        if f is not None and f > 0:
            disc_d = parse_date(s.get('DiscDate')) or parse_date(s.get('CurPerEn'))
            adj = cumulative_adj_factor_after(quotes_sorted, disc_d)
            return (f * adj, disc_d)
    # フォールバック: 直近FY実績
    for s in sorted_stmts:
        if s.get('CurPerType') in ('FY', '4Q'):
            f = safe_float(s.get('DivAnn'))
            if f is not None and f > 0:
                disc_d = parse_date(s.get('DiscDate')) or parse_date(s.get('CurPerEn'))
                adj = cumulative_adj_factor_after(quotes_sorted, disc_d)
                return (f * adj, disc_d)
    return None


def compute_yield_dist_at(
    statements: list[dict],
    quotes_sorted: list[dict],
    target_date: date,
) -> dict[str, float] | None:
    """
    target以前の5年間の月次利回り分布 (分割調整済)。
    """
    five_years_ago = date(target_date.year - 5, target_date.month, 1)

    div_history_by_year: dict[int, float] = {}
    for s in statements:
        if s.get('CurPerType') not in ('FY', '4Q'):
            continue
        per_end = parse_date(s.get('CurPerEn'))
        v = safe_float(s.get('DivAnn'))
        disc_d = parse_date(s.get('DiscDate')) or per_end
        if per_end and v is not None:
            adj = cumulative_adj_factor_after(quotes_sorted, disc_d)
            div_history_by_year[per_end.year] = v * adj

    monthly: dict[str, tuple[date, float]] = {}
    for q in quotes_sorted:
        d = parse_date(q.get('Date'))
        if d is None or d < five_years_ago or d > target_date:
            continue
        close = safe_float(q.get('AdjustmentClose')) or safe_float(q.get('Close'))
        if close is None or close <= 0:
            continue
        ym = f'{d.year}-{d.month:02d}'
        prev = monthly.get(ym)
        if prev is None or d > prev[0]:
            monthly[ym] = (d, close)

    yields: list[float] = []
    for _ym, (d, close) in monthly.items():
        dps = div_history_by_year.get(d.year)
        if dps is None or dps <= 0:
            continue
        yields.append((dps / close) * 100.0)

    if len(yields) < 12:
        return None

    yields.sort()
    return {
        'q25': quantile(yields, 0.25),
        'q50': quantile(yields, 0.50),
        'q75': quantile(yields, 0.75),
        'n': len(yields),
    }


def get_latest_fy_metric_at(statements: list[dict], key: str) -> float | None:
    """target以前のdisclosure内で最新FYのキー値"""
    fy_list = [s for s in statements if s.get('CurPerType') in ('FY', '4Q')]
    fy_list.sort(key=lambda s: parse_date(s.get('CurPerEn')) or date.min, reverse=True)
    for s in fy_list:
        v = safe_float(s.get(key))
        if v is not None:
            return v
    return None


def compute_box_reliability(box_info: dict | None) -> float:
    """
    ボックス信頼度を 0〜1 で返す。

    内訳 (重み):
      ADX 40%      : 低いほど横ばい (22以下が望ましい)
      値幅 20%      : 8〜20%が理想 (中央値14%が満点)
      ATR比率 20%   : 1.5%以上が望ましい
      タッチ回数 20% : 上下各2回以上で満点
    """
    if box_info is None:
        return 0.0

    # ADX (40%)
    adx = box_info.get('adx')
    if adx is None:
        adx_score = 0.0
    else:
        adx_score = max(0.0, min(1.0, (22 - adx) / 22))

    # 値幅 (20%) - width_pct は generate.py で × 100 されている
    width_pct = box_info.get('width_pct', 0)
    if 8 <= width_pct <= 20:
        # 中央値14%が最も理想
        deviation = abs(width_pct - 14) / 6  # 14±6
        width_score = max(0.0, 1.0 - deviation * 0.3)
    elif width_pct < 8:
        width_score = max(0.0, width_pct / 8 * 0.5)
    else:
        width_score = max(0.0, 1.0 - (width_pct - 20) / 10) * 0.5

    # ATR比率 (20%) - atr_ratio も × 100 された値
    atr_ratio = box_info.get('atr_ratio') or 0
    atr_score = min(atr_ratio / 1.5, 1.0)

    # タッチ回数 (20%)
    upper = box_info.get('upper_touches', 0)
    lower = box_info.get('lower_touches', 0)
    touch_score = min(min(upper, lower) / 2, 1.0)

    return (
        0.4 * adx_score
        + 0.2 * width_score
        + 0.2 * atr_score
        + 0.2 * touch_score
    )


def compute_buy_score(current_yield: float, box_reliability: float) -> float:
    """
    BUY 優先順位スコア (B戦略・後方互換用)。

    yield × (0.5 + 0.5 × box_reliability)
      box_reliability=0 → score = yield × 0.5  (50%減点)
      box_reliability=1 → score = yield × 1.0  (満点、ボックス完璧)
    """
    return current_yield * (0.5 + 0.5 * box_reliability)


# ============================================================================
# 戦略レジストリ (--compare で全戦略を試す)
# ============================================================================


def score_A_yield_only(info: dict) -> float:
    """A: 利回り単純順"""
    return info.get('current_yield') or 0


def score_B_box_yield(info: dict) -> float:
    """B: ボックス信頼度 × 利回り"""
    y = info.get('current_yield') or 0
    box_rel = info.get('box_reliability') or 0
    return y * (0.5 + 0.5 * box_rel)


def score_C_q75_deviation(info: dict) -> float:
    """C: Q75乖離率順 (歴史的にどれだけ割安か)"""
    y = info.get('current_yield') or 0
    q75 = info.get('q75')
    if not q75 or q75 <= 0:
        return 0
    return (y - q75) / q75


def score_D_multi_factor(info: dict) -> float:
    """
    D: 多要素スコア
       0.4 × 利回り(正規化) + 0.2 × ROE(正規化) +
       0.2 × 自己資本比率(正規化) + 0.2 × ボックス信頼度
    """
    y = info.get('current_yield') or 0
    roe = info.get('roe') or 0
    eq = info.get('equity_ratio') or 0  # decimal: 0.5 = 50%
    box_rel = info.get('box_reliability') or 0
    yield_norm = min(y / 5.0, 1.0)        # 5%以上で満点
    roe_norm = min(roe / 15.0, 1.0)        # ROE 15%以上で満点
    eq_norm = min(eq / 0.6, 1.0)           # 自己資本比率 60%以上で満点
    return 0.4 * yield_norm + 0.2 * roe_norm + 0.2 * eq_norm + 0.2 * box_rel


def score_E_yield_x_roe(info: dict) -> float:
    """E: 利回り × ROE (質と価格両方)"""
    y = info.get('current_yield') or 0
    roe = info.get('roe') or 0
    return y * (roe / 10.0)


STRATEGIES = {
    'A': {
        'name': '利回り単純順',
        'desc': 'yield高い順',
        'score_fn': score_A_yield_only,
    },
    'B': {
        'name': 'ボックス × 利回り',
        'desc': 'yield × (0.5 + 0.5 × box信頼度)',
        'score_fn': score_B_box_yield,
    },
    'C': {
        'name': 'Q75乖離率順',
        'desc': '(yield - Q75) / Q75',
        'score_fn': score_C_q75_deviation,
    },
    'D': {
        'name': '多要素スコア',
        'desc': '0.4y + 0.2ROE + 0.2eq + 0.2box',
        'score_fn': score_D_multi_factor,
    },
    'E': {
        'name': '利回り × ROE',
        'desc': 'yield × ROE / 10',
        'score_fn': score_E_yield_x_roe,
    },
}


def screen_stock_at(
    code: str,
    sector33: str,
    statements_filtered: list[dict],
    quotes_sorted_filtered: list[dict],
    target_date: date,
) -> tuple[bool, dict[str, Any]]:
    """
    target_date時点で銘柄を 8 条件スクリーニング。

    Returns:
        (passes, info)
        info: {
            'price', 'forecast_dps', 'current_yield',
            'q25', 'q75', 'q50',
            'per', 'pbr', 'payout', 'equity_ratio',
            'div_history',
        }
    """
    info: dict[str, Any] = {}

    if not quotes_sorted_filtered or not statements_filtered:
        return (False, info)

    # 価格
    price_pair = get_last_trading_close_at_or_before(quotes_sorted_filtered, target_date)
    if not price_pair:
        return (False, info)
    _last_date, price = price_pair
    info['price'] = price

    # 予想DPS
    fdps_pair = compute_forecast_dps_at(statements_filtered, quotes_sorted_filtered)
    if not fdps_pair or fdps_pair[0] is None or fdps_pair[0] <= 0:
        return (False, info)
    forecast_dps, _ = fdps_pair
    info['forecast_dps'] = forecast_dps

    current_yield = forecast_dps / price * 100.0
    info['current_yield'] = current_yield

    # 利回り分布
    dist = compute_yield_dist_at(statements_filtered, quotes_sorted_filtered, target_date)
    if dist is None:
        return (False, info)
    info.update({'q25': dist['q25'], 'q50': dist['q50'], 'q75': dist['q75']})

    # PER, PBR
    eps = get_latest_fy_metric_at(statements_filtered, 'EPS')
    bps = get_latest_fy_metric_at(statements_filtered, 'BPS')
    # 分割調整
    fy_list = [s for s in statements_filtered if s.get('CurPerType') in ('FY', '4Q')]
    fy_list.sort(key=lambda s: parse_date(s.get('CurPerEn')) or date.min, reverse=True)
    if fy_list:
        latest_fy = fy_list[0]
        disc_d = parse_date(latest_fy.get('DiscDate')) or parse_date(latest_fy.get('CurPerEn'))
        adj = cumulative_adj_factor_after(quotes_sorted_filtered, disc_d)
        eps = (eps * adj) if eps is not None else None
        bps = (bps * adj) if bps is not None else None
    per = (price / eps) if (eps is not None and eps > 0) else None
    pbr = (price / bps) if (bps is not None and bps > 0) else None
    info['per'] = per
    info['pbr'] = pbr

    # 配当性向 (DPS / EPS から計算)
    div_actual = get_latest_fy_metric_at(statements_filtered, 'DivAnn')
    if fy_list and div_actual is not None:
        latest_fy = fy_list[0]
        disc_d = parse_date(latest_fy.get('DiscDate')) or parse_date(latest_fy.get('CurPerEn'))
        adj = cumulative_adj_factor_after(quotes_sorted_filtered, disc_d)
        div_adj = div_actual * adj
        if eps is not None and eps > 0:
            payout = (div_adj / eps) * 100.0
        else:
            payout = None
    else:
        payout = None
    info['payout'] = payout

    # 自己資本比率
    equity_ratio = None
    sorted_all = sorted(
        statements_filtered,
        key=lambda s: parse_date(s.get('CurPerEn')) or date.min,
        reverse=True,
    )
    for s in sorted_all:
        v = safe_float(s.get('EqAR'))
        if v is not None:
            equity_ratio = v
            break
    info['equity_ratio'] = equity_ratio

    # 8 条件チェック
    div_history = extract_annual_dividends(statements_filtered, quotes_sorted=quotes_sorted_filtered)
    info['div_history'] = div_history

    # ボックス相場評価 (BUY優先順位用) - 直近の価格データから判定
    box_info = evaluate_box(quotes_sorted_filtered)
    info['box_info'] = box_info
    info['box_reliability'] = compute_box_reliability(box_info)

    # ROE = 純利益 / 自己資本 × 100 (戦略Dで使用)
    np_val = get_latest_fy_metric_at(statements_filtered, 'NP')
    eq_val = get_latest_fy_metric_at(statements_filtered, 'Eq')
    if np_val is not None and eq_val is not None and eq_val > 0:
        info['roe'] = (np_val / eq_val) * 100.0
    else:
        info['roe'] = 0.0

    sales_history = extract_annual_metric(statements_filtered, 'Sales')
    op_history = extract_annual_metric(statements_filtered, 'OP')
    np_history = extract_annual_metric(statements_filtered, 'NP')

    c1 = check_dividend_history(div_history, code)[0]      # 減配ゼロ
    c2 = check_stability(sales_history)                    # 売上安定性
    c3 = check_stability(op_history)                       # 営利安定性
    c4 = check_stability(np_history)                       # 純利安定性
    c5 = check_payout_ratio(payout)                        # 配当性向 ≤50%
    c6 = check_equity_ratio(equity_ratio, sector33)        # 自己資本比率 ≥50%
    c7 = check_valuation(per, pbr)                         # PER × PBR ≤22.5
    c8 = check_min_yield(current_yield, code)              # 最低利回り ≥3%

    checks = [c1, c2, c3, c4, c5, c6, c7, c8]
    info['checks'] = checks
    # None は判定不能（データ不足）→ False扱い
    passes = all(c is True for c in checks)
    info['passes'] = passes
    return (passes, info)


# ============================================================================
# Backtest engine
# ============================================================================


class Position:
    __slots__ = ('code', 'name', 'qty', 'cost_basis', 'open_date', 'cost_total')

    def __init__(self, code: str, name: str, qty: float, price: float, open_date: date):
        self.code = code
        self.name = name
        self.qty = qty                    # 株数
        self.cost_basis = price           # 1株あたり買値（取得単価）
        self.open_date = open_date
        self.cost_total = qty * price     # 取得総額

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.cost_basis) * self.qty


class Portfolio:
    def __init__(self, initial_capital: float, max_positions: int):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.positions: dict[str, Position] = {}
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []  # 日別評価額
        self.dividends_received_total = 0.0       # 累積配当（税引後）
        self.realized_capital_gain_total = 0.0    # 累積実現キャピタルゲイン (税引後)
        self.total_commissions_paid = 0.0         # 累積手数料
        self.total_taxes_paid = 0.0               # 累積税金

    def position_size_yen(self) -> float:
        """1ポジションあたりの目標金額"""
        return self.initial_capital / self.max_positions

    def buy(self, code: str, name: str, price: float, dt: date, target_yen: float) -> bool:
        if price <= 0 or target_yen <= 0:
            return False
        gross = target_yen / (1 + Config.BUY_COMMISSION)  # 手数料込みでtarget_yen 使う想定
        qty = math.floor(gross / price / 100) * 100      # 100株単位
        if qty <= 0:
            return False
        cost = qty * price
        commission = cost * Config.BUY_COMMISSION
        total_cost = cost + commission
        if total_cost > self.cash:
            return False
        self.cash -= total_cost
        self.total_commissions_paid += commission
        # 既存ポジションがあったら平均化（通常無いが）
        if code in self.positions:
            old = self.positions[code]
            new_qty = old.qty + qty
            new_cost_total = old.cost_total + cost
            new_basis = new_cost_total / new_qty
            old.qty = new_qty
            old.cost_basis = new_basis
            old.cost_total = new_cost_total
        else:
            self.positions[code] = Position(code, name, qty, price, dt)
        self.trades.append({
            'date': dt.isoformat(),
            'code': code,
            'name': name,
            'action': 'BUY',
            'price': round(price, 2),
            'qty': qty,
            'value': round(cost, 0),
            'commission': round(commission, 0),
            'pnl': 0,
        })
        return True

    def sell(self, code: str, price: float, dt: date, reason: str = ''):
        if code not in self.positions:
            return
        pos = self.positions[code]
        proceeds = pos.qty * price
        commission = proceeds * Config.SELL_COMMISSION
        gain = (price - pos.cost_basis) * pos.qty - commission
        tax = max(0, gain) * Config.CAPITAL_GAINS_TAX
        net_proceeds = proceeds - commission - tax
        self.cash += net_proceeds
        # 集計
        self.total_commissions_paid += commission
        self.total_taxes_paid += tax
        self.realized_capital_gain_total += (gain - tax)  # 税引後実現損益
        self.trades.append({
            'date': dt.isoformat(),
            'code': code,
            'name': pos.name,
            'action': f'SELL ({reason})',
            'price': round(price, 2),
            'qty': pos.qty,
            'value': round(proceeds, 0),
            'commission': round(commission + tax, 0),
            'pnl': round(gain - tax, 0),
        })
        del self.positions[code]

    def receive_dividends(self, dividends: dict[str, float], dt: date):
        """配当受領 (税引後)"""
        for code, amount in dividends.items():
            if code not in self.positions:
                continue
            pos = self.positions[code]
            gross = amount * pos.qty
            tax = gross * Config.DIVIDEND_TAX
            net = gross - tax
            self.cash += net
            self.dividends_received_total += net
            self.total_taxes_paid += tax
            if gross > 0:
                self.trades.append({
                    'date': dt.isoformat(),
                    'code': code,
                    'name': pos.name,
                    'action': 'DIVIDEND',
                    'price': round(amount, 2),
                    'qty': pos.qty,
                    'value': round(gross, 0),
                    'commission': round(tax, 0),  # 税金
                    'pnl': round(net, 0),
                })

    def total_value(self, prices: dict[str, float]) -> float:
        v = self.cash
        for code, pos in self.positions.items():
            p = prices.get(code, pos.cost_basis)
            v += pos.market_value(p)
        return v


# ============================================================================
# Performance metrics
# ============================================================================


def compute_metrics(
    equity_curve: list[dict],
    initial_capital: float,
    portfolio: 'Portfolio',
    final_prices: dict[str, float],
) -> dict:
    if not equity_curve:
        return {}
    final = equity_curve[-1]['value']
    total_return = (final / initial_capital - 1) * 100

    # 期間 (年)
    start_d = parse_date(equity_curve[0]['date'])
    end_d = parse_date(equity_curve[-1]['date'])
    years = max((end_d - start_d).days / 365.25, 1e-6)
    cagr = ((final / initial_capital) ** (1 / years) - 1) * 100

    # 月次リターン
    monthly_values: dict[str, float] = {}
    for row in equity_curve:
        d = parse_date(row['date'])
        ym = f'{d.year}-{d.month:02d}'
        monthly_values[ym] = row['value']
    sorted_monthly = sorted(monthly_values.items())
    monthly_returns = []
    for i in range(1, len(sorted_monthly)):
        prev = sorted_monthly[i - 1][1]
        cur = sorted_monthly[i][1]
        if prev > 0:
            monthly_returns.append(cur / prev - 1)

    # ボラティリティ (年率) + シャープレシオ
    if len(monthly_returns) > 1:
        mean = sum(monthly_returns) / len(monthly_returns)
        var = sum((x - mean) ** 2 for x in monthly_returns) / (len(monthly_returns) - 1)
        vol_monthly = math.sqrt(var)
        vol_annual = vol_monthly * math.sqrt(12) * 100
        sharpe = (mean * 12) / (vol_monthly * math.sqrt(12)) if vol_monthly > 0 else 0
    else:
        vol_annual = 0
        sharpe = 0

    # 最大ドローダウン
    peak = equity_curve[0]['value']
    max_dd = 0
    for row in equity_curve:
        v = row['value']
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # 配当・キャピタルゲインの内訳
    realized_gain = portfolio.realized_capital_gain_total
    div_total = portfolio.dividends_received_total
    # 未実現損益: 現在保有銘柄の評価益
    unrealized_gain = 0.0
    for code, pos in portfolio.positions.items():
        p = final_prices.get(code, pos.cost_basis)
        unrealized_gain += pos.unrealized_pnl(p)
    total_capital_gain = realized_gain + unrealized_gain

    # 配当が総リターンに占める割合
    total_profit = final - initial_capital
    if total_profit > 0:
        div_contribution = div_total / total_profit * 100
    else:
        div_contribution = 0

    # 配当の年率換算（初期資金に対して）
    div_yield_total = div_total / initial_capital * 100
    div_yield_annual = ((1 + div_yield_total / 100) ** (1 / years) - 1) * 100

    # 取引回数
    n_trades = sum(1 for t in portfolio.trades if t['action'] == 'BUY')
    n_sells = sum(1 for t in portfolio.trades if t['action'].startswith('SELL'))
    n_dividends = sum(1 for t in portfolio.trades if t['action'] == 'DIVIDEND')

    # 勝率 (実現益 > 0 のSELL/全SELL)
    wins = sum(1 for t in portfolio.trades
               if t['action'].startswith('SELL') and t['pnl'] > 0)
    win_rate = (wins / n_sells * 100) if n_sells > 0 else 0

    return {
        'period_start': equity_curve[0]['date'],
        'period_end': equity_curve[-1]['date'],
        'years': round(years, 2),
        'initial_capital_yen': initial_capital,
        'final_value_yen': final,
        'total_profit_yen': round(total_profit, 0),
        '---リターン---': '',
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'volatility_annual_pct': round(vol_annual, 2),
        'sharpe_ratio': round(sharpe, 3),
        'max_drawdown_pct': round(max_dd, 2),
        '---配当 vs キャピタルゲイン---': '',
        'total_dividend_income_yen': round(div_total, 0),
        'total_capital_gain_yen': round(total_capital_gain, 0),
        '  realized_capital_gain_yen': round(realized_gain, 0),
        '  unrealized_capital_gain_yen': round(unrealized_gain, 0),
        'dividend_contribution_pct': round(div_contribution, 1),
        'annualized_dividend_yield_pct': round(div_yield_annual, 2),
        '---取引統計---': '',
        'total_buys': n_trades,
        'total_sells': n_sells,
        'total_dividends_received': n_dividends,
        'win_rate_pct': round(win_rate, 1),
        'total_commissions_paid_yen': round(portfolio.total_commissions_paid, 0),
        'total_taxes_paid_yen': round(portfolio.total_taxes_paid, 0),
    }


# ============================================================================
# CSV writers
# ============================================================================


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        log.warning('Empty data, skipping write to %s', path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fns = fieldnames or list(rows[0].keys())
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================================
# Main backtest
# ============================================================================


def fetch_all_data(args) -> tuple[dict[str, dict], date, date]:
    """
    バックテスト用データを取得 (全戦略で共通利用)。
    Returns:
        (stock_data, start_date, end_date)
    """
    api_key = os.environ.get('J_QUANTS_API_KEY')
    if not api_key:
        raise RuntimeError('J_QUANTS_API_KEY env var not set')

    Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    client = JQuantsClient(api_key)

    universe = fetch_universe(client, force_refresh=args.no_cache)

    log.info('Pre-fetching data for %d stocks...', len(universe))
    stock_data: dict[str, dict] = {}
    for i, u in enumerate(universe):
        if i % 50 == 0:
            log.info('  [%d/%d] %s %s', i, len(universe), u['code'], u['name'])
        quotes, statements = fetch_stock_history(
            client, u['code'], start, end,
            force_refresh=args.no_cache,
        )
        if quotes and statements:
            stock_data[u['code']] = {
                'name': u['name'],
                'sector33': u['sector33'],
                'quotes_sorted': sorted(quotes, key=lambda q: parse_date(q.get('Date')) or date.min),
                'statements': statements,
            }

    log.info('Loaded data for %d/%d stocks', len(stock_data), len(universe))
    return stock_data, start, end


def run_strategy(
    stock_data: dict[str, dict],
    start: date,
    end: date,
    initial_capital: float,
    max_positions: int,
    strategy_id: str,
) -> tuple[Portfolio, dict]:
    """
    指定の戦略でバックテストを実行。

    Returns:
        (portfolio, metrics)
    """
    score_fn = STRATEGIES[strategy_id]['score_fn']
    strategy_name = STRATEGIES[strategy_id]['name']
    log.info('=== Running strategy [%s] %s ===', strategy_id, strategy_name)

    rebalance_dates = get_monthly_first_dates(start, end)
    portfolio = Portfolio(initial_capital, max_positions)
    paid_dividends_seen: set[tuple[str, int]] = set()

    for ri, rdate in enumerate(rebalance_dates):
        # 1. 価格取得
        prices_today: dict[str, float] = {}
        for code, sd in stock_data.items():
            quotes_filtered = filter_quotes_at_or_before(sd['quotes_sorted'], rdate)
            p = get_last_trading_close_at_or_before(quotes_filtered, rdate)
            if p:
                prices_today[code] = p[1]

        # 2. 配当受領
        for code in list(portfolio.positions.keys()):
            sd = stock_data.get(code)
            if not sd:
                continue
            stmts_filt = filter_statements_disclosed_before(sd['statements'], rdate)
            for s in stmts_filt:
                if s.get('CurPerType') not in ('FY', '4Q'):
                    continue
                per_end = parse_date(s.get('CurPerEn'))
                if per_end is None:
                    continue
                key = (code, per_end.year)
                if key in paid_dividends_seen:
                    continue
                pos = portfolio.positions[code]
                if pos.open_date <= per_end:
                    div = safe_float(s.get('DivAnn'))
                    if div is None or div <= 0:
                        paid_dividends_seen.add(key)
                        continue
                    disc_d = parse_date(s.get('DiscDate')) or per_end
                    adj = cumulative_adj_factor_after(sd['quotes_sorted'], disc_d)
                    div_adj = div * adj
                    portfolio.receive_dividends({code: div_adj}, rdate)
                paid_dividends_seen.add(key)

        # 3. SELL シグナル
        sells = []
        for code, pos in list(portfolio.positions.items()):
            sd = stock_data.get(code)
            if not sd:
                continue
            stmts_filt = filter_statements_disclosed_before(sd['statements'], rdate)
            quotes_filt = filter_quotes_at_or_before(sd['quotes_sorted'], rdate)
            _, info = screen_stock_at(code, sd['sector33'], stmts_filt, quotes_filt, rdate)
            cur_y = info.get('current_yield')
            q25 = info.get('q25')
            if cur_y is not None and q25 is not None and cur_y <= q25:
                sells.append((code, 'YIELD_Q25'))
                continue
            div_h = info.get('div_history', [])
            if len(div_h) >= 2 and div_h[-2] > 0:
                if (div_h[-1] - div_h[-2]) / div_h[-2] < -0.10:
                    sells.append((code, 'DPS_CUT'))

        for code, reason in sells:
            price = prices_today.get(code)
            if price:
                portfolio.sell(code, price, rdate, reason)

        # 4. BUY 候補の発掘 (戦略ごとのスコア)
        buy_candidates = []
        for code, sd in stock_data.items():
            if code in portfolio.positions:
                continue
            stmts_filt = filter_statements_disclosed_before(sd['statements'], rdate)
            quotes_filt = filter_quotes_at_or_before(sd['quotes_sorted'], rdate)
            passes, info = screen_stock_at(code, sd['sector33'], stmts_filt, quotes_filt, rdate)
            cur_y = info.get('current_yield')
            q75 = info.get('q75')
            if not passes or cur_y is None or q75 is None or cur_y < q75:
                continue
            score = score_fn(info)
            buy_candidates.append({
                'code': code,
                'name': sd['name'],
                'yield': cur_y,
                'q75': q75,
                'box_reliability': info.get('box_reliability', 0.0),
                'roe': info.get('roe', 0.0),
                'score': score,
            })

        # スコア高い順
        buy_candidates.sort(key=lambda x: -x['score'])

        # 5. BUY 実行
        slots_available = portfolio.max_positions - len(portfolio.positions)
        target_per_pos = portfolio.position_size_yen()
        bought_count = 0
        for cand in buy_candidates:
            if bought_count >= slots_available:
                break
            if portfolio.cash < target_per_pos * 0.5:
                break
            price = prices_today.get(cand['code'])
            if not price:
                continue
            if portfolio.buy(cand['code'], cand['name'], price, rdate, target_per_pos):
                bought_count += 1

        # 6. 評価額記録
        total = portfolio.total_value(prices_today)
        portfolio.equity_curve.append({
            'date': rdate.isoformat(),
            'value': round(total, 0),
            'cash': round(portfolio.cash, 0),
            'positions': len(portfolio.positions),
            'cumulative_dividends': round(portfolio.dividends_received_total, 0),
        })

        if ri % 12 == 0:
            log.info('  [%s] Value: %s, Pos: %d',
                     rdate.isoformat(),
                     f'{int(total):,}',
                     len(portfolio.positions))

    # 最終時点の価格
    final_prices: dict[str, float] = {}
    for code, sd in stock_data.items():
        if code in portfolio.positions:
            quotes_filtered = filter_quotes_at_or_before(sd['quotes_sorted'], end)
            p = get_last_trading_close_at_or_before(quotes_filtered, end)
            if p:
                final_prices[code] = p[1]

    metrics = compute_metrics(portfolio.equity_curve, initial_capital, portfolio, final_prices)
    return portfolio, metrics


def run_backtest(args) -> int:
    """シングル戦略バックテスト"""
    stock_data, start, end = fetch_all_data(args)
    initial_capital = args.capital
    max_positions = args.positions
    strategy_id = args.strategy

    log.info('=== DIVIDEND HEIST BACKTEST ===')
    log.info('Period: %s ~ %s', start, end)
    log.info('Capital: %s yen, Max positions: %d', f'{initial_capital:,}', max_positions)
    log.info('Strategy: [%s] %s', strategy_id, STRATEGIES[strategy_id]['name'])

    portfolio, metrics = run_strategy(stock_data, start, end, initial_capital, max_positions, strategy_id)

    log.info('=== METRICS ===')
    for k, v in metrics.items():
        log.info('  %-30s %s', k, v)

    summary_rows = [{'metric': k, 'value': v} for k, v in metrics.items()]
    write_csv(Config.OUTPUT_DIR / 'summary.csv', summary_rows, ['metric', 'value'])
    write_csv(Config.OUTPUT_DIR / 'equity_curve.csv', portfolio.equity_curve)
    write_csv(Config.OUTPUT_DIR / 'trades.csv', portfolio.trades)

    monthly_rows = []
    if portfolio.equity_curve:
        for i in range(1, len(portfolio.equity_curve)):
            prev = portfolio.equity_curve[i - 1]
            cur = portfolio.equity_curve[i]
            ret_pct = (cur['value'] / prev['value'] - 1) * 100 if prev['value'] > 0 else 0
            monthly_rows.append({
                'date': cur['date'],
                'value': cur['value'],
                'return_pct': round(ret_pct, 3),
            })
    write_csv(Config.OUTPUT_DIR / 'monthly_returns.csv', monthly_rows)
    log.info('Output written to: %s', Config.OUTPUT_DIR)
    return 0


def run_compare(args) -> int:
    """全戦略を比較実行"""
    stock_data, start, end = fetch_all_data(args)
    initial_capital = args.capital
    max_positions = args.positions

    log.info('=== STRATEGY COMPARISON ===')
    log.info('Period: %s ~ %s', start, end)

    all_results = {}
    for strategy_id in STRATEGIES.keys():
        portfolio, metrics = run_strategy(
            stock_data, start, end, initial_capital, max_positions, strategy_id,
        )
        all_results[strategy_id] = {
            'portfolio': portfolio,
            'metrics': metrics,
        }

    # 比較サマリー作成
    comparison_rows = []
    metrics_keys_to_show = [
        'total_return_pct', 'cagr_pct', 'volatility_annual_pct', 'sharpe_ratio',
        'max_drawdown_pct',
        'total_dividend_income_yen', 'total_capital_gain_yen',
        'dividend_contribution_pct', 'annualized_dividend_yield_pct',
        'total_buys', 'total_sells', 'win_rate_pct',
    ]
    for strategy_id, result in all_results.items():
        m = result['metrics']
        row = {
            'strategy_id': strategy_id,
            'strategy_name': STRATEGIES[strategy_id]['name'],
            'description': STRATEGIES[strategy_id]['desc'],
        }
        for k in metrics_keys_to_show:
            row[k] = m.get(k, '')
        comparison_rows.append(row)

    # CAGR順にソート
    comparison_rows.sort(key=lambda x: -float(x.get('cagr_pct') or 0))

    write_csv(
        Config.OUTPUT_DIR / 'comparison.csv',
        comparison_rows,
        ['strategy_id', 'strategy_name', 'description'] + metrics_keys_to_show,
    )

    # 各戦略のequity curve をまとめて1ファイルに
    combined_equity = []
    for strategy_id, result in all_results.items():
        for row in result['portfolio'].equity_curve:
            combined_equity.append({
                'date': row['date'],
                'strategy': strategy_id,
                'value': row['value'],
            })
    write_csv(Config.OUTPUT_DIR / 'comparison_equity_curves.csv', combined_equity)

    # ランキング表示
    log.info('')
    log.info('=== RANKING (CAGR順) ===')
    log.info('%-3s %-25s %-12s %-10s %-10s', 'ID', 'Strategy', 'Total Ret', 'CAGR', 'Sharpe')
    for row in comparison_rows:
        log.info('%-3s %-25s %-12s %-10s %-10s',
                 row['strategy_id'],
                 row['strategy_name'],
                 f"{row['total_return_pct']}%",
                 f"{row['cagr_pct']}%",
                 row['sharpe_ratio'])

    log.info('')
    log.info('Output written to: %s', Config.OUTPUT_DIR)
    log.info('  comparison.csv             (戦略別サマリー)')
    log.info('  comparison_equity_curves.csv (全戦略の評価額推移)')
    return 0


def run_multi_period(args) -> int:
    """
    複数期間 × 全戦略を比較。
    各期間で全戦略を実行し、安定的に強い戦略を発見する。

    デフォルト期間 (--periods で上書き可能):
      2020-04-01 ~ 2022-04-01
      2022-04-01 ~ 2024-04-01
      2024-04-01 ~ 2026-04-01
    """
    # 期間リストの構築
    if args.periods:
        periods = []
        for p_str in args.periods.split(','):
            s, e = p_str.split(':')
            periods.append((date.fromisoformat(s.strip()), date.fromisoformat(e.strip())))
    else:
        periods = [
            (date(2020, 4, 1), date(2022, 4, 1)),
            (date(2022, 4, 1), date(2024, 4, 1)),
            (date(2024, 4, 1), date(2026, 4, 1)),
        ]

    earliest = min(p[0] for p in periods)
    latest = max(p[1] for p in periods)

    # データを最広範囲で1度だけ取得 (全期間カバー)
    args_fetch = argparse.Namespace(**vars(args))
    args_fetch.start = earliest.isoformat()
    args_fetch.end = latest.isoformat()
    log.info('Fetching data for the widest range %s ~ %s', earliest, latest)
    stock_data, _, _ = fetch_all_data(args_fetch)

    log.info('=== MULTI-PERIOD STRATEGY COMPARISON ===')
    log.info('%d periods × %d strategies = %d backtests',
             len(periods), len(STRATEGIES), len(periods) * len(STRATEGIES))

    initial_capital = args.capital
    max_positions = args.positions

    # 結果収集: {(period_label, strategy_id): metrics}
    all_results: dict[tuple[str, str], dict] = {}
    period_labels: list[str] = []

    for pi, (p_start, p_end) in enumerate(periods):
        period_label = f'{p_start.year}-{p_end.year}'
        period_labels.append(period_label)
        log.info('')
        log.info('========================================')
        log.info('  PERIOD %d/%d: %s', pi + 1, len(periods), period_label)
        log.info('========================================')

        for strategy_id in STRATEGIES.keys():
            try:
                _portfolio, metrics = run_strategy(
                    stock_data, p_start, p_end,
                    initial_capital, max_positions, strategy_id,
                )
                all_results[(period_label, strategy_id)] = metrics
            except Exception as e:
                log.error('Period %s, Strategy %s failed: %s',
                          period_label, strategy_id, e)
                all_results[(period_label, strategy_id)] = {}

    # ---- 詳細CSV: 期間別 × 戦略別の全指標 ----
    detailed_rows = []
    for period_label in period_labels:
        for strategy_id in STRATEGIES.keys():
            m = all_results.get((period_label, strategy_id), {})
            detailed_rows.append({
                'period': period_label,
                'strategy_id': strategy_id,
                'strategy_name': STRATEGIES[strategy_id]['name'],
                'total_return_pct': m.get('total_return_pct', ''),
                'cagr_pct': m.get('cagr_pct', ''),
                'sharpe_ratio': m.get('sharpe_ratio', ''),
                'max_drawdown_pct': m.get('max_drawdown_pct', ''),
                'volatility_annual_pct': m.get('volatility_annual_pct', ''),
                'total_dividend_income_yen': m.get('total_dividend_income_yen', ''),
                'dividend_contribution_pct': m.get('dividend_contribution_pct', ''),
                'annualized_dividend_yield_pct': m.get('annualized_dividend_yield_pct', ''),
                'win_rate_pct': m.get('win_rate_pct', ''),
                'total_buys': m.get('total_buys', ''),
                'total_sells': m.get('total_sells', ''),
            })
    write_csv(Config.OUTPUT_DIR / 'multi_period_detailed.csv', detailed_rows)

    # ---- ランキング表 (戦略ごとの集約) ----
    ranking_rows = []
    for strategy_id in STRATEGIES.keys():
        cagrs = []
        sharpes = []
        ddowns = []
        div_yields = []
        for period_label in period_labels:
            m = all_results.get((period_label, strategy_id), {})
            try:
                cagr = float(m.get('cagr_pct') or 0)
                cagrs.append(cagr)
            except (TypeError, ValueError):
                pass
            try:
                sh = float(m.get('sharpe_ratio') or 0)
                sharpes.append(sh)
            except (TypeError, ValueError):
                pass
            try:
                dd = float(m.get('max_drawdown_pct') or 0)
                ddowns.append(dd)
            except (TypeError, ValueError):
                pass
            try:
                dy = float(m.get('annualized_dividend_yield_pct') or 0)
                div_yields.append(dy)
            except (TypeError, ValueError):
                pass

        if not cagrs:
            continue

        # 集約指標
        avg_cagr = sum(cagrs) / len(cagrs)
        min_cagr = min(cagrs)
        max_cagr = max(cagrs)
        # 標準偏差
        if len(cagrs) > 1:
            var = sum((x - avg_cagr) ** 2 for x in cagrs) / (len(cagrs) - 1)
            std_cagr = math.sqrt(var)
        else:
            std_cagr = 0
        # 安定性スコア = 平均CAGR / 標準偏差 (高いほど安定して強い)
        consistency_score = (avg_cagr / std_cagr) if std_cagr > 0.5 else avg_cagr / 0.5

        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
        avg_dd = sum(ddowns) / len(ddowns) if ddowns else 0
        avg_div = sum(div_yields) / len(div_yields) if div_yields else 0

        row = {
            'strategy_id': strategy_id,
            'strategy_name': STRATEGIES[strategy_id]['name'],
            'description': STRATEGIES[strategy_id]['desc'],
        }
        # 各期間のCAGR
        for i, period_label in enumerate(period_labels):
            row[f'cagr_{period_label}'] = cagrs[i] if i < len(cagrs) else ''
        # 集約
        row['avg_cagr'] = round(avg_cagr, 2)
        row['min_cagr'] = round(min_cagr, 2)
        row['max_cagr'] = round(max_cagr, 2)
        row['std_cagr'] = round(std_cagr, 2)
        row['consistency_score'] = round(consistency_score, 2)
        row['avg_sharpe'] = round(avg_sharpe, 3)
        row['avg_max_dd'] = round(avg_dd, 2)
        row['avg_dividend_yield'] = round(avg_div, 2)

        ranking_rows.append(row)

    # 安定性スコア順にソート (高い = 全期間で安定して強い)
    ranking_rows.sort(key=lambda x: -x['consistency_score'])

    # CSV出力
    field_order = [
        'strategy_id', 'strategy_name', 'description',
    ] + [f'cagr_{p}' for p in period_labels] + [
        'avg_cagr', 'min_cagr', 'max_cagr', 'std_cagr',
        'consistency_score', 'avg_sharpe', 'avg_max_dd', 'avg_dividend_yield',
    ]
    write_csv(Config.OUTPUT_DIR / 'multi_period_ranking.csv', ranking_rows, field_order)

    # ---- ログ出力: 一目でわかるランキング ----
    log.info('')
    log.info('==============================================================')
    log.info('  RANKING by Consistency Score (高いほど全期間で安定して強い)')
    log.info('==============================================================')
    header = f'{"ID":<3} {"Strategy":<25}'
    for p in period_labels:
        header += f' {p:>10}'
    header += f' {"AVG":>8} {"MIN":>8} {"Score":>8}'
    log.info(header)
    log.info('-' * len(header))
    for row in ranking_rows:
        line = f'{row["strategy_id"]:<3} {row["strategy_name"]:<25}'
        for p in period_labels:
            v = row.get(f'cagr_{p}', 0)
            line += f' {v:>9.2f}%'
        line += f' {row["avg_cagr"]:>7.2f}%'
        line += f' {row["min_cagr"]:>7.2f}%'
        line += f' {row["consistency_score"]:>8.2f}'
        log.info(line)

    log.info('')
    log.info('==============================================================')
    log.info('Output written to: %s', Config.OUTPUT_DIR)
    log.info('  multi_period_detailed.csv  (期間×戦略の全指標)')
    log.info('  multi_period_ranking.csv   (戦略別ランキング、安定性スコア順)')
    log.info('==============================================================')

    return 0


# ============================================================================
# Entry point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='DIVIDEND HEIST バックテスト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
戦略一覧:
  A  利回り単純順        (yield高い順)
  B  ボックス × 利回り   (yield × (0.5 + 0.5 × box信頼度))   [デフォルト]
  C  Q75乖離率順         ((yield - Q75) / Q75)
  D  多要素スコア         (0.4y + 0.2ROE + 0.2eq + 0.2box)
  E  利回り × ROE        (yield × ROE / 10)

使用例:
  # B戦略のみ実行 (デフォルト)
  python3 backtest.py

  # 全戦略を1期間で比較
  python3 backtest.py --compare --start 2024-01-01 --end 2025-12-31

  # 全戦略を複数期間で比較 (おすすめ)
  python3 backtest.py --multi-period

  # カスタム期間を指定
  python3 backtest.py --multi-period --periods "2021-01-01:2023-01-01,2023-01-01:2025-01-01"
""",
    )
    parser.add_argument('--start', default='2021-04-01', help='開始日 YYYY-MM-DD')
    parser.add_argument('--end', default='2026-04-30', help='終了日 YYYY-MM-DD')
    parser.add_argument('--capital', type=float, default=10_000_000, help='初期資金 (円)')
    parser.add_argument('--positions', type=int, default=20, help='最大保有銘柄数')
    parser.add_argument('--no-cache', action='store_true', help='キャッシュを使わず再取得')
    parser.add_argument(
        '--strategy', default='B', choices=list(STRATEGIES.keys()),
        help='単体実行する戦略 (A/B/C/D/E、デフォルト: B)',
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='全戦略を比較実行し comparison.csv を出力 (1期間)',
    )
    parser.add_argument(
        '--multi-period', action='store_true',
        help='複数期間 × 全戦略を比較実行 (おすすめ。デフォルト期間: 2020-2022, 2022-2024, 2024-2026)',
    )
    parser.add_argument(
        '--periods', default='',
        help='--multi-period 用のカスタム期間 (例: "2021-01-01:2023-01-01,2023-01-01:2025-01-01")',
    )
    args = parser.parse_args()

    if args.multi_period:
        return run_multi_period(args)
    elif args.compare:
        return run_compare(args)
    else:
        return run_backtest(args)


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIVIDEND HEIST - 高配当スイング自動シグナル生成スクリプト
================================================================

設計対応表 v1.4 に基づく実装。

J-Quants Standard プランの API を叩いて、東証プライム × 時価総額300億円以上の
銘柄に対して以下を計算し、data/results.json に出力する。

  - スクリーニング判定（8条件）
  - バケット分類（コア・累進 / コア・DOE / スイング / 対象外）
  - シグナル判定（BUY / SELL / NEUTRAL）
  - ボックス判定（ADX, 値幅, タッチ, ATR）
  - 撤退判定（累進撤回 / 大幅減配 / 営業利益急減）
  - 決算予定日と財務推移

Usage:
    export J_QUANTS_API_KEY="your_refresh_token"
    python generate.py

GitHub Actions で毎日 19:30 JST に実行する想定。
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

# ============================================================================
# (1) ロギング設定
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('dividend-heist')

# ============================================================================
# (2) 閾値・設定（変更しやすさを保証する一元管理）
# ============================================================================

# --- スクリーニング閾値 ---
DIVIDEND_HISTORY_PERIODS = 10        # 減配チェック期間（過去10期）
REVENUE_STABILITY_PERIODS = 5        # 業績安定性チェック期間（過去5期）
REVENUE_DROP_THRESHOLD = -0.10       # -10%超下落
REVENUE_CONSECUTIVE_LIMIT = 2        # 2期連続でアウト
PAYOUT_RATIO_MAX = 50.0              # 配当性向 ≤ 50%
EQUITY_RATIO_MIN = 0.50              # 自己資本比率 ≥ 50%
GRAHAM_THRESHOLD = 22.5              # PER × PBR ≤ 22.5
MIN_YIELD_THRESHOLD = 3.0            # 最低利回り ≥ 3.0%

# --- ボックス判定閾値 ---
BOX_LOOKBACK_DAYS = 60               # 60日のレンジで判定
BOX_ADX_PERIOD = 14                  # ADX期間
BOX_ADX_THRESHOLD = 22.0             # ADX < 22 でトレンドなし
BOX_WIDTH_MIN = 0.08                 # 値幅 8%
BOX_WIDTH_MAX = 0.20                 # 値幅 20%
BOX_TOUCH_TOLERANCE = 0.01           # 上下限 ±1% 圏内
BOX_TOUCH_MIN = 2                    # 上下それぞれ2回以上
BOX_ATR_MIN = 0.015                  # ATR/終値 ≥ 1.5%

# --- シグナル判定（利回り分布） ---
YIELD_LOOKBACK_YEARS = 5             # 過去5年の月次利回りで分布計算
YIELD_LOOKBACK_DAYS = YIELD_LOOKBACK_YEARS * 365 + 60  # 余裕を持って取得

# --- 撤退判定（緊急シグナル） ---
EMERGENCY_DPS_DROP = -0.10           # DPS 10%超減配
EMERGENCY_OP_DROP = -0.20            # 営業利益YoY -20%以下

# --- ユニバース ---
MARKET_CODE_PRIME = '0111'           # 東証プライム
MARKET_CAP_MIN_OKU = 300             # 時価総額300億円以上

# --- API設定 ---
JQUANTS_BASE = 'https://api.jquants.com'
API_SLEEP_SEC = 0.1                  # レートリミット対策
API_TIMEOUT_SEC = 30
API_MAX_RETRIES = 3

# --- 出力 ---
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'results.json'
OUTPUT_DIR = OUTPUT_PATH.parent

# ============================================================================
# (3) 累進配当銘柄リスト
# ============================================================================

# Tier 1: 日経累進高配当株指数 30銘柄
NIKKEI_PROGRESSIVE_30: set[str] = {
    '4272',  # 日本化薬
    '4502',  # 武田薬品工業
    '8593',  # 三菱HCキャピタル
    '4521',  # 科研製薬
    '5938',  # LIXIL
    '4503',  # アステラス製薬
    '8439',  # 東京センチュリー
    '7956',  # ピジョン
    '9364',  # 上組
    '3861',  # 王子ホールディングス
    '4042',  # 東ソー
    '4208',  # UBE
    '4528',  # 小野薬品工業
    '8309',  # 三井住友トラストグループ
    '8725',  # MS&AD
    '4182',  # 三菱ガス化学
    '4205',  # 日本ゼオン
    '7313',  # テイ・エス テック
    '8252',  # 丸井グループ
    '1719',  # 安藤ハザマ
    '1928',  # 積水ハウス
    '4041',  # 日本曹達
    '5020',  # ENEOSホールディングス
    '8473',  # SBIホールディングス
    '1870',  # 矢作建設工業
    '3431',  # 宮地エンジニアリンググループ
    '5201',  # AGC
    '3291',  # 飯田グループホールディングス
    '4183',  # 三井化学
    '8130',  # サンゲツ
}

# Tier 2: 累進配当宣言銘柄（公式指数未採用）
PROGRESSIVE_DECLARED: set[str] = {
    '8058',  # 三菱商事
    '8001',  # 伊藤忠商事
    '8031',  # 三井物産
    '8002',  # 丸紅
    '8053',  # 住友商事
    '9433',  # KDDI
    '9434',  # ソフトバンク
    '8306',  # 三菱UFJフィナンシャル・グループ
    '8316',  # 三井住友フィナンシャルグループ
    '8411',  # みずほフィナンシャルグループ
    '8766',  # 東京海上ホールディングス
    '8630',  # SOMPOホールディングス
    '1605',  # INPEX
    '5108',  # ブリヂストン
    '7203',  # トヨタ自動車
    '7011',  # 三菱重工業
    '8801',  # 三井不動産
}

PROGRESSIVE_DIVIDEND_LIST: set[str] = NIKKEI_PROGRESSIVE_30 | PROGRESSIVE_DECLARED

# DOE採用銘柄
DOE_LIST: set[str] = {
    '2502',  # アサヒグループホールディングス
    '7011',  # 三菱重工業（DOE採用）
}

# 「減配ゼロ判定 + 最低利回り」免除リスト
DIVIDEND_CHECK_EXEMPT: set[str] = PROGRESSIVE_DIVIDEND_LIST | DOE_LIST

# 自己資本比率チェック免除セクター（金融4業種+不動産業）
EQUITY_RATIO_EXEMPT_SECTORS: set[str] = {
    '7050',  # 銀行業
    '7100',  # 証券、商品先物取引業
    '7150',  # 保険業
    '7200',  # その他金融業
    '8050',  # 不動産業
}

# ============================================================================
# (4) J-Quants API クライアント
# ============================================================================

class JQuantsClient:
    """
    J-Quants V2 API クライアント。

    認証フロー:
      1) 環境変数 J_QUANTS_API_KEY からリフレッシュトークンを読む
      2) /v1/token/auth_refresh で ID トークンを取得（24時間有効）
      3) 以降の API 呼び出しは Authorization: Bearer <id_token>

    注: J-Quants の現行エンドポイントは /v1/... のパス。
        設計対応表の "/v2/" 表記は API バージョンではなく "V2 認証方式" のこと。
    """

    def __init__(self, refresh_token: str):
        if not refresh_token:
            raise ValueError('J_QUANTS_API_KEY (refresh token) is empty')
        self.refresh_token = refresh_token
        self.id_token: str | None = None
        self.session = requests.Session()
        self._authenticate()

    def _authenticate(self) -> None:
        """リフレッシュトークンから ID トークンを取得。"""
        url = f'{JQUANTS_BASE}/v1/token/auth_refresh'
        params = {'refreshtoken': self.refresh_token}
        log.info('Authenticating with J-Quants...')
        try:
            resp = self.session.post(url, params=params, timeout=API_TIMEOUT_SEC)
            resp.raise_for_status()
            data = resp.json()
            self.id_token = data.get('idToken')
            if not self.id_token:
                raise RuntimeError(f'No idToken in response: {data}')
            log.info('Authenticated successfully.')
        except requests.HTTPError as e:
            log.error('Authentication failed: HTTP %s - %s', e.response.status_code, e.response.text)
            raise

    def _headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.id_token}'}

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        GET リクエスト + ページング処理 + リトライ。

        J-Quants はカーソルベースのページングを使う：
        レスポンスに `pagination_key` があれば次ページが存在。
        """
        url = f'{JQUANTS_BASE}{path}'
        all_data: dict[str, list[Any]] = defaultdict(list)
        scalar_data: dict[str, Any] = {}
        pagination_key: str | None = None

        while True:
            req_params = dict(params or {})
            if pagination_key:
                req_params['pagination_key'] = pagination_key

            for attempt in range(API_MAX_RETRIES):
                try:
                    resp = self.session.get(
                        url,
                        params=req_params,
                        headers=self._headers(),
                        timeout=API_TIMEOUT_SEC,
                    )
                    if resp.status_code == 401:
                        # ID トークン期限切れ → 再認証して 1 回だけリトライ
                        log.warning('401 received, re-authenticating...')
                        self._authenticate()
                        resp = self.session.get(
                            url,
                            params=req_params,
                            headers=self._headers(),
                            timeout=API_TIMEOUT_SEC,
                        )
                    if resp.status_code == 429:
                        # レートリミット → バックオフ
                        wait = 2 ** attempt
                        log.warning('429 rate limited, backing off %ds', wait)
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == API_MAX_RETRIES - 1:
                        raise
                    wait = 2 ** attempt
                    log.warning('Request failed (attempt %d): %s, retrying in %ds',
                                attempt + 1, e, wait)
                    time.sleep(wait)
            else:
                raise RuntimeError(f'Failed after {API_MAX_RETRIES} retries: {url}')

            payload = resp.json()
            for key, value in payload.items():
                if key == 'pagination_key':
                    continue
                if isinstance(value, list):
                    all_data[key].extend(value)
                else:
                    scalar_data[key] = value

            pagination_key = payload.get('pagination_key')
            if not pagination_key:
                break
            time.sleep(API_SLEEP_SEC)

        result: dict[str, Any] = dict(scalar_data)
        result.update(all_data)
        return result

    # --- ユニバース取得 ---

    def get_listed_info(self, target_date: str | None = None) -> list[dict[str, Any]]:
        """上場銘柄一覧を取得（東証プライムフィルタは呼び出し側で実施）。"""
        params = {}
        if target_date:
            params['date'] = target_date
        data = self.get('/v1/listed/info', params)
        return data.get('info', [])

    # --- 株価取得 ---

    def get_daily_quotes(self, code: str, from_date: str, to_date: str) -> list[dict[str, Any]]:
        """指定銘柄の日次株価を取得。"""
        params = {'code': code, 'from': from_date, 'to': to_date}
        data = self.get('/v1/prices/daily_quotes', params)
        return data.get('daily_quotes', [])

    # --- 財務取得 ---

    def get_statements(self, code: str) -> list[dict[str, Any]]:
        """指定銘柄の財務情報（過去全期）を取得。"""
        params = {'code': code}
        data = self.get('/v1/fins/statements', params)
        return data.get('statements', [])

    # --- 決算予定 ---

    def get_announcement(self) -> list[dict[str, Any]]:
        """決算発表予定一覧（全銘柄、3週間先まで）を取得。"""
        data = self.get('/v1/fins/announcement', {})
        return data.get('announcement', [])

# ============================================================================
# (5) ヘルパー: 数値変換・期間整形
# ============================================================================

def safe_float(value: Any) -> float | None:
    """空文字や None を None に変換した上で float へ。"""
    if value is None or value == '' or value == '－':
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except ValueError:
        return None


def normalize_code(code: str) -> str:
    """銘柄コードを4桁文字列に正規化（J-Quantsは末尾0付きの5桁を返すことがある）。"""
    code = str(code).strip()
    if len(code) == 5 and code.endswith('0'):
        return code[:4]
    return code


def format_period(s: dict[str, Any]) -> str:
    """財務情報1件から表示用の期間ラベルを作る（例: "2024Q4"）。"""
    period_end = s.get('CurrentPeriodEndDate', '')
    type_code = s.get('TypeOfCurrentPeriod', '')
    year = period_end[:4] if period_end else '----'
    return f'{year}{type_code}' if type_code else year


# ============================================================================
# (6) スクリーニング: 配当履歴
# ============================================================================

def extract_annual_dividends(statements: list[dict[str, Any]]) -> list[float]:
    """
    財務一覧から年度単位の DPS（DivAnn 実績）を時系列で抽出。

    決算期種別 'FY' の DivAnn を採用し、古い順に並べる。
    """
    annual = []
    for s in statements:
        if s.get('TypeOfCurrentPeriod') != 'FY':
            continue
        div = safe_float(s.get('ResultDividendPerShareAnnual'))
        if div is None:
            div = safe_float(s.get('DivAnn'))
        period = parse_date(s.get('CurrentPeriodEndDate'))
        if div is not None and period is not None:
            annual.append((period, div))
    annual.sort(key=lambda x: x[0])
    return [d for _, d in annual]


def check_dividend_history(div_history: list[float], code: str) -> tuple[bool | None, int]:
    """
    減配ゼロ判定。

    Returns:
        (judgment, periods_used)
        - judgment: True=通過, False=減配あり, None=データ不足/免除
    """
    if code in DIVIDEND_CHECK_EXEMPT:
        return True, len(div_history)

    if len(div_history) < 3:
        return None, len(div_history)

    history = div_history[-DIVIDEND_HISTORY_PERIODS:]
    cuts = sum(1 for i in range(1, len(history)) if history[i] < history[i - 1])
    return cuts == 0, len(history)


# ============================================================================
# (7) スクリーニング: 業績安定性
# ============================================================================

def extract_annual_metric(statements: list[dict[str, Any]], key: str) -> list[float]:
    """財務一覧から年度単位の指定指標を時系列で抽出。"""
    annual = []
    for s in statements:
        if s.get('TypeOfCurrentPeriod') != 'FY':
            continue
        v = safe_float(s.get(key))
        period = parse_date(s.get('CurrentPeriodEndDate'))
        if v is not None and period is not None:
            annual.append((period, v))
    annual.sort(key=lambda x: x[0])
    return [v for _, v in annual]


def check_stability(values: list[float]) -> bool | None:
    """
    過去5期の前期比 < -10% が2期連続で発生していなければ True。

    Returns:
        True=安定, False=2期連続急減あり, None=データ不足
    """
    if len(values) < 3:
        return None

    series = values[-REVENUE_STABILITY_PERIODS:]
    if len(series) < 2:
        return None

    drops: list[bool] = []
    for i in range(1, len(series)):
        prev = series[i - 1]
        curr = series[i]
        if prev <= 0:
            drops.append(False)
            continue
        change = (curr - prev) / abs(prev)
        drops.append(change < REVENUE_DROP_THRESHOLD)

    consecutive = 0
    for d in drops:
        if d:
            consecutive += 1
            if consecutive >= REVENUE_CONSECUTIVE_LIMIT:
                return False
        else:
            consecutive = 0
    return True


# ============================================================================
# (8) スクリーニング: 配当性向・自己資本比率・割安性・最低利回り
# ============================================================================

def check_payout_ratio(payout: float | None) -> bool | None:
    if payout is None:
        return None
    return payout <= PAYOUT_RATIO_MAX


def check_equity_ratio(equity_ratio: float | None, sector_code: str) -> bool | None:
    if sector_code in EQUITY_RATIO_EXEMPT_SECTORS:
        return True  # 免除
    if equity_ratio is None:
        return None
    return equity_ratio >= EQUITY_RATIO_MIN


def check_valuation(per: float | None, pbr: float | None) -> bool | None:
    if per is None or pbr is None:
        return None
    if per <= 0 or pbr <= 0:
        return False
    return (per * pbr) <= GRAHAM_THRESHOLD


def check_min_yield(yield_pct: float | None, code: str) -> bool | None:
    if code in DIVIDEND_CHECK_EXEMPT:
        return True
    if yield_pct is None:
        return None
    return yield_pct >= MIN_YIELD_THRESHOLD


# ============================================================================
# (9) シグナル判定（利回り分布 Q25/Q75）
# ============================================================================

def quantile(sorted_values: list[float], q: float) -> float:
    """簡易な線形補間 quantile（NumPy 不要）。"""
    if not sorted_values:
        return float('nan')
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def calculate_yield_distribution(
    quotes: list[dict[str, Any]],
    div_history_by_year: dict[int, float],
) -> dict[str, float]:
    """
    過去5年の月次配当利回りから Q25 / median / Q75 を計算。

    各月末時点の利回り = 期中DPS実績 / 月末終値
    """
    monthly: dict[str, float] = {}  # 'YYYY-MM' → 月末終値
    for q in quotes:
        d = parse_date(q.get('Date'))
        close = safe_float(q.get('Close'))
        if d is None or close is None or close <= 0:
            continue
        ym = f'{d.year}-{d.month:02d}'
        # 月末を更新（最新の日付の Close を取る）
        prev = monthly.get(ym)
        if prev is None or d > prev[0]:
            monthly[ym] = (d, close)

    yields: list[float] = []
    for ym, (d, close) in monthly.items():
        dps = div_history_by_year.get(d.year)
        if dps is None or dps <= 0:
            continue
        y = (dps / close) * 100.0
        yields.append(y)

    if len(yields) < 12:
        return {'q25': float('nan'), 'median': float('nan'), 'q75': float('nan'), 'n': len(yields)}

    yields.sort()
    return {
        'q25': quantile(yields, 0.25),
        'median': quantile(yields, 0.50),
        'q75': quantile(yields, 0.75),
        'n': len(yields),
    }


def determine_signal(current_yield: float | None, dist: dict[str, float]) -> str:
    """BUY / SELL / NEUTRAL を判定。"""
    if current_yield is None or current_yield <= 0:
        return 'NEUTRAL'
    q75 = dist.get('q75')
    q25 = dist.get('q25')
    if q75 is None or math.isnan(q75) or q25 is None or math.isnan(q25):
        return 'NEUTRAL'
    if current_yield >= q75:
        return 'BUY'
    if current_yield <= q25:
        return 'SELL'
    return 'NEUTRAL'


# ============================================================================
# (10) ボックス相場判定（ADX, 値幅, タッチ, ATR）
# ============================================================================

def calculate_true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def calculate_adx_and_atr(
    bars: list[dict[str, float]],
    period: int = BOX_ADX_PERIOD,
) -> tuple[float | None, float | None]:
    """
    Wilder's smoothing による ADX(period) と ATR(period) を計算。

    bars: [{'high': float, 'low': float, 'close': float}, ...] の時系列順。
    """
    if len(bars) < period * 2:
        return None, None

    tr_list: list[float] = []
    plus_dm: list[float] = []
    minus_dm: list[float] = []

    for i in range(1, len(bars)):
        h, l, c = bars[i]['high'], bars[i]['low'], bars[i]['close']
        ph, pl, pc = bars[i - 1]['high'], bars[i - 1]['low'], bars[i - 1]['close']
        tr = calculate_true_range(h, l, pc)
        tr_list.append(tr)
        up = h - ph
        down = pl - l
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)

    if len(tr_list) < period:
        return None, None

    # Wilder smoothing
    atr = sum(tr_list[:period]) / period
    plus_di_sm = sum(plus_dm[:period]) / period
    minus_di_sm = sum(minus_dm[:period]) / period

    dx_list: list[float] = []

    def calc_dx(plus_di: float, minus_di: float) -> float:
        denom = plus_di + minus_di
        if denom == 0:
            return 0.0
        return abs(plus_di - minus_di) / denom * 100.0

    plus_di = (plus_di_sm / atr) * 100.0 if atr > 0 else 0.0
    minus_di = (minus_di_sm / atr) * 100.0 if atr > 0 else 0.0
    dx_list.append(calc_dx(plus_di, minus_di))

    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di_sm = (plus_di_sm * (period - 1) + plus_dm[i]) / period
        minus_di_sm = (minus_di_sm * (period - 1) + minus_dm[i]) / period
        plus_di = (plus_di_sm / atr) * 100.0 if atr > 0 else 0.0
        minus_di = (minus_di_sm / atr) * 100.0 if atr > 0 else 0.0
        dx_list.append(calc_dx(plus_di, minus_di))

    if len(dx_list) < period:
        return None, atr

    adx = sum(dx_list[:period]) / period
    for i in range(period, len(dx_list)):
        adx = (adx * (period - 1) + dx_list[i]) / period

    return adx, atr


def evaluate_box(quotes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    過去 BOX_LOOKBACK_DAYS 営業日でボックス相場かを判定。

    Returns:
        dict | None - 60日未満ならNone、それ以外は判定結果
    """
    bars: list[dict[str, float]] = []
    for q in quotes:
        h = safe_float(q.get('High'))
        l = safe_float(q.get('Low'))
        c = safe_float(q.get('Close'))
        if h is None or l is None or c is None:
            continue
        bars.append({'high': h, 'low': l, 'close': c})

    if len(bars) < BOX_LOOKBACK_DAYS:
        return None

    recent = bars[-BOX_LOOKBACK_DAYS:]
    closes = [b['close'] for b in recent]
    highs = [b['high'] for b in recent]
    lows = [b['low'] for b in recent]

    upper = max(highs)
    lower = min(lows)
    avg_close = sum(closes) / len(closes)

    width_pct = ((upper - lower) / avg_close) if avg_close > 0 else 0.0

    upper_touches = sum(
        1 for c in closes if abs(c - upper) / upper <= BOX_TOUCH_TOLERANCE
    ) if upper > 0 else 0
    lower_touches = sum(
        1 for c in closes if abs(c - lower) / lower <= BOX_TOUCH_TOLERANCE
    ) if lower > 0 else 0

    adx, atr = calculate_adx_and_atr(recent, BOX_ADX_PERIOD)
    last_close = closes[-1]
    atr_ratio = (atr / last_close * 100.0) if (atr is not None and last_close > 0) else None

    is_box = (
        adx is not None and adx < BOX_ADX_THRESHOLD
        and BOX_WIDTH_MIN <= width_pct <= BOX_WIDTH_MAX
        and upper_touches >= BOX_TOUCH_MIN
        and lower_touches >= BOX_TOUCH_MIN
        and atr_ratio is not None and atr_ratio >= BOX_ATR_MIN * 100.0
    )

    return {
        'is_box': is_box,
        'adx': round(adx, 2) if adx is not None else None,
        'width_pct': round(width_pct * 100.0, 2),
        'upper': round(upper, 2),
        'lower': round(lower, 2),
        'upper_touches': upper_touches,
        'lower_touches': lower_touches,
        'atr_ratio': round(atr_ratio, 2) if atr_ratio is not None else None,
    }


# ============================================================================
# (11) 撤退判定（緊急シグナル）
# ============================================================================

def check_emergency_exit(
    code: str,
    div_history: list[float],
    op_history: list[float],
    progressive_status_lost: bool = False,
) -> tuple[bool, list[str]]:
    """
    撤退判定。3条件のいずれかで True。

    Returns:
        (emergency_flag, reasons)
    """
    reasons: list[str] = []

    # 1) 累進配当撤回
    if code in PROGRESSIVE_DIVIDEND_LIST and progressive_status_lost:
        reasons.append('累進配当撤回')

    # 2) DPS 10%超減配
    if len(div_history) >= 2:
        prev, curr = div_history[-2], div_history[-1]
        if prev > 0:
            change = (curr - prev) / prev
            if change <= EMERGENCY_DPS_DROP:
                reasons.append(f'DPS{abs(change) * 100:.1f}%減配')

    # 3) 営業利益 YoY -20% 以下
    if len(op_history) >= 2:
        prev, curr = op_history[-2], op_history[-1]
        if prev > 0:
            change = (curr - prev) / prev
            if change <= EMERGENCY_OP_DROP:
                reasons.append(f'営業利益YoY{change * 100:.1f}%')

    return bool(reasons), reasons


# ============================================================================
# (12) バケット分類
# ============================================================================

def classify_bucket(code: str, screening_pass: bool) -> tuple[str, str | None]:
    """
    バケット判定。

    Returns:
        (bucket, core_type)
        bucket  : 'コア' / 'スイング' / '対象外'
        core_type: 'progressive' / 'doe' / None
    """
    if not screening_pass:
        return ('対象外', None)
    if code in PROGRESSIVE_DIVIDEND_LIST:
        return ('コア', 'progressive')
    if code in DOE_LIST:
        return ('コア', 'doe')
    return ('スイング', None)

# ============================================================================
# (13) 財務推移 / 決算予定 / その他の整形
# ============================================================================

def build_financials_history(statements: list[dict[str, Any]], n: int = 8) -> list[dict[str, Any]]:
    """
    直近 n 期分の財務サマリを抽出（FY/Q1/Q2/Q3 含む全期）。

    Returns:
        新しい順 → 古い順、最大 n 件
    """
    rows: list[tuple[date, dict[str, Any]]] = []
    for s in statements:
        period_end = parse_date(s.get('CurrentPeriodEndDate'))
        if period_end is None:
            continue
        rows.append((period_end, {
            'period': format_period(s),
            'sales': safe_float(s.get('NetSales')),
            'op': safe_float(s.get('OperatingProfit')),
            'np': safe_float(s.get('Profit')),
            'eps': safe_float(s.get('EarningsPerShare')),
            'dps': safe_float(s.get('ResultDividendPerShareAnnual')),
        }))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows[:n]]


def is_imminent_business_day(target: date | None, today: date) -> bool:
    """
    target が today の翌営業日（月〜金、土日跨ぎ考慮）に該当するなら True。

    祝日カレンダーは扱わない（簡易判定）。
    """
    if target is None:
        return False
    weekday = today.weekday()  # Mon=0 ... Sun=6
    if weekday <= 3:  # Mon-Thu → 翌日
        next_bday = today + timedelta(days=1)
    elif weekday == 4:  # Fri → 月曜
        next_bday = today + timedelta(days=3)
    elif weekday == 5:  # Sat → 月曜
        next_bday = today + timedelta(days=2)
    else:  # Sun → 月曜
        next_bday = today + timedelta(days=1)
    return target == next_bday


def get_latest_full_year_metric(statements: list[dict[str, Any]], key: str) -> float | None:
    """直近の FY データから指定指標を取得。"""
    fy_list = [s for s in statements if s.get('TypeOfCurrentPeriod') == 'FY']
    fy_list.sort(key=lambda s: parse_date(s.get('CurrentPeriodEndDate')) or date.min)
    if not fy_list:
        return None
    return safe_float(fy_list[-1].get(key))


def get_forecast_dps(statements: list[dict[str, Any]]) -> float | None:
    """会社予想の年間 DPS（FDivAnn 相当）を取得。最新の Q から取る。"""
    sorted_stmts = sorted(
        statements,
        key=lambda s: (parse_date(s.get('DisclosedDate')) or date.min),
        reverse=True,
    )
    for s in sorted_stmts:
        f = safe_float(s.get('ForecastDividendPerShareAnnual'))
        if f is not None and f > 0:
            return f
    # フォールバック: 直近FYの実績
    return get_latest_full_year_metric(statements, 'ResultDividendPerShareAnnual')


def get_latest_payout_ratio(statements: list[dict[str, Any]]) -> float | None:
    """最新FYの配当性向を取得。なければ DivAnn / EPS から計算。"""
    fy_list = [s for s in statements if s.get('TypeOfCurrentPeriod') == 'FY']
    fy_list.sort(key=lambda s: parse_date(s.get('CurrentPeriodEndDate')) or date.min)
    if not fy_list:
        return None
    latest = fy_list[-1]
    payout = safe_float(latest.get('ResultPayoutRatioAnnual'))
    if payout is not None:
        return payout
    div = safe_float(latest.get('ResultDividendPerShareAnnual'))
    eps = safe_float(latest.get('EarningsPerShare'))
    if div is not None and eps is not None and eps > 0:
        return (div / eps) * 100.0
    return None


def get_equity_ratio_latest(statements: list[dict[str, Any]]) -> float | None:
    """最新の自己資本比率（EquityToAssetRatio）。"""
    sorted_stmts = sorted(
        statements,
        key=lambda s: parse_date(s.get('CurrentPeriodEndDate')) or date.min,
    )
    for s in reversed(sorted_stmts):
        ratio = safe_float(s.get('EquityToAssetRatio'))
        if ratio is not None:
            return ratio
    return None


def get_per_pbr(price: float, statements: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """PER, PBR を計算。EPS/BPS は最新FYから取る。"""
    eps = get_latest_full_year_metric(statements, 'EarningsPerShare')
    bps = get_latest_full_year_metric(statements, 'BookValuePerShare')
    per = (price / eps) if (eps is not None and eps > 0) else None
    pbr = (price / bps) if (bps is not None and bps > 0) else None
    return per, pbr


# ============================================================================
# (14) 個別銘柄の処理
# ============================================================================

def process_stock(
    client: JQuantsClient,
    code: str,
    name: str,
    sector33: str,
    sector33_name: str,
    market_cap_oku: float,
    today: date,
    announcement_map: dict[str, date],
) -> dict[str, Any] | None:
    """1銘柄分のデータ取得 → 計算 → 結果 dict を返す。"""
    log.debug('Processing %s %s', code, name)
    time.sleep(API_SLEEP_SEC)

    # --- 株価取得（過去 YIELD_LOOKBACK_DAYS 日）---
    to_date = today.strftime('%Y-%m-%d')
    from_date = (today - timedelta(days=YIELD_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    try:
        quotes = client.get_daily_quotes(code, from_date, to_date)
    except Exception as e:
        log.warning('quotes fail %s: %s', code, e)
        return None

    if not quotes:
        log.debug('no quotes for %s', code)
        return None

    quotes.sort(key=lambda q: q.get('Date', ''))
    last_quote = quotes[-1]
    price = safe_float(last_quote.get('Close'))
    if price is None or price <= 0:
        return None

    time.sleep(API_SLEEP_SEC)

    # --- 財務取得 ---
    try:
        statements = client.get_statements(code)
    except Exception as e:
        log.warning('statements fail %s: %s', code, e)
        return None

    # --- 指標抽出 ---
    div_annual = extract_annual_dividends(statements)
    sales_history = extract_annual_metric(statements, 'NetSales')
    op_history = extract_annual_metric(statements, 'OperatingProfit')
    np_history = extract_annual_metric(statements, 'Profit')

    forecast_dps = get_forecast_dps(statements)
    current_yield = (forecast_dps / price * 100.0) if (forecast_dps and forecast_dps > 0) else None
    payout = get_latest_payout_ratio(statements)
    equity_ratio = get_equity_ratio_latest(statements)
    per, pbr = get_per_pbr(price, statements)
    per_pbr = (per * pbr) if (per is not None and pbr is not None) else None

    # --- スクリーニング ---
    div_judgment, div_periods_used = check_dividend_history(div_annual, code)
    s_results = {
        'dividend': div_judgment,
        'revenue': check_stability(sales_history),
        'op_profit': check_stability(op_history),
        'net_profit': check_stability(np_history),
        'payout': check_payout_ratio(payout),
        'equity': check_equity_ratio(equity_ratio, sector33),
        'valuation': check_valuation(per, pbr),
        'min_yield': check_min_yield(current_yield, code),
    }
    # None（免除/データなし）は通過扱い
    screening_pass = all(v is None or v is True for v in s_results.values())

    # --- バケット ---
    bucket, core_type = classify_bucket(code, screening_pass)

    # --- シグナル（利回り分布）---
    div_history_by_year: dict[int, float] = {}
    for s in statements:
        if s.get('TypeOfCurrentPeriod') != 'FY':
            continue
        d = parse_date(s.get('CurrentPeriodEndDate'))
        v = safe_float(s.get('ResultDividendPerShareAnnual'))
        if d and v is not None:
            div_history_by_year[d.year] = v

    yield_dist = calculate_yield_distribution(quotes, div_history_by_year)
    signal = determine_signal(current_yield, yield_dist)

    # --- ボックス ---
    box = evaluate_box(quotes)

    # --- 撤退判定 ---
    emergency, reasons = check_emergency_exit(code, div_annual, op_history)

    # --- 決算予定 ---
    next_earnings = announcement_map.get(code)
    earnings_imminent = is_imminent_business_day(next_earnings, today)

    # --- 財務推移 ---
    financials_history = build_financials_history(statements, n=8)

    # --- JSON 組み立て ---
    return {
        'code': code,
        'name': name,
        'sector': sector33_name,
        'sector33_code': sector33,
        'market_cap_oku': round(market_cap_oku),
        'price': round(price, 2),

        'bucket': bucket,
        'core_type': core_type,
        'screening_pass': screening_pass,
        'signal': signal,
        'is_box': box['is_box'] if box else False,
        'emergency_exit': emergency,
        'emergency_reasons': reasons,
        'earnings_imminent': earnings_imminent,
        'next_earnings_date': next_earnings.isoformat() if next_earnings else None,

        'forecast_dps': round(forecast_dps, 2) if forecast_dps is not None else None,
        'current_yield': round(current_yield, 2) if current_yield is not None else None,
        'yield_q25': round(yield_dist['q25'], 2) if not math.isnan(yield_dist['q25']) else None,
        'yield_median': round(yield_dist['median'], 2) if not math.isnan(yield_dist['median']) else None,
        'yield_q75': round(yield_dist['q75'], 2) if not math.isnan(yield_dist['q75']) else None,
        'yield_sample_n': yield_dist['n'],

        'per': round(per, 2) if per is not None else None,
        'pbr': round(pbr, 2) if pbr is not None else None,
        'per_pbr': round(per_pbr, 2) if per_pbr is not None else None,
        'payout_ratio': round(payout, 2) if payout is not None else None,
        'equity_ratio': round(equity_ratio, 4) if equity_ratio is not None else None,

        'screening': s_results,
        'screening_meta': {
            'dividend_periods_used': div_periods_used,
        },

        'box': box if box else None,

        'financials_history': financials_history,

        'links': {
            'minkabu': f'https://minkabu.jp/stock/{code}',
            'irbank': f'https://irbank.net/{code}/',
            'tdnet': 'https://www.release.tdnet.info/inbs/I_main_00.html',
        },
    }


# ============================================================================
# (15) ユニバース取得 + フィルタ
# ============================================================================

def build_universe(client: JQuantsClient) -> list[dict[str, Any]]:
    """
    上場銘柄一覧を取得して東証プライム × 時価総額 300億円以上にフィルタ。

    時価総額は J-Quants の listed_info に直接含まれないため、ここでは
    プライム銘柄を全件取得し、時価総額フィルタは前日株価×発行株式数で
    後段（process_stock 直前）で行う。listed_info に MarketCapitalization
    がある場合はそれを優先利用。
    """
    log.info('Fetching listed info...')
    info = client.get_listed_info()
    log.info('Got %d listed entries.', len(info))

    universe = []
    for row in info:
        market_code = row.get('MarketCode', '')
        if market_code != MARKET_CODE_PRIME:
            continue

        code = normalize_code(row.get('Code', ''))
        if not code:
            continue

        # 時価総額: listed_info にあれば使う（円単位）
        mcap_yen = safe_float(row.get('MarketCapitalization'))
        mcap_oku = (mcap_yen / 1e8) if mcap_yen is not None else None

        sector33 = row.get('Sector33Code', '')
        sector33_name = row.get('Sector33CodeName', '')
        name = row.get('CompanyName', '') or row.get('CompanyNameEnglish', '')

        universe.append({
            'code': code,
            'name': name,
            'sector33': sector33,
            'sector33_name': sector33_name,
            'market_cap_oku': mcap_oku,
        })

    # 時価総額フィルタ（None は後段で個別計算する場合もあるが、ここでは確定値のみ通す）
    if any(u['market_cap_oku'] is not None for u in universe):
        filtered = [u for u in universe if u['market_cap_oku'] is not None
                    and u['market_cap_oku'] >= MARKET_CAP_MIN_OKU]
        log.info('Universe after market_cap >= %d 億円: %d',
                 MARKET_CAP_MIN_OKU, len(filtered))
        return filtered
    else:
        log.warning('listed_info に MarketCapitalization なし、時価総額フィルタは後段で実施')
        return universe


def build_announcement_map(client: JQuantsClient) -> dict[str, date]:
    """決算発表予定 → {code: 発表予定日} のマップを作る。"""
    log.info('Fetching earnings announcements...')
    try:
        ann = client.get_announcement()
    except Exception as e:
        log.warning('announcement fetch failed: %s', e)
        return {}

    result: dict[str, date] = {}
    for row in ann:
        code = normalize_code(row.get('Code', ''))
        d = parse_date(row.get('Date'))
        if code and d:
            # 同一銘柄複数あれば直近のみ採用
            if code not in result or d < result[code]:
                result[code] = d

    log.info('Announcement map size: %d', len(result))
    return result


# ============================================================================
# (16) メイン処理
# ============================================================================

def main() -> int:
    api_key = os.environ.get('J_QUANTS_API_KEY')
    if not api_key:
        log.error('J_QUANTS_API_KEY 環境変数が設定されていません')
        return 2

    today = date.today()
    log.info('=== DIVIDEND HEIST generate.py start (today=%s) ===', today)

    # クライアント
    try:
        client = JQuantsClient(api_key)
    except Exception as e:
        log.error('Authentication failed: %s', e)
        return 3

    # ユニバース
    universe = build_universe(client)
    log.info('Universe size: %d', len(universe))

    # 決算予定
    announcement_map = build_announcement_map(client)

    # 銘柄ループ
    stocks: list[dict[str, Any]] = []
    failures = 0
    n_total = len(universe)
    for i, u in enumerate(universe, 1):
        code = u['code']
        try:
            result = process_stock(
                client,
                code=code,
                name=u['name'],
                sector33=u['sector33'],
                sector33_name=u['sector33_name'],
                market_cap_oku=u['market_cap_oku'] or 0.0,
                today=today,
                announcement_map=announcement_map,
            )
            if result is not None:
                stocks.append(result)
        except Exception as e:
            log.warning('Failed %s: %s', code, e)
            failures += 1

        if i % 50 == 0:
            log.info('Progress: %d/%d (failures=%d)', i, n_total, failures)

    # 統計
    bucket_counts: dict[str, int] = defaultdict(int)
    signal_counts: dict[str, int] = defaultdict(int)
    for s in stocks:
        bucket_counts[s['bucket']] += 1
        signal_counts[s['signal']] += 1

    log.info('=== Summary ===')
    log.info('Processed: %d / %d (failures=%d)', len(stocks), n_total, failures)
    for k, v in bucket_counts.items():
        log.info('  bucket %s: %d', k, v)
    for k, v in signal_counts.items():
        log.info('  signal %s: %d', k, v)

    # 出力
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at': datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'universe_count': len(universe),
        'processed_count': len(stocks),
        'failure_count': failures,
        'thresholds': {
            'min_yield': MIN_YIELD_THRESHOLD,
            'graham': GRAHAM_THRESHOLD,
            'payout_max': PAYOUT_RATIO_MAX,
            'equity_min': EQUITY_RATIO_MIN,
            'box_adx_max': BOX_ADX_THRESHOLD,
            'box_width_min': BOX_WIDTH_MIN,
            'box_width_max': BOX_WIDTH_MAX,
        },
        'stocks': stocks,
    }
    with OUTPUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    log.info('Wrote %s (%d stocks)', OUTPUT_PATH, len(stocks))
    return 0


if __name__ == '__main__':
    sys.exit(main())

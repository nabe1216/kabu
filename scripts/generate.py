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
BOX_TOUCH_TOLERANCE = 0.02           # 上下限 ±2% 圏内
BOX_TOUCH_MIN = 1                    # 上下それぞれ1回以上
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

# --- API設定 (J-Quants V2) ---
JQUANTS_BASE = 'https://api.jquants.com'
API_SLEEP_SEC = 0.55                 # Standard 120 req/min = 0.5s/req に余裕を持たせる
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
# DOE (Dividend on Equity) = 自己資本配当率を一定以上に維持する株主還元方針
# DOE採用銘柄はスクリーニングをバイパスして常に「コア・DOE」として扱う
DOE_LIST: set[str] = {
    '2502',  # アサヒグループホールディングス
    '7011',  # 三菱重工業（DOE採用）
    '8595',  # ジャフコグループ (DOE 8%)
    '6770',  # アルプスアルパイン (DOE)
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

    認証: 環境変数 J_QUANTS_API_KEY (= ダッシュボードで発行した API Key) を
    `x-api-key` ヘッダーに付けて送るだけ。
    V1 のような refresh token → ID token の交換は不要。

    エンドポイントは /v2/... に統一。レスポンスは全て `{"data": [...], "pagination_key": "..."}` の形式。
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError('J_QUANTS_API_KEY is empty')
        self.api_key = api_key
        self.session = requests.Session()
        log.info('Using J-Quants V2 API key authentication.')

    def _headers(self) -> dict[str, str]:
        return {'x-api-key': self.api_key}

    def get(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        GET リクエスト + ページング処理 + リトライ。

        V2 はカーソルベースのページング。レスポンスに `pagination_key` があれば次ページ。
        全ページ分の `data` 配列を結合して返す。
        """
        url = f'{JQUANTS_BASE}{path}'
        all_rows: list[dict[str, Any]] = []
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
                    if resp.status_code == 429:
                        # レートリミット超過 → 指数バックオフ
                        wait = 2 ** (attempt + 1)
                        log.warning('429 rate limited, backing off %ds', wait)
                        time.sleep(wait)
                        continue
                    if resp.status_code in (401, 403):
                        log.error('Auth failed: HTTP %s - %s',
                                  resp.status_code, resp.text[:300])
                        resp.raise_for_status()
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
            # V2 は `data` キーに統一されているが、念のため他のキーもフォールバックで取れるようにする
            rows = payload.get('data')
            if rows is None:
                # フォールバック: V2 移行途中でキー名が違う場合に備える
                for key, value in payload.items():
                    if key != 'pagination_key' and isinstance(value, list):
                        rows = value
                        break
            if rows:
                all_rows.extend(rows)

            pagination_key = payload.get('pagination_key')
            if not pagination_key:
                break
            time.sleep(API_SLEEP_SEC)

        return all_rows

    # --- 株価カラム名の正規化 (V2の短縮形 → V1相当) ---

    @staticmethod
    def _normalize_quote(q: dict[str, Any]) -> dict[str, Any]:
        """V2の短縮カラム名を V1 形式 (Open/High/Low/Close 等) に揃える。"""
        mapping = {
            'O': 'Open', 'H': 'High', 'L': 'Low', 'C': 'Close',
            'Vo': 'Volume', 'Va': 'TurnoverValue',
            'AdjO': 'AdjustmentOpen', 'AdjH': 'AdjustmentHigh',
            'AdjL': 'AdjustmentLow', 'AdjC': 'AdjustmentClose',
            'AdjVo': 'AdjustmentVolume', 'AdjFactor': 'AdjustmentFactor',
        }
        out = dict(q)
        for short, long_name in mapping.items():
            if short in out and long_name not in out:
                out[long_name] = out[short]
        return out

    # --- ユニバース取得 ---

    def get_listed_info(self, target_date: str | None = None) -> list[dict[str, Any]]:
        """
        上場銘柄一覧 (Listed Issue Master) を取得。
        V2: /v2/equities/master
        """
        params = {}
        if target_date:
            params['date'] = target_date
        return self.get('/v2/equities/master', params)

    # --- 株価取得 ---

    def get_daily_quotes(self, code: str, from_date: str, to_date: str) -> list[dict[str, Any]]:
        """
        指定銘柄の日次株価 (Stock Prices OHLC) を取得。
        V2: /v2/equities/bars/daily
        カラム名は V1 互換に正規化して返す。
        """
        params = {'code': code, 'from': from_date, 'to': to_date}
        rows = self.get('/v2/equities/bars/daily', params)
        return [self._normalize_quote(r) for r in rows]

    # --- 財務取得 ---

    def get_statements(self, code: str) -> list[dict[str, Any]]:
        """
        指定銘柄の財務情報（サマリ）を取得。
        V2: /v2/fins/summary (V1 の /fins/statements 後継)
        """
        params = {'code': code}
        return self.get('/v2/fins/summary', params)

    # --- 決算予定 ---

    def get_announcement(self) -> list[dict[str, Any]]:
        """
        決算発表予定 (Earnings Calendar) を取得。
        V2: /v2/equities/earnings-calendar
        """
        return self.get('/v2/equities/earnings-calendar', {})


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
    period_end = s.get('CurPerEn', '')
    type_code = s.get('CurPerType', '')
    year = period_end[:4] if period_end else '----'
    return f'{year}{type_code}' if type_code else year


# ============================================================================
# (6) スクリーニング: 配当履歴
# ============================================================================

def extract_annual_dividends(
    statements: list[dict[str, Any]],
    quotes_sorted: list[dict[str, Any]] | None = None,
) -> list[float]:
    """
    財務一覧から年度単位の DPS（DivAnn 実績）を時系列で抽出。

    quotes_sorted を渡すと、分割調整して連続性を保つ。
    例: 1:5分割があった場合、過去の DivAnn (分割前=高い値) を 0.2倍して現在ベースに揃える。
    これにより減配判定の誤動作 (分割を減配と誤認) を防ぐ。

    決算期種別 'FY' / '4Q' の DivAnn を採用し、古い順に並べる。
    """
    annual = []
    for s in statements:
        if s.get('CurPerType') not in ('FY', '4Q'):
            continue
        div = safe_float(s.get('DivAnn'))
        period = parse_date(s.get('CurPerEn'))
        if div is None or period is None:
            continue
        # 分割調整
        disc_date = parse_date(s.get('DiscDate')) or period
        adj = cumulative_adj_factor_after(quotes_sorted, disc_date) if quotes_sorted else 1.0
        annual.append((period, div * adj))
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
        if s.get('CurPerType') not in ('FY', '4Q'):
            continue
        v = safe_float(s.get(key))
        period = parse_date(s.get('CurPerEn'))
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
) -> dict[str, Any]:
    """
    過去5年の月次配当利回りから Q25 / median / Q75 を計算 + 時系列データを返す。

    各月末時点の利回り = 期中DPS実績 / 月末終値
    """
    monthly: dict[str, tuple[date, float]] = {}  # 'YYYY-MM' → (月末日付, 月末調整済終値)
    for q in quotes:
        d = parse_date(q.get('Date'))
        # 株価は AdjustmentClose (分割調整済) を優先使用、なければ Close (素値)
        close = safe_float(q.get('AdjustmentClose')) or safe_float(q.get('Close'))
        if d is None or close is None or close <= 0:
            continue
        ym = f'{d.year}-{d.month:02d}'
        # 月末を更新（最新の日付の Close を取る）
        prev = monthly.get(ym)
        if prev is None or d > prev[0]:
            monthly[ym] = (d, close)

    yields: list[float] = []
    history: list[dict[str, Any]] = []  # 時系列データ
    for ym, (d, close) in sorted(monthly.items()):
        dps = div_history_by_year.get(d.year)
        if dps is None or dps <= 0:
            continue
        y = (dps / close) * 100.0
        yields.append(y)
        history.append({
            'date': d.isoformat(),
            'yield': round(y, 3),
            'price': round(close, 2),
            'dps': round(dps, 2),
        })

    if len(yields) < 12:
        return {
            'q25': float('nan'), 'median': float('nan'), 'q75': float('nan'),
            'n': len(yields),
            'history': history,
        }

    yields_sorted = sorted(yields)
    return {
        'q25': quantile(yields_sorted, 0.25),
        'median': quantile(yields_sorted, 0.50),
        'q75': quantile(yields_sorted, 0.75),
        'n': len(yields_sorted),
        'history': history,
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

def is_progressive_or_doe(code: str) -> bool:
    """累進配当 or DOE 採用銘柄か"""
    return code in PROGRESSIVE_DIVIDEND_LIST or code in DOE_LIST


def compute_industry_leaders(stocks: list[dict[str, Any]]) -> set[str]:
    """
    全銘柄プロセス後に呼ぶ。「業界首位級」銘柄コードのセットを返す。

    定義:
      - 同じ33業種コード内で、時価総額 TOP3
      - または TOPIX Core30 銘柄 (bucket='TOPIX Core30' から判定)

    引数:
      stocks: process_stock の戻り値リスト (market_cap_oku, sector33_code, bucket を含む)
    """
    leaders: set[str] = set()

    # 1. TOPIX Core30 銘柄を全部追加
    for s in stocks:
        if s.get('bucket') == 'TOPIX Core30':
            leaders.add(s['code'])

    # 2. 同業種33コード内で時価総額 TOP3
    sector_groups: dict[str, list[tuple[str, float]]] = {}
    for s in stocks:
        sector33 = s.get('sector33_code', '')
        mcap = s.get('market_cap_oku', 0) or 0
        if not sector33 or mcap <= 0:
            continue
        sector_groups.setdefault(sector33, []).append((s['code'], mcap))

    for sector33, members in sector_groups.items():
        members.sort(key=lambda x: -x[1])
        for code, _ in members[:3]:
            leaders.add(code)

    return leaders


def classify_tier(
    code: str,
    industry_leaders: set[str],
) -> tuple[str, float]:
    """
    銘柄を Tier に分類し、(tier, weight) を返す。

    Tier S: 累進/DOE銘柄 AND 業界首位級 → 重み 4.0
    Tier A: 累進/DOE銘柄 OR 業界首位級   → 重み 2.0
    Tier B: スクリーニングPASSのみ         → 重み 1.0

    注: 呼び出し側で「screening_pass=True」を前提とする。
    """
    is_qual = is_progressive_or_doe(code)
    is_leader = code in industry_leaders

    if is_qual and is_leader:
        return ('S', 4.0)
    elif is_qual or is_leader:
        return ('A', 2.0)
    else:
        return ('B', 1.0)


def classify_bucket(code: str, screening_pass: bool) -> tuple[str, str | None]:
    """
    バケット判定。

    重要な設計判断:
      累進配当宣言銘柄 / DOE採用銘柄は、その株主還元方針自体が
      投資判断の根拠になるため、スクリーニング結果に関わらず常に「コア」に
      分類する。スクリーニング結果は別途 screening_pass フラグで保持されるので、
      UI 側で「コアだが財務スクリーニングは FAIL」のような表示が可能。

      これにより、業績変動の大きい VC (ジャフコG等) や周期性の強い業界の
      DOE 銘柄も、追跡対象から外れない。

    Returns:
        (bucket, core_type)
        bucket  : 'コア' / 'スイング' / '対象外'
        core_type: 'progressive' / 'doe' / None
    """
    # 累進・DOE銘柄: 常にコア (screening_pass を問わない)
    if code in PROGRESSIVE_DIVIDEND_LIST:
        return ('コア', 'progressive')
    if code in DOE_LIST:
        return ('コア', 'doe')
    # 上記以外: スクリーニング結果でスイング or 対象外
    if not screening_pass:
        return ('対象外', None)
    return ('スイング', None)

# ============================================================================
# (13) 財務推移 / 決算予定 / その他の整形
# ============================================================================

def build_financials_history(
    statements: list[dict[str, Any]],
    n: int = 8,
    quotes_sorted: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    直近 n 期分の財務サマリを抽出（FY/Q1/Q2/Q3 含む全期）。

    quotes_sorted を渡すと、1株あたり値 (EPS, DPS) を分割調整して連続性を保つ。
    Sales/OP/NP/Eq は総額系なので調整不要。

    Returns:
        新しい順 → 古い順、最大 n 件
    """
    rows: list[tuple[date, dict[str, Any]]] = []
    for s in statements:
        period_end = parse_date(s.get('CurPerEn'))
        if period_end is None:
            continue
        disc_date = parse_date(s.get('DiscDate')) or period_end
        adj = cumulative_adj_factor_after(quotes_sorted, disc_date) if quotes_sorted else 1.0

        np_val = safe_float(s.get('NP'))
        eq_val = safe_float(s.get('Eq'))
        eps_raw = safe_float(s.get('EPS'))
        dps_raw = safe_float(s.get('DivAnn'))
        eps_val = eps_raw * adj if eps_raw is not None else None
        dps_val = dps_raw * adj if dps_raw is not None else None
        roe_val = None
        if np_val is not None and eq_val is not None and eq_val > 0:
            roe_val = (np_val / eq_val) * 100.0
        rows.append((period_end, {
            'period': format_period(s),
            'sales': safe_float(s.get('Sales')),  # 総額: 調整不要
            'op': safe_float(s.get('OP')),         # 総額: 調整不要
            'np': np_val,                           # 総額: 調整不要
            'eps': eps_val,                         # 1株: 分割調整済
            'dps': dps_val,                         # 1株: 分割調整済
            'eq': eq_val,                           # 総額: 調整不要
            'roe': roe_val,                         # 比率: 不変
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


def _sorted_fy_statements(statements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """FY/4Q の disclosure を CurPerEn 降順 (新しい順) に返す。"""
    fy_list = [s for s in statements if s.get('CurPerType') in ('FY', '4Q')]
    fy_list.sort(key=lambda s: parse_date(s.get('CurPerEn')) or date.min, reverse=True)
    return fy_list


def cumulative_adj_factor_after(
    quotes: list[dict[str, Any]] | None,
    since_date: date | None,
) -> float:
    """
    since_date より後に発生した株式分割の累積調整係数を返す。

    J-Quants V2 の AdjustmentFactor は分割日のレコードに記録される。
    例: 1株を5株に分割した日のレコード → AdjustmentFactor = 0.2
        1株を3株に分割した日のレコード → AdjustmentFactor = 1/3 ≈ 0.333
        分割なしの日のレコード         → AdjustmentFactor = 1.0

    用途:
      - 1株あたりの値 (DPS, EPS, BPS) を分割後ベースに換算:
          new_value = old_value × cumulative_factor
      - 発行株式数 (総数) を分割後ベースに換算:
          new_shares = old_shares / cumulative_factor

    quotes が None / 空 / since_date が None なら 1.0 を返す。
    """
    if not since_date or not quotes:
        return 1.0
    factor = 1.0
    for q in quotes:
        d = parse_date(q.get('Date'))
        if d is None or d <= since_date:
            continue
        adj = safe_float(q.get('AdjustmentFactor'))
        if adj is not None and adj > 0 and abs(adj - 1.0) > 1e-6:
            factor *= adj
    return factor


def get_latest_full_year_metric(statements: list[dict[str, Any]], key: str) -> float | None:
    """
    直近 FY 系 disclosure から指定指標を取得する (分割調整なし)。

    最新 FY エントリに値が無ければ、データ欠損とみなして1つ前の FY を参照。
    分割調整が必要な場合は ..._with_date 版を使ってください。
    """
    for s in _sorted_fy_statements(statements):
        v = safe_float(s.get(key))
        if v is not None:
            return v
    return None


def get_latest_full_year_metric_with_date(
    statements: list[dict[str, Any]], key: str,
) -> tuple[float | None, date | None]:
    """
    直近 FY から (value, disclosure_date) を返す。disclosure_date は分割調整に使う。
    """
    for s in _sorted_fy_statements(statements):
        v = safe_float(s.get(key))
        if v is not None:
            d = parse_date(s.get('DiscDate')) or parse_date(s.get('CurPerEn'))
            return v, d
    return None, None



def get_forecast_dps_with_date(
    statements: list[dict[str, Any]],
) -> tuple[float | None, date | None]:
    """
    今期予想 DPS と、その値を取得した disclosure_date のタプルを返す。
    分割調整に必要なため disclosure_date を一緒に返す。
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
    # Pass 1: 種類に応じて NxFDivAnn / FDivAnn を選ぶ
    for s in sorted_stmts:
        per_type = s.get('CurPerType', '')
        if per_type in ('FY', '4Q'):
            f = safe_float(s.get('NxFDivAnn'))
        else:
            f = safe_float(s.get('FDivAnn'))
        if f is not None and f > 0:
            d = parse_date(s.get('DiscDate')) or parse_date(s.get('CurPerEn'))
            return f, d
    # Pass 2: 何でもいいので FDivAnn / NxFDivAnn から拾う
    for s in sorted_stmts:
        for key in ('FDivAnn', 'NxFDivAnn'):
            f = safe_float(s.get(key))
            if f is not None and f > 0:
                d = parse_date(s.get('DiscDate')) or parse_date(s.get('CurPerEn'))
                return f, d
    # Pass 3: 直近FYの実績 + その日付
    fb_v, fb_d = get_latest_full_year_metric_with_date(statements, 'DivAnn')
    return fb_v, fb_d


def get_forecast_dps(statements: list[dict[str, Any]]) -> float | None:
    """互換のため (value のみ) を返すラッパー。分割調整は呼び出し側で別途行う。"""
    v, _d = get_forecast_dps_with_date(statements)
    return v


def get_latest_payout_ratio(statements: list[dict[str, Any]]) -> float | None:
    """
    配当性向 (%) を直近 FY から計算する。

    最新 FY に DivAnn / EPS があればそれで計算 (透明・単位明確)。
    無ければ前の FY に遡る。それでも無ければ API の PayoutRatioAnn で
    フォールバック (decimal 検出して %% に正規化)。
    """
    for fy in _sorted_fy_statements(statements):
        div = safe_float(fy.get('DivAnn'))
        eps = safe_float(fy.get('EPS'))
        if div is not None and eps is not None and eps > 0:
            return (div / eps) * 100.0
    # フォールバック: API field
    for fy in _sorted_fy_statements(statements):
        payout_raw = safe_float(fy.get('PayoutRatioAnn'))
        if payout_raw is not None:
            return payout_raw * 100.0 if payout_raw < 3 else payout_raw
    return None


def get_equity_ratio_latest(statements: list[dict[str, Any]]) -> float | None:
    """
    最新の自己資本比率 (EqAR, decimal形式 0.5 = 50%)。

    FY/4Q だけでなく四半期 disclosure も含めて、最も新しい有効値を返す。
    """
    sorted_stmts = sorted(
        statements,
        key=lambda s: parse_date(s.get('CurPerEn')) or date.min,
        reverse=True,
    )
    for s in sorted_stmts:
        ratio = safe_float(s.get('EqAR'))
        if ratio is not None:
            return ratio
    return None


def get_per_pbr(price: float, statements: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """PER, PBR を計算。EPS/BPS は新しい FY から順に取得 (フォールバック付き)。"""
    eps = get_latest_full_year_metric(statements, 'EPS')
    bps = get_latest_full_year_metric(statements, 'BPS')
    per = (price / eps) if (eps is not None and eps > 0) else None
    pbr = (price / bps) if (bps is not None and bps > 0) else None
    return per, pbr


def get_shares_outstanding(statements: list[dict[str, Any]], np_value: float | None) -> float | None:
    """
    発行済株式数を推定する。

    優先順位:
    1. ShOutFY (フィールド明示)
    2. NumIssShFY などの代替フィールド名 (V2 で異なる場合に備える)
    3. NP / EPS から逆算 (両方とも揃っていれば)
    """
    # 候補フィールド名 (V2 仕様書で確認できる正式名は ShOutFY だが、念のため別名も試す)
    for key in ('ShOutFY', 'NumIssShFY', 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'):
        v = get_latest_full_year_metric(statements, key)
        if v is not None and v > 0:
            return v
    # 最後の手段: NP / EPS から逆算 (どちらも 直近FY から取得)
    eps = get_latest_full_year_metric(statements, 'EPS')
    if np_value is not None and np_value > 0 and eps is not None and eps > 0:
        return np_value / eps
    return None


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

    # --- 株価データを日付昇順ソート (分割調整係数の計算で使う) ---
    quotes_sorted = sorted(quotes, key=lambda q: parse_date(q.get('Date')) or date.min)

    # --- 時価総額の計算 (V2の listed_info には含まれないため、価格 × 発行株数で算出) ---
    if market_cap_oku <= 0:
        # 株式数は分割調整が必要 (1:5分割なら、過去FYの shares × 5 = 現在の shares)
        latest_np_for_shares = get_latest_full_year_metric(statements, 'NP')
        shares_raw, shares_date = (None, None)
        for s in _sorted_fy_statements(statements):
            v = safe_float(s.get('ShOutFY'))
            if v is not None and v > 0:
                shares_raw = v
                shares_date = parse_date(s.get('DiscDate')) or parse_date(s.get('CurPerEn'))
                break
        if shares_raw is None:
            # NP / EPS から逆算 (EPS は分割調整が必要)
            shares_raw = get_shares_outstanding(statements, latest_np_for_shares)
            # この場合 shares_date は EPS の disclosure 日、ただし既に NP/EPS で打ち消し合うため
            # 結果は分割調整済みと等価。以下では shares_date=None として扱う (係数=1)
            shares_date = None
        if shares_raw is not None and shares_raw > 0:
            # 分割があれば総株式数は増える (1:5分割なら × 5、つまり factor で割る)
            adj_shares = cumulative_adj_factor_after(quotes_sorted, shares_date)
            shares = shares_raw / adj_shares if adj_shares > 0 else shares_raw
            market_cap_oku = (price * shares) / 1e8
        else:
            market_cap_oku = 0.0  # 不明扱い

    # --- 指標抽出 (extract系は分割調整済) ---
    div_annual = extract_annual_dividends(statements, quotes_sorted=quotes_sorted)
    sales_history = extract_annual_metric(statements, 'Sales')   # 総額: 調整不要
    op_history = extract_annual_metric(statements, 'OP')          # 総額: 調整不要
    np_history = extract_annual_metric(statements, 'NP')          # 総額: 調整不要

    # --- DPS forecast: 分割調整 ---
    forecast_dps_raw, forecast_dps_date = get_forecast_dps_with_date(statements)
    forecast_dps_adj = cumulative_adj_factor_after(quotes_sorted, forecast_dps_date)
    forecast_dps = (forecast_dps_raw * forecast_dps_adj) if forecast_dps_raw is not None else None
    current_yield = (forecast_dps / price * 100.0) if (forecast_dps and forecast_dps > 0) else None

    # 異常検知ログ
    if current_yield is not None and current_yield > 15.0:
        log.warning(
            'ANOMALY %s %s: yield=%.2f%% (DPS_raw=%s adj=%.3f → %.2f, price=%s, disc_date=%s)',
            code, name, current_yield, forecast_dps_raw, forecast_dps_adj,
            forecast_dps or 0, price, forecast_dps_date,
        )

    # --- EPS, BPS: 分割調整 ---
    eps_raw, eps_date = get_latest_full_year_metric_with_date(statements, 'EPS')
    bps_raw, bps_date = get_latest_full_year_metric_with_date(statements, 'BPS')
    eps_adj = cumulative_adj_factor_after(quotes_sorted, eps_date)
    bps_adj = cumulative_adj_factor_after(quotes_sorted, bps_date)
    eps = (eps_raw * eps_adj) if eps_raw is not None else None
    bps = (bps_raw * bps_adj) if bps_raw is not None else None
    per = (price / eps) if (eps is not None and eps > 0) else None
    pbr = (price / bps) if (bps is not None and bps > 0) else None
    per_pbr = (per * pbr) if (per is not None and pbr is not None) else None

    # --- 配当性向: 分割調整した DPS と EPS から計算 (両方とも分割調整済み) ---
    # ただし DivAnn (実績) と forecast_dps は別の disclosure 由来の場合があるので慎重に
    # 直近FYの DivAnn / EPS で計算する (両方とも同じ disclosure からとれば調整不要だが、念のため別々に取得)
    div_actual_raw, div_actual_date = get_latest_full_year_metric_with_date(statements, 'DivAnn')
    if div_actual_raw is not None and eps_raw is not None and eps_raw > 0:
        # 同じ FY disclosure 内で計算する (分割調整は両方とも同じ係数なので相殺)
        # → 直接 raw 値で計算 OK
        # ただし div_actual_raw と eps_raw が異なる FY から来ている場合に備えて、
        # それぞれ調整してから割り算する
        div_adj = cumulative_adj_factor_after(quotes_sorted, div_actual_date)
        div_for_payout = div_actual_raw * div_adj
        eps_for_payout = eps  # already adjusted
        payout = (div_for_payout / eps_for_payout) * 100.0 if eps_for_payout > 0 else None
    else:
        payout = None

    equity_ratio = get_equity_ratio_latest(statements)

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
    # --- ヒストリカル DPS: 分割調整して連続性を保つ ---
    # 古い disclosure の DPS は分割係数で割引 (1:5分割があれば × 0.2 で現在の規模に揃える)
    div_history_by_year: dict[int, float] = {}
    for s in statements:
        if s.get('CurPerType') not in ('FY', '4Q'):
            continue
        d = parse_date(s.get('CurPerEn'))
        v = safe_float(s.get('DivAnn'))
        disc_d = parse_date(s.get('DiscDate')) or d
        if d and v is not None:
            adj = cumulative_adj_factor_after(quotes_sorted, disc_d)
            div_history_by_year[d.year] = v * adj

    yield_dist = calculate_yield_distribution(quotes, div_history_by_year)
    signal = determine_signal(current_yield, yield_dist)

    # --- ボックス ---
    box = evaluate_box(quotes)

    # --- 撤退判定 ---
    emergency, reasons = check_emergency_exit(code, div_annual, op_history)

    # --- 決算予定 ---
    next_earnings = announcement_map.get(code)
    earnings_imminent = is_imminent_business_day(next_earnings, today)

    # --- 財務推移 (1株あたり値は分割調整済) ---
    financials_history = build_financials_history(statements, n=8, quotes_sorted=quotes_sorted)

    # --- ROE 計算: 直近FYの ROE = NP / Eq * 100 ---
    latest_np = get_latest_full_year_metric(statements, 'NP')
    latest_eq = get_latest_full_year_metric(statements, 'Eq')
    roe = None
    if latest_np is not None and latest_eq is not None and latest_eq > 0:
        roe = (latest_np / latest_eq) * 100.0

    # --- 直近FY の主要指標を top-level convenience として ---
    latest_sales = get_latest_full_year_metric(statements, 'Sales')
    latest_op = get_latest_full_year_metric(statements, 'OP')

    # --- JSON 組み立て ---
    # --- 決算月 (CurPerEn の月から推定) ---
    # 直近FYのCurPerEn を見て、その月を「期末月」とする
    # 例: CurPerEn=2025-03-31 → fiscal_month=3 (3月期決算)
    fiscal_month: int | None = None
    for s in _sorted_fy_statements(statements):
        per_end = parse_date(s.get('CurPerEn'))
        if per_end is not None:
            fiscal_month = per_end.month
            break
    # 配当権利確定月: 期末配当=決算月、中間配当=決算月の6ヶ月前
    interim_month: int | None = None
    if fiscal_month is not None:
        interim_month = ((fiscal_month - 6 - 1) % 12) + 1  # -6ヶ月して1-12に正規化

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

        'fiscal_month': fiscal_month,
        'interim_month': interim_month,

        'forecast_dps': round(forecast_dps, 2) if forecast_dps is not None else None,
        'current_yield': round(current_yield, 2) if current_yield is not None else None,
        'yield_q25': round(yield_dist['q25'], 2) if not math.isnan(yield_dist['q25']) else None,
        'yield_median': round(yield_dist['median'], 2) if not math.isnan(yield_dist['median']) else None,
        'yield_q75': round(yield_dist['q75'], 2) if not math.isnan(yield_dist['q75']) else None,
        'yield_sample_n': yield_dist['n'],
        'yield_history': yield_dist.get('history', []),

        'per': round(per, 2) if per is not None else None,
        'pbr': round(pbr, 2) if pbr is not None else None,
        'per_pbr': round(per_pbr, 2) if per_pbr is not None else None,
        'payout_ratio': round(payout, 2) if payout is not None else None,
        'equity_ratio': round(equity_ratio, 4) if equity_ratio is not None else None,
        'roe': round(roe, 2) if roe is not None else None,

        'latest_sales_oku': round(latest_sales / 1e8) if latest_sales is not None else None,
        'latest_op_oku': round(latest_op / 1e8) if latest_op is not None else None,
        'latest_np_oku': round(latest_np / 1e8) if latest_np is not None else None,

        'screening': s_results,
        'screening_meta': {
            'dividend_periods_used': div_periods_used,
        },

        'box': box if box else None,

        'financials_history': financials_history,

        # 過去250日分（約1年）の株価履歴 (自前チャート用)
        # ダッシュボードで 1M/3M/6M/1Y の切替に対応
        'price_history': [
            {
                'd': q.get('Date'),
                'c': round(safe_float(q.get('Close')) or 0, 2),
                'o': round(safe_float(q.get('Open')) or 0, 2),
                'h': round(safe_float(q.get('High')) or 0, 2),
                'l': round(safe_float(q.get('Low')) or 0, 2),
                'v': int(safe_float(q.get('Volume')) or 0),
            }
            for q in quotes_sorted[-250:]
            if safe_float(q.get('Close')) and safe_float(q.get('Close')) > 0
        ],

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
    上場銘柄一覧 (V2: /v2/equities/master) を取得して
    東証プライム × TOPIX Large70 + Mid400 にフィルタ。

    V2 では listed_info に MarketCapitalization が含まれないため、
    TOPIX Scale category (ScaleCat) で大型〜中型株 (約470銘柄) を抽出する。
    これは設計の「東証プライム × 時価総額300億円以上 (約500銘柄)」と
    実質的に等価。

    フィールド名は V2 で全て短縮形:
      Code, CoName, S33 (Sector33), S33Nm, Mkt (MarketCode), ScaleCat
    """
    log.info('Fetching listed info (V2)...')
    info = client.get_listed_info()
    log.info('Got %d listed entries.', len(info))

    # 大型・中型株を表す ScaleCat の値
    SCALE_TARGETS = {
        'TOPIX Core30',
        'TOPIX Large70',
        'TOPIX Mid400',
    }

    universe = []
    skipped_market = 0
    skipped_scale = 0
    for row in info:
        market_code = row.get('Mkt', '')
        if market_code != MARKET_CODE_PRIME:
            skipped_market += 1
            continue

        scale_cat = row.get('ScaleCat', '') or ''
        if scale_cat not in SCALE_TARGETS:
            skipped_scale += 1
            continue

        code = normalize_code(row.get('Code', ''))
        if not code:
            continue

        sector33 = row.get('S33', '')
        sector33_name = row.get('S33Nm', '')
        name = row.get('CoName', '') or row.get('CoNameEn', '')

        universe.append({
            'code': code,
            'name': name,
            'sector33': sector33,
            'sector33_name': sector33_name,
            'market_cap_oku': None,  # V2では取得不可、process_stock内で計算
            'scale_cat': scale_cat,
        })

    log.info('Universe filter: prime_skip=%d, scale_skip=%d, kept=%d',
             skipped_market, skipped_scale, len(universe))
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
        # V2 では `ScheduledDate` がメイン、フォールバックで `Date` も確認
        d = (parse_date(row.get('ScheduledDate'))
             or parse_date(row.get('Date'))
             or parse_date(row.get('PublicationDate')))
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

    # --- Tier 判定 (全銘柄プロセス後に一括計算) ---
    log.info('Computing industry leaders...')
    industry_leaders = compute_industry_leaders(stocks)
    log.info('Industry leaders: %d stocks', len(industry_leaders))

    tier_counts: dict[str, int] = defaultdict(int)
    for s in stocks:
        code = s['code']
        is_qual = is_progressive_or_doe(code)
        is_leader = code in industry_leaders
        tier, weight = classify_tier(code, industry_leaders)
        s['tier'] = tier
        s['tier_weight'] = weight
        s['is_industry_leader'] = is_leader
        s['is_progressive_or_doe'] = is_qual
        # screening_pass の中だけで Tier 集計 (実際にBUY候補となるもの)
        if s.get('screening_pass'):
            tier_counts[tier] += 1

    log.info('=== Tier distribution (screening pass only) ===')
    for t in ['S', 'A', 'B']:
        log.info('  Tier %s: %d', t, tier_counts[t])

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

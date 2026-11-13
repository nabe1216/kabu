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
GRAHAM_THRESHOLD = 40.0              # PER × PBR ≤ 40 (B3: 異常値だけ弾く緩い床。業種バイアス緩和)
MIN_YIELD_THRESHOLD = 3.0            # 最低利回り ≧ 3.0%

# 買いの判定。
#   yield    … スクリーニングを通り、利回りが足切り以上なら BUY（待たない）
#   quantile … 利回りが過去3年の Q75 以上で BUY、Q25 以下で SELL（これまで）
# 64条件の総当たりで yield が86%の条件で上回ったため、yield を採用。
# 2026年9月に pbr（低PBR順）へ変えたが、11月13日に yield（利回りの高い順）へ戻した。
#   20年分の検証（long_lab・long_lab2）：ルールを作るときに見ていない期間（2009〜2018年）で利回り順が全条件で上回り（+4.5pt）、
#   17年通しでも 13.6％ 対 13.3％・最大下落 24.8％ 対 27.8％。低PBR順が上だったのは、それを選んだ最近の期間だけ。
#   差は偶然と区別できない大きさ。低PBR順は variant_engine の並走記録（PBR）で12月から比べ続ける。
BUY_MODE = 'yield'
# 低PBR順のとき、全銘柄の中で PBR が安い側何％までを買うか。
# 80 なら「安い側20％」。検証で 10/20/30％ いずれも崩れなかったので、
# 最初に決めた 20％ を使う。
PBR_TOP_PCT = 80

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
YIELD_LOOKBACK_YEARS = 9             # 過去5年の月次利回りで分布計算
YIELD_LOOKBACK_DAYS = YIELD_LOOKBACK_YEARS * 365 + 60  # 余裕を持って取得
# 分割・併合を読むためだけに、株価はもっと前までさかのぼって取る（2026年11月6日）。
# 10期の配当を「期末の7日前」の基準で直すには、いちばん古い期の期末の7日前からの分割・併合が要る。
# J-Quants のスタンダードは10年分。取れなければ（3銘柄続けて失敗したら）9年分に戻し、
# 分割・併合が読めない古い期は減配の比べから外す（extract_annual_dividends の known_since）。
# 利回りの分布・ボックス・株価の表示は、これまでどおり YIELD_LOOKBACK_DAYS の範囲だけを使う。
SPLIT_LOOKBACK_YEARS = 10
# 分位（Q25/Q75）の計算に使う期間。取得期間より短くできる。
YIELD_DISTRIBUTION_YEARS = 3         # 出口が来ないときは 2 に下げる

# --- 撤退判定（緊急シグナル） ---
EMERGENCY_DPS_DROP = -0.10           # DPS 10%超減配
EMERGENCY_OP_DROP = -0.20            # 営業利益YoY -20%以下
# 減配・業績急変の印が立っても、条件に当たれば売らずに持ち続ける（2026年11月14日。ワタナベさんの方針「多少の差なら塩漬け」）
#   いまの形：K1（浅い減配（−30％未満）は持つ）。検証はプロジェクトの hold2_plan_2026-11-14 / hold2_result_2026-11-14
#   cut_min       … 減配で売るのは、年間配当がこの率以上減ったときだけ（それより浅い減配は持つ）
#   op_hold_yield … 業績急変だけ（減配の印なし）で売るときは、利回りがこの値（％）以上なら持つ
#   hold_yield    … 利回りがこの値以上なら持つ ／ hold_tiers … この Tier なら持つ
#   always_red    … 営業利益が赤字に転落したら必ず売る ／ always_nodiv … 無配になったら必ず売る
#   持っているあいだも、印のある銘柄は新しく買わない・買い増さない。条件から外れた日に売る
EMERGENCY_HOLD: dict = {'cut_min': 0.30}

# --- ユニバース ---
MARKET_CODE_PRIME = '0111'           # 東証プライム
MARKET_CAP_MIN_OKU = 300             # 時価総額300億円以上
# 対象の範囲。
#   core  … TOPIX Core30 / Large70 / Mid400（約489社）
#   prime … 東証プライムの全銘柄（約1,557社）
#
# 検証では prime のほうが基準を上回る幅が大きかった
# （+8.9pt -> +11.2pt）が、取得に70〜90分かかる。
UNIVERSE_SCOPE = 'small1'

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

# 2026年11月13日に調べ直した（それまでのリストは2025年ごろの手入力で、確認日の記録がなかった）。
#   「配当を守る会社」＝ ①日経累進高配当株指数の構成銘柄 ＋ ②会社の公式資料（決算短信・有価証券報告書・中期経営計画・
#   配当のお知らせ）に「減配しない」（累進配当・維持または増配・減額は原則行わない）と書いている会社
#   - DOE だけの会社、「増配を目指す」「増配に努める」だけの会社、下限だけを書いている会社は入れない
#   - 見直しは年1回。日経累進高配当株指数の定期入れ替え（毎年6月末）のあと、7月に行う（DIVIDEND_POLICY_CHECKED を更新）
#   - ②は大型株を中心に確かめた分だけ。中小型で累進配当を書いている会社は、まだ入っていない（EDINET の有価証券報告書を
#     全文で調べれば拾える。API キーが要る）
DIVIDEND_POLICY_CHECKED = '2026-11-13'

# ① 日経累進高配当株指数（愛称：しっかりインカム）30銘柄。2026年6月30日の定期入れ替えのあと
#   出典：日経「日経累進高配当株指数の定期入れ替え」（2026年6月12日）。入れ替えで12銘柄が入り、12銘柄が出た
#   （出た12：飯田G・宮地エンジ・三菱ガス化学・三井化学・日本ゼオン・日本化薬・アステラス・小野薬品・ENEOS・AGC・
#    三井住友トラスト・SBI。このうち小野薬品・三井住友トラストは②で残る）
NIKKEI_PROGRESSIVE_30: set[str] = {
    '4502',  # 武田薬品工業
    '8593',  # 三菱HCキャピタル
    '4521',  # 科研製薬
    '5938',  # LIXIL
    '8439',  # 東京センチュリー
    '7956',  # ピジョン
    '9364',  # 上組
    '3861',  # 王子ホールディングス
    '4042',  # 東ソー
    '4208',  # UBE
    '8725',  # MS&AD
    '7313',  # テイ・エス テック
    '8252',  # 丸井グループ
    '1719',  # 安藤ハザマ
    '1928',  # 積水ハウス
    '4041',  # 日本曹達
    '1870',  # 矢作建設工業
    '8130',  # サンゲツ
    '2127',  # 日本M&Aセンターホールディングス（2026年6月に採用）
    '2379',  # ディップ（同）
    '3231',  # 野村不動産ホールディングス（同）
    '4540',  # ツムラ（同）
    '4613',  # 関西ペイント（同）
    '4633',  # サカタインクス（同）
    '5021',  # コスモエネルギーホールディングス（同）
    '5602',  # 栗本鐵工所（同）
    '7164',  # 全国保証（同）
    '7458',  # 第一興商（同）
    '8750',  # 第一ライフグループ（同）
    '8803',  # 平和不動産（同）
}

# ② 公式資料で「減配しない」と書いている会社（2026年11月13日に確かめた。かっこは条件・期限・言い方）
#   出典：各社の決算短信・有価証券報告書・中期経営計画・配当のお知らせ（2026年の最新）。第三者の整理
#   （たぐ「累進配当 銘柄 一覧」「累進配当はいつまで？」2026年10月・エルマネ「累進配当を掲げる企業一覧」2026年9月）で当たりをつけ、
#   会社の言葉を確かめた
#   外したもの（それまで入っていた）：トヨタ・ソフトバンク・ブリヂストン（「増配に努める／目指す」だけ）、
#   KDDI（「安定増配」「配当性向40％超」）、東京海上（「利益成長に応じて持続的に高める」）、SOMPO（「利益成長にあわせた増配を原則」）
#   入れなかったもの：キヤノン（「2026年12月期は下げない」の今期だけ）、住友不動産（配当性向35％到達までの条件つき）
PROGRESSIVE_DECLARED: set[str] = {
    '8058',  # 三菱商事
    '8001',  # 伊藤忠商事
    '8031',  # 三井物産（2029年3月期まで）
    '8002',  # 丸紅（2028年3月期まで）
    '8053',  # 住友商事
    '8015',  # 豊田通商（2028年3月期まで）
    '2768',  # 双日（株主資本が増えている限り・2027年3月期まで）
    '8306',  # 三菱UFJフィナンシャル・グループ（CET1比率9.5％以上の条件つき）
    '8316',  # 三井住友フィナンシャルグループ
    '8411',  # みずほフィナンシャルグループ
    '8309',  # 三井住友トラストグループ（CET1比率9.5％以上の条件つき）
    '7186',  # 横浜フィナンシャルグループ（1年ごとに決め直す）
    '8334',  # 群馬銀行
    '7182',  # ゆうちょ銀行（中期経営計画 2029年3月期まで）
    '8729',  # ソニーフィナンシャルグループ（原則として）
    '7181',  # かんぽ生命保険（原則として・2029年3月期まで）
    '4528',  # 小野薬品工業
    '4568',  # 第一三共（中期経営計画 2031年3月期まで）
    '4151',  # 協和キリン
    '2502',  # アサヒグループホールディングス
    '2503',  # キリンホールディングス（原則として）
    '2802',  # 味の素
    '2871',  # ニチレイ
    '1605',  # INPEX（2027年12月期まで）
    '7011',  # 三菱重工業
    '8801',  # 三井不動産
    '8802',  # 三菱地所
    '3289',  # 東急不動産ホールディングス
    '1808',  # 長谷工コーポレーション
    '3099',  # 三越伊勢丹ホールディングス
    '3402',  # 東レ
    '3407',  # 旭化成（目指す・2028年3月期まで）
    '4204',  # 積水化学工業（原則として・2029年3月期まで）
    '5019',  # 出光興産
    '5301',  # 東海カーボン
    '5706',  # 三井金属
    '7004',  # カナデビア
    '7269',  # スズキ
    '7270',  # SUBARU（目指す）
    '7741',  # HOYA
    '7912',  # 大日本印刷
    '9007',  # 小田急電鉄
    '9531',  # 東京瓦斯
    '9503',  # 関西電力（努める・2027年3月期から）
    '9104',  # 商船三井（2031年3月期まで）
    '3382',  # セブン&アイ・ホールディングス
}

# ③ 有価証券報告書の【配当政策】に「減配しない」と書いている会社（2026年11月13日に追加）
#   scripts/edinet_policy.py が EDINET（金融庁）から毎月読み取り、data/dividend_policy.json に出す。中小型も拾える
#   - 「今期だけ」の約束（例：キヤノンの「2026年12月期は下げない」）は入れない
#   - ③は Tier（配当を守る会社）にだけ使い、減配の条件の免除には使わない（免除をどうするかは未決のため。
#     ①②の免除は、これまでどおり）
def _load_edinet_policy() -> tuple[set[str], dict]:
    try:
        js = json.loads((Path(__file__).parent.parent / 'data' / 'dividend_policy.json').read_text(encoding='utf-8'))
    except Exception:
        return set(), {}
    codes = {c for c, v in (js.get('companies') or {}).items()
             if v.get('progressive') and '今期だけ' not in (v.get('flags') or [])}
    return codes, js


EDINET_DECLARED, EDINET_POLICY = _load_edinet_policy()

PROGRESSIVE_DIVIDEND_LIST: set[str] = NIKKEI_PROGRESSIVE_30 | PROGRESSIVE_DECLARED | EDINET_DECLARED

# DOE採用銘柄（2026年11月13日に空にした）
# DOE は「純資産に対して何％を配当に回すか」の目安で、「減配しない」約束ではない。DOE を掲げる会社は600社ほどあり、
# それまでの4社（アサヒ・三菱重工・ジャフコ・アルプスアルパイン）は手で選んだ一部だった。アサヒ・三菱重工は②に入っている
DOE_LIST: set[str] = set()

# 「減配ゼロ判定 + 最低利回り」免除リスト
DIVIDEND_CHECK_EXEMPT: set[str] = NIKKEI_PROGRESSIVE_30 | PROGRESSIVE_DECLARED | DOE_LIST   # ③は入れない

# 自己資本比率チェック免除セクター（金融4業種+不動産業）
EQUITY_RATIO_EXEMPT_SECTORS: set[str] = {
    '7050',  # 銀行業
    '7100',  # 証券、商品先物取引業
    '7150',  # 保険業
    '7200',  # その他金融業
    '8050',  # 不動産業
}

# 業種別の自己資本比率最低基準（緩和テーブル）
# 業界特性で自己資本比率が低めになりがちな業種は閾値を緩める。
# 33% を基準とすることで、累進配当銘柄の積水ハウス、大和ハウス、5大商社、商船三井 等を拾える。
# それ以外の業種は EQUITY_RATIO_MIN (50%) を適用。
EQUITY_RATIO_RELAXED_BY_SECTOR: dict[str, float] = {
    '2050': 0.33,  # 建設業 (積水ハウス42.7%、大和ハウス34.4%、大林組40.0%、鹿島37.2%など)
    '5100': 0.33,  # 海運業 (商船三井48.2%、船舶投資が重い業界特性)
    '6050': 0.33,  # 卸売業 (5大商社、在庫保有で自己資本比率低めの傾向)
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
        if STMTS_CACHE_DAYS > 0:
            c = _load_stmts_cache()
            hit = c['data'].get(code)
            if hit is not None:
                return hit
            got = self.get('/v2/fins/summary', params)
            c['data'][code] = got
            if len(c['data']) % 100 == 0:
                _save_stmts_cache(force=False)
            return got
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

# 財務データを何日もたせるか。0 ならキャッシュを使わない。
# 株価と財務で1銘柄あたり2回APIを呼ぶため、財務を毎日取らなければ
# 実行時間がほぼ半分になる。決算の反映はその日数ぶん遅れる。
STMTS_CACHE_DAYS = 7
# 2026年11月8日：92MB まで大きくなり GitHub の上限（100MB）に近づいたので、圧縮して保存する（約6MB）。
# 古い圧縮なしのファイルがあれば読み、圧縮して保存したあとに消す（daily.yml がその削除も保存する）。
_STMTS_CACHE_PATH = OUTPUT_DIR / 'stmts_cache.json.gz'
_STMTS_CACHE_LEGACY = OUTPUT_DIR / 'stmts_cache.json'
_stmts_cache = None


def _load_stmts_cache():
    """財務データのキャッシュを読む。壊れていれば空で始める。"""
    global _stmts_cache
    if _stmts_cache is not None:
        return _stmts_cache
    _stmts_cache = {'saved_at': '', 'data': {}}
    src = _STMTS_CACHE_PATH if _STMTS_CACHE_PATH.exists() else _STMTS_CACHE_LEGACY
    if STMTS_CACHE_DAYS > 0 and src.exists():
        try:
            import gzip
            opener = gzip.open if src.suffix == '.gz' else open
            with opener(src, 'rt', encoding='utf-8') as f:
                c = json.load(f)
            saved = parse_date(c.get('saved_at'))
            if saved is not None:
                age = (date.today() - saved).days
                if age < STMTS_CACHE_DAYS:
                    _stmts_cache = c
                    log.info('財務データのキャッシュを使います（%d日前・%d銘柄）',
                             age, len(c.get('data', {})))
                else:
                    log.info('財務データのキャッシュが%d日前なので取り直します', age)
            else:
                _stmts_cache = c
                log.info('作りかけのキャッシュを引き継ぎます（%d銘柄）',
                         len(c.get('data', {})))
        except Exception as e:
            log.warning('財務データのキャッシュを読めません: %s', e)
    return _stmts_cache


def _save_stmts_cache(force=True):
    """取得した財務データを保存する。"""
    if STMTS_CACHE_DAYS <= 0 or _stmts_cache is None:
        return
    if not _stmts_cache.get('data'):
        return
    try:
        _stmts_cache['saved_at'] = date.today().isoformat() if force else ''
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        import gzip
        tmp = _STMTS_CACHE_PATH.with_suffix('.tmp')
        with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=6) as f:
            json.dump(_stmts_cache, f, ensure_ascii=False, separators=(',', ':'))
        tmp.replace(_STMTS_CACHE_PATH)
        if _STMTS_CACHE_LEGACY.exists():
            _STMTS_CACHE_LEGACY.unlink()
        log.info('財務データのキャッシュを保存しました（%d銘柄・%.1fMB）',
                 len(_stmts_cache['data']), _STMTS_CACHE_PATH.stat().st_size / 1e6)
    except Exception as e:
        log.warning('財務データのキャッシュを保存できません: %s', e)


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
    known_since: date | None = None,
) -> list[float]:
    """
    財務一覧から年度単位の DPS（DivAnn 実績）を時系列で抽出。

    quotes_sorted を渡すと、分割調整して連続性を保つ。
    例: 1:5分割があった場合、過去の DivAnn (分割前=高い値) を 0.2倍して現在ベースに揃える。
    これにより減配判定の誤動作 (分割を減配と誤認) を防ぐ。

    決算期種別 'FY' / '4Q' の DivAnn を採用し、古い順に並べる。
    """
    # 2026年11月6日：分割・併合の基準は「期末の7日前」で直す（actual_dps_adjusted。V3）。
    # 同じ期の開示が2つ以上あるとき（訂正など）は、あとで開示した方を使う（同じ期を2回数えない）。
    # known_since：株価を取り始めた日（取った範囲の都合で、それより前の分割・併合が見えないとき）。
    # 期末の7日前がそれより前の期は、いまの基準に直せるかわからないので使わない
    # （株価の範囲の外の分割で、古い期の配当が分割前の額のまま残り「減配」に見えるのを防ぐ。例：サンドラッグ 2017年3月期）。
    events = split_events(quotes_sorted)
    by_period: dict[date, tuple[date, float, float | None]] = {}
    for s in statements:
        if s.get('CurPerType') not in ('FY', '4Q'):
            continue
        period = parse_date(s.get('CurPerEn'))
        if period is None:
            continue
        if known_since is not None and period - timedelta(days=DIV_REC_DAYS) < known_since:
            continue
        v = actual_dps_adjusted(s, events)
        if v is None:
            continue
        disc = parse_date(s.get('DiscDate')) or period
        # 2026年11月10日：期末の7日前〜開示の日に分割・併合があった期は、「すでに分割後の基準で書いた」読み方も持っておく
        # （本の検証の道具で見つけた。期末を基準日にした分割で、実績の配当を分割後に直して載せる会社が約2％ある）
        alt = None
        fe = _fy_end_of(s)
        raw = safe_float(s.get('DivAnn'))
        if fe is not None and raw is not None and any(fe - timedelta(days=DIV_REC_DAYS) <= d <= disc for d, _f in events):
            alt = raw * _factor_after(events, disc)
        if period not in by_period or disc >= by_period[period][0]:
            by_period[period] = (disc, v, alt)
    out: list[float] = []
    for p in sorted(by_period):
        _d, v, alt = by_period[p]
        if alt is not None and out and out[-1] > 0 and v > 0 and alt > 0 \
                and abs(math.log(alt / out[-1])) < abs(math.log(v / out[-1])):
            v = alt
        out.append(v)
    return out


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
    """自己資本比率チェック

    - 金融4業種＋不動産業: 完全免除
    - 建設業・海運業・卸売業: 35%以上で許容（業界特性で低めになるため）
    - その他: 50%以上が必要（デフォルト）
    """
    if sector_code in EQUITY_RATIO_EXEMPT_SECTORS:
        return True  # 免除
    if equity_ratio is None:
        return None
    # 業種別の緩和閾値があればそれを使う、なければデフォルト (50%)
    threshold = EQUITY_RATIO_RELAXED_BY_SECTOR.get(sector_code, EQUITY_RATIO_MIN)
    return equity_ratio >= threshold


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
    if yield_pct <= 0:
        return False   # 無配 (利回り0以下) は高配当戦略の対象外
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

    # 分位の計算は直近 YIELD_DISTRIBUTION_YEARS 年ぶんだけを使う。
    # 期間が長いと分布が動かず、買ったあと売りシグナルが出にくくなる。
    # 表示用の history は全期間のまま残すので、チャートは変わらない。
    dist_months = YIELD_DISTRIBUTION_YEARS * 12
    yields_for_dist = yields[-dist_months:] if len(yields) > dist_months else yields
    yields_sorted = sorted(yields_for_dist)
    return {
        'q25': quantile(yields_sorted, 0.25),
        'median': quantile(yields_sorted, 0.50),
        'q75': quantile(yields_sorted, 0.75),
        'n': len(yields_sorted),
        'history': history,
    }


def calculate_valuation_history(
    quotes_sorted: list[dict[str, Any]],
    statements: list[dict[str, Any]],
    yield_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    各月末時点での PER / PBR / グレアム指数を計算する。

    yield_history のデータ点と同期。各月末で「その時点で公表済みの最新EPS/BPS」を引いて、
    分割調整を行った上で PER = price/EPS, PBR = price/BPS を算出する。
    """
    if not yield_history or not statements:
        return []

    # 各 statement から (disc_date, eps_raw, bps_raw) のタイムラインを構築
    fin_timeline: list[dict[str, Any]] = []
    for s in statements:
        d = parse_date(s.get('DiscDate')) or parse_date(s.get('CurPerEn'))
        if d is None:
            continue
        eps_raw = safe_float(s.get('EPS'))
        np_val = safe_float(s.get('NP'))
        eq_val = safe_float(s.get('Eq'))
        bps_raw = None
        if eq_val is not None and np_val is not None and eps_raw is not None and eps_raw != 0:
            shares_raw = np_val / eps_raw
            if shares_raw and shares_raw > 0:
                bps_raw = eq_val / shares_raw
        if eps_raw is None and bps_raw is None:
            continue
        fin_timeline.append({
            'date': d,
            'eps_raw': eps_raw,
            'bps_raw': bps_raw,
        })
    fin_timeline.sort(key=lambda x: x['date'])

    if not fin_timeline:
        return []

    # 各月末時点で直近の財務データを引いて PER/PBR を計算
    result: list[dict[str, Any]] = []
    for h in yield_history:
        try:
            d = date.fromisoformat(h['date'])
        except Exception:
            continue
        price = h.get('price')
        if not price or price <= 0:
            continue

        # その日以前で最新の財務データを引く
        latest = None
        for ft in fin_timeline:
            if ft['date'] <= d:
                latest = ft
            else:
                break
        if latest is None:
            continue

        # 分割調整: その時点から最新までの累積分割係数で調整
        adj = cumulative_adj_factor_after(quotes_sorted, latest['date'])
        eps_adj = latest['eps_raw'] * adj if latest['eps_raw'] is not None else None
        bps_adj = latest['bps_raw'] * adj if latest['bps_raw'] is not None else None

        per = price / eps_adj if eps_adj and eps_adj > 0 else None
        pbr = price / bps_adj if bps_adj and bps_adj > 0 else None
        gpi = per * pbr if (per is not None and pbr is not None) else None

        result.append({
            'date': h['date'],
            'per': round(per, 2) if per is not None else None,
            'pbr': round(pbr, 2) if pbr is not None else None,
            'graham': round(gpi, 2) if gpi is not None else None,
        })

    return result


def compute_valuation_cheap_flags(
    valuation_history: list[dict],
    current_per: float | None,
    current_pbr: float | None,
) -> dict:
    """過去5年月次 PER/PBR の中央値(Q50)を計算し、現在値が中央値以下
    (= 過去水準比で割安) かのフラグを返す。B3構成の割安ラベル用。
    PER は分母(EPS)が荒れるため有効点12未満は判定不能(None)。"""
    per_series = sorted(
        h['per'] for h in valuation_history
        if h.get('per') is not None and h['per'] > 0
    )
    pbr_series = sorted(
        h['pbr'] for h in valuation_history
        if h.get('pbr') is not None and h['pbr'] > 0
    )
    per_median = quantile(per_series, 0.50) if len(per_series) >= 12 else None
    pbr_median = quantile(pbr_series, 0.50) if len(pbr_series) >= 12 else None
    per_cheap = (
        current_per is not None and per_median is not None
        and current_per <= per_median
    )
    pbr_cheap = (
        current_pbr is not None and pbr_median is not None
        and current_pbr <= pbr_median
    )
    return {
        'per_median': round(per_median, 2) if per_median is not None else None,
        'pbr_median': round(pbr_median, 2) if pbr_median is not None else None,
        'per_cheap': per_cheap,
        'pbr_cheap': pbr_cheap,
    }


def determine_signal(current_yield: float | None, dist: dict[str, float]) -> str:
    """BUY / SELL / NEUTRAL を判定。"""
    if current_yield is None or current_yield <= 0:
        return 'NEUTRAL'
    # 新しいルール：その銘柄の過去と比べて安くなるのを待たず、
    # 利回りが足切り以上なら買う候補にする（スクリーニングは別で判定）。
    if BUY_MODE in ('yield', 'pbr'):
        return 'BUY' if current_yield >= MIN_YIELD_THRESHOLD else 'NEUTRAL'
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


def emergency_hold_decision(s: dict[str, Any], rule: dict | None = None) -> tuple[bool, str]:
    """減配・業績急変の印が立った銘柄を、売らずに持ち続けるか（EMERGENCY_HOLD）。(持つか, 理由)"""
    rule = EMERGENCY_HOLD if rule is None else rule
    if not s.get('emergency_exit') or not rule:
        return False, ''
    dc, oc = s.get('emg_div_chg'), s.get('emg_op_chg')
    cut = dc is not None and dc <= EMERGENCY_DPS_DROP
    opd = oc is not None and oc <= EMERGENCY_OP_DROP
    if not (cut or opd):
        return False, ''          # 減配・業績急変以外（累進配当の撤回など）は、いままでどおり売る
    yld = s.get('current_yield')
    c_ok = cut and ('cut_min' not in rule or dc <= -float(rule['cut_min']))
    o_ok = opd and ('op_min' not in rule or oc <= -float(rule['op_min']))
    sell = c_ok or o_ok
    why = ''
    if not sell:
        why = f'浅い減配（{dc * 100:.0f}%）' if cut else ''
    if sell and rule.get('op_hold_yield') is not None and not c_ok and o_ok \
            and yld is not None and yld >= float(rule['op_hold_yield']):
        sell = False
        why = (f'浅い減配（{dc * 100:.0f}%）・' if cut else '') + \
            f'業績の悪化（営業利益{oc * 100:.0f}%）だけで利回り{yld:.1f}%'
    if sell and rule.get('hold_yield') is not None and yld is not None and yld >= float(rule['hold_yield']):
        sell, why = False, f'利回り{yld:.1f}%'
    if sell and rule.get('hold_tiers') and s.get('tier') in set(rule['hold_tiers']):
        sell, why = False, f'Tier {s.get("tier")}'
    if rule.get('always_red') and opd and oc <= -1.0:
        sell = True
    if rule.get('always_nodiv') and cut and dc <= -0.995:
        sell = True
    return (False, '') if sell else (True, why or '持ち続ける条件に当たる')


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
        if dps_raw is not None and s.get('CurPerType') in ('FY', '4Q') and quotes_sorted:
            dps_val = actual_dps_adjusted(s, split_events(quotes_sorted))   # 2026年11月6日：V3
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


SAFETY_FIN_S33 = {'7050', '7100', '7150', '7200'}   # 銀行・証券・保険・その他金融（営業利益の意味が違うので安全域を出さない）


def safety_margin(statements: list[dict[str, Any]], market_cap_oku: float, sector33: str) -> float | None:
    """2026年11月10日：山口揚平の安全域（保守的。34段目の Y1）＝（営業利益の前々期・前期・今期予想の平均×10 ＋ 現預金 − 負債の合計）÷ 時価総額。
    負債の合計＝総資産 − 純資産（いちばん新しい開示）。並走記録（variant_engine）の買う順番に使う。本番の売買には使わない。"""
    if sector33 in SAFETY_FIN_S33 or not market_cap_oku or market_cap_oku <= 0:
        return None
    fy = _sorted_fy_statements(statements)
    ops = [safe_float(x.get('OP')) for x in fy[:2]]
    op_f = None
    fy_end = parse_date(fy[0].get('CurFYEn') or fy[0].get('CurPerEn')) if fy else None
    for x in sorted(statements, key=lambda q: q.get('DiscDate') or '', reverse=True):
        fe = parse_date(x.get('CurFYEn') or x.get('CurPerEn'))
        if x.get('CurPerType') in ('FY', '4Q'):
            if fe == fy_end:
                op_f = safe_float(x.get('NxFOP'))
            break
        if fy_end is not None and fe is not None and fe > fy_end:
            op_f = safe_float(x.get('FOP'))
            if op_f is not None:
                break
    vals = [v for v in ops + [op_f] if v is not None]
    if len(vals) < 2 or sum(vals) <= 0:
        return None
    last = next((x for x in sorted(statements, key=lambda q: q.get('DiscDate') or '', reverse=True)
                 if safe_float(x.get('TA')) and safe_float(x.get('Eq')) is not None and safe_float(x.get('CashEq')) is not None), None)
    if last is None:
        return None
    net_cash = safe_float(last['CashEq']) - (safe_float(last['TA']) - safe_float(last['Eq']))
    return (10.0 * sum(vals) / len(vals) + net_cash) / (market_cap_oku * 1e8)


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


# ── 分割・併合の基準のずれを直す（2026年11月6日。20年分の検証 long_lab16 で採った「直した計算 V3」と同じ考え方）──
# 1) 実績の年間配当は、開示日ではなく「期末の7日前」より後の分割・併合で、いまの株数の基準に直す。
#    期末が基準日の株式分割では、決算短信の年間配当は分割前の株数で払った額のまま載るので、
#    開示日より後の分割だけで直すと、翌期と比べて「減配」に見えていた（KDDI・野村不動産HD・第一ライフなど）。
# 2) 中間の基準日（期末の6か月前）の7日前から、期末の7日前までに分割・併合した期は、
#    年間配当に分割前と分割後の基準が混ざるので、減配の比べには使わない。
# 3) 予想配当は、開示から DIV_EV_DAYS 日以内に DIV_EV_MIN 倍以上（または 1/DIV_EV_MIN 以下）の分割・併合があれば、
#    「分割前の基準で出した」「すでに分割後の基準で出した」の2通りに読み、直前の開示の予想に近い方を採る。
# 20年分の検証では、この直し方が配当金情報（実際に決まった配当）と2014〜2025年の12年すべてで±5％以内に合った。
DIV_REC_DAYS = 7
DIV_EV_DAYS = 270
DIV_EV_MIN = 1.25

# 長い期間の株価の取得の状態（3銘柄続けて失敗したら、その回は9年分に戻す）
_LONG_QUOTES: dict[str, Any] = {'on': True, 'fail': 0, 'used': 0}


def split_lookback_from(today: date) -> date:
    """分割・併合を読むための株価の取り始め（SPLIT_LOOKBACK_YEARS 年前の14日あと。プランの範囲の端を避ける）。"""
    import calendar
    y = today.year - SPLIT_LOOKBACK_YEARS
    d = date(y, today.month, min(today.day, calendar.monthrange(y, today.month)[1]))
    return d + timedelta(days=14)


def split_events(quotes: list[dict[str, Any]] | None) -> list[tuple[date, float]]:
    """株価の記録から、分割・併合の日と係数（1でない日だけ）を日付順に返す。"""
    out: list[tuple[date, float]] = []
    for q in quotes or []:
        d = parse_date(q.get('Date'))
        adj = safe_float(q.get('AdjustmentFactor'))
        if d is not None and adj is not None and adj > 0 and abs(adj - 1.0) > 1e-6:
            out.append((d, adj))
    out.sort(key=lambda x: x[0])
    return out


def _factor_on_or_after(events: list[tuple[date, float]], since: date) -> float:
    f = 1.0
    for d, a in events:
        if d >= since:
            f *= a
    return f


def _factor_after(events: list[tuple[date, float]], since: date) -> float:
    f = 1.0
    for d, a in events:
        if d > since:
            f *= a
    return f


def _fy_end_of(s: dict[str, Any]) -> date | None:
    return parse_date(s.get('CurFYEn')) or parse_date(s.get('CurPerEn'))


def _months_before(d: date, months: int) -> date:
    y, m = d.year, d.month - months
    while m <= 0:
        m += 12
        y -= 1
    import calendar
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def actual_dps_adjusted(s: dict[str, Any], events: list[tuple[date, float]]) -> float | None:
    """実績の年間配当（DivAnn）を、いまの株数の基準に直す（V3）。期の途中で分割・併合した期は None。"""
    v = safe_float(s.get('DivAnn'))
    fe = _fy_end_of(s)
    if v is None or fe is None:
        return None
    if events:
        a_ = _months_before(fe, 6) - timedelta(days=DIV_REC_DAYS)
        b_ = fe - timedelta(days=DIV_REC_DAYS)
        if any(a_ <= d < b_ for d, _f in events):
            return None
    return v * _factor_on_or_after(events, fe - timedelta(days=DIV_REC_DAYS))


def forecast_dps_timeline(
    statements: list[dict[str, Any]], events: list[tuple[date, float]],
) -> dict[int, float]:
    """開示ごとの予想配当（なければ実績）を、いまの株数の基準に直した値（V3）。キーは statements の位置。"""
    order = sorted(
        range(len(statements)),
        key=lambda i: ((parse_date(statements[i].get('DiscDate')) or parse_date(statements[i].get('CurPerEn')) or date.min), i),
    )
    out: dict[int, float] = {}
    prev = None
    for i in order:
        s = statements[i]
        d = parse_date(s.get('DiscDate')) or parse_date(s.get('CurPerEn'))
        if d is None:
            continue
        per = s.get('CurPerType', '')
        v = safe_float(s.get('NxFDivAnn')) if per in ('FY', '4Q') else safe_float(s.get('FDivAnn'))
        if v is None or v <= 0:
            for key in ('FDivAnn', 'NxFDivAnn'):
                v = safe_float(s.get(key))
                if v is not None and v > 0:
                    break
        if v is None or v <= 0:
            continue
        val = v * _factor_after(events, d)
        nxt = next(((ed, ef) for ed, ef in events if ed > d), None)
        if (prev is not None and prev > 0 and nxt is not None and (nxt[0] - d).days <= DIV_EV_DAYS
                and (nxt[1] >= DIV_EV_MIN or nxt[1] <= 1.0 / DIV_EV_MIN)):
            alt = val / nxt[1]          # すでに分割・併合のあとの基準で出していた、と読んだ値
            if abs(math.log(alt / prev)) < abs(math.log(val / prev)):
                val = alt
        out[i] = val
        prev = val
    return out


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


def forecast_dps_v3(statements: list[dict[str, Any]], events: list[tuple[date, float]]) -> float | None:
    """いまの予想配当を、分割・併合の基準の取り違えを直して返す（2026年11月6日。V3）。
    どの開示の値を使うかは get_forecast_dps_with_date と同じ（新しい期の予想 → なければ何でも → なければ直近の実績）。"""
    tl = forecast_dps_timeline(statements, events)
    order = sorted(
        range(len(statements)),
        key=lambda i: (parse_date(statements[i].get('CurPerEn')) or parse_date(statements[i].get('DiscDate')) or date.min),
        reverse=True,
    )
    for i in order:                      # Pass 1：種類に応じて NxFDivAnn / FDivAnn
        s = statements[i]
        f = safe_float(s.get('NxFDivAnn')) if s.get('CurPerType', '') in ('FY', '4Q') else safe_float(s.get('FDivAnn'))
        if f is not None and f > 0:
            return tl.get(i)
    for i in order:                      # Pass 2：FDivAnn / NxFDivAnn のどちらか
        s = statements[i]
        if any((safe_float(s.get(k)) or 0) > 0 for k in ('FDivAnn', 'NxFDivAnn')):
            return tl.get(i)
    for s in _sorted_fy_statements(statements):   # Pass 3：直近の実績
        if safe_float(s.get('DivAnn')) is not None:
            return actual_dps_adjusted(s, events)
    return None


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

    # --- 株価取得 ---
    # 分割・併合を読むために SPLIT_LOOKBACK_YEARS 年前から取る（取れなければ YIELD_LOOKBACK_DAYS 日に戻す）。
    # 利回りの分布・ボックス・表示に使う quotes は、これまでどおり YIELD_LOOKBACK_DAYS 日の範囲に絞る。
    to_date = today.strftime('%Y-%m-%d')
    short_from = today - timedelta(days=YIELD_LOOKBACK_DAYS)
    quotes_all: list[dict[str, Any]] | None = None
    if _LONG_QUOTES['on']:
        long_from = split_lookback_from(today)
        try:
            quotes_all = client.get_daily_quotes(code, long_from.strftime('%Y-%m-%d'), to_date)
            _LONG_QUOTES['used'] += 1
            _LONG_QUOTES['fail'] = 0
        except Exception as e:
            quotes_all = None
            _LONG_QUOTES['fail'] += 1
            log.warning('long quotes fail %s: %s', code, e)
            if _LONG_QUOTES['fail'] >= 3:
                _LONG_QUOTES['on'] = False
                log.warning('%d年前からの株価が取れないので、%d年分に戻す（古い期は減配の比べから外す）',
                            SPLIT_LOOKBACK_YEARS, YIELD_LOOKBACK_YEARS)
    if quotes_all is None:
        try:
            quotes_all = client.get_daily_quotes(code, short_from.strftime('%Y-%m-%d'), to_date)
        except Exception as e:
            log.warning('quotes fail %s: %s', code, e)
            return None

    if not quotes_all:
        log.debug('no quotes for %s', code)
        return None

    quotes_all.sort(key=lambda q: q.get('Date', ''))
    quotes = [q for q in quotes_all if (parse_date(q.get('Date')) or date.min) >= short_from]
    if not quotes:
        log.debug('no recent quotes for %s', code)
        return None
    # 株価の記録が9年分の範囲の始まり（＋14日）までに始まっていれば、それより前が見えないのは取った範囲の都合
    # （J-Quants が黙って範囲を縮めて返したときも同じ）。その日より前の分割・併合は読めないので、古い期を減配の比べから外す。
    # 記録がもっと遅く始まるのは、そのあとに上場した会社（上場前の分割は記録にない。これまでどおり扱う）。
    first_q = parse_date(quotes_all[0].get('Date'))
    split_known_since = first_q if (first_q is not None and first_q <= short_from + timedelta(days=14)) else None
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
    # 2026年11月6日：分割・併合を読むので、長く取った quotes_all を使う（表示の株価は最後の250日だけ）
    quotes_sorted = sorted(quotes_all, key=lambda q: parse_date(q.get('Date')) or date.min)

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
    div_annual = extract_annual_dividends(statements, quotes_sorted=quotes_sorted,
                                          known_since=split_known_since)
    sales_history = extract_annual_metric(statements, 'Sales')   # 総額: 調整不要
    op_history = extract_annual_metric(statements, 'OP')          # 総額: 調整不要
    np_history = extract_annual_metric(statements, 'NP')          # 総額: 調整不要

    # --- DPS forecast: 分割調整 ---
    # 2026年11月6日：分割・併合の基準の取り違えを直す（forecast_dps_v3。V3）
    split_ev = split_events(quotes_sorted)
    forecast_dps_raw, forecast_dps_date = get_forecast_dps_with_date(statements)
    forecast_dps = forecast_dps_v3(statements, split_ev)
    if forecast_dps is None and forecast_dps_raw is not None:
        forecast_dps = forecast_dps_raw * cumulative_adj_factor_after(quotes_sorted, forecast_dps_date)
    forecast_dps_adj = (forecast_dps / forecast_dps_raw) if (forecast_dps is not None and forecast_dps_raw) else 1.0
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
        # 2026年11月6日：配当は「期末の7日前」の基準で直す（V3）。EPS は決算短信で分割後の基準に
        # 直して載ることが多いので、開示日の基準のまま（これまでどおり）
        div_for_payout = None
        for _s in _sorted_fy_statements(statements):
            if safe_float(_s.get('DivAnn')) is not None:
                div_for_payout = actual_dps_adjusted(_s, split_ev)
                break
        if div_for_payout is None:
            div_for_payout = div_actual_raw * cumulative_adj_factor_after(quotes_sorted, div_actual_date)
        eps_for_payout = eps  # already adjusted
        payout = (div_for_payout / eps_for_payout) * 100.0 if (eps_for_payout and eps_for_payout > 0) else None
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
    # 2026年11月6日：実績の年間配当は「期末の7日前」の基準で直す（V3）。期の途中で分割・併合した期は使わない
    div_history_by_year: dict[int, float] = {}
    _dh_disc: dict[int, date] = {}
    for s in statements:
        if s.get('CurPerType') not in ('FY', '4Q'):
            continue
        d = parse_date(s.get('CurPerEn'))
        if d is None:
            continue
        v = actual_dps_adjusted(s, split_ev)
        disc_d = parse_date(s.get('DiscDate')) or d
        if v is not None and (d.year not in _dh_disc or disc_d >= _dh_disc[d.year]):
            div_history_by_year[d.year] = v
            _dh_disc[d.year] = disc_d

    yield_dist = calculate_yield_distribution(quotes, div_history_by_year)
    signal = determine_signal(current_yield, yield_dist)

    # --- ボックス ---
    box = evaluate_box(quotes)

    # --- 撤退判定 ---
    emergency, reasons = check_emergency_exit(code, div_annual, op_history)
    # 持ち続ける条件（emergency_hold_decision）で使う、減り方の深さ
    emg_div_chg = ((div_annual[-1] - div_annual[-2]) / div_annual[-2]
                   if len(div_annual) >= 2 and div_annual[-2] > 0 else None)
    emg_op_chg = ((op_history[-1] - op_history[-2]) / op_history[-2]
                  if len(op_history) >= 2 and op_history[-2] > 0 else None)

    # --- 決算予定 ---
    next_earnings = announcement_map.get(code)
    earnings_imminent = is_imminent_business_day(next_earnings, today)

    # --- 財務推移 (1株あたり値は分割調整済) ---
    financials_history = build_financials_history(statements, n=8, quotes_sorted=quotes_sorted)

    # --- 月次 PER/PBR/グレアム指数 履歴 ---
    valuation_history = calculate_valuation_history(
        quotes_sorted, statements, yield_dist.get('history', [])
    )

    # --- 過去中央値割安フラグ (B3: 表示ラベル用) ---
    valuation_flags = compute_valuation_cheap_flags(valuation_history, per, pbr)

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
        'emg_div_chg': round(emg_div_chg, 4) if emg_div_chg is not None else None,
        'emg_op_chg': round(emg_op_chg, 4) if emg_op_chg is not None else None,
        # 直近の株式分割・併合（毎日の点検で、分割と重なる「減配」がないかを見る）
        'last_split': ({'date': split_ev[-1][0].isoformat(), 'factor': split_ev[-1][1]} if split_ev else None),
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
        'per_median': valuation_flags['per_median'],
        'pbr_median': valuation_flags['pbr_median'],
        'per_cheap': valuation_flags['per_cheap'],
        'pbr_cheap': valuation_flags['pbr_cheap'],
        'payout_ratio': round(payout, 2) if payout is not None else None,
        'equity_ratio': round(equity_ratio, 4) if equity_ratio is not None else None,
        'roe': round(roe, 2) if roe is not None else None,

        'latest_sales_oku': round(latest_sales / 1e8) if latest_sales is not None else None,
        'latest_op_oku': round(latest_op / 1e8) if latest_op is not None else None,
        'safety_y1': (lambda v: round(v, 3) if v is not None else None)(safety_margin(statements, market_cap_oku, sector33)),
        'latest_np_oku': round(latest_np / 1e8) if latest_np is not None else None,

        'screening': s_results,
        'screening_meta': {
            'dividend_periods_used': div_periods_used,
        },

        'box': box if box else None,

        'financials_history': financials_history,
        'valuation_history': valuation_history,

        # 過去250日分（約1年）の株価履歴 (自前チャート用)
        # AdjustmentClose系を使用することで株式分割を考慮した連続的なチャートになる
        # フォールバック: AdjustmentXxx がなければ生のXxxを使用
        'price_history': [
            {
                'd': q.get('Date'),
                'c': round(safe_float(q.get('AdjustmentClose')) or safe_float(q.get('Close')) or 0, 2),
                'o': round(safe_float(q.get('AdjustmentOpen')) or safe_float(q.get('Open')) or 0, 2),
                'h': round(safe_float(q.get('AdjustmentHigh')) or safe_float(q.get('High')) or 0, 2),
                'l': round(safe_float(q.get('AdjustmentLow')) or safe_float(q.get('Low')) or 0, 2),
                'v': int(safe_float(q.get('AdjustmentVolume')) or safe_float(q.get('Volume')) or 0),
            }
            for q in quotes_sorted[-250:]
            if (safe_float(q.get('AdjustmentClose')) or safe_float(q.get('Close')))
            and (safe_float(q.get('AdjustmentClose')) or safe_float(q.get('Close'))) > 0
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
        # 段階に応じて規模区分で絞る。
        #   core   … Core30 / Large70 / Mid400
        #   small1 … 上記 + TOPIX Small 1
        #   prime  … 絞らない（社数が多く時間内に収まらない）
        _targets = set(SCALE_TARGETS)
        if UNIVERSE_SCOPE == 'small1':
            _targets |= {'TOPIX Small 1'}
        if UNIVERSE_SCOPE != 'prime' and scale_cat not in _targets:
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

    # ── 低PBR順（BUY_MODE == 'pbr'）──
    # 全銘柄の中で PBR が安い側 PBR_TOP_PCT% に入る銘柄だけを BUY に残し、
    # 同じ Tier の中では PBR の低い順に買えるよう buy_sort を付ける。
    _pbrs = sorted([(x['code'], x['pbr']) for x in stocks
                    if x.get('pbr') is not None and x['pbr'] > 0], key=lambda t: -t[1])
    _n = len(_pbrs)
    _rank = {c: (i + 1) / _n * 100 for i, (c, _) in enumerate(_pbrs)} if _n else {}
    for x in stocks:
        x['pbr_rank'] = round(_rank.get(x['code'], 0.0), 1)
        x['signal_pre_rank'] = x.get('signal')        # 2026年11月10日：低PBRの足切りの前の合図（並走記録 variant_engine が使う）
        if BUY_MODE == 'pbr':
            x['buy_sort'] = x['pbr'] if x.get('pbr') else 99.0
            if x.get('signal') == 'BUY' and _rank.get(x['code'], 0.0) < PBR_TOP_PCT:
                x['signal'] = 'NEUTRAL'
        else:
            x['buy_sort'] = -(x.get('current_yield') or 0)

    _save_stmts_cache()

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
        # 減配・業績急変でも持ち続けるか（Tier を使う形があるので、Tier を決めたあとに判定する）
        _h, _hw = emergency_hold_decision(s)
        s['emergency_hold'] = _h
        s['emergency_hold_reason'] = _hw
        s['is_industry_leader'] = is_leader
        s['is_progressive_or_doe'] = is_qual
        # どこで「配当を守る会社」と判定したか（画面の詳細に出す）
        _c4 = str(code)[:4]
        _ep = (EDINET_POLICY.get('companies') or {}).get(_c4) or {}
        _src = [n for n, ok in (('日経累進高配当株指数', _c4 in NIKKEI_PROGRESSIVE_30),
                                ('会社の資料で確認', _c4 in PROGRESSIVE_DECLARED),
                                ('有価証券報告書', _c4 in EDINET_DECLARED)) if ok]
        s['div_policy'] = ({'src': _src, 'phrase': _ep.get('phrase'), 'flags': _ep.get('flags') or [],
                            'snippet': _ep.get('snippet'), 'submitted': _ep.get('submitted')} if _src else None)
        # screening_pass の中だけで Tier 集計 (実際にBUY候補となるもの)
        if s.get('screening_pass'):
            tier_counts[tier] += 1

    log.info('=== Tier distribution (screening pass only) ===')
    for t in ['S', 'A', 'B']:
        log.info('  Tier %s: %d', t, tier_counts[t])

    log.info('=== Summary ===')
    log.info('Processed: %d / %d (failures=%d)', len(stocks), n_total, failures)
    log.info('Long quotes (%d年前から・分割の読み取り用): %d銘柄で取得・%s',
             SPLIT_LOOKBACK_YEARS, _LONG_QUOTES['used'], '有効' if _LONG_QUOTES['on'] else '途中で9年分に戻した')
    for k, v in bucket_counts.items():
        log.info('  bucket %s: %d', k, v)
    for k, v in signal_counts.items():
        log.info('  signal %s: %d', k, v)

    # 待機資金・配分案に使う 1306 のいちばん新しい終値（2026年11月10日。11月13日に results.json にも出すよう前へ移した）
    etf_px, etf_date = None, None
    try:
        to_d = date.today()
        q = client.get_daily_quotes('13060', (to_d - timedelta(days=14)).isoformat(), to_d.isoformat())
        q = [x for x in q if safe_float(x.get('Close')) is not None]
        if q:
            q.sort(key=lambda x: x.get('Date', ''))
            etf_px = safe_float(q[-1].get('Close'))
            etf_date = str(q[-1].get('Date', ''))[:10]
            log.info('1306 close %s (%s)', etf_px, etf_date)
    except Exception as e:
        log.warning('1306 の株価が取れませんでした（待機資金は前の値段で評価）: %s', e)

    # 出力
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at': datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'universe_count': len(universe),
        'processed_count': len(stocks),
        'failure_count': failures,
        'thresholds': {
            'min_yield': MIN_YIELD_THRESHOLD, 'buy_mode': BUY_MODE, 'emergency_hold_rule': {'name': 'K1 浅い減配（−30％未満）は持つ', 'rule': EMERGENCY_HOLD},
            'dividend_policy_checked': DIVIDEND_POLICY_CHECKED, 'edinet_policy_checked': EDINET_POLICY.get('checked_at'),
            'edinet_policy_count': len(EDINET_DECLARED),
            'graham': GRAHAM_THRESHOLD,
            'payout_max': PAYOUT_RATIO_MAX,
            'equity_min': EQUITY_RATIO_MIN,
            'box_adx_max': BOX_ADX_THRESHOLD,
            'box_width_min': BOX_WIDTH_MIN,
            'box_width_max': BOX_WIDTH_MAX,
        },
        # 株価の最後の日（毎日の点検で、データが古くなっていないかを見る）
        'latest_price_date': max(
            (s['price_history'][-1].get('d') for s in stocks if s.get('price_history')), default=None),
        # 2026年11月13日：配分案で余りを 1306 に回すための値段（売買単位は10口。2026年4月1日に1口を10口に分割したあと）
        'etf1306': {'code': '1306', 'name': 'NEXT FUNDS TOPIX連動型上場投信', 'close': etf_px, 'date': etf_date, 'unit': 10},
        'stocks': stocks,
    }
    # 2026年11月8日：履歴を data/history/ に分けて書く（results.json が67MBまで大きくなったため）。
    # stocks そのものは変えないので、このあとの検証運用は今までどおり履歴つきで動く。
    from results_io import write_results
    sizes = write_results(payload, OUTPUT_PATH)

    log.info('Wrote %s (%d stocks) + history %d files (%.1fMB total)', OUTPUT_PATH, len(stocks),
             len(sizes) - 1, sum(sizes.values()) / 1e6)

    # === ポートフォリオ自動売買シミュレーション ===
    try:
        from portfolio_engine import update_portfolio
        log.info('Updating portfolio (forward test)...')
        portfolio_state = update_portfolio(stocks, etf_px)
        try:      # 2026年11月10日：買う順番の並走記録（12月1日から。本番の売買には影響しない）
            from variant_engine import update_variants
            update_variants(stocks, etf_px)
        except Exception as e:
            log.error('Variant tracks failed: %s', e, exc_info=True)
        total_return = portfolio_state['total_value'] - portfolio_state['initial_cash']
        return_pct = total_return / portfolio_state['initial_cash'] * 100
        log.info(
            'Portfolio: total=¥%s, return=¥%s (%+.2f%%), holdings=%d',
            f"{portfolio_state['total_value']:,}",
            f"{total_return:+,}",
            return_pct,
            len(portfolio_state['holdings']),
        )
    except Exception as e:
        log.error('Portfolio update failed: %s', e, exc_info=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())

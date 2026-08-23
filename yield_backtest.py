#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配当利回りルールの比較（yield_backtest.py）

  ※ scripts/backtest.py（既存の検証運用）とは別物です。
    名前が衝突しないよう yield_ を付けています。

複数のルールを、同じ期間・同じデータで走らせて比較します。
「相場によって早めた方がいいのか」を議論ではなく数字で決めるための道具です。

■ 未来のデータを使わないための決まりごと
  ある時点 t の判定に使ってよいのは次だけです。
    株価   … t 以前の終値
    配当   … t 以前に開示された予想DPS（DiscDate <= t）
    分布   … t より前の期間だけで作った利回り分布
  ここを守らないと、実際には取れなかった好成績が出ます。
  すべての計算を「その日までのデータ」に限定しています。

■ 比較するルール（VARIANTS で自由に足せます）
  fixed      いまの実装。Q75で買い、Q25で売る
  tranche    3分割。Q75/Q85/Q95 で1/3ずつ買い、Q25/Q15/Q5 で1/3ずつ売る
  dynamic    条件で必要パーセンタイルを上下させる（増配率・市場内順位）
  cross      自分の過去比ではなく、その日の市場内順位で判定

■ 使い方
  python yield_backtest.py --years 2            直近2年で全ルールを比較
  python yield_backtest.py --only fixed,tranche 一部だけ比較
  python yield_backtest.py --limit 80           試し実行（80銘柄）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("backtest")

JQUANTS_BASE = "https://api.jquants.com"
API_SLEEP = 0.55
MARKET_PRIME = "0111"
SCALE_TARGETS = {"TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"}
OUTDIR = Path("data")
CACHE = OUTDIR / "yield_backtest_cache.pkl"

# ══════════════════════════════════════════
# 実運用の設定（portfolio_engine.py と同じ）
#   これまでのバックテストは「等金額・8〜25銘柄」という簡略版だった。
#   実際は Tier 別に予算が決まっていて、1000万では3〜4銘柄で資金が尽きる。
#   その条件で各ルールがどうなるかを確かめるために用意する。
# ══════════════════════════════════════════
TIER_BUDGET = {"S": 4_000_000, "A": 2_000_000, "B": 1_000_000}

# 累進配当銘柄（日経累進高配当株指数30 ＋ 宣言銘柄）
PROGRESSIVE = {
    "4272","4502","8593","4521","5938","4503","8439","7956","9364","3861",
    "4042","4208","4528","8309","8725","4182","4205","7313","8252","1719",
    "1928","4041","5020","8473","1870","3431","5201","3291","4183","8130",
    "8058","8001","8031","8002","8053","9433","9434","8306","8316","8411",
    "8766","8630","1605","5108","7203","7011","8801",
}
DOE = {"2502","7011","8595","6770"}


# ══════════════════════════════════════════
# 比較するルール
#   ここを編集すれば、いくらでも条件を足せます。
#   entry / exit はパーセンタイル。数字が小さいほど「安いところで買う」。
# ══════════════════════════════════════════
VARIANTS: dict[str, dict[str, Any]] = {
    "fixed": {
        "label": "固定Q75（現行）",
        "entry": [75], "exit": [25],
    },
    "fixed65": {
        "label": "固定Q65（緩め）",
        "entry": [65], "exit": [35],
    },
    "tranche": {
        "label": "3分割 Q75/85/95",
        "entry": [75, 85, 95], "exit": [25, 15, 5],
    },
    "tranche_wide": {
        "label": "3分割 Q60/75/90",
        "entry": [60, 75, 90], "exit": [40, 25, 10],
    },
    "dynamic": {
        "label": "条件で閾値を可変",
        "entry": [75], "exit": [25], "dynamic": True,
    },
    "dynamic_tranche": {
        "label": "3分割＋可変",
        "entry": [75, 85, 95], "exit": [25, 15, 5], "dynamic": True,
    },
    "cross": {
        "label": "市場内順位で判定",
        "entry": [75], "exit": [25], "cross": True,
    },
    # ── 出口が来ない問題への対策 ──
    "exit_median": {
        "label": "Q75買い・Q50売り",
        # Q25まで待つと株価がQ75/Q25倍まで上がる必要がある。
        # 中央値で降りれば必要な上昇幅が半分以下になり、出口が現実的に来る。
        "entry": [75], "exit": [50],
    },
    "exit_gain10": {
        "label": "Q75買い・+10％で売り",
        # 分布ではなく取得単価からの上昇率で降りる。
        # 分布のどこで買っても出口までの距離が一定になる。
        "entry": [75], "exit": [], "gain_exit": 0.10,
    },
    "exit_gain15": {
        "label": "Q75買い・+15％で売り",
        "entry": [75], "exit": [], "gain_exit": 0.15,
    },
    "rotate": {
        "label": "入れ替え（枠が埋まったら弱いものと交換）",
        # 枠が常に埋まっているなら、売りは「条件を満たしたら」ではなく
        # 「もっと良い候補が現れたら」で起こすほうが自然。
        # 横ばい相場では順位が入れ替わるので、自然に売買が増える。
        "entry": [75], "exit": [25], "rotate": 15.0,
    },
    "rotate_median": {
        "label": "入れ替え＋Q50売り",
        "entry": [75], "exit": [50], "rotate": 15.0,
    },
    # ── 組み合わせ ──
    # 利益確定だけだと下げ相場で一度も出口が来ない。
    # 入れ替えを併せると、相場の方向に関係なく建玉が回るようになる。
    "rotate_gain15": {
        "label": "入れ替え＋15％利確",
        "entry": [75], "exit": [25], "gain_exit": 0.15, "rotate": 15.0,
    },
    "rotate_gain10": {
        "label": "入れ替え＋10％利確",
        "entry": [75], "exit": [25], "gain_exit": 0.10, "rotate": 15.0,
    },
    # ── 入れ替えのしきい値を振って感度を見る ──
    # 差が何ポイント開いたら交換するか。小さいほど頻繁に入れ替わる。
    "rotate_narrow": {
        "label": "入れ替え（差8pt・頻繁）",
        "entry": [75], "exit": [25], "rotate": 8.0,
    },
    "rotate_wide": {
        "label": "入れ替え（差25pt・慎重）",
        "entry": [75], "exit": [25], "rotate": 25.0,
    },
    # ── 中央値を起点にした段階買い ──
    # Q75は9年分布では遠すぎるので、中央値から積み増していく形にする。
    "med_tranche": {
        "label": "中央値から3分割（Q50/65/80）",
        "entry": [50, 65, 80], "exit": [40, 25, 10],
    },
    "med_tranche_gain": {
        "label": "中央値3分割＋10％利確",
        "entry": [50, 65, 80], "exit": [], "gain_exit": 0.10,
    },
    "med_gain_rotate": {
        "label": "中央値買い＋10％利確＋入れ替え",
        "entry": [50], "exit": [30], "gain_exit": 0.10, "rotate": 25.0,
    },
    # ── 平均値を起点にした段階買い ──
    # 「平均より上か」を、標準偏差いくつぶん離れているかで測る。
    # 順位ではなく距離を見るので、分布が偏っているときに違いが出る。
    "mean_tranche": {
        "label": "平均から3分割（0/+0.5σ/+1σ）",
        "measure": "z", "entry": [0.0, 0.5, 1.0], "exit": [-0.3, -0.7, -1.2],
    },
    "mean_tranche_gain": {
        "label": "平均3分割＋10％利確",
        "measure": "z", "entry": [0.0, 0.5, 1.0], "exit": [], "gain_exit": 0.10,
    },
    # ── 伸びる銘柄を持ち続けるための売り方 ──
    # 一律+10％で切ると、まだ上がる銘柄まで手放してしまう。
    # 「いつ降りるか」を変えた4案を並べて比べる。
    "hold_if_cheap": {
        "label": "利確10％。ただし中央値より安ければ持つ",
        "entry": [50, 65, 80], "exit": [], "gain_exit": 0.10,
        "gain_hold_above": 50.0,
    },
    "trail_run": {
        "label": "+10％で見張り開始・高値から7％下げたら売り",
        "entry": [50, 65, 80], "exit": [],
        "trail_arm": 0.10, "trail": 0.07,
    },
    "gain_by_tier": {
        "label": "Tier別利確（S20％/A15％/B10％）",
        "entry": [50, 65, 80], "exit": [],
        "gain_exit_by_tier": {"S": 0.20, "A": 0.15, "B": 0.10},
    },
    # ── Tier S をどこまで引っぱれるか ──
    # 質の高い銘柄は+20％より伸びる余地があるのでは、という検証。
    # 上げすぎると出口が来なくなるので、決済率も併せて見る。
    "tier_s30": {
        "label": "Tier別利確（S30％/A20％/B10％）",
        "entry": [50, 65, 80], "exit": [],
        "gain_exit_by_tier": {"S": 0.30, "A": 0.20, "B": 0.10},
    },
    "tier_s40": {
        "label": "Tier別利確（S40％/A25％/B12％）",
        "entry": [50, 65, 80], "exit": [],
        "gain_exit_by_tier": {"S": 0.40, "A": 0.25, "B": 0.12},
    },
    "tier_s_hold": {
        "label": "Sは利確しない（A15％/B10％）",
        # 99 は事実上「到達しない」＝ S は利確で売らない、の意味
        "entry": [50, 65, 80], "exit": [],
        "gain_exit_by_tier": {"S": 99.0, "A": 0.15, "B": 0.10},
    },
    "tier_s_hold_rot": {
        "label": "Sは利確せず・入れ替えあり",
        "entry": [50, 65, 80], "exit": [],
        "gain_exit_by_tier": {"S": 99.0, "A": 0.15, "B": 0.10},
        "rotate": 25.0,
    },
    # ── S銘柄の買い場を逃さないための案 ──
    # S（累進配当×業界首位）は上昇しやすく、9年分位では割高判定になって
    # ほとんど買えない。Tierごとに買いの基準をずらして拾いにいく。
    "s_loose": {
        "label": "S緩め（S:Q30/45/60・A:Q40/55/70・B:Q50/65/80）",
        "entry": [50, 65, 80],
        "entry_by_tier": {"S": [30, 45, 60], "A": [40, 55, 70], "B": [50, 65, 80]},
        "exit": [], "gain_exit_by_tier": {"S": 0.20, "A": 0.15, "B": 0.10},
    },
    "s_loose_wide": {
        "label": "S大幅緩め（S:Q20/35/50・A:Q35/50/65・B:Q50/65/80）",
        "entry": [50, 65, 80],
        "entry_by_tier": {"S": [20, 35, 50], "A": [35, 50, 65], "B": [50, 65, 80]},
        "exit": [], "gain_exit_by_tier": {"S": 0.20, "A": 0.15, "B": 0.10},
    },
    "s_loose_hold": {
        "label": "S緩め・Sは利確しない",
        "entry": [50, 65, 80],
        "entry_by_tier": {"S": [30, 45, 60], "A": [40, 55, 70], "B": [50, 65, 80]},
        "exit": [], "gain_exit_by_tier": {"S": 99.0, "A": 0.15, "B": 0.10},
    },
    # ── いま実際に動いているルール（比較の土台）──
    # portfolio_engine.py と同じ：Q75で全額買い、Q25で売り、Tier順に拾う。
    # これを測らないと「どれだけ良くなるのか」が分からない。
    "current_live": {
        "label": "★現行ルール（Q75買い・Q25売り・Tier順）",
        "entry": [75], "exit": [25], "priority": "tier",
    },

    # ── 予算配分だけを変えた版（買いの基準は現行のまま）──
    # 1銘柄あたりの比重を下げると、それだけで分散が効くのかを見る。
    "live_budget15": {
        "label": "現行の買い方＋予算を15銘柄ぶんに",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
    },
    "live_budget20": {
        "label": "現行の買い方＋予算を20銘柄ぶんに",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 20,
    },
    "live_budget10": {
        "label": "現行の買い方＋予算を10銘柄ぶんに",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 10,
    },
    "live_budget25": {
        "label": "現行の買い方＋予算を25銘柄ぶんに",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 25,
    },
    "live_budget15_flat": {
        "label": "予算15銘柄・Tier重みなし（均等）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 1.0, "A": 1.0, "B": 1.0},
    },
    "live_budget15_strong": {
        "label": "予算15銘柄・Sを厚く（S3.0/A1.5/B1.0）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 3.0, "A": 1.5, "B": 1.0},
    },

    # ── 税金の繰り延べを測るための版 ──
    # 売らなければ譲渡益課税が発生しない。
    # 「どこまでが繰延の効果か」を切り分けるために、
    # 買い方は現行のまま、売らない Tier だけを変えて並べる。
    "live15_holdS": {
        "label": "現行の買い方＋予算15＋Sは売らない",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"],
    },
    "live15_holdSA": {
        "label": "現行の買い方＋予算15＋SとAは売らない",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A"],
    },
    "live15_holdall": {
        "label": "買うだけで売らない（比較の基準）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"],
    },
    "med15_holdS": {
        "label": "中央値3分割＋予算15＋Sは売らない",
        "entry": [50, 65, 80], "exit": [], "priority": "tier",
        "gain_exit_by_tier": {"S": 99.0, "A": 0.15, "B": 0.10},
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"],
    },

    # ── 減配したら手放すかどうかの比較 ──
    "live15_holdS_cut": {
        "label": "Sは売らない＋減配したら手放す",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "live15_holdall_cut": {
        "label": "売らない＋減配だけ手放す",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },

    # ── 予算も買いの基準も変えた版 ──
    "full_new15": {
        "label": "中央値3分割＋Tier別利確＋予算15銘柄ぶん",
        "entry": [50, 65, 80], "exit": [], "priority": "tier",
        "gain_exit_by_tier": {"S": 0.20, "A": 0.15, "B": 0.10},
        "budget_weighted": True, "target_names": 15,
    },
    "full_new15_hold": {
        "label": "上記＋Sは利確しない",
        "entry": [50, 65, 80], "exit": [], "priority": "tier",
        "gain_exit_by_tier": {"S": 99.0, "A": 0.15, "B": 0.10},
        "budget_weighted": True, "target_names": 15,
    },

    # ── Tier順に拾う（実運用の portfolio_engine と同じ優先順位）──
    "tier_first": {
        "label": "Tier順で拾う（S→A→B）",
        "entry": [50, 65, 80], "exit": [], "priority": "tier",
        "gain_exit_by_tier": {"S": 0.20, "A": 0.15, "B": 0.10},
    },
    "tier_first_loose": {
        "label": "Tier順＋S緩め（S:Q30/45/60）",
        "entry": [50, 65, 80], "priority": "tier",
        "entry_by_tier": {"S": [30, 45, 60], "A": [40, 55, 70], "B": [50, 65, 80]},
        "exit": [], "gain_exit_by_tier": {"S": 0.20, "A": 0.15, "B": 0.10},
    },
    "tier_first_hold": {
        "label": "Tier順＋S緩め・Sは利確しない",
        "entry": [50, 65, 80], "priority": "tier",
        "entry_by_tier": {"S": [30, 45, 60], "A": [40, 55, 70], "B": [50, 65, 80]},
        "exit": [], "gain_exit_by_tier": {"S": 99.0, "A": 0.15, "B": 0.10},
    },
    "keep_progressive": {
        "label": "利確10％。累進配当銘柄は利確しない",
        "entry": [50, 65, 80], "exit": [], "gain_exit": 0.10,
        "keep_progressive": True, "rotate": 25.0,
    },
    "mean_gain_rotate": {
        "label": "平均買い＋10％利確＋入れ替え",
        "measure": "z", "entry": [0.0], "exit": [-0.5],
        "gain_exit": 0.10, "rotate": 0.6,
    },
}

# 可変ルールの加減点。必要パーセンタイルを下げる＝買いやすくする。
# 条件を増やすほど過去に合わせただけの数字になりやすいので、3つに絞っています。
DYNAMIC_RULES = {
    "growth": (-10, "増配率が年5％超"),
    "cross_top": (-10, "市場内で上位10％"),
    "regime_tight": (+10, "利回り分布が上振れしやすい局面"),
}
PCT_FLOOR, PCT_CEIL = 50.0, 90.0


# ══════════════════════════════════════════
# データ取得
# ══════════════════════════════════════════
class RangeTooLong(Exception):
    """指定した期間がプランの範囲を超えているときに投げる"""


class JQ:
    def __init__(self, key: str):
        self.key = key
        self.s = requests.Session()

    def get(self, path: str, params: dict | None = None) -> list[dict]:
        rows, pk = [], None
        while True:
            p = dict(params or {})
            if pk:
                p["pagination_key"] = pk
            for attempt in range(3):
                try:
                    r = self.s.get(f"{JQUANTS_BASE}{path}", params=p,
                                   headers={"x-api-key": self.key}, timeout=30)
                    if r.status_code == 429:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    if r.status_code in (401, 403):
                        sys.exit(f"認証に失敗しました（{r.status_code}）")
                    if r.status_code == 400:
                        # 期間が長すぎる場合にこれが返る。呼び出し側で短くして再試行する。
                        raise RangeTooLong(r.text[:150])
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            j = r.json()
            data = j.get("data")
            if data is None:
                data = next((v for k, v in j.items()
                             if k != "pagination_key" and isinstance(v, list)), [])
            rows.extend(data or [])
            pk = j.get("pagination_key")
            if not pk:
                return rows
            time.sleep(API_SLEEP)


def norm_code(c: str) -> str:
    c = str(c).strip()
    return c[:4] if len(c) == 5 and c.endswith("0") else c


def fnum(v: Any) -> float | None:
    if v is None or v == "" or v == "－":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pdate(s: Any) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# 取得できた年数を覚えておく。1銘柄目で分かれば以降は無駄な試行をしない。
_ok_years: int | None = None
FETCH_CANDIDATES = (9, 8, 7, 5, 4, 3, 2)   # 上限は9年（調査済み）


def fetch_bars(jq: JQ, code: str, want_years: int) -> list[dict]:
    """株価を取得する。期間が長すぎて弾かれたら、自動的に短くして試す。

    J-Quants は契約プランによって遡れる年数が決まっており、
    それを超える期間を指定すると 400 が返ります。
    どこまで遡れるかを実際に試して見つけます。
    """
    global _ok_years
    today = date.today()
    cands = [_ok_years] if _ok_years else \
            [y for y in FETCH_CANDIDATES if y <= want_years] or [2]
    for y in cands:
        frm = (today - timedelta(days=365 * y + 30)).isoformat()
        try:
            rows = jq.get("/v2/equities/bars/daily",
                          {"code": code, "from": frm, "to": today.isoformat()})
            if _ok_years is None:
                _ok_years = y
                log.info("さかのぼれる期間: %d年（プランの上限に合わせました）", y)
            return rows
        except RangeTooLong:
            continue
    return []


def fetch_all(jq: JQ, years: int, limit: int,
              scale_filter: bool = True) -> dict:
    """株価と財務をまとめて取得する。時間がかかるのでキャッシュします。"""
    log.info("銘柄一覧を取得中…")
    info = jq.get("/v2/equities/master", {})
    uni = []
    for row in info:
        if row.get("Mkt") != MARKET_PRIME:
            continue
        # 大型〜中型に絞ると、業績が崩れて中小型に落ちた会社が
        # 最初から入らない（生き残りだけを見ることになる）。
        # prime を選べば、その偏りが減る。
        if scale_filter and (row.get("ScaleCat") or "") not in SCALE_TARGETS:
            continue
        code = norm_code(row.get("Code", ""))
        if code:
            uni.append({"code": code,
                        "name": row.get("CoName", "") or row.get("CoNameEn", ""),
                        "sector": row.get("S33Nm", ""),
                        # 業界首位級の判定に使う。以前ここが抜けていて
                        # Tier S が1社も出ない状態になっていた。
                        "scale": row.get("ScaleCat", "")})
    if limit:
        uni = uni[:limit]
    log.info("対象 %d銘柄", len(uni))

    store = {"universe": uni, "quotes": {}, "stmts": {}}
    ok = 0
    for i, u in enumerate(uni, 1):
        time.sleep(API_SLEEP)
        try:
            q = fetch_bars(jq, u["code"], 9)   # いつでも上限まで取っておく
        except Exception as e:
            log.warning("株価取得失敗 %s: %s", u["code"], e)
            continue
        if not q:
            continue
        ok += 1
        time.sleep(API_SLEEP)
        try:
            st = jq.get("/v2/fins/summary", {"code": u["code"]})
        except Exception:
            st = []
        store["quotes"][u["code"]] = q
        store["stmts"][u["code"]] = st
        if i % 50 == 0:
            log.info("  取得 %d/%d（成功 %d）", i, len(uni), ok)
    if ok == 0:
        sys.exit("株価を1銘柄も取得できませんでした。APIキーと契約プランをご確認ください。")
    log.info("取得できた銘柄: %d / %d", ok, len(uni))
    return store


# ══════════════════════════════════════════
# 時点を守った指標づくり
# ══════════════════════════════════════════
def quotes_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    close = df.get("AdjC", df.get("AdjustmentClose", df.get("C", df.get("Close"))))
    adj = df.get("AdjFactor", df.get("AdjustmentFactor"))
    out = pd.DataFrame({"date": df["Date"],
                        "close": pd.to_numeric(close, errors="coerce"),
                        "adj": pd.to_numeric(adj, errors="coerce").fillna(1.0)})
    return out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def dps_timeline(stmts: list[dict], px: pd.DataFrame) -> pd.DataFrame:
    """開示日つきの予想DPS系列を作る。

    その日に「すでに開示されていた」DPSしか使わないのが要点。
    分割は、開示日より後に起きたぶんだけ調整して現在基準に揃える。
    """
    if px.empty:
        return pd.DataFrame()
    # 分割係数は日付順に累積させておき、任意の日以降の累積を引けるようにする
    adj = px.set_index("date")["adj"].replace(0, 1.0)
    cum = adj[::-1].cumprod()[::-1]        # その日以降の累積係数
    cum_after = cum.shift(-1).fillna(1.0)

    recs = []
    for s in stmts:
        d = pdate(s.get("DiscDate")) or pdate(s.get("CurPerEn"))
        if d is None:
            continue
        per = s.get("CurPerType", "")
        v = fnum(s.get("NxFDivAnn")) if per in ("FY", "4Q") else fnum(s.get("FDivAnn"))
        if v is None or v <= 0:
            v = fnum(s.get("DivAnn"))
        if v is None or v <= 0:
            continue
        ts = pd.Timestamp(d)
        # 開示日以降の分割ぶんだけ調整
        idx = cum_after.index.searchsorted(ts)
        factor = float(cum_after.iloc[idx]) if idx < len(cum_after) else 1.0
        recs.append({"date": ts, "dps": v * factor})
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values("date")
    return df.groupby("date", as_index=False)["dps"].last()


def shares_outstanding(stmts: list[dict]) -> float | None:
    """発行済株式数。ShOutFY があればそれ、無ければ 純利益÷EPS で逆算する。"""
    fy = [x for x in stmts if x.get("CurPerType") in ("FY", "4Q")]
    fy.sort(key=lambda x: str(x.get("CurPerEn", "")), reverse=True)
    for x in fy:
        v = fnum(x.get("ShOutFY"))
        if v and v > 0:
            return v
    for x in fy:
        np_, eps = fnum(x.get("NP")), fnum(x.get("EPS"))
        if np_ and eps and eps > 0:
            return np_ / eps
    return None


def fiscal_months(stmts: list[dict]) -> tuple[int | None, int | None]:
    """決算月と中間配当の権利月（決算月の6か月前）を返す。"""
    fy = [x for x in stmts if x.get("CurPerType") in ("FY", "4Q")]
    fy.sort(key=lambda x: str(x.get("CurPerEn", "")), reverse=True)
    for x in fy:
        d = pdate(x.get("CurPerEn"))
        if d:
            return d.month, ((d.month - 6 - 1) % 12) + 1
    return None, None


def assign_tiers(store: dict, last_price: dict[str, float]) -> dict[str, str]:
    """Tier を判定する（generate.py と同じ考え方）。

      S … 累進配当/DOE銘柄 かつ 業界首位級
      A … どちらか一方
      B … それ以外
    業界首位級 = TOPIX Core30 または 33業種内で時価総額TOP3。
    """
    mcap: dict[str, float] = {}
    for u in store["universe"]:
        code = u["code"]
        sh = shares_outstanding(store["stmts"].get(code, []))
        px = last_price.get(code)
        if sh and px:
            mcap[code] = px * sh

    leaders = {u["code"] for u in store["universe"] if u.get("scale") == "TOPIX Core30"}
    by_sector: dict[str, list[tuple[str, float]]] = {}
    for u in store["universe"]:
        c, sec = u["code"], u.get("sector") or ""
        if sec and c in mcap:
            by_sector.setdefault(sec, []).append((c, mcap[c]))
    for s33, members in by_sector.items():
        members.sort(key=lambda x: -x[1])
        for c, _ in members[:3]:
            leaders.add(c)

    tiers = {}
    for u in store["universe"]:
        c = u["code"]
        qual = c in PROGRESSIVE or c in DOE
        lead = c in leaders
        tiers[c] = "S" if (qual and lead) else ("A" if (qual or lead) else "B")
    return tiers


def build_panel(store: dict, years: int, lookback: int = 36) -> pd.DataFrame:
    """月末ごとの「その時点で分かっていた」利回りの表を作る。"""
    frames = []
    last_price: dict[str, float] = {}
    for code, qrows in store["quotes"].items():
        px0 = quotes_to_df(qrows)
        if not px0.empty:
            last_price[code] = float(px0["close"].iloc[-1])
    tiers = assign_tiers(store, last_price)
    names_by_code = {u["code"]: u.get("name", u["code"]) for u in store["universe"]}

    for code, qrows in store["quotes"].items():
        px = quotes_to_df(qrows)
        if len(px) < 300:
            continue
        dps = dps_timeline(store["stmts"].get(code, []), px)
        if dps.empty:
            continue

        m = px.set_index("date")["close"].resample("ME").last().dropna()
        d = dps.set_index("date")["dps"].reindex(m.index, method="ffill")
        y = (d / m * 100.0).replace([np.inf, -np.inf], np.nan)

        fm, im = fiscal_months(store["stmts"].get(code, []))
        f = pd.DataFrame({"date": m.index, "code": code,
                          "price": m.values, "dps": d.values, "yield": y.values})
        f["tier"] = tiers.get(code, "B")
        f["name"] = names_by_code.get(code, code)
        # 累進配当・DOE銘柄は「減配しにくい」と宣言している。
        # 利確の扱いを変えるかどうかを試せるようにフラグを持たせる。
        f["progressive"] = (code in PROGRESSIVE) or (code in DOE)
        # 減配の検知。直近1年の最高額を下回ったら減配とみなす。
        # 配当利回りで割安さを測る戦略では、減配は「買った根拠が消える」出来事。
        prev_max = f["dps"].rolling(12, min_periods=2).max().shift(1)
        f["dps_cut"] = (f["dps"] < prev_max * 0.999).fillna(False)
        f["fiscal_month"] = fm if fm else 0
        f["interim_month"] = im if im else 0
        # 増配率（過去2年）。これも過去だけを見る
        f["dps_growth"] = f["dps"].pct_change(24) * 100.0 / 2.0
        frames.append(f.dropna(subset=["yield"]))

    if not frames:
        sys.exit("パネルを作れませんでした。データ取得をご確認ください。")
    panel = pd.concat(frames, ignore_index=True)

    # 各時点で、過去の分布における自分の位置（未来は見ない）。
    # min_periods は lookback と同じにする。半分で計算を許すと
    # 「36か月の分位」と言いながら実際は18か月で出している、という
    # ラベルと中身の食い違いが起きるため。
    # J-Quants で遡れるのが5年程度なので、分布は36か月で作る。
    # 60か月にすると分布づくりだけでデータを使い切り、検証する期間が残らない。
    panel = panel.sort_values(["code", "date"])
    panel["pct_own"] = (
        panel.groupby("code")["yield"]
        .transform(lambda s: s.rolling(lookback, min_periods=lookback)   # 窓を満たすまで計算しない
                   .apply(lambda w: (w.iloc[-1] >= w[:-1]).mean() * 100, raw=False))
    )
    # その日の市場内での順位
    panel["pct_cross"] = panel.groupby("date")["yield"].rank(pct=True) * 100.0

    # 平均値からの離れ具合（zスコア）。
    # パーセンタイルは順位しか見ないが、こちらは「どれだけ離れているか」を測る。
    # 過去だけを使うため、当日を除いた窓で平均と標準偏差を出す。
    def _z(s: pd.Series) -> pd.Series:
        past = s.shift(1)
        m = past.rolling(lookback - 1, min_periods=lookback - 1).mean()
        sd = past.rolling(lookback - 1, min_periods=lookback - 1).std()
        return (s - m) / sd.replace(0, np.nan)
    panel["zscore"] = panel.groupby("code")["yield"].transform(_z)

    start = panel["date"].max() - pd.DateOffset(years=years)
    return panel[panel["date"] >= start].dropna(subset=["pct_own"]).reset_index(drop=True)


# ══════════════════════════════════════════
# 売買ルール
# ══════════════════════════════════════════
def required_pct(base: float, row: pd.Series, dynamic: bool) -> float:
    """必要パーセンタイル。条件が揃うほど下がる＝買いやすくなる。"""
    if not dynamic:
        return base
    p = base
    if row.get("dps_growth", 0) is not None and row.get("dps_growth", 0) > 5:
        p += DYNAMIC_RULES["growth"][0]
    if row.get("pct_cross", 0) >= 90:
        p += DYNAMIC_RULES["cross_top"][0]
    if row.get("mkt_yield_z", 0) > 1.0:
        p += DYNAMIC_RULES["regime_tight"][0]
    return float(np.clip(p, PCT_FLOOR, PCT_CEIL))


def simulate(panel: pd.DataFrame, cfg: dict, capital: float = 3_000_000,
             max_names: int = 15, tier_budget: bool = False,
             dividends: bool = False, slip_bps: float = 0.0,
             fee_bps: float = 0.0, tax_rate: float = 0.0) -> dict:
    """月末ごとに判定して売買する。等金額・分割建玉。

    出口は3通りを組み合わせられる。
      exit       … 利回りの分位が下がったら降りる（従来）
      gain_exit  … 取得単価から一定率上がったら降りる
      rotate     … 枠が埋まっているとき、もっと良い候補と入れ替える
    """
    entry, exits = cfg["entry"], cfg.get("exit", [])
    dyn, cross = cfg.get("dynamic", False), cfg.get("cross", False)
    # measure: "pct"（パーセンタイル）か "z"（平均から何σ離れているか）
    measure = cfg.get("measure", "pct")
    # Tierごとに買いの基準を変える。
    # 質の高い銘柄（S）は多少の割高を許容しないと、買い場が来ないため。
    entry_by_tier = cfg.get("entry_by_tier")
    # 予算の決め方。
    #   固定額 … S400万/A200万/B100万（現行）。1000万では3〜5銘柄で尽きる。
    #   比率   … 総資産 ÷ 目標銘柄数 × Tierの重み。資産が増えれば自動で広がる。
    weighted = cfg.get("budget_weighted", False)
    target_n = cfg.get("target_names", 15)
    tw = cfg.get("tier_weight", {"S": 2.0, "A": 1.5, "B": 1.0})

    def entry_for(row):
        if entry_by_tier:
            return entry_by_tier.get(row.get("tier", "B"), entry)
        return entry

    def level(row):
        """割安さの指標。大きいほど割安。"""
        if measure == "z":
            v = row.get("zscore")
            return float(v) if v is not None and not pd.isna(v) else -99.0
        return row["pct_cross"] if cross else row["pct_own"]
    gain_exit = cfg.get("gain_exit")
    gain_by_tier = cfg.get("gain_exit_by_tier")     # Tierごとに利確ラインを変える
    hold_above = cfg.get("gain_hold_above")         # まだ割安なら利確を見送る
    keep_prog = cfg.get("keep_progressive", False)  # 累進配当銘柄は利確しない
    # 売らない Tier。売却しなければ課税されないので、税金が繰り延べられる。
    # 「塩漬け」が税制上どれだけ有利かを測るために用意する。
    hold_tiers = set(cfg.get("hold_tiers", []))
    # 減配したら手放すか。実運用の「緊急撤退」に相当する。
    exit_on_cut = cfg.get("exit_on_cut", False)
    min_yield = cfg.get("min_yield", 0.0)
    trail_arm = cfg.get("trail_arm")                # この率まで上がったら見張り開始
    trail = cfg.get("trail")                        # 高値からこの率下げたら売る
    rotate = cfg.get("rotate")
    n_tr = len(entry)

    # 売買にかかる費用。
    #   slip … 約定のずれ。成行で買えば少し高く、売れば少し安くなる。
    #   fee  … 手数料。
    #   tax  … 譲渡益と配当への課税。損は繰り越して相殺する（損益通算）。
    slip = slip_bps / 10000.0
    fee = fee_bps / 10000.0
    loss_pool = 0.0        # 相殺できる損失の残り

    dates = sorted(panel["date"].unique())
    cash = capital
    # code -> {"units": n, "lots": [(株数, 取得単価), ...]}
    pos: dict[str, dict] = {}
    curve, trades = [], []
    diag = {"adjusted": 0, "changed": 0, "full_slots": 0, "months": 0,
            "opened": 0, "closed": 0, "hold_months": [], "rotations": 0,
            # Tierごとに「建てた数」と「どこまで含み益が伸びたか」を記録する。
            # 利確ラインを変えても結果が動かないとき、
            # そもそもその銘柄を持っていないのか、
            # 持っていても伸びていないのかを切り分けるため。
            "tier_opened": {}, "tier_peak": []}
    opened_at: dict[str, int] = {}

    mkt = panel.groupby("date")["yield"].median()
    mkt_z = (mkt - mkt.rolling(36, min_periods=12).mean()) / \
            mkt.rolling(36, min_periods=12).std()

    def value_of(day):
        v = cash
        for c, st in pos.items():
            if c in day.index:
                v += sum(sh for sh, _ in st["lots"]) * day.loc[c, "price"]
        return v

    for mi, dt in enumerate(dates):
        day = panel[panel["date"] == dt].set_index("code")
        day = day.assign(mkt_yield_z=float(mkt_z.get(dt, 0) or 0))

        # ── 売り ──
        for code in list(pos.keys()):
            if code not in day.index:
                continue
            row = day.loc[code]
            st = pos[code]
            price = row["price"]
            # 減配は「売らない Tier」でも例外として手放す
            if exit_on_cut and bool(row.get("dps_cut")) and st["lots"]:
                for sh, pr in st["lots"]:
                    eff = price * (1 - slip) * (1 - fee)
                    cash += sh * eff
                    gain = (eff - pr) * sh
                    if tax_rate > 0:
                        if gain > 0:
                            taxable = max(0.0, gain - loss_pool)
                            loss_pool = max(0.0, loss_pool - gain)
                            t = taxable * tax_rate
                            cash -= t
                            diag["tax"] = diag.get("tax", 0.0) + t
                        else:
                            loss_pool += -gain
                    diag["fee"] = diag.get("fee", 0.0) + sh * price * (slip + fee)
                    trades.append({"code": code, "date": dt, "side": "sell",
                                   "price": eff, "shares": sh})
                diag["cut_exits"] = diag.get("cut_exits", 0) + 1
                if st.get("sh_sum"):
                    avg = st["cost_sum"] / st["sh_sum"]
                    g_ = st.get("peak", avg) / avg - 1
                    diag["tier_peak"].append((st.get("tier", "B"), g_))
                    for e_ in reversed(diag.get("trade_log", [])):
                        if e_["code"] == code and e_["result"] == "保有中":
                            e_["peak_gain"] = g_
                            e_["result"] = "売却"
                            break
                del pos[code]
                diag["closed"] += 1
                if code in opened_at:
                    diag["hold_months"].append(mi - opened_at.pop(code))
                continue

            if hold_tiers and row.get("tier", "B") in hold_tiers:
                continue          # この Tier は売らない

            # ── 利益が乗ったときの降り方 ──
            if st["lots"]:
                cost = sum(sh * pr for sh, pr in st["lots"]) / \
                       sum(sh for sh, _ in st["lots"])
                gain = price / cost - 1
                st["peak"] = max(st.get("peak", price), price)

                # 利確ライン。Tier別の指定があればそちらを優先する。
                thr = None
                if gain_by_tier:
                    thr = gain_by_tier.get(row.get("tier", "B"))
                elif gain_exit is not None:
                    thr = gain_exit

                # まだ割安なら利確を見送る、という判断も試せるようにする。
                # 「+10％だが過去の中央値よりまだ安い」なら伸びしろが残っている、
                # という考え方。
                skip = False
                if hold_above is not None and level(row) >= hold_above:
                    skip = True
                if keep_prog and bool(row.get("progressive")):
                    skip = True

                sold = False
                if thr is not None and not skip and gain >= thr:
                    sold = True
                # 高値からの下落で降りる（伸びるだけ伸ばしてから降りる）
                elif trail is not None and trail_arm is not None:
                    if st["peak"] / cost - 1 >= trail_arm and \
                       price <= st["peak"] * (1 - trail):
                        sold = True

                if sold:
                    for sh, pr in st["lots"]:
                        eff = price * (1 - slip) * (1 - fee)
                        cash += sh * eff
                        gain = (eff - pr) * sh
                        if tax_rate > 0:
                            if gain > 0:
                                taxable = max(0.0, gain - loss_pool)
                                loss_pool = max(0.0, loss_pool - gain)
                                t = taxable * tax_rate
                                cash -= t
                                diag["tax"] = diag.get("tax", 0.0) + t
                            else:
                                loss_pool += -gain
                        diag["fee"] = diag.get("fee", 0.0) + sh * price * (slip + fee)
                        trades.append({"code": code, "date": dt, "side": "sell",
                                       "price": eff, "shares": sh})
                    st["lots"], st["units"] = [], 0

            # 利回りの分位で降りる
            if st["units"] > 0 and exits:
                p = level(row)
                want = len(entry_for(row)) - sum(1 for e in exits if p <= e)
                while st["units"] > want and st["units"] > 0:
                    sh, pr = st["lots"].pop()
                    eff = price * (1 - slip) * (1 - fee)
                    cash += sh * eff
                    gain = (eff - pr) * sh
                    if tax_rate > 0:
                        if gain > 0:
                            taxable = max(0.0, gain - loss_pool)
                            loss_pool = max(0.0, loss_pool - gain)
                            t = taxable * tax_rate
                            cash -= t
                            diag["tax"] = diag.get("tax", 0.0) + t
                        else:
                            loss_pool += -gain
                    diag["fee"] = diag.get("fee", 0.0) + sh * price * (slip + fee)
                    trades.append({"code": code, "date": dt, "side": "sell",
                                   "price": eff, "shares": sh})
                    st["units"] -= 1

            if st["units"] == 0:
                if st.get("sh_sum"):
                    avg = st["cost_sum"] / st["sh_sum"]
                    g_ = st.get("peak", avg) / avg - 1
                    diag["tier_peak"].append((st.get("tier", "B"), g_))
                    for e_ in reversed(diag.get("trade_log", [])):
                        if e_["code"] == code and e_["result"] == "保有中":
                            e_["peak_gain"] = g_
                            e_["result"] = "売却"
                            break
                del pos[code]
                diag["closed"] += 1
                if code in opened_at:
                    diag["hold_months"].append(mi - opened_at.pop(code))

        # ── 買い候補 ──
        diag["months"] += 1
        cands = []
        for code, row in day.iterrows():
            if min_yield > 0 and row.get("yield", 0) < min_yield:
                continue        # 利回りが低すぎる銘柄は最初から除く
            p = level(row)
            ent = entry_for(row)
            need = [required_pct(e, row, dyn) if measure == "pct" else e
                    for e in ent]
            want = sum(1 for nd in need if p >= nd)
            if dyn and measure == "pct" and need != list(map(float, ent)):
                diag["adjusted"] += 1
                if want != sum(1 for e in ent if p >= e):
                    diag["changed"] += 1
            have = pos.get(code, {}).get("units", 0)
            if want > have:
                cands.append((p, code, want - have, row["price"],
                              row.get("tier", "B")))
        # 並べ替え方。
        #   pct  … 割安な順（既定）。ただしS銘柄は割安度が低く出るため、
        #          Bに枠を先取りされてSが永久に買えなくなる。
        #   tier … Tier順（S→A→B）。実運用の portfolio_engine と同じ優先順位。
        if cfg.get("priority") == "tier":
            _ord = {"S": 0, "A": 1, "B": 2}
            cands.sort(key=lambda x: (_ord.get(x[4], 9), -x[0]))
        else:
            cands.sort(key=lambda x: -x[0])

        # ── 入れ替え ──
        # 「枠が埋まったら」だけを条件にすると、資金が先に尽きる運用では
        # 一度も発火しない。実際に効くのは「新規で買えないとき」なので、
        # 枠が埋まっている場合と、資金が足りない場合の両方で検討する。
        blocked = False
        if rotate and cands:
            if tier_budget:
                _t = day.loc[cands[0][1], "tier"] if "tier" in day.columns else "B"
                need_cash = TIER_BUDGET.get(_t, TIER_BUDGET["B"]) / n_tr
            else:
                need_cash = value_of(day) / max_names / n_tr
            blocked = len(pos) >= max_names or cash < need_cash
        if rotate and blocked and cands:
            held = [(level(day.loc[c]), c) for c in pos if c in day.index]
            if held:
                held.sort()
                worst_p, worst_c = held[0]
                best = next(((p, c) for p, c, _, _, _ in cands if c not in pos), None)
                if best and best[0] - worst_p >= rotate:
                    st = pos.pop(worst_c)
                    pr0 = day.loc[worst_c, "price"]
                    pr = pr0 * (1 - slip) * (1 - fee)
                    for sh, _c in st["lots"]:
                        cash += sh * pr
                        diag["fee"] = diag.get("fee", 0.0) + sh * pr0 * (slip + fee)
                        trades.append({"code": worst_c, "date": dt, "side": "sell",
                                       "price": pr, "shares": sh})
                    diag["closed"] += 1
                    diag["rotations"] += 1
                    if worst_c in opened_at:
                        diag["hold_months"].append(mi - opened_at.pop(worst_c))

        if len(pos) >= max_names:
            diag["full_slots"] += 1

        # ── 配当（権利月に 年間DPS ÷ 2 を受け取る）──
        if dividends:
            for code, st in pos.items():
                if code not in day.index:
                    continue
                row = day.loc[code]
                mth = pd.Timestamp(dt).month
                for key in ("fiscal_month", "interim_month"):
                    if int(row.get(key, 0) or 0) == mth and row["dps"] > 0:
                        held = sum(sh for sh, _ in st["lots"])
                        gross = row["dps"] * 0.5 * held
                        net = gross * (1 - tax_rate)
                        cash += net
                        diag["dividend"] = diag.get("dividend", 0) + net
                        diag["tax"] = diag.get("tax", 0.0) + (gross - net)

        # ── 買い ──
        total = value_of(day)
        for p, code, add, price, _tier in cands:
            if len(pos) >= max_names and code not in pos:
                continue
            # Tier別予算か、等金額か
            nt = len(entry_for(day.loc[code])) if entry_by_tier else n_tr
            tier = day.loc[code, "tier"] if "tier" in day.columns else "B"
            if weighted:
                # 総資産に対する比率で決める。重みの平均で割って、
                # 目標銘柄数ぶんに収まるようにする。
                avg_w = sum(tw.values()) / len(tw)
                unit_size = total / target_n * (tw.get(tier, 1.0) / avg_w) / nt
            elif tier_budget:
                unit_size = TIER_BUDGET.get(tier, TIER_BUDGET["B"]) / nt
            else:
                unit_size = total / max_names / nt
            for _ in range(add):
                if cash < unit_size or unit_size < price:
                    break
                # 100株単位（実運用に合わせる）
                sh = int(unit_size // price // 100) * 100 if tier_budget \
                     else int(unit_size // price)
                if sh <= 0:
                    break
                buy_eff = price * (1 + slip) * (1 + fee)
                cash -= sh * buy_eff
                diag["fee"] = diag.get("fee", 0.0) + sh * price * (slip + fee)
                st = pos.setdefault(code, {"units": 0, "lots": [], "peak": price})
                if st["units"] == 0:
                    diag["opened"] += 1
                    opened_at[code] = mi
                    tg = day.loc[code, "tier"] if "tier" in day.columns else "B"
                    diag["tier_opened"][tg] = diag["tier_opened"].get(tg, 0) + 1
                    diag.setdefault("trade_log", []).append({
                        "tier": tg, "code": code,
                        "name": day.loc[code].get("name", code),
                        "date": str(pd.Timestamp(dt).date())[:7],
                        "price": price, "peak_gain": 0.0, "result": "保有中"})
                st["tier"] = day.loc[code, "tier"] if "tier" in day.columns else "B"
                st["cost_sum"] = st.get("cost_sum", 0.0) + sh * buy_eff
                st["sh_sum"] = st.get("sh_sum", 0) + sh
                st["lots"].append((sh, buy_eff))
                st["units"] += 1
                trades.append({"code": code, "date": dt, "side": "buy",
                               "price": price, "shares": sh})

        curve.append({"date": dt, "value": value_of(day),
                      "cash": cash, "names": len(pos)})

    # 期末に残っている建玉も、到達した含み益として記録しておく
    for code, st in pos.items():
        if st.get("sh_sum"):
            avg = st["cost_sum"] / st["sh_sum"]
            g_ = st.get("peak", avg) / avg - 1
            diag["tier_peak"].append((st.get("tier", "B"), g_))
            for e_ in reversed(diag.get("trade_log", [])):
                if e_["code"] == code and e_["result"] == "保有中":
                    e_["peak_gain"] = g_
                    break

    return {"curve": pd.DataFrame(curve), "trades": pd.DataFrame(trades),
            "capital": capital, "diag": diag}


def metrics(res: dict) -> dict:
    c = res["curve"]
    if c.empty:
        return {}
    v = c["value"].to_numpy()
    cap = res["capital"]
    yrs = max((c["date"].iloc[-1] - c["date"].iloc[0]).days / 365.25, 0.5)
    total = v[-1] / cap - 1
    cagr = (v[-1] / cap) ** (1 / yrs) - 1
    dd = float((1 - v / np.maximum.accumulate(v)).max())
    r = pd.Series(v).pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0
    t = res["trades"]
    d = res.get("diag", {})
    # 保有枠が埋まっていた月の割合。高いほど「閾値を緩めても効かない」状態。
    saturated = d["full_slots"] / d["months"] * 100 if d.get("months") else 0.0
    # 出口の来やすさ。買った建玉のうち、何割が実際に手仕舞えたか。
    opened, closed = d.get("opened", 0), d.get("closed", 0)
    exit_rate = closed / opened * 100 if opened else 0.0
    hold = d.get("hold_months", [])
    avg_hold = float(np.mean(hold)) if hold else float("nan")
    return {"総リターン": total * 100, "年率": cagr * 100,
            "最大下落": dd * 100, "シャープ": sharpe,
            "売買回数": len(t), "平均保有銘柄": float(c["names"].mean()),
            "判定変化": d.get("changed", 0), "枠飽和率": saturated,
            "費用": d.get("fee", 0.0), "税金": d.get("tax", 0.0),
            "決済率": exit_rate, "平均保有月数": avg_hold,
            "入れ替え": d.get("rotations", 0)}


# ── 比較の基準：対象銘柄を等金額で買って持ち続けた場合 ──
# ルールが本当に価値を生んでいるのか、それとも相場が上がっただけなのか。
# 銘柄選択も売買判断も一切しない場合の成績を出して並べる。
def buy_and_hold(pn: pd.DataFrame, tax_rate: float) -> dict | None:
    dts = sorted(pn["date"].unique())
    if len(dts) < 12:
        return None
    first = pn[pn["date"] == dts[0]].set_index("code")
    codes = list(first.index)
    if not codes:
        return None
    # 初日に等金額で買い、あとは何もしない
    w = 1.0 / len(codes)
    base_px = first["price"].to_dict()
    vals, div_total = [], 0.0
    for dt in dts:
        day = pn[pn["date"] == dt].set_index("code")
        v = 0.0
        for c in codes:
            if c in day.index and base_px.get(c):
                v += w * day.loc[c, "price"] / base_px[c]
                m = pd.Timestamp(dt).month
                for key in ("fiscal_month", "interim_month"):
                    if int(day.loc[c].get(key, 0) or 0) == m and day.loc[c, "dps"] > 0:
                        g = w * (day.loc[c, "dps"] * 0.5) / base_px[c]
                        div_total += g * (1 - tax_rate)
            else:
                v += w        # 上場廃止などは取得時の値で据え置く
        vals.append(v + div_total)
    arr = np.array(vals)
    yrs = max((pd.Timestamp(dts[-1]) - pd.Timestamp(dts[0])).days / 365.25, 0.5)
    dd = float((1 - arr / np.maximum.accumulate(arr)).max())
    r = pd.Series(arr).pct_change().dropna()
    return {"総リターン": (arr[-1] - 1) * 100,
            "年率": (arr[-1] ** (1 / yrs) - 1) * 100,
            "最大下落": dd * 100,
            "シャープ": float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0}


# ══════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=2,
                    help="検証する年数。取得できるのが約5年なので、"
                         "分布づくりに3年使い、残りが検証期間になります")
    ap.add_argument("--lookback", type=int, default=36,
                    help="利回り分布を作る月数（既定36＝3年）")
    ap.add_argument("--diagnose-tier", action="store_true",
                    help="Tier判定の中身を調べて表示する（売買はしない）")
    ap.add_argument("--sweep", default="",
                    help="分位の期間を振って比較する。例: 12,24,36,48,60,72,84\n"
                         "検証期間は --years で固定されるので、比較が成立する")
    ap.add_argument("--limit", type=int, default=0, help="試し実行。先頭N銘柄")
    ap.add_argument("--only", default="", help="比較するルールをカンマ区切りで指定")
    ap.add_argument("--capital", type=float, default=0,
                    help="元本。未指定なら通常300万・実運用モードで1000万")
    ap.add_argument("--show-trades", action="store_true",
                    help="建玉の明細を表示する（どの銘柄をいつ買ったか）")
    ap.add_argument("--walk", type=int, default=0,
                    help="この年数の窓を1年ずつずらして何度も検証する。"
                         "例: 4 なら 2019-2023、2020-2024… と繰り返す。"
                         "特定の期間がたまたま良かっただけかを見分けられる")
    ap.add_argument("--min-yield", type=float, default=0.0,
                    help="この利回り（％）を下回る銘柄は買わない。"
                         "実運用のスクリーニングに相当する部分")
    ap.add_argument("--universe", choices=["core", "prime"], default="core",
                    help="core＝大型〜中型（既定）、prime＝プライム全銘柄。"
                         "prime にすると中小型に落ちた会社も入り、偏りが減る")
    ap.add_argument("--slip-bps", type=float, default=0.0,
                    help="約定のずれ（bps）。10なら片道0.1％")
    ap.add_argument("--fee-bps", type=float, default=0.0,
                    help="手数料（bps）。主要ネット証券の現物は無料コースあり")
    ap.add_argument("--tax", type=float, default=0.0,
                    help="譲渡益と配当への税率（％）。日本の特定口座なら20.315")
    ap.add_argument("--realistic", action="store_true",
                    help="実運用と同じ条件で回す（1000万・Tier別予算・20銘柄・配当あり）")
    ap.add_argument("--max-names", type=int, default=0,
                    help="同時に持つ最大銘柄数。未指定なら通常15・実運用モードで20")
    ap.add_argument("--refetch", action="store_true", help="キャッシュを無視して取り直す")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    if CACHE.exists() and not args.refetch:
        log.info("キャッシュから読み込みます（取り直すなら --refetch）")
        store = pd.read_pickle(CACHE)
    else:
        key = os.environ.get("J_QUANTS_API_KEY")
        if not key:
            log.error("J_QUANTS_API_KEY が設定されていません")
            return 2
        store = fetch_all(JQ(key), args.years, args.limit,
                          scale_filter=(args.universe == "core"))
        store["universe_mode"] = args.universe
        pd.to_pickle(store, CACHE)
        log.info("キャッシュに保存しました: %s", CACHE)

    # 未指定のときだけ既定値を入れる。
    # ここで無条件に上書きすると、実運用モードで銘柄数を変えても効かなくなる。
    if args.capital <= 0:
        args.capital = 10_000_000 if args.realistic else 3_000_000
    if args.max_names <= 0:
        args.max_names = 20 if args.realistic else 15
    if args.realistic:
        log.info("実運用モード：元本%s万・Tier別予算・最大%d銘柄・配当あり",
                 f"{int(args.capital/10000):,}", args.max_names)

    # ── Tier判定の診断 ──
    if args.diagnose_tier:
        uni = store["universe"]
        print(f"\n■ Tier判定の内訳（対象 {len(uni)}銘柄）\n")

        # 1. 規模区分の値が期待どおり入っているか
        from collections import Counter
        sc = Counter((u.get("scale") or "(空)") for u in uni)
        print("【規模区分（ScaleCat）の内訳】")
        for k, v in sc.most_common():
            print(f"  {k!r:<24}{v:>5}銘柄")

        # 2. 業種コードが入っているか
        sec = Counter(1 if (u.get("sector") or "") else 0 for u in uni)
        print(f"\n【業種名（S33Nm）】 入っている {sec.get(1,0)}銘柄 / "
              f"空 {sec.get(0,0)}銘柄")
        sample = [u for u in uni if u.get("sector")][:3]
        if sample:
            print("  例: " + ", ".join(f"{u['code']}→{u['sector']!r}" for u in sample))

        # 3. 累進配当リストがユニバースに何社いるか
        codes = {u["code"] for u in uni}
        prog_in = sorted((PROGRESSIVE | DOE) & codes)
        print(f"\n【累進配当・DOE】 リスト{len(PROGRESSIVE | DOE)}社中、"
              f"対象に {len(prog_in)}社")
        print("  " + ", ".join(prog_in[:20]) + (" …" if len(prog_in) > 20 else ""))

        # 4. 業界首位級の判定結果
        last_price = {}
        for code, qrows in store["quotes"].items():
            px = quotes_to_df(qrows)
            if not px.empty:
                last_price[code] = float(px["close"].iloc[-1])
        core30 = {u["code"] for u in uni if u.get("scale") == "TOPIX Core30"}
        print(f"\n【業界首位級】")
        print(f"  TOPIX Core30 と判定 … {len(core30)}銘柄")
        mcap = {}
        for u in uni:
            sh = shares_outstanding(store["stmts"].get(u["code"], []))
            px = last_price.get(u["code"])
            if sh and px:
                mcap[u["code"]] = px * sh
        print(f"  時価総額を出せた   … {len(mcap)}銘柄 / {len(uni)}")

        tiers = assign_tiers(store, last_price)
        tc = Counter(tiers.values())
        print(f"\n【Tier判定の結果】 " +
              " / ".join(f"{k} {tc.get(k,0)}銘柄" for k in ("S", "A", "B")))
        if tc.get("S", 0) == 0:
            print("\n  Sが0社です。次のどれかが原因です。")
            if not core30:
                print("  → 規模区分の文字列が想定と違い、業界首位級を判定できていない。")
            if not mcap:
                print("  → 発行済株式数が取れず、時価総額で業種上位を出せていない。")
            if not prog_in:
                print("  → 累進配当リストの銘柄コードが対象と一致していない。")
            if core30 and prog_in:
                ov = sorted(core30 & (PROGRESSIVE | DOE))
                print(f"  → Core30と累進配当の重なりが {len(ov)}社。")
                if ov:
                    print("    " + ", ".join(ov))
        return 0

    # 古いキャッシュには規模区分が入っていない。
    # 株価を取り直すと20分かかるので、銘柄一覧だけを引き直して補う（数秒）。
    if store.get("universe") and not any(u.get("scale") for u in store["universe"]):
        key = os.environ.get("J_QUANTS_API_KEY")
        if key:
            log.info("キャッシュに規模区分がありません。銘柄一覧だけ取り直します…")
            try:
                info = JQ(key).get("/v2/equities/master", {})
                by_code = {}
                for row in info:
                    by_code[norm_code(row.get("Code", ""))] = (
                        row.get("ScaleCat", ""), row.get("S33Nm", ""))
                n = 0
                for u in store["universe"]:
                    sc, sn = by_code.get(u["code"], ("", ""))
                    if sc:
                        u["scale"] = sc
                        u["sector"] = u.get("sector") or sn
                        n += 1
                log.info("  %d銘柄に規模区分を補いました", n)
                pd.to_pickle(store, CACHE)
            except Exception as e:
                log.warning("銘柄一覧の取り直しに失敗: %s", e)

    mode = store.get("universe_mode", "core")
    if mode != args.universe:
        log.warning("キャッシュは「%s」で作られています。"
                    "「%s」で見たい場合は取り直してください（--refetch）。",
                    mode, args.universe)
    log.info("対象の種類: %s（%d銘柄）", mode, len(store.get("universe", [])))

    log.info("パネルを作成中…")
    panel = build_panel(store, args.years, args.lookback)
    if args.lookback < 24:
        log.warning("利回り分布を%dか月で作っています。"
                    "期間は前に伸びますが、分位の精度は落ちます。"
                    "結果は割り引いて見てください。", args.lookback)
    log.info("判定できる時点: %d件 / 銘柄 %d / 期間 %s〜%s",
             len(panel), panel["code"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    # ── 分位の期間を振って比べる ──
    # 検証期間（--years）は固定したまま分位の長さだけを変えるので、
    # 「順位が動いたのは期間のせいか設定のせいか」を切り分けられる。
    if args.sweep:
        lbs = [int(x) for x in args.sweep.split(",") if x.strip()]
        names = [x.strip() for x in args.only.split(",") if x.strip()] or \
                ["fixed", "rotate_wide", "rotate_gain10"]
        log.info("分位の期間を %s か月で比較します（検証期間は共通）", lbs)

        rows = []
        for lb in lbs:
            pn = build_panel(store, args.years, lb)
            if pn.empty:
                log.warning("分位%dか月：判定できる時点がありません", lb)
                continue
            span = f"{pn['date'].min().date()}〜{pn['date'].max().date()}"
            for nm in names:
                if nm not in VARIANTS:
                    continue
                m = metrics(simulate(pn, VARIANTS[nm], args.capital, args.max_names,
                                     tier_budget=args.realistic,
                                     dividends=args.realistic,
                                     slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                                     tax_rate=args.tax / 100.0))
                if m:
                    rows.append({"分位": lb, "期間": span, "ルール": VARIANTS[nm]["label"],
                                 "年率": m["年率"], "決済率": m["決済率"],
                                 "最大下落": m["最大下落"], "シャープ": m["シャープ"],
                                 "売買": m["売買回数"],
                                 "平均保有銘柄": m["平均保有銘柄"]})
            log.info("  分位 %dか月 完了（%s）", lb, span)

        if not rows:
            print("比較できる結果がありませんでした。")
            return 1
        df = pd.DataFrame(rows)

        spans = df.groupby("分位")["期間"].first()
        aligned = spans.nunique() == 1
        print(f"\n■ 分位の期間を変えたときの比較")
        if aligned:
            print(f"　 検証期間はすべて共通： {spans.iloc[0]}")
        else:
            print("　 **検証期間がそろっていません。比較として成立していません。**")
            for lb, sp in spans.items():
                print(f"　   分位{lb:>3}か月 … {sp}")
            print("　 株価が足りない可能性があります。--refetch で取り直してください。")
        print(f"　 元本 {args.capital:,.0f}円 ／ 最大 {args.max_names}銘柄"
              f"{' ／ 実運用条件' if args.realistic else ''}\n")

        for metric, unit in [("年率", "%"), ("決済率", "%"), ("最大下落", "%"),
                             ("平均保有銘柄", "銘柄")]:
            piv = df.pivot(index="分位", columns="ルール", values=metric)
            print(f"【{metric}】")
            head = "分位      " + "".join(f"{c[:14]:>16}" for c in piv.columns)
            print(head)
            print("-" * len(head))
            for lb, r in piv.iterrows():
                line = f"{lb:>3}か月   " + "".join(f"{v:>15.1f}{unit}" for v in r.values)
                print(line)
            print()

        # 期間の違いで結論が変わるかを判定する
        piv = df.pivot(index="分位", columns="ルール", values="年率")
        spread = float(piv.max().max() - piv.min().min())
        best_by_lb = piv.idxmax(axis=1)
        flipped = best_by_lb.nunique() > 1
        piv_e = df.pivot(index="分位", columns="ルール", values="決済率")
        dup = piv_e.duplicated(keep=False)
        held = df["平均保有銘柄"].mean()
        print("【読み方】")
        print(f"　 平均保有銘柄 … {held:.1f}（上限 {args.max_names}銘柄）")
        if held < args.max_names * 0.6:
            print("　 上限まで届いていません。資金が先に尽きているので、")
            print("　 上限を上げ下げしても結果は変わりません。")
            print("　 銘柄数を変えて試すなら、上限ではなく元本を動かしてください。")
        else:
            print("　 上限が実際に効いています。この銘柄数での結果として読めます。")
        if dup.any():
            same = list(piv_e.index[dup])
            print(f"　 分位 {same} の結果が同一です。")
            print("　 → その長さぶんの株価が無く、同じ範囲を見ている可能性が高い。")
            print("　   --refetch で取り直したうえで、もう一度お試しください。")
        print(f"　 年率の最大と最小の差 … {spread:.1f}ポイント")
        if flipped:
            print("　 分位の長さによって、最も成績の良いルールが入れ替わっています。")
            print("　 → 期間の選び方でルールの優劣が変わるということ。どれかを選ぶ根拠は弱い。")
        else:
            print(f"　 どの期間でも最良は同じルール（{best_by_lb.iloc[0]}）でした。")
        if spread < 3.0:
            print("　 差が小さいため、分位の期間は成績にほとんど影響していません。")
            print("　 → 期間は好みで決めてよい、という結論になります。")
        else:
            print("　 差が大きいので、期間の選択は成績に影響します。")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTDIR / "lookback_sweep.csv", index=False, encoding="utf-8-sig")
        print(f"\n書き出しました: data/lookback_sweep.csv")
        return 0

    if args.min_yield > 0:
        for v in VARIANTS.values():
            v.setdefault("min_yield", args.min_yield)
        log.info("利回り %.1f％ 未満の銘柄は買わない設定で回します", args.min_yield)

    # ── 期間をずらして何度も検証する ──
    # ひとつの期間だけで良く見えるのは、たまたまかもしれない。
    # 窓を1年ずつずらして、どの期間でも同じ結論になるかを確かめる。
    if args.walk > 0:
        nm = [x.strip() for x in args.only.split(",") if x.strip()] or \
             ["current_live", "live15_holdS", "live15_holdall"]
        d0_, d1_ = panel["date"].min(), panel["date"].max()
        wins = []
        st_ = d1_ - pd.DateOffset(years=args.walk)
        while st_ >= d0_:
            wins.append((st_, st_ + pd.DateOffset(years=args.walk)))
            st_ = st_ - pd.DateOffset(years=1)
        wins.reverse()
        if not wins:
            print("窓を作れません。--walk を短くしてください。")
            return 1
        log.info("%d年の窓を %d通り試します", args.walk, len(wins))

        rows_ = []
        for a, b in wins:
            sub = panel[(panel["date"] >= a) & (panel["date"] <= b)]
            if sub["date"].nunique() < args.walk * 10:
                continue
            bh_ = buy_and_hold(sub, args.tax / 100.0)
            for n_ in nm:
                if n_ not in VARIANTS:
                    continue
                m_ = metrics(simulate(sub, VARIANTS[n_], args.capital, args.max_names,
                                      tier_budget=args.realistic,
                                      dividends=args.realistic,
                                      slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                                      tax_rate=args.tax / 100.0))
                if m_:
                    rows_.append({"窓": f"{a.date()}〜{b.date()}",
                                  "ルール": VARIANTS[n_]["label"],
                                  "年率": m_["年率"], "最大下落": m_["最大下落"]})
            if bh_:
                rows_.append({"窓": f"{a.date()}〜{b.date()}",
                              "ルール": "（基準）全部買って放置",
                              "年率": bh_["年率"], "最大下落": bh_["最大下落"]})
            log.info("  %s 完了", a.date())

        if not rows_:
            print("結果がありません。")
            return 1
        wf = pd.DataFrame(rows_)
        for metric in ("年率", "最大下落"):
            piv = wf.pivot(index="窓", columns="ルール", values=metric)
            print(f"\n【{metric}】（{args.walk}年の窓を1年ずつずらして）\n")
            head = "窓                     " + "".join(f"{c[:16]:>18}" for c in piv.columns)
            print(head)
            print("-" * min(len(head), 160))
            for w_, r_ in piv.iterrows():
                print(f"{w_:<23}" + "".join(f"{v:>17.1f}%" for v in r_.values))
            print()

        piv = wf.pivot(index="窓", columns="ルール", values="年率")
        base_col = next((c for c in piv.columns if c.startswith("（基準）")), None)
        print("【読み方】")
        if base_col is not None:
            for c in piv.columns:
                if c == base_col:
                    continue
                win = (piv[c] > piv[base_col]).sum()
                print(f"  {c[:28]:<30}基準を上回った窓 … {win}/{len(piv)}")
            print("\n  すべての窓で上回っていれば、期間に依存しない優位と言えます。")
            print("  半分程度なら、たまたま良い期間があっただけかもしれません。")
        best = piv.drop(columns=[base_col] if base_col else []).idxmax(axis=1)
        print(f"\n  窓ごとの最良ルール … {best.nunique()}種類")
        if best.nunique() == 1:
            print(f"  どの窓でも同じルールが最良でした（{best.iloc[0]}）。")
        else:
            print("  窓によって最良のルールが入れ替わっています。")
            print("  → ひとつを選ぶ根拠は弱いということです。")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        wf.to_csv(OUTDIR / "walk_forward.csv", index=False, encoding="utf-8-sig")
        print(f"\n書き出しました: data/walk_forward.csv")
        return 0

    names = [x.strip() for x in args.only.split(",") if x.strip()] or list(VARIANTS)
    rows = []
    for name in names:
        if name not in VARIANTS:
            log.warning("未定義のルール: %s", name)
            continue
        cfg = VARIANTS[name]
        res = simulate(panel, cfg, args.capital, args.max_names,
                       tier_budget=args.realistic, dividends=args.realistic,
                       slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                       tax_rate=args.tax / 100.0)
        m = metrics(res)
        if m:
            rows.append({"ルール": cfg["label"], **m})
            log.info("  %s 完了", cfg["label"])

    bh = buy_and_hold(panel, args.tax / 100.0)


    if not rows:
        print("結果がありません。")
        return 1

    df = pd.DataFrame(rows).sort_values("年率", ascending=False)
    print(f"\n■ 比較結果（{panel['date'].min().date()} 〜 "
          f"{panel['date'].max().date()} / 元本 {args.capital:,.0f}円）\n")
    print(f"{'ルール':<26}{'総':>8}{'年率':>7}{'最大下落':>9}{'シャープ':>9}"
          f"{'売買':>6}{'決済率':>8}{'保有月数':>9}")
    print("-" * 84)
    for _, x in df.iterrows():
        hold = "－" if pd.isna(x["平均保有月数"]) else f"{x['平均保有月数']:.1f}"
        print(f"{x['ルール']:<26}{x['総リターン']:>7.1f}%{x['年率']:>6.1f}%"
              f"{x['最大下落']:>8.1f}%{x['シャープ']:>9.2f}"
              f"{x['売買回数']:>6.0f}{x['決済率']:>7.0f}%{hold:>9}")
    # ── Tier別の内訳（1つ目のルールについて）──
    first = VARIANTS[names[0]]
    d0 = simulate(panel, first, args.capital, args.max_names,
                  tier_budget=args.realistic, dividends=args.realistic,
                  slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                  tax_rate=args.tax / 100.0)["diag"]
    # 建玉の明細。どの銘柄をいつ買って、いくらまで伸びたかを見る。
    if args.show_trades:
        tl = d0.get("trade_log", [])
        if tl:
            print(f"\n■ 建玉の明細（{first['label']} の場合・最大40件）\n")
            print(f"{'Tier':<6}{'コード':<8}{'銘柄':<18}{'買った月':<12}"
                  f"{'取得単価':>10}{'到達益':>9}{'結果':>9}")
            print("-" * 74)
            for x in tl[:40]:
                print(f"{x['tier']:<6}{x['code']:<8}{str(x['name'])[:16]:<18}"
                      f"{x['date']:<12}{x['price']:>10,.0f}{x['peak_gain']*100:>8.1f}%"
                      f"{x['result']:>9}")

    tp = d0.get("tier_peak", [])
    if tp:
        print(f"\n■ Tier別の内訳（{first['label']} の場合）\n")
        print(f"{'Tier':<6}{'建玉数':>7}{'平均の到達益':>13}"
              f"{'+10%到達':>10}{'+20%到達':>10}{'+30%到達':>10}{'+40%到達':>10}")
        print("-" * 70)
        for t in ("S", "A", "B"):
            g = [x for tt, x in tp if tt == t]
            if not g:
                print(f"{t:<6}{0:>7}{'—':>13}{'—':>10}{'—':>10}{'—':>10}{'—':>10}")
                continue
            n = len(g)
            print(f"{t:<6}{n:>7}{np.mean(g)*100:>12.1f}%"
                  f"{sum(1 for x in g if x >= 0.10)/n*100:>9.0f}%"
                  f"{sum(1 for x in g if x >= 0.20)/n*100:>9.0f}%"
                  f"{sum(1 for x in g if x >= 0.30)/n*100:>9.0f}%"
                  f"{sum(1 for x in g if x >= 0.40)/n*100:>9.0f}%")
        print("\n  到達益＝建てたあと、含み益が最大でどこまで伸びたか。")
        print("  ここが利確ラインに届いていなければ、その設定は結果に効かない。")
        # ユニバース全体のTier構成も出す
        if "tier" in panel.columns:
            comp = panel.groupby("code")["tier"].first().value_counts()
            print("\n  対象銘柄のTier構成： " +
                  " / ".join(f"{k} {v}銘柄" for k, v in comp.items()))

    if args.tax > 0 or args.slip_bps > 0 or args.fee_bps > 0:
        print(f"\n■ 費用の内訳（元本 {args.capital:,.0f}円に対して）\n")
        print(f"{'ルール':<30}{'手数料+ずれ':>14}{'税金':>13}{'合計':>13}{'元本比':>9}")
        print("-" * 80)
        for _, x in df.iterrows():
            tot = x["費用"] + x["税金"]
            print(f"{x['ルール'][:28]:<30}{x['費用']:>13,.0f}円{x['税金']:>12,.0f}円"
                  f"{tot:>12,.0f}円{tot/args.capital*100:>8.1f}%")
        print(f"\n  条件： 約定のずれ 片道{args.slip_bps:.0f}bps ／ "
              f"手数料 片道{args.fee_bps:.0f}bps ／ 税率{args.tax:.3f}％")
        print("  ※ 税金は譲渡益と配当にかかります。損は繰り越して相殺しています。")

    if bh:
        print(f"\n■ 比較の基準：対象{panel['code'].nunique()}銘柄を等金額で買って持ち続けた場合\n")
        print(f"  総リターン {bh['総リターン']:>7.1f}%   年率 {bh['年率']:>5.1f}%   "
              f"最大下落 {bh['最大下落']:>5.1f}%   シャープ {bh['シャープ']:.2f}")
        best = df["年率"].max()
        diff = best - bh["年率"]
        print(f"\n  最良のルールとの差 … {diff:+.1f}ポイント")
        if diff < 1.0:
            print("  ルールで選んでも、全部買って持つのと変わりません。")
            print("  → この期間の成績は、銘柄選択ではなく相場そのものによるものです。")
        elif diff < 3.0:
            print("  差はわずかです。銘柄選択の効果は限定的とみるべきです。")
        else:
            print("  基準を明確に上回っています。銘柄選択に意味があったと言えます。")
        print("  ※ 基準は初日に等金額で買って放置した場合。配当は課税後で加算しています。")

    cut = d0.get("cut_exits", 0) if "d0" in dir() else 0
    print("\n  決済率＝買った建玉のうち実際に売れた割合。低いほど「出口が来ない」状態。")
    print("  保有月数＝売れたものの平均保有期間。")

    sat = df["枠飽和率"].mean()
    print(f"\n  枠飽和率 {sat:.0f}%（保有が上限 {args.max_names}銘柄 に達していた月の割合）")
    if sat > 60:
        print("  → 枠が常に埋まっているため、買いの閾値を緩めても結果はほとんど変わりません。")
        print("     この状態では「いつ買うか」より「どれを優先するか」が効きます。")
        print("     --max-names を増やすか、資金に対して銘柄数を絞ってお試しください。")
    zero = df[(df["判定変化"] == 0) & (df["ルール"].str.contains("可変"))]
    if len(zero):
        print("  → 可変ルールが一度も判定を変えていません。条件が厳しすぎる可能性があります。")

    df.to_csv(OUTDIR / "yield_backtest.csv", index=False, encoding="utf-8-sig")
    with (OUTDIR / "yield_backtest.json").open("w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
                   "years": args.years, "results": df.to_dict("records")},
                  f, ensure_ascii=False, indent=2)
    print(f"\n書き出しました: data/yield_backtest.csv, data/yield_backtest.json")
    print("\n※ 過去の成績であり、将来の結果を保証するものではありません。")
    if args.tax > 0 or args.slip_bps > 0 or args.fee_bps > 0:
        print("※ 手数料・税金・約定のずれを含めた数字です。")
    else:
        print("※ 手数料・税金・約定のずれは含めていません。実際の成績はこれより下がります。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

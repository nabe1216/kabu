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
import re
import logging
import os
import io
import sys
import time
import unicodedata
import zlib
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
# 時間の上限への備え
#   検証のワークフローは180分で強制終了される。強制終了だと、
#   取り直し途中の株価も計算の結果も、何も残らない（2026年10月8日に起きた）。
#   そうなる前に自分で止まり、取った分を保存して、次の実行で続きから取る。
#   分はこのプログラムが動き始めてからの時間。前後の準備と保存に10分ほど見ておく。
# ══════════════════════════════════════════
_T0 = time.monotonic()
FETCH_BUDGET_MIN = float(os.environ.get("FETCH_BUDGET_MIN", "130"))
# これより遅く取り終えたら、計算は次の実行に回す。
# 計算そのものは1,600社×9年の合成データで3分ほど（通常の比較2.5分・窓ずらし3分）。
COMPUTE_START_LIMIT_MIN = float(os.environ.get("COMPUTE_START_LIMIT_MIN", "140"))
FETCH_SAVE_EVERY_MIN = 15.0      # 取得の途中もこの間隔で保存する
EXIT_UNFINISHED = 3              # 途中で止めたときの終了コード（緑にしないため）


def _elapsed_min() -> float:
    return (time.monotonic() - _T0) / 60.0


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

    # ── 市場全体の状態で買う量を変える ──
    # 市場の利回り中央値が過去3年平均より高い＝全体が安い、と判断して厚く買う。
    # 逆に低いときは薄く買い、現金を残す。
    "regime_mild": {
        "label": "市場が安いとき厚く（1.5倍/0.7倍）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "hold_tiers": ["S"],
        "regime_scale": {"cheap": 1.5, "rich": 0.7,
                         "cheap_z": 0.5, "rich_z": -0.5},
    },
    "regime_strong": {
        "label": "市場が安いとき厚く（2.5倍/0.3倍）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "hold_tiers": ["S"],
        "regime_scale": {"cheap": 2.5, "rich": 0.3,
                         "cheap_z": 0.5, "rich_z": -0.5},
    },
    "regime_wait": {
        "label": "安いときだけ買う（3倍/買わない）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "hold_tiers": ["S"],
        "regime_scale": {"cheap": 3.0, "rich": 0.0,
                         "cheap_z": 0.3, "rich_z": 0.0},
    },

    # ── TOPIX の下落率で買う量を変える ──
    # 「高値から15％下げたら暴落」という素直な定義。
    # 利回り中央値より直感的で、指数だけ見れば判断できる。
    "topix_dd15": {
        "label": "TOPIXが高値から15％安で厚く（2倍/0.7倍）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "hold_tiers": ["S"],
        "regime_scale": {"use": "topix", "cheap": 2.0, "rich": 0.7,
                         "cheap_dd": -15.0, "rich_dd": -3.0},
    },
    "topix_dd10": {
        "label": "TOPIXが高値から10％安で厚く（1.5倍/0.8倍）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "hold_tiers": ["S"],
        "regime_scale": {"use": "topix", "cheap": 1.5, "rich": 0.8,
                         "cheap_dd": -10.0, "rich_dd": -3.0},
    },

    # ── 買い増しを許す ──
    # さらに下がったところで買い増すと平均取得単価が下がるが、
    # 1銘柄への比重が増える。分散と取得単価のどちらを取るか。
    "holdS_addon": {
        "label": "Sは持つ＋買い増しを許す（最大3回）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "add_on": True, "add_on_max": 3,
    },

    # ── Tier の重みを変える ──
    "holdS_wS3": {
        "label": "Sは持つ＋Sを厚く（S3.0/A1.5/B1.0）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 3.0, "A": 1.5, "B": 1.0},
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "holdS_wFlat": {
        "label": "Sは持つ＋重みなし（均等）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 1.0, "A": 1.0, "B": 1.0},
        "hold_tiers": ["S"], "exit_on_cut": True,
    },

    # ── 財務で選ぶ。探索で高配当を上回った3つを、本番と同じ条件で確かめる ──
    # 買う条件は「全銘柄の中で安い側20％」。並べ方は安い順。
    # 売り方・予算・減配撤退は本番と同じ。利回りの足切りは総当たりの欄で 0 と 4 を振る
    # （4 のときは「利回り4％以上の中で PBR が低い順」＝いまのルールとの組み合わせになる）。
    "pbr_low": {
        "label": "低PBR順に買う（安い側20％から）",
        "entry": [80], "exit": [], "measure": "pbr",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "pbr_low_tier": {
        "label": "低PBR順に買う（Tier優先）",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "per_low": {
        "label": "低PER順に買う（安い側20％から）",
        "entry": [80], "exit": [], "measure": "per",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "small_low": {
        "label": "時価総額の小さい順に買う（小さい側20％から）",
        "entry": [80], "exit": [], "measure": "size",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },

    # ── A. スクリーニング8条件は効いているか ──
    # 検証ツールにはこれまで利回りの足切りしか入っていなかった。
    # 本番と同じ8条件を入れて、入れない場合と比べる。
    "xs_any_tier_scr": {
        "label": "利回りの高い順（Tier優先）＋8条件",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "pbr_low_tier_scr": {
        "label": "低PBR順（Tier優先）＋8条件",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier", "screen": True,
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    # ── C. 低PBR順の結果は、細かい条件で崩れないか ──
    "pbr_low_tier_10": {
        "label": "低PBR順（Tier優先）安い側10％から",
        "entry": [90], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "pbr_low_tier_30": {
        "label": "低PBR順（Tier優先）安い側30％から",
        "entry": [70], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "value_low_tier": {
        "label": "PBR＋PERの割安順（Tier優先）",
        "entry": [80], "exit": [], "measure": "value", "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "pbr_low_tier_slip30": {
        "label": "低PBR順（Tier優先）約定のずれ30bps",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier", "slip_bps": 30,
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    # ── 低PBR順に変えたことで、前提が変わった部分の確かめ直し ──
    # 本番は「減配」と「業績急変（営業利益−20％）」の両方で売る。
    # pbr_low_tier は減配だけなので、本番と同じ売り方の版を基準にする。
    "pbr_live": {
        "label": "低PBR順・本番と同じ売り方（減配＋業績急変）",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "pbr_nocut": {
        "label": "低PBR順・減配でも業績急変でも売らない",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"],
    },
    "pbr_live_t10": {
        "label": "低PBR順・本番の売り方・目標10銘柄",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 10,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "pbr_live_t20": {
        "label": "低PBR順・本番の売り方・目標20銘柄",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 20,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },

    # ── 足切りどうしを、本番のルールで直接比べる ──
    "pbr_live_y25": {
        "label": "低PBR順・本番の売り方・足切り2.5％",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 2.5,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "pbr_live_y30": {
        "label": "低PBR順・本番の売り方・足切り3.0％",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "pbr_live_y35": {
        "label": "低PBR順・本番の売り方・足切り3.5％",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.5,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "pbr_live_y40": {
        "label": "低PBR順・本番の売り方・足切り4.0％",
        "entry": [80], "exit": [], "measure": "pbr", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 4.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },

    # ── 割安と稼ぐ力を合わせる（35年のデータで最も安定していた考え方）──
    "pbrroe70_live": {
        "label": "割安70＋稼ぐ力30の順・本番の売り方・足切り3％",
        "entry": [80], "exit": [], "measure": "pbrroe70", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "pbrroe50_live": {
        "label": "割安50＋稼ぐ力50の順・本番の売り方・足切り3％",
        "entry": [80], "exit": [], "measure": "pbrroe50", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "pbrroe30_live": {
        "label": "割安30＋稼ぐ力70の順・本番の売り方・足切り3％",
        "entry": [80], "exit": [], "measure": "pbrroe30", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    # ── 利回り順（9月26日の本番）を、本番の売り方・足切り3％で ──
    # 株式分割の調整を入れたあとで、9月27日の「低PBR順に切り替える」判断を確かめ直すための相手。
    # 利回りは配当も株価も分割調整済みなので、もともと分割の歪みを受けていない。
    "yield_live_y30": {
        "label": "利回りの高い順・本番の売り方・足切り3％",
        "entry": [0], "exit": [], "priority": "tier", "cross": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    # ── 本番と同じく、8条件を通った銘柄だけから買う版（2026年10月9日に追加）──
    # 上の比較（pbr_live_y30・pbrroe50_live・yield_live_y30 など）は、売り方は本番と同じだが、
    # 買う対象に8条件のスクリーニングをかけていなかった（全銘柄から選んでいた）。
    # 本番の portfolio_engine は screening_pass を通った銘柄からしか買わないので、同じ形で確かめ直す。
    # 名前の先頭を変えてあるのは、総当たりの勝敗表の見出し（先頭10文字）で見分けられるようにするため。
    "yield_live_y30_scr": {
        "label": "利回り順・足切り3％・8条件あり・本番の売り方",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "yield_live_y40_scr": {
        "label": "利回り順・足切り4％・8条件あり・本番の売り方",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 4.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    "pbrroe50_live_scr": {
        "label": "割安50＋稼ぐ力50・足切り3％・8条件あり・本番の売り方",
        "entry": [80], "exit": [], "measure": "pbrroe50", "priority": "tier", "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },
    # ── うねりで細かく売買する（2026年10月9日に追加）──
    # 土台は yield_live_y30_scr（利回り順・8条件・足切り3％・緊急時だけ売る）。
    # そこに「レンジの中にいて上限近くまで来たら、その値段で降りる」を足す。
    #   ・取得単価より上のときだけ降りる（安値で投げない）
    #   ・降りた分の現金は、その月末に通常の順番（Tier→利回り）で買い直す。
    #     同じ銘柄がまだ候補の上位なら、月末の値段で買い戻すことになる。
    #   ・レンジの判定は「レンジの設定」の欄（既定 60日・値幅8〜20％・上限−2％で降りる）
    # 以前の swing_full・swing_B_only は、9月以前の Q75 を土台にしていた。
    # 本番の売らない設定（hold_tiers）は、レンジで降りる処理より前で止めてしまうため、
    # 往復させる Tier は hold_tiers から外す。
    "swing_all_y30_scr": {
        "label": "うねり・全Tier・足切り3％・8条件あり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "exit_on_cut": True, "exit_on_op": True,
        "range_exit": {"fraction": 1.0, "min_gain": 0.0},
    },
    "swing_B_y30_scr": {
        "label": "うねり・Bだけ・足切り3％・8条件あり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A"], "exit_on_cut": True, "exit_on_op": True,
        "range_exit": {"fraction": 1.0, "min_gain": 0.0},
    },
    # ── 売りルールの確かめ直し（2026年10月11日に追加）──
    # 土台は利回り順・足切り3％・8条件。3つとも、緊急撤退の判定を本番と同じにし
    # （実績の年間配当が前の通期より10％以上減る／営業利益が前の通期より20％以上減る）、
    # その銘柄は買い候補から外す（本番の portfolio_engine と同じ）。違うのは「持っている銘柄を売るか」だけ。
    #   sell_live_scr … 本番どおり。減配でも業績急変でも売る
    #   sell_cut_scr  … 減配のときだけ売る（業績急変では売らない）
    #   sell_none_scr … 売らない
    # 上の yield_live_y30_scr は以前の判定（予想配当が少しでも下がったら減配・業績急変はほぼ働かない・
    # 買い候補から外さない）。以前の結果と照らし合わせるために残してある。
    "sell_live_scr": {
        "label": "売り：本番どおり・利回り順・3％・8条件",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
    },
    "sell_cut_scr": {
        "label": "売り：減配だけ・利回り順・3％・8条件",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": False,
        "emg_rule": "actual", "skip_emg_buy": True,
    },
    "sell_none_scr": {
        "label": "売り：売らない・利回り順・3％・8条件",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": False, "exit_on_op": False,
        "emg_rule": "actual", "skip_emg_buy": True,
    },
    # ── 価格で損切りする（2026年10月11日に追加）──
    # 土台は sell_live_scr（本番どおり：利回り順・3％・8条件・減配か業績急変で売り、その銘柄は買わない）。
    # そこに「平均取得単価から一定率下がったら全部売る」を足す。
    #   ・判定は日々の終値。最初に線を割った日の終値で売る（約定のずれは片道の設定どおり）
    #   ・損切りした銘柄は12か月買い直さない。売ったお金は、その月末に通常の順番で別の銘柄を買う
    #   ・損切りで出た損は配当とも相殺する（特定口座で配当を口座で受け取る前提）。
    #     比べる相手の stop_none_scr も同じ扱いにしてある。違いは損切りの有無だけ
    # sell_live_scr は、前回の結果（平均21.5％）が再現できるかの確認用に並べる。
    "stop_none_scr": {
        "label": "損切りなし・本番どおり・損は配当と相殺",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
    },
    "stop10_scr": {
        "label": "損切り10％・12か月買い直さない・本番どおり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
        "stop_loss": 0.10, "stop_cooldown": 12,
    },
    "stop20_scr": {
        "label": "損切り20％・12か月買い直さない・本番どおり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
        "stop_loss": 0.20, "stop_cooldown": 12,
    },
    "stop30_scr": {
        "label": "損切り30％・12か月買い直さない・本番どおり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
        "stop_loss": 0.30, "stop_cooldown": 12,
    },
    # ── 落ち方で売り方を変える（2026年10月11日に追加）──
    # 土台は sell_live_scr（本番どおり：実績の配当−10％以上か営業利益−20％以上で売る）。
    # 変えるのは「何をもって減配・業績急変とみなすか」だけ。買い候補から外す判定も同じ基準に揃える。
    #   emg_deep_scr   … 深いときだけ売る（配当−30％以上か赤字転落）。軽い減配・減益は持ち続ける
    #   emg_both_scr   … 減配と業績急変が重なったときだけ売る（片方だけなら持ち続ける）
    #   emg_fc10_scr   … 本番の判定に加えて、予想の配当が実績より10％以上低くなった時点でも売る
    #                    （通期の決算を待たない。本番より早く動く）
    #   emg_fcany_scr  … 予想の配当が実績を少しでも（1％以上）下回ったら売る（いちばん敏感な形）
    "emg_deep_scr": {
        "label": "深いときだけ売る（配当−30％か赤字転落）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
        "emg_cut_thr": 0.30, "emg_op_thr": 1.0,
    },
    "emg_both_scr": {
        "label": "減配と業績急変が重なったときだけ売る",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
        "emg_require_both": True,
    },
    "emg_fc10_scr": {
        "label": "予想の減配（−10％）でも売る・本番の判定も残す",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
        "emg_forecast": 0.10,
    },
    "emg_fcany_scr": {
        "label": "予想が少しでも下がったら売る（−1％）・本番の判定も残す",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
        "emg_forecast": 0.01,
    },
    # ── 上がった銘柄を売って、売買を増やす（2026年10月11日に追加）──
    # 土台は sell_live_scr（本番どおり）。減配・業績急変の売りはそのまま残し、その上に足す。
    # 売ったお金は、その月末に通常の順番（Tier→利回り）で別の銘柄を買う。
    #   act_y25_scr   … 利回りが2.5％を割ったら売る（株価が上がって割高になった銘柄の乗り換え）
    #   act_y20_scr   … 同じく2.0％（売る回数は少なめ）
    #   act_tp50_scr  … 取得単価から+50％で全部売る。12か月は買い直さない
    #   act_tp30h_scr … +30％で半分売る。残りは持ち続け、そこからさらに+30％でまた半分
    "act_y25_scr": {
        "label": "利回り2.5％割れで売る・乗り換え・本番どおり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
        "yield_exit": 2.5,
    },
    "act_y20_scr": {
        "label": "利回り2.0％割れで売る・乗り換え・本番どおり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
        "yield_exit": 2.0,
    },
    "act_tp50_scr": {
        "label": "利確+50％で全部・12か月買い直さない・本番どおり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
        "tp_gain": 0.50, "tp_frac": 1.0, "act_cooldown": 12,
    },
    "act_tp30h_scr": {
        "label": "利確+30％で半分・残りは持つ・本番どおり",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True,
        "tp_gain": 0.30, "tp_frac": 0.5,
    },
    # ── 売らずに動く・税金を減らしながら動く（2026年10月11日に追加）──
    # 土台は本番どおり。どれも「損は配当とも相殺する」（特定口座で配当を口座で受け取る前提）。
    # 比べる相手の base_tax_scr も同じ扱い（中身は stop_none_scr と同じ。名前だけ分かりやすくした）。
    #   harvest_scr   … 12月末に含み損の銘柄を全部売り、同じ株数をすぐ買い戻す（年末の損出し）。
    #                   含み損が2％未満の銘柄は売らない。約定のずれ（片道0.1％）で往復0.2％かかるので、
    #                   含み損が1％程度だと、戻る税金（損の20.315％）より費用の方が大きくなるため
    #   harvest10_scr … 同じく、含み損が10％以上の銘柄だけ（売買の回数を抑える）
    #   topup_scr     … 毎月、買いのあとに残った現金で、目標を下回っている持ち株を買い増す
    #   combo_scr     … 買い増しと12月の損出しの両方
    "base_tax_scr": {
        "label": "本番どおり（損は配当と相殺）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
    },
    "harvest_scr": {
        "label": "12月に含み損を売って買い戻す（損出し・−2％以上）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
        "harvest": {"month": 12, "min_loss": 0.02},
    },
    "harvest10_scr": {
        "label": "12月に含み損10％以上を売って買い戻す",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
        "harvest": {"month": 12, "min_loss": 0.10},
    },
    "topup_scr": {
        "label": "配当で比率の下がった持ち株を買い増す",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
        "topup": True,
    },
    "combo_scr": {
        "label": "配当で買い増し＋12月の損出し",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "tax_div_offset": True,
        "topup": True, "harvest": {"month": 12, "min_loss": 0.02},
    },
    # ── 配当の権利確定月の前に、買いの条件をゆるめる（2026年10月11日に追加）──
    # 土台は本番どおり。どれも配当を受け取る月を正しくしてある（div_split。期末だけの会社は期末に全額）。
    # 比べる相手の base_div_scr も同じ扱い。
    # 次の月末に配当の権利が確定する銘柄に限って（＝権利確定月に入ってすぐ買う形）、
    #   rec_y25_scr   … 利回りの足切りを3％→2.5％にゆるめる（8条件は必須のまま）
    #   rec_miss1_scr … 8条件のうち1つ満たしていなくても買う（利回り3％は必須のまま）
    #   rec_both_scr  … 両方ゆるめる
    #   rec_first_scr … 条件はそのまま。同じ Tier の中で、こうした銘柄を先に買う
    "base_div_scr": {
        "label": "本番どおり（配当の時期を正確に）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
    },
    "rec_y25_scr": {
        "label": "権利月の前は利回り2.5％まで買う",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "rec_relax": {"yield": 2.5, "miss": 0},
    },
    "rec_miss1_scr": {
        "label": "権利月の前は8条件の1つ欠けまで買う",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "rec_relax": {"yield": None, "miss": 1},
    },
    "rec_both_scr": {
        "label": "権利月の前は両方ゆるめる（2.5％・1つ欠け）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "rec_relax": {"yield": 2.5, "miss": 1},
    },
    # ── 8条件を1つずつ外す（2026年10月11日に追加）──
    # 土台は本番どおり（配当の時期を正確に）。利回り3％の足切りはそのまま。
    # 外した条件が「効いている」なら本番がその形に勝ち、「飾り」なら差が出ない。
    "drop_div_scr": {
        "label": "外す：10期減配なし",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "screen_drop": ["div"],
    },
    "drop_sales_scr": {
        "label": "外す：売上の安定",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "screen_drop": ["sales"],
    },
    "drop_op_scr": {
        "label": "外す：営業利益の安定",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "screen_drop": ["op"],
    },
    "drop_np_scr": {
        "label": "外す：純利益の安定",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "screen_drop": ["np"],
    },
    "drop_pay_scr": {
        "label": "外す：配当性向50％以下",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "screen_drop": ["pay"],
    },
    "drop_eq_scr": {
        "label": "外す：自己資本比率",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "screen_drop": ["eq"],
    },
    "drop_val_scr": {
        "label": "外す：PER×PBR40以下",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "screen_drop": ["val"],
    },
    "drop_all_scr": {
        "label": "外す：利回り以外の7つ全部",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "screen_drop": "all",
    },
    # ── 2回目の候補（2026年10月11日に追加）──
    # 土台は本番どおり（配当の時期を正確に）。変えるのは1か所だけ。
    #   tier_none_scr   … Tier をやめる。順番は利回りの高い順だけ、予算は全銘柄同じ（総資産÷15）
    #   tier_order_scr  … Tier の順番だけやめる（利回りの高い順）。予算の重み（S2・A1.5・B1）は残す
    #   sector3_scr     … 同じ業種（33業種）は3銘柄まで
    #   ycap6_scr       … 利回り6％超は買わない
    #   exfirst_scr     … 同じ Tier の中で、権利落ちの直後の銘柄を先に買う
    "tier_none_scr": {
        "label": "Tierなし・利回り順・予算は同じ",
        "entry": [0], "exit": [], "priority": "pct", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "tier_weight": {"S": 1.0, "A": 1.0, "B": 1.0},
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
    },
    "tier_order_scr": {
        "label": "Tierの順番なし・予算の重みは残す",
        "entry": [0], "exit": [], "priority": "pct", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
    },
    "sector3_scr": {
        "label": "同じ業種は3銘柄まで",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "max_per_sector": 3,
    },
    "ycap6_scr": {
        "label": "利回り6％超は買わない",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "max_yield": 6.0,
    },
    "exfirst_scr": {
        "label": "権利落ち直後の銘柄を先に買う",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "ex_first": True,
    },
    # ── 3回目の候補（2026年10月11日に追加）──
    # 土台は本番どおり（配当の時期を正確に）。変えるのは1か所だけ。
    #   names10_scr  … 目標10銘柄（1銘柄の予算が厚くなる。集中）
    #   names25_scr  … 目標25銘柄（保有の上限は30。分散）
    #   cash20_scr   … 総資産の2割は常に現金で残す（売りはしないので、買いを控えて守る形）
    #   tier_sa_scr  … S と A だけを買う（B は買わない）。2回目で Tier の順番が効いていたため
    "names10_scr": {
        "label": "目標10銘柄（集中）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 10, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
    },
    "names25_scr": {
        "label": "目標25銘柄（分散・上限30）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 25, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "max_names": 30,
    },
    "cash20_scr": {
        "label": "現金を常に2割残す",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "cash_floor": 0.20,
    },
    "tier_sa_scr": {
        "label": "SとAだけ買う（Bは買わない）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tiers_allowed": ["S", "A"],
    },
    # ── 5回目の候補（2026年10月12日に追加）：Tier の後出しを外す ──
    #   本番の Tier は「いまの時価総額・いまの Core30・いまの累進配当/DOE のリスト」で決めている。
    #   これを過去の全期間に当てはめると、あとで大きくなった会社や、あとで累進配当を宣言した会社を
    #   最初から知っていたことになる。add_pit_tiers がその月末の情報だけで判定し直した列を使う。
    "tier_pit_mix_scr": {
        "label": "首位級だけその時点で判定（累進配当はいまのリスト）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
    },
    "tier_pit_lead_scr": {
        "label": "後出しなし（その時点の首位級だけでTier）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_lead",
    },
    # ── 7回目の候補（2026年10月12日に追加）：基準は「後出しなし（tier_pit_lead_scr）」 ──
    #   drop_skip_pit     … 株価の下落だけで利回りが上がった銘柄は買わない
    #                       （12か月で株価が20％以上下がり、1株配当は増えていない銘柄）
    #   tier_pit_grow_scr … 累進配当・DOEのリスト（宣言の時期が分からず後出しになる）の代わりに、
    #                       その時点までの増配の実績で Tier を決める（grow_timeline：直近3回の比べで
    #                       2回以上増配・減配なし）。S＝首位級かつ増配、A＝どちらか、B＝それ以外
    "drop_skip_pit": {
        "label": "株価の下落だけで利回りが上がった銘柄は買わない",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_lead", "skip_price_drop": 0.20,
    },
    "tier_pit_grow_scr": {
        "label": "後出しなし（首位級＋増配の実績でTier）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_grow",
    },
    # ── 8回目の候補（2026年10月12日に追加）：S は持ち続け、A・B はうねりの中で売り買いする ──
    #   相手は「首位級だけその時点で判定（累進配当はいまのリスト）」（tier_pit_mix_scr。全部持ち続ける）。
    #   A・B の銘柄がうねりの中にいれば、上の線で売り、下の線に戻るまで買い直さない。
    #   売って空いたお金は、その月末に、ほかの買える候補へ回る（いつもの買いの順番）。
    #   うねりの取り方は BOX_SETS の3つ（短め・中くらい・長め）。結果を見る前に決めたもの。
    "swing_s_pit": {
        "label": "A・Bだけうねり取り（短め：20日・幅3〜8％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "hold_tiers": ["S"],
        "range_exit": {"fraction": 1.0, "min_gain": 0.0, "top_col": "box_top_s"},
        "swing_entry": {"tag": "s", "tiers": ["A", "B"]},
    },
    "swing_m_pit": {
        "label": "A・Bだけうねり取り（中くらい：60日・幅4〜12％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "hold_tiers": ["S"],
        "range_exit": {"fraction": 1.0, "min_gain": 0.0, "top_col": "box_top_m"},
        "swing_entry": {"tag": "m", "tiers": ["A", "B"]},
    },
    "swing_l_pit": {
        "label": "A・Bだけうねり取り（長め：120日・幅6〜18％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "hold_tiers": ["S"],
        "range_exit": {"fraction": 1.0, "min_gain": 0.0, "top_col": "box_top_l"},
        "swing_entry": {"tag": "l", "tiers": ["A", "B"]},
    },
    # ── 9回目の候補（2026年10月12日に追加）：うねりを日ごとに売り買いする ──
    #   相手は8回目と同じ「首位級だけその時点で判定（tier_pit_mix_scr。全部持ち続ける）」。
    #   S は持ち続け、A・B はうねりの中で、売り値に届いた日に売り、買い値に届いた日に買う（月に何往復でも）。
    #   うねりの取り方は DAILY_BOX_SETS の3つ（細かい・短め・中くらい）。結果を見る前に決めたもの。
    "swing_d_xs_pit": {
        "label": "日ごとにうねり取り（細かい：20日・幅2〜5％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "swing_daily": {"tag": "xs", "tiers": ["A", "B"]},
    },
    "swing_d_s_pit": {
        "label": "日ごとにうねり取り（短め：20日・幅3〜8％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "swing_daily": {"tag": "s", "tiers": ["A", "B"]},
    },
    "swing_d_m_pit": {
        "label": "日ごとにうねり取り（中くらい：60日・幅4〜12％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "swing_daily": {"tag": "m", "tiers": ["A", "B"]},
    },
    # ── 10回目の候補（2026年10月12日に追加）：手で引くような水平線でうねりを取る ──
    #   相手は9回目と同じ「首位級だけその時点で判定（tier_pit_mix_scr。全部持ち続ける）」。
    #   S は持ち続け、A・B は水平線の中で、上の線の手前（幅の10％下）に高値が届いた日に売り、
    #   下の線の手前（幅の10％上）に安値が届いた日に買う。線の引き方は DAILY_BOX_SETS の h1・h2・h3。
    #   売って空いたお金は、その月末に、ほかの買える候補へ回る（いつもの買いの順番）。
    "line_1y_pit": {
        "label": "1年線でうねり取り（250日・幅5〜25％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "swing_daily": {"tag": "h1", "tiers": ["A", "B"]},
    },
    "line_2y_pit": {
        "label": "2年線でうねり取り（500日・幅6〜35％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "swing_daily": {"tag": "h2", "tiers": ["A", "B"]},
    },
    "line_3y_pit": {
        "label": "3年線でうねり取り（750日・幅8〜50％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix",
        "swing_daily": {"tag": "h3", "tiers": ["A", "B"]},
    },
    # ── 11回目の候補（2026年10月12日に追加）：増配の実績がある銘柄だけで回す ──
    #   増配の実績 … 直近3回の通期どうしの比べのうち2回以上で年間配当（実績）が増え、記録のある範囲（最大10期）で
    #               一度も減っていない（grow_timeline。その時点で分かっていた数字だけ。7回目と同じ定義）
    #   新しく買うのは、この実績がある銘柄だけ（月末の買いも、線の下での買いも）。持っている銘柄は売る理由にしない。
    #   比べ方 … ① 1年・2年・3年線のうねり取りを「増配の実績だけ・全部持ち続ける」と比べる
    #            ② 4つとも、今の本番（tier_pit_mix_scr）と比べる
    "safe_hold_pit": {
        "label": "増配の実績だけ・全部持ち続ける",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit",
    },
    "safe_line_1y_pit": {
        "label": "増配の実績だけ・1年線でうねり取り",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit",
        "swing_daily": {"tag": "h1", "tiers": ["A", "B"]},
    },
    "safe_line_2y_pit": {
        "label": "増配の実績だけ・2年線でうねり取り",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit",
        "swing_daily": {"tag": "h2", "tiers": ["A", "B"]},
    },
    "safe_line_3y_pit": {
        "label": "増配の実績だけ・3年線でうねり取り",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit",
        "swing_daily": {"tag": "h3", "tiers": ["A", "B"]},
    },
    # ── 12回目の候補（2026年10月12日に追加）：「増配の実績だけ・2年線」を崩しにいく検証 ──
    #   銘柄を決まった方法で半分に分けた A群・B群（code_half）と、2年線の設定を少し変えたもの。
    "safe_hold_a": {
        "label": "増配だけ・A群・持ち続ける",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit", "code_half": 0,
    },
    "safe_2y_a": {
        "label": "増配だけ・A群・2年線",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit", "code_half": 0,
        "swing_daily": {"tag": "h2", "tiers": ["A", "B"]},
    },
    "safe_hold_b": {
        "label": "増配だけ・B群・持ち続ける",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit", "code_half": 1,
    },
    "safe_2y_b": {
        "label": "増配だけ・B群・2年線",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit", "code_half": 1,
        "swing_daily": {"tag": "h2", "tiers": ["A", "B"]},
    },
    "safe_2y_t15": {
        "label": "増配だけ・2年線（触れた判定の幅1.5％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit",
        "swing_daily": {"tag": "h2t15", "tiers": ["A", "B"]},
    },
    "safe_2y_t25": {
        "label": "増配だけ・2年線（触れた判定の幅2.5％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit",
        "swing_daily": {"tag": "h2t25", "tiers": ["A", "B"]},
    },
    "safe_2y_i05": {
        "label": "増配だけ・2年線（売り買いの位置5％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit",
        "swing_daily": {"tag": "h2i05", "tiers": ["A", "B"]},
    },
    "safe_2y_i15": {
        "label": "増配だけ・2年線（売り買いの位置15％）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "tier_col": "tier_pit_mix", "require_col": "grow_pit",
        "swing_daily": {"tag": "h2i15", "tiers": ["A", "B"]},
    },
    "rec_first_scr": {
        "label": "権利月が近い銘柄を先に買う（条件そのまま）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": True,
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "emg_rule": "actual", "skip_emg_buy": True, "div_split": True,
        "rec_relax": {"yield": None, "miss": 0, "first": True},
    },

    "switch36_flip": {
        "label": "市況で切り替え（36か月）・切り替わった月だけ入れ替える",
        "entry": [80], "exit": [], "measure": "switch36", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
        "sell_on_flip": True,
    },
    "switch36_live": {
        "label": "市況で切り替え（因子の勢い36か月）・本番の売り方・足切り3％",
        "entry": [80], "exit": [], "measure": "switch36", "priority": "tier",
        "budget_weighted": True, "target_names": 15, "min_yield_fixed": 3.0,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True, "exit_on_op": True,
    },

    # ── B. 業種に偏らない低PBR ──
    "pbr_sec_tier": {
        "label": "業種の中で低PBR順（Tier優先）",
        "entry": [80], "exit": [], "measure": "pbr_sec", "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },

    # ── 「過去3年と比べて安いとき」に買う仕組みは効いているか ──
    # 探索で「高配当・買って放置」がいまのルールを上回ったため、
    # 本番と同じ条件（スクリーニング・Tier別予算・売らない）で確かめる。
    # 変えるのは「どの銘柄を、いつ買うか」だけ。
    "xs_any_tier": {
        "label": "分位なし：利回りの高い順に買う（Tier優先）",
        "entry": [0], "exit": [], "priority": "tier", "cross": True,
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "xs_any_yield": {
        "label": "分位なし：利回りの高い順に買う（Tierを見ない）",
        "entry": [0], "exit": [], "cross": True,
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "xs80_tier": {
        "label": "全銘柄の利回り上位20％から買う（Tier優先）",
        "entry": [80], "exit": [], "priority": "tier", "cross": True,
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },
    "own_any_tier": {
        "label": "分位で待たない：条件を満たせばすぐ買う（Tier優先）",
        "entry": [0], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"], "exit_on_cut": True,
    },

    # ── 質（Tier）で入れ替える ──
    # B ばかり持っているときに S が買い候補になったら入れ替えるべきか。
    # 売却益に20.315％の税金がかかるので、それを取り戻せるかが問われる。
    "swap_2step": {
        "label": "2段階上なら入れ替え（B→S のみ）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 2, "max_year": 4},
    },
    "swap_1step": {
        "label": "1段階上なら入れ替え（B→A も）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 4},
    },
    "swap_2step_win": {
        "label": "2段階上＋含み益のときだけ入れ替え",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 2, "max_year": 4, "only_win": True},
    },
    "swap_2step_rare": {
        "label": "2段階上＋年2回まで",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 2, "max_year": 2},
    },

    # ── 入れ替えを、含み損益と配当の時期まで見て判断する ──
    # 含み損の銘柄を売れば税金はかからず、損失は繰り越して相殺できる。
    # 逆に含み益の銘柄を売ると、その場で2割が消える。
    # また権利確定月の直前に売ると、その回の配当を丸ごと逃す。
    "swap_loss": {
        "label": "含み損の銘柄だけ入れ替える（税金がかからない）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 2, "max_year": 4, "only_loss": True},
    },
    "swap_nodiv": {
        "label": "権利月の3か月前は入れ替えない",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 2, "max_year": 4, "avoid_div_months": 3},
    },
    "swap_smart": {
        "label": "含み損のみ＋権利月の3か月前は避ける",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 2, "max_year": 4,
                      "only_loss": True, "avoid_div_months": 3},
    },
    "swap_smart_1": {
        "label": "同上・1段階上でも入れ替え",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 6,
                      "only_loss": True, "avoid_div_months": 3},
    },

    # ── 十分に上がり、利回りも下がった銘柄を、質の高い銘柄に乗り換える ──
    # 少額の利確ではなく、ある程度の含み益が出ていて、
    # かつ利回りが下がって割高側に入った銘柄だけを対象にする。
    "swap_ripe20": {
        "label": "含み益20％以上＋Q25以下を S/A に乗り換え",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 6,
                      "min_gain": 0.20, "max_pct": 25},
    },
    "swap_ripe30": {
        "label": "含み益30％以上＋Q25以下を S/A に乗り換え",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 6,
                      "min_gain": 0.30, "max_pct": 25},
    },
    "swap_ripe50": {
        "label": "含み益50％以上＋Q25以下を S/A に乗り換え",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 6,
                      "min_gain": 0.50, "max_pct": 25},
    },
    "swap_ripe20_med": {
        "label": "含み益20％以上＋中央値以下を S/A に乗り換え",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 6,
                      "min_gain": 0.20, "max_pct": 50},
    },
    "swap_ripe20_nodiv": {
        "label": "含み益20％以上＋Q25以下＋権利月は避ける",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 6, "min_gain": 0.20,
                      "max_pct": 25, "avoid_div_months": 3},
    },

    # ── 業績急変で撤退するか ──
    # 実運用には入っているのに、検証では一度も試していなかった。
    "exit_op_only": {
        "label": "業績急変だけで撤退（減配では売らない）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True, "exit_on_op": True,
    },
    "exit_none": {
        "label": "減配でも業績急変でも売らない",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A", "B"],
    },

    # ── 新規で買えないときだけ入れ替える ──
    # 資金があるなら買い増しで質を上げればよく、税金を払う必要がない。
    # 枠が埋まっている、または現金が足りないときだけ入れ替える。
    "swap_stuck": {
        "label": "買えないときだけ入れ替え（含み益20％＋Q25以下＋権利月回避）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 6, "min_gain": 0.20,
                      "max_pct": 25, "avoid_div_months": 3,
                      "only_when_stuck": True},
    },
    "swap_stuck_loose": {
        "label": "買えないときだけ入れ替え（含み益の条件なし）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 1, "max_year": 6,
                      "max_pct": 25, "avoid_div_months": 3,
                      "only_when_stuck": True},
    },
    "swap_stuck_gap2": {
        "label": "買えないときだけ入れ替え（2段階上のみ）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "swap_tier": {"min_gap": 2, "max_year": 6, "min_gain": 0.20,
                      "max_pct": 25, "avoid_div_months": 3,
                      "only_when_stuck": True},
    },

    # ── 外部要因が逆風の業種を避ける ──
    # 金利が下降局面なら銀行を買わない、原油が下降局面なら資源を買わない、
    # 円高局面なら輸出関連を買わない。判定は前月までの値だけで行う。
    "factor_skip": {
        "label": "逆風の業種は買わない",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "factor_rule": {"mode": "skip"},
    },
    "factor_thin": {
        "label": "逆風の業種は半分の金額で買う",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "factor_rule": {"mode": "thin", "thin": 0.5},
    },
    "factor_thin30": {
        "label": "逆風の業種は3割の金額で買う",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "factor_rule": {"mode": "thin", "thin": 0.3},
    },

    # ── 中核を厚く持ち、二軍だけ往復する ──
    # S（累進配当かつ業界首位）はレンジになりにくく上に抜けやすい。
    # B はレンジに収まりやすい。性質で扱いを分ける。
    "swing_B_only": {
        "label": "SとAは持つ・Bだけレンジ往復",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A"], "exit_on_cut": True,
        "range_exit": {"fraction": 1.0, "min_gain": 0.0},
    },
    "swing_B_only_wS3": {
        "label": "SとAは持つ・Bだけ往復＋Sを厚く（S3.0）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 3.0, "A": 1.5, "B": 1.0},
        "hold_tiers": ["S", "A"], "exit_on_cut": True,
        "range_exit": {"fraction": 1.0, "min_gain": 0.0},
    },
    "swing_B_only_wS4": {
        "label": "SとAは持つ・Bだけ往復＋Sをさらに厚く（S4.0）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 4.0, "A": 1.5, "B": 1.0},
        "hold_tiers": ["S", "A"], "exit_on_cut": True,
        "range_exit": {"fraction": 1.0, "min_gain": 0.0},
    },
    "holdSA_noswing_wS3": {
        "label": "SとAは持つ・往復なし＋Sを厚く（比較用）",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 3.0, "A": 1.5, "B": 1.0},
        "hold_tiers": ["S", "A"], "exit_on_cut": True,
    },

    # ── Tier分けに価値があるか ──
    # S/A/B を判定して S 優先で買う仕組みが、成績に寄与しているのか。
    # 利回り順に買うだけでも同じなら、Tier判定は不要な複雑さ。
    "no_tier": {
        "label": "Tierを使わず利回り順に買う・減配撤退",
        "entry": [75], "exit": [], "priority": "pct",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 1.0, "A": 1.0, "B": 1.0},
        "exit_on_cut": True,
    },
    "no_tier_random": {
        "label": "Tierを使わず銘柄コード順に買う・減配撤退",
        "entry": [75], "exit": [], "priority": "code",
        "budget_weighted": True, "target_names": 15,
        "tier_weight": {"S": 1.0, "A": 1.0, "B": 1.0},
        "exit_on_cut": True,
    },

    # ── 高配当で買い、レンジの上限で一度降りる ──
    # 土台は「Sは売らない＋減配撤退」のまま。
    # そこに「レンジの中にいて上限まで来たら降りる」を足す。
    # 外れても持ち続けるだけなので、失敗しても元の戦略に戻るだけ。
    "swing_full": {
        "label": "高配当＋レンジ上限で全部降りる",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "exit_on_cut": True,
        "range_exit": {"fraction": 1.0, "min_gain": 0.0},
    },
    "swing_half": {
        "label": "高配当＋レンジ上限で半分だけ降りる",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "exit_on_cut": True,
        "range_exit": {"fraction": 0.5, "min_gain": 0.0},
    },
    "swing_gain3": {
        "label": "高配当＋レンジ上限（3％以上の益があるときだけ）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "exit_on_cut": True,
        "range_exit": {"fraction": 1.0, "min_gain": 0.03},
    },
    "swing_holdS": {
        "label": "高配当＋レンジ上限（SはそのままAとBだけ降りる）",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "exit_on_cut": True,
        "hold_tiers": ["S"],
        "range_exit": {"fraction": 1.0, "min_gain": 0.0},
    },

    # ── 買いの閾値を振る ──
    # 分位を9年→3年、足切りを3％→4％に変えたのに、
    # 「どこまで安ければ買うか」だけが当初のQ75のまま検証されていなかった。
    # 3つは互いに影響し合うので、他を変えたならここも確かめる必要がある。
    #   Q60 … 過去3年の上位40％に入れば買う（買いやすい）
    #   Q85 … 上位15％に入らないと買わない（厳しい）
    "entryQ60": {
        "label": "Q60で買う・Sは売らない＋減配撤退",
        "entry": [60], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "entryQ65": {
        "label": "Q65で買う・Sは売らない＋減配撤退",
        "entry": [65], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "entryQ70": {
        "label": "Q70で買う・Sは売らない＋減配撤退",
        "entry": [70], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "entryQ75": {
        "label": "Q75で買う・Sは売らない＋減配撤退",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "entryQ80": {
        "label": "Q80で買う・Sは売らない＋減配撤退",
        "entry": [80], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "entryQ85": {
        "label": "Q85で買う・Sは売らない＋減配撤退",
        "entry": [85], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "entryQ90": {
        "label": "Q90で買う・Sは売らない＋減配撤退",
        "entry": [90], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },

    # ── S は持ち続け、A と B の扱いだけ変える ──
    # 中核（S）は税金を繰り延べて複利で回し、周辺（A・B）で現金を作る、
    # という組み合わせ。売らない良さと、現金が入る安心を両立できるか。
    "holdS_ab_half50": {
        "label": "Sは持つ・AとBは50％で半分売る",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "partial_gain": {"threshold": 0.50, "fraction": 0.5},
    },
    "holdS_ab_half30": {
        "label": "Sは持つ・AとBは30％で半分売る",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "partial_gain": {"threshold": 0.30, "fraction": 0.5},
    },
    "holdS_ab_gain20": {
        "label": "Sは持つ・AとBは20％で全部売る",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True, "gain_exit": 0.20,
    },
    "holdSA_cut": {
        "label": "SとAは持つ・BだけQ25で売る",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S", "A"], "exit_on_cut": True,
    },

    # ── 部分利確（現金も入り、残りは走り続ける）──
    # 全部売ると税金を一度に払い、伸びしろも失う。
    # 一部だけ売れば、その中間を取れるのではないか、という検証。
    "half20": {
        "label": "20％上がるたびに半分売る",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "exit_on_cut": True,
        "partial_gain": {"threshold": 0.20, "fraction": 0.5},
    },
    "half30": {
        "label": "30％上がるたびに半分売る",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "exit_on_cut": True,
        "partial_gain": {"threshold": 0.30, "fraction": 0.5},
    },
    "third20": {
        "label": "20％上がるたびに3分の1売る",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "exit_on_cut": True,
        "partial_gain": {"threshold": 0.20, "fraction": 0.34},
    },
    "half50": {
        "label": "50％上がるたびに半分売る",
        "entry": [75], "exit": [], "priority": "tier",
        "budget_weighted": True, "target_names": 15, "exit_on_cut": True,
        "partial_gain": {"threshold": 0.50, "fraction": 0.5},
    },

    # ── 安定性を高めるための制約 ──
    # 年率ではなく「ばらつきの小ささ」「最悪のときのマシさ」を狙う。
    "stable_sector3": {
        "label": "Sは売らない＋減配撤退＋1業種3銘柄まで",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True, "max_per_sector": 3,
    },
    "stable_sector2": {
        "label": "同上・1業種2銘柄まで",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True, "max_per_sector": 2,
    },
    "stable_cash20": {
        "label": "Sは売らない＋減配撤退＋現金2割を残す",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 15,
        "hold_tiers": ["S"], "exit_on_cut": True, "cash_floor": 0.20,
    },
    "stable_wide": {
        "label": "Sは売らない＋減配撤退＋目標25銘柄",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 25,
        "hold_tiers": ["S"], "exit_on_cut": True,
    },
    "stable_all": {
        "label": "業種3・現金2割・25銘柄をすべて",
        "entry": [75], "exit": [25], "priority": "tier",
        "budget_weighted": True, "target_names": 25,
        "hold_tiers": ["S"], "exit_on_cut": True,
        "max_per_sector": 3, "cash_floor": 0.20,
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
class NotAllowed(Exception):
    """契約プランに含まれない、または認証が通らないときに投げる"""


class RangeTooLong(Exception):
    """指定した期間がプランの範囲を超えているときに投げる"""


class RateLimited(Exception):
    """429（呼び出しすぎ）が続いて取れなかったときに投げる。

    以前はここで黙って「データなし」を返していたため、
    混み合ったときに、その銘柄が検証から静かに消えていた。
    """


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
            # 429（呼び出しすぎ）は待てば通るので長めに粘る（合計で約3分）。
            # 通信エラー・5xx は3回まで。どちらも、だめなら例外にする
            # （黙って空のデータを返すと、その銘柄が検証から消える）。
            net_err = 0
            for attempt in range(7):
                try:
                    r = self.s.get(f"{JQUANTS_BASE}{path}", params=p,
                                   headers={"x-api-key": self.key}, timeout=30)
                    if r.status_code == 429:
                        wait = min(60, 2 ** (attempt + 1))
                        ra = str(r.headers.get("Retry-After") or "").strip()
                        if ra.isdigit():
                            wait = max(wait, min(120, int(ra)))
                        time.sleep(wait)
                        continue
                    if r.status_code in (401, 403):
                        # 契約プランに含まれない項目でも403が返る。
                        # 呼び出し側で「使えない」と扱えるよう例外にする。
                        raise NotAllowed(f"{r.status_code} {r.text[:120]}")
                    if r.status_code == 400:
                        # 期間が長すぎる場合にこれが返る。呼び出し側で短くして再試行する。
                        raise RangeTooLong(r.text[:150])
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    net_err += 1
                    if net_err >= 3:
                        raise
                    time.sleep(2 ** net_err)
            else:
                raise RateLimited(f"429 が続きました: {path} {params}")
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


# 為替は公開データセットから取る（J-Quantsの契約には含まれないため）
FX_URL = ("https://raw.githubusercontent.com/datasets/exchange-rates/"
          "main/data/daily.csv")


def fetch_fx() -> pd.DataFrame:
    """ドル円の日次を取る。取れなければ空を返す。

    2020〜2026年は円安が大きく進んだ時期で、
    高配当バリュー株（商社・銀行・輸出）はその恩恵を受けている可能性がある。
    成績のどれだけが円安由来かを切り分けるために使う。
    """
    try:
        import urllib.request
        with urllib.request.urlopen(FX_URL, timeout=60) as r:
            raw = r.read().decode("utf-8", "ignore")
        df = pd.read_csv(io.StringIO(raw))
        jp = df[df["Country"].astype(str).str.contains("Japan", case=False,
                                                       na=False)].copy()
        if jp.empty:
            raise ValueError("日本の行が見つかりません")
        jp["date"] = pd.to_datetime(jp["Date"])
        jp["usdjpy"] = pd.to_numeric(jp["Exchange rate"], errors="coerce")
        jp = jp.dropna(subset=["usdjpy"]).sort_values("date")
        log.info("ドル円を取得しました（%d日分 / %s〜%s）", len(jp),
                 jp["date"].min().date(), jp["date"].max().date())
        return jp[["date", "usdjpy"]].reset_index(drop=True)
    except Exception as e:
        log.warning("ドル円を取得できませんでした（%s）。為替の分析は省略します。", e)
        return pd.DataFrame()


SP500_URL = ("https://raw.githubusercontent.com/datasets/s-and-p-500/"
             "main/data/data.csv")
VIX_URL = ("https://raw.githubusercontent.com/datasets/finance-vix/"
           "main/data/vix-daily.csv")

# 金利・商品。列名がまちまちなので、日付以外の最初の数値列を使う。
OTHER_SRC = {
    "us10y": ("米10年金利", "銀行株が動く。期間中 0.7％→4.6％",
              "https://raw.githubusercontent.com/datasets/"
              "bond-yields-us-10y/main/data/monthly.csv"),
    "brent": ("原油（ブレント）", "商社・エネルギー株が動く",
              "https://raw.githubusercontent.com/datasets/"
              "oil-prices/main/data/brent-daily.csv"),
    "gold": ("金", "不安のときに買われる。株と逆に動きやすい",
             "https://raw.githubusercontent.com/datasets/"
             "gold-prices/main/data/monthly.csv"),
}


def _fetch_csv(url: str) -> pd.DataFrame:
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r:
        return pd.read_csv(io.StringIO(r.read().decode("utf-8", "ignore")))


# 33業種のうち、外部要因に強く反応するもの。
# 「どの業種がどの要因に反応するか」は自分で決めるしかないため、
# 後付けにならないよう、一般に知られている対応だけを使う。
SECTOR_FACTOR = {
    # 金利が上がると利ざやが広がる
    "rate": ["銀行業", "保険業", "証券、商品先物取引業", "その他金融業"],
    # 資源価格に直結する
    "oil": ["鉱業", "石油・石炭製品", "卸売業", "海運業"],
    # 円安が追い風になる（海外売上の比率が高い）
    "fx": ["輸送用機器", "電気機器", "精密機器", "機械"],
}


def factor_trend(sr: pd.Series, win: int = 6) -> pd.Series:
    """その指標が上向きか下向きかを、月ごとに返す。

    前月までの値だけを使う（当月を含めると後出しになる）。
    win か月前と比べて上なら +1、下なら −1。
    """
    prev = sr.shift(1)
    return np.sign(prev - prev.shift(win)).fillna(0)


def fetch_markets() -> dict:
    """外部の市場データを取る。取れなかったものは入らない。

    S&P500 … 米国株の影響を切り分ける
    VIX    … 恐怖指数。市場が不安なときに上がる。暴落局面の代用になる
    """
    out = {}
    try:
        d = _fetch_csv(SP500_URL)
        d["date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["sp500"] = pd.to_numeric(d["SP500"], errors="coerce")
        d = d.dropna(subset=["date", "sp500"]).sort_values("date")
        out["sp500"] = d.set_index("date")["sp500"]
        log.info("S&P500 を取得しました（%d件 / %s〜%s）", len(d),
                 d["date"].min().date(), d["date"].max().date())
    except Exception as e:
        log.warning("S&P500 を取得できませんでした（%s）", e)
    try:
        d = _fetch_csv(VIX_URL)
        d["date"] = pd.to_datetime(d["DATE"], errors="coerce")
        d["vix"] = pd.to_numeric(d["CLOSE"], errors="coerce")
        d = d.dropna(subset=["date", "vix"]).sort_values("date")
        out["vix"] = d.set_index("date")["vix"]
        log.info("VIX（恐怖指数）を取得しました（%d件 / %s〜%s）", len(d),
                 d["date"].min().date(), d["date"].max().date())
    except Exception as e:
        log.warning("VIX を取得できませんでした（%s）", e)
    try:
        fx = fetch_fx()
        if not fx.empty:
            out["usdjpy"] = fx.set_index("date")["usdjpy"]
    except Exception as e:
        log.warning("ドル円を取得できませんでした（%s）", e)

    for key, (jname, _note, url) in OTHER_SRC.items():
        try:
            d = _fetch_csv(url)
            dc = next(c for c in d.columns if "date" in c.lower())
            vc = next(c for c in d.columns if c != dc)
            d[dc] = pd.to_datetime(d[dc], errors="coerce")
            d[vc] = pd.to_numeric(d[vc], errors="coerce")
            d = d.dropna(subset=[dc, vc]).sort_values(dc)
            out[key] = d.set_index(dc)[vc]
            log.info("%s を取得しました（%d件）", jname, len(d))
        except Exception as e:
            log.warning("%s を取得できませんでした（%s）", jname, e)
    return out


# ══════════════════════════════════════════
# ストレステストの部品（2026年10月12日に追加）
# ══════════════════════════════════════════
STRESS_SCEN = [
    ("逆風が全部来る", {"fx": -0.20, "rate": -1.5, "oil": -0.40, "spx": -0.25},
     "円高20％・米金利1.5pt低下・原油40％安・米国株25％安"),
    ("リーマン級", {"fx": -0.25, "rate": -2.0, "oil": -0.60, "spx": -0.45},
     "円高25％・米金利2pt低下・原油60％安・米国株45％安"),
    ("円高だけ", {"fx": -0.20, "rate": 0, "oil": 0, "spx": 0}, "円高20％"),
    ("金利低下だけ", {"fx": 0, "rate": -1.5, "oil": 0, "spx": 0}, "米金利1.5pt低下"),
    ("原油安だけ", {"fx": 0, "rate": 0, "oil": -0.40, "spx": 0}, "原油40％安"),
    ("米国株安だけ", {"fx": 0, "rate": 0, "oil": 0, "spx": -0.30}, "米国株30％安"),
]
STRESS_FACTORS = ["fx", "rate", "oil", "spx"]
# 単独の要因のシナリオと、その要因
STRESS_SINGLE = {"円高だけ": "fx", "金利低下だけ": "rate", "原油安だけ": "oil", "米国株安だけ": "spx"}


def stress_factor_changes(mk: dict) -> pd.DataFrame:
    """外部要因の月ごとの変化（月末の資産の増減と時期をそろえたもの）。

      fx   … ドル円の変化率（＋なら円安）。日次のデータなので月末の値で比べる
      oil  … 原油（ブレント）の変化率。日次のデータなので月末の値で比べる
      rate … 米10年金利の変化（％ポイント）
      spx  … 米国株（S&P500）の変化率
    米10年金利と S&P500 のデータは「その月の平均」で、月末の値ではない。
    月平均どうしの変化は、前の月の半ばから今月の半ばまでの動きになり、
    月末から月末までの資産の増減と半月ずれる。そのまま使うと反応の強さが小さく出るので、
    その月と翌月の変化の平均を使って時期をそろえる（2026年10月12日に追加）。
    """
    need = ["usdjpy", "us10y", "brent", "sp500"]
    fac = pd.DataFrame({k: mk[k].resample("ME").last() for k in need}).dropna()

    def _center(x: pd.Series) -> pd.Series:
        return (x + x.shift(-1)) / 2

    return pd.DataFrame({
        "fx": fac["usdjpy"].pct_change(),
        "rate": _center(fac["us10y"].diff()),
        "oil": fac["brent"].pct_change(),
        "spx": _center(fac["sp500"].pct_change()),
    }).dropna()


def stress_fit(curve: pd.DataFrame, dX: pd.DataFrame, scen=None) -> dict | None:
    """月ごとの資産の増減を外部要因の変化で説明する式を作り、シナリオの1年の増減を出す。

    返す値：coef（切片＋各要因の反応の強さ）、r2（説明できる割合）、n（月数）、
            worst12（実際の最悪の12か月）、hit（シナリオ名 → 外部要因の影響だけの1年の増減）、
            hit_all（同じく、いつもの伸び＝切片×12 を足したもの）、ret（月ごとの増減）
    """
    scen = STRESS_SCEN if scen is None else scen
    v = curve.set_index("date")["value"].resample("ME").last().pct_change().dropna()
    df = pd.concat([v.rename("r"), dX], axis=1, join="inner").dropna()
    if len(df) < 24:
        return None
    X = np.column_stack([np.ones(len(df))] + [df[k].to_numpy() for k in STRESS_FACTORS])
    y = df["r"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = float(1 - resid.var() / y.var()) if y.var() > 0 else 0.0
    worst12 = float(((1 + v).rolling(12).apply(np.prod, raw=True) - 1).min()) if len(v) >= 12 \
        else float("nan")
    hit, hit_all = {}, {}
    for sname, shock, _d in scen:
        h = float(sum(coef[i + 1] * shock[k] for i, k in enumerate(STRESS_FACTORS)))
        hit[sname] = h
        hit_all[sname] = h + float(coef[0]) * 12
    return {"coef": coef, "r2": r2, "n": len(df), "worst12": worst12,
            "hit": hit, "hit_all": hit_all, "ret": v}


def stress_bad_months(rets: dict, dX: pd.DataFrame, q: float = 0.10) -> list:
    """実際に逆風だった月（各要因の変化が悪い側から1割の月）の、月ごとの増減の平均。

    返す値：[(要因, 月数, その境目の変化, {名前: 平均の増減}), …]
    「悪い側」はどの要因も下がる側（円高・金利低下・原油安・米国株安）。
    """
    out = []
    # 外部要因のデータは何十年分もあるので、資産の増減がそろっている期間だけに絞ってから1割を選ぶ
    idx = None
    for r in rets.values():
        ix = r.dropna().index
        idx = ix if idx is None else idx.intersection(ix)
    d_ = dX.loc[dX.index.intersection(idx)] if idx is not None else dX.iloc[0:0]
    for k in STRESS_FACTORS:
        s_ = d_[k].dropna()
        if s_.empty:
            continue
        thr = float(s_.quantile(q))
        months = s_[s_ <= thr].index
        avg = {nm: float(r.reindex(months).mean()) for nm, r in rets.items()}
        out.append((k, len(months), thr, avg))
    return out


def fetch_topix(jq: "JQ", years: int = 9) -> pd.DataFrame:
    """TOPIX の日次終値を取る。

    V2 での項目名が確かめられていないため、考えられる経路を順に試す。
    取れなければ空を返し、呼び出し側で「なし」として扱う。
    """
    today = date.today()
    frm = (today - timedelta(days=365 * years + 30)).isoformat()
    to = today.isoformat()
    paths = [("/v2/indices/topix", {"from": frm, "to": to}),
             ("/v2/markets/indices/topix", {"from": frm, "to": to}),
             ("/v1/indices/topix", {"from": frm, "to": to})]
    for path, params in paths:
        try:
            rows = jq.get(path, params)
        except NotAllowed:
            continue      # 契約に含まれない。次の経路を試す
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        dcol = next((c for c in df.columns if c.lower() in ("date", "d")), None)
        ccol = next((c for c in df.columns
                     if c.lower() in ("close", "c", "closeprice")), None)
        if not dcol or not ccol:
            continue
        out = pd.DataFrame({"date": pd.to_datetime(df[dcol]),
                            "close": pd.to_numeric(df[ccol], errors="coerce")})
        out = out.dropna().sort_values("date").reset_index(drop=True)
        if len(out) > 100:
            log.info("TOPIX を取得しました（%s / %d日分）", path, len(out))
            return out
    log.warning("TOPIX を取得できませんでした（契約プランに含まれない可能性）。"
                "指数との比較と、指数を使う判定は省略します。")
    return pd.DataFrame()


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


def save_store(store: dict, note: str = "") -> None:
    """取得したデータを保存する。書いている途中で止められても壊れないよう、
    別名に書いてから置き換える。"""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    pd.to_pickle(store, tmp)
    os.replace(tmp, CACHE)
    if note:
        log.info("  %s（%d / %d銘柄・開始から%.0f分）", note,
                 len(store.get("quotes", {})), len(store.get("universe", [])),
                 _elapsed_min())


def _fetch_one(jq: JQ, code: str) -> tuple[list | None, list | None]:
    """1銘柄の株価と財務を取る。取れなければ例外。株価が空なら (None, None)。"""
    time.sleep(API_SLEEP)
    q = fetch_bars(jq, code, 9)   # いつでも上限まで取っておく
    if not q:
        return None, None
    time.sleep(API_SLEEP)
    try:
        st = jq.get("/v2/fins/summary", {"code": code})
    except NotAllowed:
        st = []                   # 契約に含まれない。財務なしとして扱う
    # それ以外の失敗は例外のまま返す。財務だけ欠けた銘柄を混ぜず、次の実行で取り直す
    return q, st


def fetch_all(jq: JQ, years: int, limit: int,
              scale_filter: bool = True, store: dict | None = None) -> dict:
    """株価と財務をまとめて取得する。時間がかかるのでキャッシュします。

    途中で止まっても続きから取れるようにしてある。
      ・使ってよい時間（FETCH_BUDGET_MIN）に達したら、取った分を保存して止まる
      ・取得中も一定の間隔で保存する（強制終了に備える）
      ・途中のデータ（complete が False）を渡すと、まだの銘柄だけを取る
    取り終えたら store["complete"] = True、途中なら False にして返す。
    """
    if store and store.get("universe") and store.get("complete") is False:
        uni = store["universe"]
        store.setdefault("quotes", {})
        store.setdefault("stmts", {})
        store["resume_runs"] = int(store.get("resume_runs", 0)) + 1
        log.info("前回の続きから取得します（取得済み %d / %d銘柄・%d回目の続き）",
                 len(store["quotes"]), len(uni), store["resume_runs"])
    else:
        log.info("銘柄一覧を取得中…")
        try:
            info = jq.get("/v2/equities/master", {})
        except NotAllowed as e:
            sys.exit(f"認証に失敗しました（{e}）。APIキーをご確認ください。")
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
        store = {"universe": uni, "quotes": {}, "stmts": {}, "complete": False}
    log.info("対象 %d銘柄（使ってよい時間 %.0f分・開始から%.0f分）",
             len(uni), FETCH_BUDGET_MIN, _elapsed_min())

    def _run(codes: list[str]) -> tuple[list[str], bool]:
        """codes を順に取る。(取れなかった銘柄, 時間切れで止めたか) を返す。"""
        failed: list[str] = []
        last_save = time.monotonic()
        for i, code in enumerate(codes, 1):
            if _elapsed_min() >= FETCH_BUDGET_MIN:
                return failed, True
            try:
                q, st = _fetch_one(jq, code)
            except Exception as e:
                log.warning("取得失敗 %s: %s", code, e)
                failed.append(code)
                continue
            if q is not None:
                store["quotes"][code] = q
                store["stmts"][code] = st
            if i % 50 == 0:
                log.info("  取得 %d/%d（全体で取得済み %d / %d銘柄・開始から%.0f分）",
                         i, len(codes), len(store["quotes"]), len(uni), _elapsed_min())
            if (time.monotonic() - last_save) / 60 >= FETCH_SAVE_EVERY_MIN:
                save_store(store, "途中まで保存しました")
                last_save = time.monotonic()
        return failed, False

    todo = [u["code"] for u in uni if u["code"] not in store["quotes"]]
    failed, stopped = _run(todo)
    if failed and not stopped:
        log.info("取れなかった %d銘柄を、もう一度だけ試します…", len(failed))
        failed, stopped = _run(failed)

    store["failed_codes"] = failed
    store["stopped_by_time"] = stopped
    if stopped:
        log.warning("使ってよい時間（%.0f分）に達したので取得を止めました。"
                    "取得済み %d / %d銘柄", FETCH_BUDGET_MIN,
                    len(store["quotes"]), len(uni))
        store["complete"] = False
        return store
    if not store["quotes"]:
        sys.exit("株価を1銘柄も取得できませんでした。APIキーと契約プランをご確認ください。")
    log.info("取得できた銘柄: %d / %d", len(store["quotes"]), len(uni))
    # 取れなかった銘柄が多いまま「取り終えた」にすると、偏ったデータで結果が出る。
    # 次の実行でもう一度取る。ただし3回続いたら、残りは無いものとして進める。
    too_many = len(failed) > max(10, int(len(uni) * 0.02))
    if too_many and int(store.get("resume_runs", 0)) < 3:
        log.warning("取れなかった銘柄が %d あります（%s …）。次の実行でもう一度取ります。",
                    len(failed), ", ".join(failed[:10]))
        store["complete"] = False
        return store
    if failed:
        log.warning("取れなかった銘柄 %d を除いて進めます: %s", len(failed),
                    ", ".join(failed[:20]) + (" …" if len(failed) > 20 else ""))
    try:
        store["topix"] = fetch_topix(jq)
    except Exception as e:
        log.warning("TOPIX の取得に失敗: %s", e)
        store["topix"] = pd.DataFrame()
    store["complete"] = True
    return store


def print_unfinished(store: dict, compute_only: bool) -> None:
    """途中で止めたことを、Summary の最後に分かる形で出す。"""
    n_ok, n_all = len(store.get("quotes", {})), len(store.get("universe", []))
    print()
    if compute_only:
        print("■ まだ終わっていません（株価は取り終えましたが、計算の時間が残っていません）")
    elif store.get("stopped_by_time"):
        print("■ まだ終わっていません（株価の取得を途中で止めました）")
    else:
        print(f"■ まだ終わっていません（取れなかった銘柄が {len(store.get('failed_codes') or []):,} あるため、"
              "次の実行でもう一度取ります）")
    print(f"　 取得済み {n_ok:,} / {n_all:,}銘柄。取った分は保存しました"
          f"（開始から{_elapsed_min():.0f}分）。")
    print("　 同じ設定のまま、もう一度「Run workflow」を押してください。"
          + ("今度は計算から始まります。" if compute_only else "続きから取ります。"))
    print("　 180分の上限で強制終了されると、取った分も結果も消えるため、その前に止めています。")


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


def op_timeline(stmts: list[dict]) -> pd.DataFrame:
    """開示日つきの営業利益（通期）の系列を作る。

    その日に開示されていた数字だけを使う。
    前年同期と比べて急減したかどうかの判定に使う。
    """
    recs = []
    for s in stmts:
        if s.get("CurPerType") not in ("FY", "4Q"):
            continue
        d = pdate(s.get("DiscDate")) or pdate(s.get("CurPerEn"))
        if d is None:
            continue
        v = fnum(s.get("OpProfit")) or fnum(s.get("OperatingProfit")) \
            or fnum(s.get("OpPrf")) or fnum(s.get("OP"))
        if v is None:
            continue
        recs.append({"date": pd.Timestamp(d), "op": float(v)})
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values("date")
    return df.groupby("date", as_index=False)["op"].last()


# スクリーニング8条件のうち、利回り以外の7つ（2026年10月11日に追加。条件を1つずつ外す検証で使う）
SCREEN_KEYS = ("div", "sales", "op", "np", "pay", "eq", "val")
SCREEN_LABELS = {"div": "10期減配なし", "sales": "売上の安定", "op": "営業利益の安定",
                 "np": "純利益の安定", "pay": "配当性向50％以下", "eq": "自己資本比率",
                 "val": "PER×PBR40以下"}

# 財務の項目名（V2 は短い略称）。候補を順に試す。
FUND_FIELDS = {
    "eps":   ("EPS",),
    "bps":   ("BPS",),
    "np":    ("NP", "NetProfit", "Profit"),
    "eq":    ("Eq", "Equity", "NetAssets", "TotalEquity"),
    "sales": ("Sales", "NetSales", "Revenue"),
    "sh":    ("ShOutFY", "ShOut", "SharesOutstanding"),
    "op":    ("OP", "OpProfit", "OperatingProfit"),
    "eqar":  ("EqAR",),
    "payout": ("PayoutRatioAnn",),
    "divann": ("DivAnn",),
    # 中間配当の実績（2026年10月11日に追加）。年間配当のうち中間で払った分。
    # 期末だけの会社（中間が0）を見分け、配当を受け取る月を正しくするために使う。
    "div2q": ("Div2Q", "ResultDividendPerShare2ndQuarter"),
}


def _stable(values: list) -> bool | None:
    """本番の check_stability と同じ。過去5期で前期比−10%超が2期続いていなければ True。"""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if len(vals) < 3:
        return None
    ser = vals[-5:]
    run = 0
    for a, b in zip(ser[:-1], ser[1:]):
        if a <= 0:
            run = 0
            continue
        if (b - a) / abs(a) < -0.10:
            run += 1
            if run >= 2:
                return False
        else:
            run = 0
    return True


def _no_cut(values: list) -> bool | None:
    """本番の check_dividend_history と同じ。過去10期で一度も減配していなければ True。"""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if len(vals) < 3:
        return None
    h = vals[-10:]
    return all(h[i] >= h[i - 1] for i in range(1, len(h)))


def fund_timeline(stmts: list[dict], px: pd.DataFrame | None = None) -> pd.DataFrame:
    """開示日つきの通期の財務項目の系列。前年の純利益も持たせる（成長率用）。

    株式分割の調整（2026年10月9日に追加）
      株価（AdjC）は、過去の値を今の株数の基準に割り戻してある。
      ところが決算の1株あたりの数字（EPS・BPS・年間配当）と株数は、開示した時点の株数のまま。
      そのまま割ると、あとで分割した銘柄ほどPBR・PERが分割の倍率だけ低く出る
      （例：三菱重工は1:10分割の前、PBRが約1.0なのに0.1と計算されていた）。
      これは「あとで分割する＝そのあと値上がりした」という未来の情報で選ぶことになる。
      px（分割の調整係数 adj つきの株価）を渡すと、開示日より後に起きた分割のぶんだけ
      1株あたりの数字を掛け、株数を割って、今の基準に揃える。
      本番の generate.py（split_adjustment_factor）と同じ考え方。
    """
    recs = []
    for st in stmts:
        if st.get("CurPerType") not in ("FY", "4Q"):
            continue
        d = pdate(st.get("DiscDate")) or pdate(st.get("CurPerEn"))
        if d is None:
            continue
        rec = {"date": pd.Timestamp(d), "fy": str(st.get("CurPerEn") or "")[:10]}
        for k, names in FUND_FIELDS.items():
            v = None
            for nm in names:
                v = fnum(st.get(nm))
                if v is not None:
                    break
            rec[k] = v
        recs.append(rec)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).sort_values("date").groupby("date", as_index=False).last()
    if px is not None and not px.empty and "adj" in px.columns:
        adj = px.set_index("date")["adj"].replace(0, 1.0).fillna(1.0)
        cum = adj[::-1].cumprod()[::-1]                    # その日を含め、それ以降の累積係数
        pos = cum.index.searchsorted(df["date"].values, side="right")   # 開示日より後の最初の日
        fac = np.array([float(cum.iloc[p]) if p < len(cum) else 1.0 for p in pos])
        for c in ("eps", "bps", "divann", "div2q"):        # 1株あたりの数字
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") * fac
        if "sh" in df.columns:                             # 株数
            df["sh"] = pd.to_numeric(df["sh"], errors="coerce") / fac
    df["np_prev"] = df["np"].shift(1)
    # 本番の緊急撤退（generate.py の check_emergency_exit）と同じ判定（2026年10月11日に追加）
    #   div_cut10 … 実績の年間配当が、前の通期より10％以上減った
    #   op_cut20  … 営業利益が、前の通期より20％以上減った（赤字転落を含む）
    # 前の通期は「期末日が違う直前の開示」。同じ通期の再開示（訂正など）と比べると、
    # 印が途中で消えてしまうため。1株あたりの配当は上で分割調整してあるので、分割を減配と取り違えない。
    _fy = df["fy"].tolist() if "fy" in df.columns else [""] * len(df)

    def _prev_fy(vals):
        out = []
        for i in range(len(vals)):
            pv = np.nan
            for j in range(i - 1, -1, -1):
                if _fy[i] and _fy[j] and _fy[j] < _fy[i]:
                    pv = vals[j]
                    break
            out.append(pv)
        return np.array(out, dtype=float)

    # 落ち方の深さも持たせる（2026年10月11日に追加）
    #   div_chg … 実績の年間配当の、前の通期からの変化率（−0.3 なら30％減。−1.0 は無配）
    #   op_chg  … 営業利益の変化率（−1.0 以下は赤字転落）
    #   前の通期が0以下のときは比べられないので空（印も立たない）
    for col_, key_, thr_, chg_col in (("div_cut10", "divann", -0.10, "div_chg"),
                                      ("op_cut20", "op", -0.20, "op_chg")):
        if key_ in df.columns:
            cur_ = pd.to_numeric(df[key_], errors="coerce").to_numpy(dtype=float)
            prv_ = _prev_fy(cur_)
            with np.errstate(divide="ignore", invalid="ignore"):
                chg_ = (cur_ - prv_) / prv_
            df[col_] = ((prv_ > 0) & (chg_ <= thr_)).astype(float)
            df[chg_col] = np.where(prv_ > 0, chg_, np.nan)
        else:
            df[col_] = 0.0
            df[chg_col] = np.nan
    # その開示の時点までの履歴だけで判定する（あとから分かる情報は使わない）
    for col, fn, key in (("ok_sales", _stable, "sales"), ("ok_op", _stable, "op"),
                         ("ok_np", _stable, "np"), ("ok_div", _no_cut, "divann")):
        out = []
        for i in range(len(df)):
            r = fn(df[key].iloc[:i + 1].tolist()) if key in df.columns else None
            out.append(np.nan if r is None else float(r))
        df[col] = out
    return df


def add_fund_columns(store: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """パネルに財務項目を足す。その月までに開示されていた数字だけを使う。

    1株あたりの数字（EPS・BPS）と株数は、株価と同じく分割調整して今の基準に揃える
    （fund_timeline に株価を渡す）。揃えないと、あとで分割した銘柄のPBRが低く出る。
    """
    panel = panel.copy()
    cols = ["eps", "bps", "np", "eq", "sales", "sh", "np_prev",
            "op", "eqar", "payout", "ok_sales", "ok_op", "ok_np", "ok_div",
            "divann", "div_cut10", "op_cut20", "div_chg", "op_chg", "div2q"]
    for c in cols:
        panel[c] = np.nan
    got = {c: 0 for c in cols}
    n_split = 0
    for code, g in panel.groupby("code"):
        px_ = quotes_to_df(store["quotes"].get(code, []))
        if not px_.empty and (px_["adj"].replace(0, 1.0) != 1.0).any():
            n_split += 1
        ft = fund_timeline(store["stmts"].get(code, []), px_)
        if ft.empty:
            continue
        ft = ft.set_index("date").sort_index()
        idx = g["date"]
        for c in cols:
            if c not in ft.columns:
                continue
            ser = ft[c].dropna()
            if ser.empty:
                continue
            got[c] += 1
            # その月末までに開示された最新の値
            vals = ser.reindex(ser.index.union(idx)).ffill().reindex(idx)
            panel.loc[g.index, c] = vals.to_numpy()
    for c in cols:
        log.info("  財務 %-8s … 取れた銘柄 %d社", c, got[c])
    log.info("  株式分割があった銘柄 %d社 … 1株あたりの数字を今の株数の基準に揃えました", n_split)
    # 予想の減配（2026年10月11日に追加）
    #   fc_chg … その月末に分かっていた予想の年間配当（dps）が、直近の実績の年間配当（divann）から
    #            どれだけ変わっているか。−0.1 なら「10％減の予想」。
    #   実績の減配（div_cut10）は通期の決算が出て初めて立つので、期中の下方修正から半年〜1年遅れる。
    #   予想の段階で分かっていたかを見るために持たせる。
    #   注意：会社が「無配予想」と出した場合、予想の欄が0になり、このツールは実績の配当で代用する
    #   （利回りの計算と同じ扱い）ため、無配の予想はここに出ない。
    _dv = panel["divann"].where(panel["divann"] > 0)
    panel["fc_chg"] = (panel["dps"] / _dv - 1.0).where(_dv.notna() & (panel["dps"] > 0))
    # 中間配当の割合（2026年10月11日に追加）
    #   div_int_share … 直近の通期で、年間配当のうち中間で払った割合（0なら期末だけの会社）。
    #   取れないときは空（その場合は以前どおり「中間と期末で半分ずつ」とみなす）。
    _q2 = pd.to_numeric(panel["div2q"], errors="coerce")
    panel["div_int_share"] = (_q2 / _dv).clip(0.0, 1.0).where(_dv.notna() & _q2.notna())
    _last = panel.sort_values("date").groupby("code")["div_int_share"].last()
    log.info("  配当の中間と期末の内訳 … 取れた銘柄 %d社（うち期末だけの会社 %d社）",
             int(_last.notna().sum()), int((_last == 0).sum()))
    # 本番のエンジンで使う、全銘柄の中での順位（％）。
    #   pct_pbr / pct_per … 低いほど高い値（安い順）
    #   pct_size          … 時価総額が小さいほど高い値
    _pbr = panel["price"] / panel["bps"].where(panel["bps"] > 0)
    _per = panel["price"] / panel["eps"].where(panel["eps"] > 0)
    _size = panel["price"] * panel["sh"]
    panel["_pbr"], panel["_per"], panel["_size"] = _pbr, _per, _size
    panel["pct_pbr"] = panel.groupby("date")["_pbr"].rank(pct=True, ascending=False) * 100
    panel["pct_per"] = panel.groupby("date")["_per"].rank(pct=True, ascending=False) * 100
    panel["pct_size"] = panel.groupby("date")["_size"].rank(pct=True, ascending=False) * 100
    panel = panel.drop(columns=["_pbr", "_per", "_size"])
    # 業種の中での PBR の順位（業種に偏らないように選ぶため）
    _pbr2 = panel["price"] / panel["bps"].where(panel["bps"] > 0)
    panel["_pbr2"] = _pbr2
    panel["pct_pbr_sec"] = panel.groupby(["date", "sector"])["_pbr2"].rank(
        pct=True, ascending=False) * 100
    panel = panel.drop(columns=["_pbr2"])
    # PBR と PER を合わせた割安さ
    panel["pct_value"] = (panel["pct_pbr"] + panel["pct_per"]) / 2
    # 稼ぐ力（ROE＝純利益÷自己資本）の順位。高いほど大きい値
    _roe = panel["np"] / panel["eq"].where(panel["eq"] > 0)
    panel["_roe"] = _roe
    panel["pct_roe"] = panel.groupby("date")["_roe"].rank(pct=True, ascending=True) * 100
    panel = panel.drop(columns=["_roe"])
    # 割安（PBRの低さ）と稼ぐ力（ROEの高さ）を合わせた順位。割合は wv で決める
    for _wv in (70, 50, 30):
        # 平均したままだと上位20％に入る銘柄が少なくなる（両方で上位の銘柄はまれ）ので、
        # 合わせた点数で順位をつけ直し、どの割合でも候補の数をそろえる
        panel["_mix"] = (panel["pct_pbr"] * _wv + panel["pct_roe"] * (100 - _wv)) / 100
        panel[f"pct_pbrroe{_wv}"] = panel.groupby("date")["_mix"].rank(pct=True, ascending=True) * 100
    panel = panel.drop(columns=["_mix"])
    # 市況で切り替える順位：割安寄りの月は割安70＋稼ぐ力30、稼ぐ力寄りの月は割安30＋稼ぐ力70
    if SWITCH_SIG is not None:
        _m = panel["date"] + pd.offsets.MonthEnd(0)
        _s = SWITCH_SIG.copy()
        _s.index = _s.index + pd.offsets.MonthEnd(0)
        _v = _m.map(_s)
        panel["pct_switch36"] = np.where(_v == True, panel["pct_pbrroe70"],
                                  np.where(_v == False, panel["pct_pbrroe30"], panel["pct_pbrroe50"]))
        # 合図が前の月から変わった月（この月だけ持ち株を入れ替える形で使う）
        _flip = (_s != _s.shift(1)) & _s.shift(1).notna()
        panel["switch_flip"] = _m.map(_flip).fillna(False).astype(bool)
        log.info("  合図が切り替わった月 … %s",
                 "、".join(str(d.date())[:7] for d in _flip[_flip].index
                          if d >= panel["date"].min()) or "なし")
        _n_v = int((_v == True).sum()); _n_q = int((_v == False).sum())
        log.info("  市況の切り替え … 割安寄り %d件・稼ぐ力寄り %d件（銘柄×月）", _n_v, _n_q)

    # ── スクリーニング8条件（本番の generate.py と同じ判定） ──
    # 判定できない（データ不足）ものは本番と同じく通過扱い。
    def _tri(x):
        return None if (x is None or pd.isna(x)) else bool(x)
    exempt_eq = {"銀行業", "証券、商品先物取引業", "保険業", "その他金融業", "不動産業"}
    relaxed_eq = {"建設業", "海運業", "卸売業"}
    per_ = panel["price"] / panel["eps"]
    pbr_ = panel["price"] / panel["bps"]
    pay = panel["payout"].where(panel["payout"].notna(),
                                panel["dps"] / panel["eps"].where(panel["eps"] > 0))
    pay = pay.where(pay.isna() | (pay >= 3), pay * 100)   # 0.35 のような比率なら％に
    eqr = panel["eqar"].where(panel["eqar"].isna() | (panel["eqar"] <= 1.5),
                              panel["eqar"] / 100)
    res = []
    n_fail = []            # 満たしていない条件の数（2026年10月11日に追加。条件をゆるめる検証で使う）
    # 条件ごとの通過（2026年10月11日に追加。条件を1つずつ外す検証で使う）。判定できないものは通過扱い
    per_check = {k: [] for k in SCREEN_KEYS}
    prog = panel["progressive"] if "progressive" in panel.columns else pd.Series(False, index=panel.index)
    for i in panel.index:
        sec = panel.at[i, "sector"]
        div_ok = True if bool(prog.at[i]) else _tri(panel.at[i, "ok_div"])
        eq_ok = (True if sec in exempt_eq else
                 None if pd.isna(eqr.at[i]) else
                 eqr.at[i] >= (0.33 if sec in relaxed_eq else 0.50))
        pv = pay.at[i]
        pay_ok = None if pd.isna(pv) else pv <= 50.0
        e, b = panel.at[i, "eps"], panel.at[i, "bps"]
        if pd.isna(e) or pd.isna(b):
            val_ok = None
        elif e <= 0 or b <= 0:
            val_ok = False
        else:
            val_ok = (per_.at[i] * pbr_.at[i]) <= 40.0
        checks = [div_ok, _tri(panel.at[i, "ok_sales"]), _tri(panel.at[i, "ok_op"]),
                  _tri(panel.at[i, "ok_np"]), pay_ok, eq_ok, val_ok]
        res.append(all(c is None or c for c in checks))
        n_fail.append(sum(1 for c in checks if c is not None and not c))
        for k, c in zip(SCREEN_KEYS, checks):
            per_check[k].append(c is None or bool(c))
    panel["screen_pass"] = res
    panel["screen_fail"] = n_fail
    for k in SCREEN_KEYS:
        panel["scr_" + k] = per_check[k]
    # 条件ごとに、どれだけの銘柄を落としているか（その条件だけで落ちる＝ほかは通っている）
    _only = {k: float(((~panel["scr_" + k]) & (panel["screen_fail"] == 1)).mean() * 100)
             for k in SCREEN_KEYS}
    _any = {k: float((~panel["scr_" + k]).mean() * 100) for k in SCREEN_KEYS}
    log.info("  条件ごとに落ちる割合（銘柄×月） … " + " ／ ".join(
        f"{SCREEN_LABELS[k]} {_any[k]:.0f}％（この条件だけで {_only[k]:.0f}％）" for k in SCREEN_KEYS))
    log.info("  スクリーニング8条件を通過した割合 … %.0f％（利回りの条件は別に判定）",
             panel["screen_pass"].mean() * 100)
    if got["eps"] == 0 and got["bps"] == 0:
        log.warning("財務の項目が取れていません。項目名が違う可能性があります。")
        # 手がかりとして、最初の1件の項目名を出す
        for code, st in store["stmts"].items():
            if st:
                log.warning("  項目名の例: %s", ", ".join(list(st[0].keys())[:40]))
                break
    return panel


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


def add_pit_tiers(panel: pd.DataFrame) -> pd.DataFrame:
    """その月末に分かっていた情報だけで Tier を判定し直した列を足す（2026年10月12日に追加）。

    本番の Tier（assign_tiers）は、いまの時価総額・いまの Core30・いまの累進配当/DOE のリストで決める。
    実運用ではそれで正しいが、バックテストの全期間に当てはめると「あとで大きくなった会社」
    「あとで累進配当を宣言した会社」を最初から知っていたことになる（後出し）。
    ここでは「業界首位級」を、その月末の時価総額で判定し直す。
      業界首位級 … その月末の時価総額（株価×その時点で開示されていた発行済株式数。どちらも分割調整済み）が
                   全銘柄で上位30（TOPIX Core30 の代わり）、または同じ業種（33業種）で上位3
      tier_pit_lead … 首位級なら A、それ以外は B。累進配当/DOE のリストは使わない（後出しなし）
      tier_pit_mix  … 首位級はその月末で判定し、累進配当/DOE はいまのリストのまま
                      （本番の形のうち、首位級の後出しだけを外したもの）
    累進配当/DOE を宣言した時期はデータにないので、その部分は後出しなしでは作れない。
    """
    panel = panel.copy()
    if "sh" not in panel.columns:
        raise RuntimeError("sh（発行済株式数）の列がありません。財務の項目が足されていません。")
    mc = pd.to_numeric(panel["price"], errors="coerce") * pd.to_numeric(panel["sh"], errors="coerce")
    mc = mc.where(mc > 0)
    sec = panel["sector"].fillna("").astype(str) if "sector" in panel.columns \
        else pd.Series("", index=panel.index)
    rk_all = mc.groupby(panel["date"]).rank(ascending=False, method="first")
    rk_sec = mc.groupby([panel["date"], sec]).rank(ascending=False, method="first")
    lead = ((rk_all <= 30) | ((rk_sec <= 3) & (sec != ""))).astype(bool)
    qual = panel["code"].isin(PROGRESSIVE | DOE)
    panel["lead_pit"] = lead.to_numpy()
    panel["tier_pit_lead"] = np.where(lead, "A", "B")
    panel["tier_pit_mix"] = np.where(lead & qual, "S", np.where(lead | qual, "A", "B"))

    try:
        d0, d1 = panel["date"].min(), panel["date"].max()
        first = panel[panel["date"] == d0]
        last = panel[panel["date"] == d1]
        n_lead = panel[panel["lead_pit"]].groupby("date").size()
        log.info("  後出しなしの業界首位級 … 1か月あたり平均 %.0f銘柄（その月末の時価総額で判定）",
                 float(n_lead.mean()) if len(n_lead) else 0.0)

        if "tier" in panel.columns:
            now_sa = set(last.loc[last["tier"].isin(["S", "A"]), "code"])
            was_lead = set(first.loc[first["lead_pit"], "code"])
            lead_now = set(last.loc[last["lead_pit"], "code"])
            # いまの Tier で S・A なのに、期間の最初の月には首位級でもリストの会社でもなかった銘柄
            late = sorted((now_sa & lead_now) - was_lead - set(PROGRESSIVE | DOE))
            nm_ = first.set_index("code")["name"].to_dict() if "name" in first.columns else {}
            nm_.update(last.set_index("code")["name"].to_dict() if "name" in last.columns else {})
            ex = "、".join(str(nm_.get(c, c))[:10] for c in late[:8])
            log.info("  いまは首位級なのに、%s には首位級でなかった銘柄 … %d社%s",
                     str(pd.Timestamp(d0).date())[:7], len(late), f"（例：{ex}）" if ex else "")
            same = (last["tier"] == last["tier_pit_mix"]).mean() * 100 if len(last) else float("nan")
            log.info("  最後の月（%s）で、本番の Tier と首位級だけ判定し直した Tier が一致する割合 … %.0f％",
                     str(pd.Timestamp(d1).date())[:7], same)
    except Exception as e:      # 記録だけなので、失敗しても検証は続ける
        log.warning("  後出しなしの Tier の記録に失敗: %s", e)
    return panel


def grow_timeline(stmts: list[dict], px: pd.DataFrame) -> pd.DataFrame:
    """開示日ごとに「増配の実績がある会社か」を判定した系列（2026年10月12日に追加）。

    増配の実績がある … 直近3回の通期どうしの比べ（前の通期→次の通期）のうち2回以上で年間配当（実績）が増え、
                       記録のある範囲（最大10期）で一度も減っていない
    その開示の時点で分かっていた通期の数字だけを使う（同じ通期の訂正は、あとの開示で置き換える）。
    比べられる通期が2回分しかないうちは、2回とも増えていることが条件になる。
    """
    ft = fund_timeline(stmts, px)
    if ft.empty or "divann" not in ft.columns:
        return pd.DataFrame()
    known: dict[str, float] = {}
    recs = []
    for _, r in ft.sort_values("date").iterrows():
        fy, v = str(r.get("fy") or ""), r.get("divann")
        if fy and v is not None and not pd.isna(v):
            known[fy] = float(v)
        vals = [known[k] for k in sorted(known)][-10:]
        if len(vals) < 3:
            recs.append({"date": r["date"], "grow": False})
            continue
        nocut = all(b >= a * 0.999 for a, b in zip(vals[:-1], vals[1:]))
        last = vals[-4:]
        ups = sum(1 for a, b in zip(last[:-1], last[1:]) if b > a * 1.001)
        recs.append({"date": r["date"], "grow": bool(nocut and ups >= 2)})
    return pd.DataFrame(recs).groupby("date", as_index=False)["grow"].last()


def add_grow_tiers(store: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """増配の実績で Tier を作った列を足す（2026年10月12日に追加）。add_pit_tiers のあとに呼ぶ。

      grow_pit      … その月末に、増配の実績がある会社か（grow_timeline）
      tier_pit_grow … 首位級（その月末の時価総額）かつ増配の実績なら S、どちらかなら A、それ以外は B
    累進配当・DOEのリスト（宣言の時期が分からず後出しになる）の代わりに使う。
    """
    if "lead_pit" not in panel.columns:
        raise RuntimeError("lead_pit の列がありません。先に add_pit_tiers を呼んでください。")
    panel = panel.copy()
    panel["grow_pit"] = False
    for code, g in panel.groupby("code"):
        px = quotes_to_df(store["quotes"].get(code, []))
        gt = grow_timeline(store["stmts"].get(code, []), px)
        if gt.empty:
            continue
        ser = gt.set_index("date")["grow"]
        vals = ser.reindex(ser.index.union(g["date"])).ffill().reindex(g["date"])
        panel.loc[g.index, "grow_pit"] = vals.fillna(False).astype(bool).to_numpy()
    lead = panel["lead_pit"].astype(bool)
    grow = panel["grow_pit"].astype(bool)
    panel["tier_pit_grow"] = np.where(lead & grow, "S", np.where(lead | grow, "A", "B"))
    try:
        n_grow = panel[grow].groupby("date").size()
        share = float(panel.loc[panel.get("screen_pass", pd.Series(True, index=panel.index)).astype(bool), "grow_pit"]
                      .mean() * 100) if "screen_pass" in panel.columns else float("nan")
        log.info("  増配の実績がある会社（直近3回の比べで2回以上増配・減配なし） … 1か月あたり平均 %.0f銘柄"
                 "（8条件を通った銘柄の %.0f％）", float(n_grow.mean()) if len(n_grow) else 0.0, share)
    except Exception as e:      # 記録だけなので、失敗しても検証は続ける
        log.warning("  増配の実績の記録に失敗: %s", e)
    return panel


def add_yield_reason_columns(store: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """利回りが上がった理由を見るための列を足す（2026年10月12日に追加）。

      px_chg12  … その月末の株価（分割調整済み。配当は含まない）が、12か月前の月末から何％変わったか
      dps_chg12 … その月末に分かっていた1株配当（予想。分割調整済み）が、12か月前の月末から何％変わったか
    どちらもその月末までの情報だけを使う。パネルは検証期間で切ってあるので、切る前の株価から計算する
    （build_panel と同じ株価・同じ1株配当の作り方）。
    """
    panel = panel.copy()
    panel["px_chg12"] = np.nan
    panel["dps_chg12"] = np.nan
    n_ok = 0
    for code, g in panel.groupby("code"):
        px = quotes_to_df(store["quotes"].get(code, []))
        if px.empty:
            continue
        m = px.set_index("date")["close"].resample("ME").last()      # 抜けた月は空のまま（12行前＝12か月前）
        dps = dps_timeline(store["stmts"].get(code, []), px)
        if dps.empty:
            continue
        d = dps.set_index("date")["dps"].reindex(m.index, method="ffill")
        pc = m / m.shift(12) - 1.0
        dc = d / d.shift(12) - 1.0
        panel.loc[g.index, "px_chg12"] = pc.reindex(g["date"]).to_numpy()
        panel.loc[g.index, "dps_chg12"] = dc.reindex(g["date"]).to_numpy()
        n_ok += 1
    drop = (panel["px_chg12"] <= -0.20) & ~(panel["dps_chg12"] > 0)
    log.info("  利回りが上がった理由 … 取れた銘柄 %d社 ／ 株価が1年で20％以上下がり、配当は増えていない銘柄×月 %.0f％",
             n_ok, float(drop.mean() * 100) if len(panel) else 0.0)
    return panel


def add_factor_columns(panel: pd.DataFrame, mk: dict,
                       win: int = 6) -> pd.DataFrame:
    """外部要因が追い風か逆風かを、月ごとにパネルへ足す。

    前月までの値だけで判定するので、後出しにはならない。
    """
    panel = panel.copy()
    src = {"rate": "us10y", "oil": "brent", "fx": "usdjpy"}
    for fac, key in src.items():
        sr = mk.get(key)
        col = f"trend_{fac}"
        if sr is None or (hasattr(sr, "empty") and sr.empty):
            panel[col] = 0.0
            continue
        m = sr.resample("ME").last().dropna()
        tr = factor_trend(m, win)
        panel[col] = panel["date"].map(tr).fillna(0.0)
    return panel


# ══════════════════════════════════════════
# うねり（箱）の判定（2026年10月12日に追加。8回目の検証）
#   目で引く「上の線・下の線」に近づけるため、最高値・最安値ではなく、
#   その前の日までの終値の90％点を上の線、10％点を下の線にする（突き抜けたひげを無視する）。
#   線どうしの幅が決めた範囲に入っていれば「うねりの中」とみなす。
#   線から1％以内に来たら「届いた」とし、売りは上の線の1％下、買いは下の線の1％上の値段で約定したとみなす
#   （指値を置いておく形。その日の終値がそこまで来た日だけ）。
# ══════════════════════════════════════════
BOX_SETS = {
    "s": {"win": 20, "w_min": 0.03, "w_max": 0.08, "label": "短め（20日・幅3〜8％）"},
    "m": {"win": 60, "w_min": 0.04, "w_max": 0.12, "label": "中くらい（60日・幅4〜12％）"},
    "l": {"win": 120, "w_min": 0.06, "w_max": 0.18, "label": "長め（120日・幅6〜18％）"},
}
BOX_Q = (0.10, 0.90)
BOX_TOUCH = 0.01


def box_signals(close: pd.Series, win: int, w_min: float, w_max: float) -> pd.DataFrame:
    """日々の終値から、うねりの線と「届いた日」を出す（その前の日までの終値だけを使う）。"""
    prev = close.shift(1)
    hi = prev.rolling(win, min_periods=win).quantile(BOX_Q[1])
    lo = prev.rolling(win, min_periods=win).quantile(BOX_Q[0])
    width = (hi - lo) / lo
    ok = (width >= w_min) & (width <= w_max)
    sell_px = hi * (1 - BOX_TOUCH)
    buy_px = lo * (1 + BOX_TOUCH)
    return pd.DataFrame({"close": close, "hi": hi, "lo": lo, "ok": ok.fillna(False),
                         "top": (ok & (close >= sell_px)).fillna(False),
                         "bot": (ok & (close <= buy_px)).fillna(False),
                         "sell_px": sell_px, "buy_px": buy_px})


def add_box_columns(store: dict, panel: pd.DataFrame, tags) -> pd.DataFrame:
    """うねりの列を月ごとにパネルへ足す（tags は BOX_SETS のキー）。

      box_top_<tag> … その月に初めて上の線に届いた日の売り値（届かなければ空）
      box_bot_<tag> … その月に初めて下の線に届いた日の買い値（届かなければ空）
      in_box_<tag>  … その月末の時点で、うねりの中にいるか
    """
    panel = panel.copy()
    tags = [t for t in tags if t in BOX_SETS]
    for t in tags:
        panel[f"box_top_{t}"] = np.nan
        panel[f"box_bot_{t}"] = np.nan
        panel[f"in_box_{t}"] = False
    for code, g in panel.groupby("code"):
        px = quotes_to_df(store["quotes"].get(code, []))
        if px.empty:
            continue
        close = px.set_index("date")["close"]
        for t in tags:
            prm = BOX_SETS[t]
            sg = box_signals(close, prm["win"], prm["w_min"], prm["w_max"])
            sg["m"] = sg.index.to_period("M")
            top = sg[sg["top"]].groupby("m")["sell_px"].first()
            bot = sg[sg["bot"]].groupby("m")["buy_px"].first()
            last_ok = sg.groupby("m")["ok"].last()
            mp = g["date"].dt.to_period("M")
            panel.loc[g.index, f"box_top_{t}"] = mp.map(top).to_numpy(dtype=float)
            panel.loc[g.index, f"box_bot_{t}"] = mp.map(bot).to_numpy(dtype=float)
            panel.loc[g.index, f"in_box_{t}"] = mp.map(last_ok).fillna(False).astype(bool).to_numpy()
    for t in tags:
        log.info("  うねり（%s） … 月末にうねりの中にいた銘柄×月 %.0f％ ／ 上の線に届いた %.0f％ ／ 下の線に届いた %.0f％",
                 BOX_SETS[t]["label"], float(panel[f"in_box_{t}"].mean() * 100),
                 float(panel[f"box_top_{t}"].notna().mean() * 100),
                 float(panel[f"box_bot_{t}"].notna().mean() * 100))
    return panel


def add_range_columns(store: dict, panel: pd.DataFrame, win: int,
                      w_min: float, w_max: float, sell_at: float) -> pd.DataFrame:
    """「その月のうちにレンジ上限へ届いたか、いくらで届いたか」を足す。

    月次のパネルだと、ひと月のあいだの往復が見えない。
    そこで日次で上限到達を調べ、月ごとに最初に届いた値段を記録する。
    レンジの上下は前日までの終値から決めるので、未来の情報は入らない。
    """
    top_px, in_box = {}, {}
    for code, rows in store["quotes"].items():
        px = quotes_to_df(rows)
        if px.empty or len(px) < win + 20:
            continue
        sr = px.set_index("date")["close"]
        prev = sr.shift(1)
        hi = prev.rolling(win, min_periods=win).max()
        lo = prev.rolling(win, min_periods=win).min()
        width = (hi - lo) / lo
        ok = (width >= w_min) & (width <= w_max)
        hit = ok & (sr >= hi * (1 - sell_at))
        if not hit.any():
            continue
        # 月ごとに、最初に届いた日の値段
        df = pd.DataFrame({"px": sr, "hit": hit, "ok": ok})
        df["m"] = df.index.to_period("M")
        for m, g in df[df["hit"]].groupby("m"):
            top_px[(code, m)] = float(g["px"].iloc[0])
        for m, g in df.groupby("m"):
            in_box[(code, m)] = bool(g["ok"].any())

    panel = panel.copy()
    mp = panel["date"].dt.to_period("M")
    keys = list(zip(panel["code"], mp))
    panel["box_top_px"] = [top_px.get(k, np.nan) for k in keys]
    panel["in_box"] = [in_box.get(k, False) for k in keys]
    return panel


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
    sector_by_code = {u["code"]: u.get("sector", "") for u in store["universe"]}

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
        f["sector"] = sector_by_code.get(code, "")
        # 累進配当・DOE銘柄は「減配しにくい」と宣言している。
        # 利確の扱いを変えるかどうかを試せるようにフラグを持たせる。
        f["progressive"] = (code in PROGRESSIVE) or (code in DOE)
        # 減配の検知。直近1年の最高額を下回ったら減配とみなす。
        # 配当利回りで割安さを測る戦略では、減配は「買った根拠が消える」出来事。
        # 営業利益の急変。前回開示と比べて大きく落ちたか。
        opt = op_timeline(store["stmts"].get(code, []))
        if not opt.empty and len(opt) >= 2:
            os_ = opt.set_index("date")["op"]
            f["op"] = f["date"].map(os_).ffill() if "date" in f.columns \
                else np.nan
            _prev = f["op"].shift(1).where(f["op"] != f["op"].shift(1)).ffill()
            f["op_drop"] = ((f["op"] < _prev * 0.8) & _prev.notna()
                            & (_prev > 0)).fillna(False)
        else:
            f["op"] = np.nan
            f["op_drop"] = False

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

    # TOPIX が高値から何％下げているか。暴落局面の判定に使う。
    # 過去12か月の高値と比べるので、未来の情報は入らない。
    tpx = store.get("topix")
    if isinstance(tpx, pd.DataFrame) and not tpx.empty:
        tm = tpx.set_index("date")["close"].resample("ME").last().dropna()
        peak = tm.rolling(12, min_periods=3).max()
        dd = (tm / peak - 1.0) * 100.0
        panel["topix"] = panel["date"].map(tm)
        panel["mkt_dd"] = panel["date"].map(dd)
    else:
        panel["topix"] = np.nan
        panel["mkt_dd"] = np.nan

    start = panel["date"].max() - pd.DateOffset(years=years)
    return panel[panel["date"] >= start].dropna(subset=["pct_own"]).reset_index(drop=True)


# 日々の終値（2026年10月11日に追加）。価格で損切りするルールの判定に使う。
#   銘柄コード → (日付の配列［1970年からのナノ秒］, 終値の配列)
#   終値はパネルと同じ「分割を調整した終値」なので、取得単価とそのまま比べられる。
#   損切りのルールを選んだときだけ main で用意する。None のときは月末の値で判定する。
DAILY_PX = None


def build_daily_arrays(store: dict) -> dict:
    """損切りの判定用に、銘柄ごとの日々の終値を配列にしておく。"""
    out = {}
    for code, rows in store["quotes"].items():
        px = quotes_to_df(rows)
        if px.empty:
            continue
        out[code] = (px["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
                     px["close"].to_numpy(dtype=float))
    return out


# ══════════════════════════════════════════
# うねりを日ごとに売り買いする（9回目。2026年10月12日に追加）
#   線の引き方は8回目と同じ（その前の日までの終値の90％点と10％点）。
#   届いたとみなす位置だけを、うねりの幅に合わせて変える：
#     売り … 上の線から「幅の15％」下　　買い … 下の線から「幅の15％」上
#   （幅3％のうねりなら線から0.45％、幅10％なら1.5％。細かいうねりでも取れる幅が残るように）
# ══════════════════════════════════════════
DAILY_BOX_SETS = {
    "xs": {"win": 20, "w_min": 0.02, "w_max": 0.05, "label": "細かい（20日・幅2〜5％）"},
    "s": {"win": 20, "w_min": 0.03, "w_max": 0.08, "label": "短め（20日・幅3〜8％）"},
    "m": {"win": 60, "w_min": 0.04, "w_max": 0.12, "label": "中くらい（60日・幅4〜12％）"},
    # 10回目（2026年10月12日に追加）：目で引く「何度も止まった上下の横線」。引き方は hline_lines を参照
    "h1": {"method": "hline", "win": 250, "piv": 10, "tol": 0.02, "w_min": 0.05, "w_max": 0.25,
           "label": "1年線（250日・幅5〜25％）"},
    "h2": {"method": "hline", "win": 500, "piv": 15, "tol": 0.02, "w_min": 0.06, "w_max": 0.35,
           "label": "2年線（500日・幅6〜35％）"},
    "h3": {"method": "hline", "win": 750, "piv": 20, "tol": 0.025, "w_min": 0.08, "w_max": 0.50,
           "label": "3年線（750日・幅8〜50％）"},
    # 12回目（2026年10月12日に追加）：2年線の設定を少しだけ変えたもの（崩れないかを見る。成績を見て選んだものではない）
    #   tol … 「線に触れた」とみなす幅（2年線は ±2％）　inset … 売り値・買い値を線から幅の何割内側に置くか（10％）
    "h2t15": {"method": "hline", "win": 500, "piv": 15, "tol": 0.015, "w_min": 0.06, "w_max": 0.35,
              "label": "2年線（触れた判定の幅1.5％）"},
    "h2t25": {"method": "hline", "win": 500, "piv": 15, "tol": 0.025, "w_min": 0.06, "w_max": 0.35,
              "label": "2年線（触れた判定の幅2.5％）"},
    "h2i05": {"method": "hline", "win": 500, "piv": 15, "tol": 0.02, "w_min": 0.06, "w_max": 0.35,
              "inset": 0.05, "label": "2年線（売り買いの位置5％）"},
    "h2i15": {"method": "hline", "win": 500, "piv": 15, "tol": 0.02, "w_min": 0.06, "w_max": 0.35,
              "inset": 0.15, "label": "2年線（売り買いの位置15％）"},
}
SWING_TOUCH_FRAC = 0.15
SWING_DAILY = None


def box_lines(close: pd.Series, win: int, w_min: float, w_max: float,
              k: float = SWING_TOUCH_FRAC) -> pd.DataFrame:
    """日ごとのうねりの線と、売り値・買い値（その前の日までの終値だけを使う）。"""
    prev = close.shift(1)
    hi = prev.rolling(win, min_periods=win).quantile(BOX_Q[1])
    lo = prev.rolling(win, min_periods=win).quantile(BOX_Q[0])
    width = (hi - lo) / lo
    ok = ((width >= w_min) & (width <= w_max)).fillna(False)
    return pd.DataFrame({"close": close, "hi": hi, "lo": lo, "ok": ok,
                         "sell_px": hi - k * (hi - lo), "buy_px": lo + k * (hi - lo)})


# ══════════════════════════════════════════
# 水平線でうねりを取る（10回目。2026年10月12日に追加）
#   目で引く「何度も止まった上下の横線」を、機械で引き直す。
#   ・山 … 前後 piv 日のなかでいちばん高い高値。谷 … いちばん安い安値（高値・安値がなければ終値）
#   ・山と谷を、高さが ±tol 以内のものどうしで集める（集まった数＝その高さに触れた回数）
#   ・上の線 … 期間の終値の真ん中より上の山で作る。触れた回数が「最多−1」以上（最低2回）の集まりのうち、
#              いちばん高い集まりを選び、その集まりの上から4分の1の高さに引く（ヒゲ1本の飛び出しは無視する）
#   ・下の線 … 真ん中より下の谷で同じように作り、いちばん安い集まりの下から4分の1の高さに引く
#   ・線として認めるのは、次をすべて満たすとき
#       上下とも2回以上触れている／幅（上÷下−1）が w_min〜w_max／期間の終値の75％以上が線の間（±tol）／
#       終値が「上の線から幅の25％」と「下の線から幅の25％」の帯を3回以上行き来している／
#       前の日の終値が線の間（±tol）にある
#   ・線は5営業日ごとに引き直す。使うのはその日より前のデータだけ（山と谷は、後ろに piv 日あって確定したものだけ）
#   ・売り … その日の高値が「上の線から幅の10％下」に届いたら、その値段で売る（指値）
#     買い … その日の安値が「下の線から幅の10％上」に届いたら、その値段で買う（指値）
#   2026年10月12日にいただいた3つのチャート（アイチコーポレーション・全国保証・アサヒグループHD）の形を
#   読み取って試し、手で引いた線に近くなる引き方を選んだ（成績を見て選んだものではない）。
# ══════════════════════════════════════════
LINE_TOUCH_FRAC = 0.10      # 売り値・買い値を、線から幅の何割内側に置くか
LINE_STEP = 5               # 線を引き直す間隔（営業日）
LINE_INSIDE_MIN = 0.75      # 期間の終値のうち、線の間にいた割合の下限
LINE_ALT_MIN = 3            # 上の帯と下の帯を行き来した回数の下限
LINE_ZONE = 0.25            # 行き来を数える帯の厚み（幅に対する割合）
LINE_EXAMPLES = ("6345", "7164", "2502")   # いただいたチャートの3銘柄（線の引き方の確認用に、いまの線を表示する）
PLACEBO_N = 20              # 12回目：ランダムな日の売り買いを何通り回すか（期間ごと）


def code_half_of(code) -> int:
    """銘柄を決まった方法で2つに分ける（12回目）。銘柄コードの CRC32 の偶奇。毎回同じ分かれ方になる。"""
    return zlib.crc32(str(code).encode("utf-8")) % 2


def quotes_hl(rows: list[dict]) -> pd.DataFrame:
    """日々の終値に、高値・安値（分割を調整したもの）を足す。行は quotes_to_df と同じ。
    調整済みの高値・安値が無ければ、生の高値・安値に「調整済み終値÷生の終値」を掛けて揃える。
    どちらも無ければ終値で代用する（src に何を使ったかを残す）。"""
    df = pd.DataFrame(rows)
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    close = pd.to_numeric(df.get("AdjC", df.get("AdjustmentClose", df.get("C", df.get("Close")))),
                          errors="coerce")
    hi = lo = None
    src = "終値"

    def _first(keys):
        return next((k for k in keys if k in df.columns), None)

    kh, kl = _first(("AdjH", "AdjustmentHigh")), _first(("AdjL", "AdjustmentLow"))
    if kh and kl:
        hi, lo, src = pd.to_numeric(df[kh], errors="coerce"), pd.to_numeric(df[kl], errors="coerce"), "高値・安値"
    else:
        kh, kl, kc = _first(("H", "High")), _first(("L", "Low")), _first(("C", "Close"))
        if kh and kl and kc:
            raw_c = pd.to_numeric(df[kc], errors="coerce")
            ratio = (close / raw_c).where(raw_c > 0)
            hi = pd.to_numeric(df[kh], errors="coerce") * ratio
            lo = pd.to_numeric(df[kl], errors="coerce") * ratio
            src = "高値・安値"
    out = pd.DataFrame({"date": df["Date"], "close": close})
    out["high"] = hi if hi is not None else np.nan
    out["low"] = lo if lo is not None else np.nan
    out = out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    out["high"] = np.fmax(out["high"].fillna(out["close"]).to_numpy(dtype=float), out["close"].to_numpy(dtype=float))
    out["low"] = np.fmin(out["low"].fillna(out["close"]).to_numpy(dtype=float), out["close"].to_numpy(dtype=float))
    out.attrs["src"] = src
    return out


def _line_thin(idx: np.ndarray, gap: int) -> np.ndarray:
    """同じ高さが続いたときに山（谷）が並ばないよう、gap 日以内の続きは最初の1つだけ残す。"""
    if len(idx) == 0:
        return idx.astype(np.int64)
    keep = [int(idx[0])]
    for i in idx[1:]:
        if int(i) - keep[-1] > gap:
            keep.append(int(i))
    return np.array(keep, dtype=np.int64)


def _line_pivots(h: np.ndarray, l: np.ndarray, piv: int) -> tuple[np.ndarray, np.ndarray]:
    """山（前後 piv 日でいちばん高い高値）と谷（いちばん安い安値）の位置。端の piv 日は確定しないので出ない。"""
    rmax = pd.Series(h).rolling(2 * piv + 1, center=True).max().to_numpy()
    rmin = pd.Series(l).rolling(2 * piv + 1, center=True).min().to_numpy()
    with np.errstate(invalid="ignore"):
        ph = np.flatnonzero(h >= rmax)
        pl = np.flatnonzero(l <= rmin)
    return _line_thin(ph, piv), _line_thin(pl, piv)


def _line_cluster(P: np.ndarray, tol: float, upper: bool) -> tuple[float, int]:
    """高さの近い山（谷）を集め、線の高さと触れた回数を返す。"""
    if len(P) == 0:
        return float("nan"), 0
    M = np.abs(P[None, :] - P[:, None]) <= tol * P[:, None]
    cnt = M.sum(axis=1)
    best = int(cnt.max())
    cand = np.flatnonzero(cnt >= max(2, best - 1))
    if len(cand) == 0:
        return float(P[int(np.argmax(cnt))]), best
    j = cand[int(np.argmax(P[cand]))] if upper else cand[int(np.argmin(P[cand]))]
    mem = P[M[j]]
    return float(np.percentile(mem, 75 if upper else 25)), int(cnt[j])


def _hline_eval(c: np.ndarray, h: np.ndarray, l: np.ndarray, ph: np.ndarray, pl: np.ndarray,
                t: int, prm: dict) -> dict:
    """t 日目の前の日までのデータで線を引く。t 日目そのものは使わない。"""
    win, piv, tol = int(prm["win"]), int(prm["piv"]), float(prm["tol"])
    a, b = t - win, t
    r = {"ok": False, "why": "", "U": float("nan"), "D": float("nan"), "nU": 0, "nD": 0,
         "width": float("nan"), "inside": float("nan"), "alts": 0}
    if a < 0:
        r["why"] = "期間が足りない"
        return r
    wc = c[a:b]
    med = float(np.median(wc))
    hi_i = ph[np.searchsorted(ph, a):np.searchsorted(ph, b - 1 - piv, side="right")]
    lo_i = pl[np.searchsorted(pl, a):np.searchsorted(pl, b - 1 - piv, side="right")]
    H = h[hi_i]
    H = H[H > med]
    L = l[lo_i]
    L = L[L < med]
    U, nU = _line_cluster(H, tol, True)
    D, nD = _line_cluster(L, tol, False)
    r.update(U=U, D=D, nU=nU, nD=nD)
    if nU < 2 or nD < 2:
        r["why"] = "触れた回数が足りない"
        return r
    w = U / D - 1.0
    r["width"] = w
    if w < prm["w_min"]:
        r["why"] = "幅が狭い"
        return r
    if w > prm["w_max"]:
        r["why"] = "幅が広い"
        return r
    ins = float(np.mean((wc >= D * (1 - tol)) & (wc <= U * (1 + tol))))
    z = LINE_ZONE * (U - D)
    lab = np.where(wc >= U - z, 1, np.where(wc <= D + z, -1, 0))
    nz = lab[lab != 0]
    alts = int(np.count_nonzero(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    r.update(inside=ins, alts=alts)
    if ins < LINE_INSIDE_MIN:
        r["why"] = "線の外にいた時間が長い"
        return r
    if alts < LINE_ALT_MIN:
        r["why"] = "上下の行き来が足りない"
        return r
    r["ok"] = True
    return r


def hline_core(hl: pd.DataFrame, prm: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """日ごとの水平線（上の線・下の線）と「線の中か」（その日より前のデータだけを使う）。hl は quotes_hl の形。"""
    c = hl["close"].to_numpy(dtype=float)
    h = hl["high"].to_numpy(dtype=float)
    l = hl["low"].to_numpy(dtype=float)
    n = len(c)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    val = np.zeros(n, dtype=bool)
    win = int(prm["win"])
    if n > win:
        ph, pl = _line_pivots(h, l, int(prm["piv"]))
        for t in range(win, n, LINE_STEP):
            r = _hline_eval(c, h, l, ph, pl, t, prm)
            if r["ok"]:
                e = min(n, t + LINE_STEP)
                hi[t:e], lo[t:e], val[t:e] = r["U"], r["D"], True
    prev = np.r_[np.nan, c[:-1]]
    tol = float(prm["tol"])
    with np.errstate(invalid="ignore"):
        ok = val & (prev >= lo * (1 - tol)) & (prev <= hi * (1 + tol))
    return hi, lo, ok


def hline_lines(hl: pd.DataFrame, prm: dict) -> pd.DataFrame:
    """日ごとの水平線と、売り値・買い値（その日より前のデータだけを使う）。hl は quotes_hl の形。
    売り値・買い値は、線から「幅の inset 割」内側（既定は LINE_TOUCH_FRAC＝10％）。"""
    hi, lo, ok = hline_core(hl, prm)
    k = float(prm.get("inset", LINE_TOUCH_FRAC))
    return pd.DataFrame({"close": hl["close"].to_numpy(dtype=float), "high": hl["high"].to_numpy(dtype=float),
                         "low": hl["low"].to_numpy(dtype=float), "hi": hi, "lo": lo, "ok": ok,
                         "sell_px": hi - k * (hi - lo), "buy_px": lo + k * (hi - lo)})


def hline_snapshot(hl: pd.DataFrame, prm: dict) -> dict:
    """いま（最後の日の終値まで）の線。表示用。"""
    c = hl["close"].to_numpy(dtype=float)
    h = hl["high"].to_numpy(dtype=float)
    l = hl["low"].to_numpy(dtype=float)
    ph, pl = _line_pivots(h, l, int(prm["piv"]))
    r = _hline_eval(c, h, l, ph, pl, len(c), prm)
    r["last"] = float(c[-1]) if len(c) else float("nan")
    r["date"] = hl["date"].iloc[-1] if len(c) else None
    # いまの終値が線のどこにいるか（線の外なら売り買いはしない）
    tol = float(prm["tol"])
    if np.isfinite(r["U"]) and np.isfinite(r["D"]) and np.isfinite(r["last"]):
        if r["last"] > r["U"] * (1 + tol):
            r["pos"] = "いまは上に抜けている"
        elif r["last"] < r["D"] * (1 - tol):
            r["pos"] = "いまは下に割れている"
        else:
            r["pos"] = "いまは線の中"
        k = LINE_TOUCH_FRAC
        r["sell_px"] = r["U"] - k * (r["U"] - r["D"])
        r["buy_px"] = r["D"] + k * (r["U"] - r["D"])
    else:
        r["pos"] = ""
    return r


def build_swing_daily(store: dict, tags) -> dict:
    """銘柄ごとに、日々の（日付, 終値, 売り値, 買い値, うねりの中か）を配列にしておく。
    水平線（10回目）は、そのうしろに（高値, 安値）も付ける。売りは高値、買いは安値で届いたかを見るため。"""
    out = {"_cal": None, "_snap": {}, "_hlsrc": {}}
    cal = set()
    for t in tags:
        out[t] = {}
    _hl_tags = [t for t in tags if DAILY_BOX_SETS[t].get("method") == "hline"]
    for code, rows in store["quotes"].items():
        px = quotes_to_df(rows)
        if px.empty:
            continue
        close = px.set_index("date")["close"]
        ds = px["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        cal.update(ds.tolist())
        hl = quotes_hl(rows) if _hl_tags else None
        if hl is not None and len(hl) != len(px):
            hl = None
        if hl is not None:
            _s = hl.attrs.get("src", "終値")
            out["_hlsrc"][_s] = out["_hlsrc"].get(_s, 0) + 1
            # 終値・高値・安値は3つの線で同じものを使い回す（メモリを節約する）
            _c_arr = hl["close"].to_numpy(dtype=float)
            _h_arr = hl["high"].to_numpy(dtype=float)
            _l_arr = hl["low"].to_numpy(dtype=float)
        _geo = {}       # 線の形（日数・山谷の幅・触れた判定の幅・幅の範囲）が同じなら、引いた線を使い回す
        for t in tags:
            prm = DAILY_BOX_SETS[t]
            if prm.get("method") == "hline":
                if hl is None:
                    continue
                _key = (prm["win"], prm["piv"], prm["tol"], prm["w_min"], prm["w_max"])
                if _key not in _geo:
                    _geo[_key] = hline_core(hl, prm)
                _hi, _lo, _ok = _geo[_key]
                _k = float(prm.get("inset", LINE_TOUCH_FRAC))
                out[t][code] = (ds, _c_arr, _hi - _k * (_hi - _lo), _lo + _k * (_hi - _lo), _ok, _h_arr, _l_arr)
                if code in LINE_EXAMPLES:
                    out["_snap"].setdefault(t, {})[code] = hline_snapshot(hl, prm)
                continue
            bl = box_lines(close, prm["win"], prm["w_min"], prm["w_max"])
            out[t][code] = (ds, bl["close"].to_numpy(dtype=float),
                            bl["sell_px"].to_numpy(dtype=float), bl["buy_px"].to_numpy(dtype=float),
                            bl["ok"].to_numpy(dtype=bool))
    out["_cal"] = np.array(sorted(cal), dtype=np.int64)
    return out


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
             fee_bps: float = 0.0, tax_rate: float = 0.0,
             ramp: int = 0, min_yield_override: float | None = None) -> dict:
    """月末ごとに判定して売買する。等金額・分割建玉。

    出口は3通りを組み合わせられる。
      exit       … 利回りの分位が下がったら降りる（従来）
      gain_exit  … 取得単価から一定率上がったら降りる
      rotate     … 枠が埋まっているとき、もっと良い候補と入れ替える
    """
    entry, exits = cfg["entry"], cfg.get("exit", [])
    dyn, cross = cfg.get("dynamic", False), cfg.get("cross", False)
    # 3回目の候補（2026年10月11日に追加）
    #   max_names     … 保有の上限をルールの側で決める（総当たりの「保有上限」より優先）。
    #                   目標25銘柄を試すときに、上限20で頭打ちにならないようにするため
    #   tiers_allowed … 新しく買う Tier を限る（例 ["S", "A"] なら B は買わない）
    if cfg.get("max_names"):
        max_names = int(cfg["max_names"])
    tiers_allowed = set(cfg["tiers_allowed"]) if cfg.get("tiers_allowed") else None
    # 5回目の候補（2026年10月12日に追加）
    #   tier_col … Tier をこの列で置き換える（add_pit_tiers が作る、後出しなしで判定した Tier）
    # 7回目の候補（2026年10月12日に追加）
    #   skip_price_drop … 12か月で株価がこの率以上下がり、1株配当（予想）が増えていない銘柄は買わない
    skip_drop = cfg.get("skip_price_drop")
    # 11回目の候補（2026年10月12日に追加）
    #   require_col … この列が True の銘柄だけを新しく買う（例 "grow_pit"＝増配の実績がある会社）。
    #                 月末の買いと、日々のうねりの買いの両方に効く。持っている銘柄を売る理由にはしない。
    require_col = cfg.get("require_col")
    # 12回目（2026年10月12日に追加）：銘柄を決まった方法で半分に分け、片方だけを買う（0 か 1）。
    #   分け方は銘柄コードの CRC32 の偶奇（毎回同じ分かれ方になる）。別々の銘柄群で同じ結果が出るかを見るため。
    code_half = cfg.get("code_half")
    if require_col and require_col not in panel.columns:
        raise RuntimeError(f"{require_col} の列がありません。増配の実績の列が足されていません。")
    # 8回目の候補（2026年10月12日に追加）
    #   swing_entry … {"tag": "m", "tiers": ["A", "B"]}。この Tier の銘柄は、うねりの中にいるあいだは
    #                 下の線に届いたときだけ（その値段で）買う。うねりの中にいなければ、いつもどおり月末に買う。
    #                 上の線で売るのは range_exit（top_col で列を選ぶ）。売った月のうちは買い直さない。
    swing_entry = cfg.get("swing_entry")
    swing_sold = {}
    # 9回目の候補（2026年10月12日に追加）
    #   swing_daily … {"tag": "m", "tiers": ["A", "B"], "min_gain": 0.0}。この Tier の持ち株は、
    #                 うねりの中で売り値に届いた日に売る。買える候補は、うねりの中で買い値に届いた日に買う。
    #                 月の中で何往復してもよい。月末の買いでは、うねりの中にいる銘柄は買わずに待つ。
    swing_daily = cfg.get("swing_daily")
    if swing_daily:
        if SWING_DAILY is None or swing_daily.get("tag", "m") not in SWING_DAILY:
            raise RuntimeError("日々のうねりの線（SWING_DAILY）が用意されていません。")
        _swd = SWING_DAILY[swing_daily.get("tag", "m")]
        _sw_tiers = set(swing_daily.get("tiers", ["A", "B"]))
        # 12回目（2026年10月12日に追加）：比べるための「ランダムな日の売り買い」
        #   placebo … {"seed": 1, "p_sell": 0.01, "p_buy": 0.01}。線の中にいる日（売り買いしてよい日）のうち、
        #             売りは p_sell、買いは p_buy の確率でランダムに選んだ日に、その日の安値〜高値の間の
        #             ランダムな値段で売り買いする。線の位置（売り値・買い値）は使わない。
        #             売る銘柄・買う候補・月末に待つ決まり・回数の目安は線の形と同じにして、日を選ぶところだけ変える。
        _plc = swing_daily.get("placebo")
        _rng = np.random.default_rng(int(_plc["seed"])) if _plc else None

        def _box_ok_at(code_, t_ns):
            arr_ = _swd.get(code_)
            if arr_ is None:
                return False
            j_ = int(np.searchsorted(arr_[0], t_ns, side="right")) - 1
            return bool(arr_[4][j_]) if j_ >= 0 else False
    if swing_entry:
        _t = swing_entry.get("tag", "m")
        if f"box_bot_{_t}" not in panel.columns:
            raise RuntimeError(f"box_bot_{_t} の列がありません。うねりの列が足されていません。")
    if skip_drop is not None and "px_chg12" not in panel.columns:
        raise RuntimeError("px_chg12 の列がありません。利回りが上がった理由の列が足されていません。")
    if cfg.get("tier_col"):
        _tc = cfg["tier_col"]
        if _tc not in panel.columns:
            raise RuntimeError(f"{_tc} の列がありません。後出しなしの Tier が足されていません。")
        panel = panel.assign(tier=panel[_tc].to_numpy())
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
        """割安さの指標。大きいほど割安（買いたい）。"""
        if measure == "z":
            v = row.get("zscore")
            return float(v) if v is not None and not pd.isna(v) else -99.0
        if measure in ("pbr", "per", "size", "pbr_sec", "value",
                       "pbrroe70", "pbrroe50", "pbrroe30", "switch36"):
            # 全銘柄の中での順位（％）。pbr/per は低いほど、size は小さいほど高い値になる
            if "pct_" + measure not in row.index:
                raise RuntimeError(f"pct_{measure} の列がありません。財務の項目が足されていません。")
            v = row.get("pct_" + measure)
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
    # 営業利益が前回開示比で2割超落ちたら撤退するか。
    # 実運用には入っているが、検証では一度も試していなかった。
    exit_on_op = cfg.get("exit_on_op", False)
    # 部分利確。上がったら「一部だけ」売る。
    #   threshold … 何％上がったら売るか（前回売った値段からの上昇率）
    #   fraction  … そのとき何割を売るか
    # 全部売ると税金を一度に払い、伸びしろも失う。
    # 一部だけなら現金も入り、残りは走り続ける。
    partial = cfg.get("partial_gain")
    # レンジの上限に届いたら降りる。
    # 土台は「売らない」戦略のまま、その上に往復を乗せる形。
    # 外れてレンジを割っても、ただ持ち続けるだけで損失にはならない。
    range_exit = cfg.get("range_exit")
    # 外部要因が逆風の業種を、どう扱うか。
    #   skip  … 買わない
    #   thin  … 薄く買う（倍率を指定）
    # 追い風か逆風かは、前月までの値だけで判定している。
    factor_rule = cfg.get("factor_rule")
    # 買い増しを許すか。既定は1銘柄1回まで。
    # さらに下がったところで買い増すと、平均取得単価が下がる代わりに
    # 1銘柄への比重が増える。効くかどうかは検証で判断する。
    add_on = cfg.get("add_on", False)
    add_on_max = cfg.get("add_on_max", 3)
    # 利回りの足切り。総当たりでは条件ごとに差し替えるので、
    # 指定があればそちらを優先する。
    min_yield = cfg.get("min_yield", 0.0) if min_yield_override is None \
        else float(min_yield_override)
    # ルールの側で足切りを固定したいとき（総当たりの足切りの軸より優先する）。
    # 足切りどうしを直接比べるために使う。
    if cfg.get("min_yield_fixed") is not None:
        min_yield = float(cfg["min_yield_fixed"])
    # スクリーニング8条件を使うか（本番と同じ絞り込み）
    use_screen = bool(cfg.get("screen", False))
    # 緊急撤退の判定のしかた（2026年10月11日に追加）
    #   既定   … 予想配当が直近1年の最高額を少しでも下回ったら「減配」（以前からの判定）
    #   actual … 本番の generate.py と同じ。実績の年間配当が前の通期より10％以上減ったら「減配」
    #            （div_cut10）、営業利益が前の通期より20％以上減ったら「業績急変」（op_cut20）
    #   注意：以前からの「業績急変」（op_drop）は、開示日と月末の日付が一致したときしか
    #   営業利益を拾っておらず、ほとんど働いていなかった（2026年10月11日に判明）。
    #   以前のルールの結果を再現できるよう、そちらはそのまま残してある。
    emg_actual = cfg.get("emg_rule") == "actual"
    # 緊急撤退に当たる銘柄（減配・業績急変）を買い候補から外すか。
    # 本番の portfolio_engine は外している。以前の検証は外しておらず、
    # 売ったその月に同じ値段で買い直すことがあった。
    skip_emg_buy = bool(cfg.get("skip_emg_buy", False))
    # 緊急撤退の判定を「落ち方」で変える（2026年10月11日に追加）。emg_rule が actual のときだけ。
    #   emg_cut_thr      … 実績の年間配当が前の通期からこの率以上減ったら「減配」（既定 0.10）
    #   emg_op_thr       … 営業利益がこの率以上減ったら「業績急変」（既定 0.20。1.0 なら赤字転落だけ）
    #   emg_require_both … 減配と業績急変の両方が重なったときだけ売る
    #   emg_forecast     … 予想の年間配当が直近の実績からこの率以上減った時点でも「減配」とみなす
    #                      （通期の決算を待たずに売る。買い候補からも外す）
    # どれも指定がなければ、以前からの列（div_cut10・op_cut20）をそのまま使う。
    # 以前の結果を1円も変えないため。
    emg_custom = any(k in cfg for k in ("emg_cut_thr", "emg_op_thr",
                                         "emg_require_both", "emg_forecast"))
    emg_cut_thr = float(cfg.get("emg_cut_thr", 0.10))
    emg_op_thr = float(cfg.get("emg_op_thr", 0.20))
    emg_both = bool(cfg.get("emg_require_both", False))
    emg_fc = cfg.get("emg_forecast")
    if emg_fc is None or emg_fc is False:
        emg_fc_thr = None
    elif emg_fc is True:
        emg_fc_thr = emg_cut_thr
    else:
        emg_fc_thr = float(emg_fc)

    def _num(v):
        return v is not None and not pd.isna(v)

    def emg_flags(row):
        """(減配か, 業績急変か) を返す。本番の generate.py と同じ「実績」の判定が土台。"""
        if not emg_custom:
            _v, _w = row.get("div_cut10"), row.get("op_cut20")
            return (_num(_v) and float(_v) > 0.5), (_num(_w) and float(_w) > 0.5)
        _dc, _oc = row.get("div_chg"), row.get("op_chg")
        cut = _num(_dc) and float(_dc) <= -emg_cut_thr
        opd = _num(_oc) and float(_oc) <= -emg_op_thr
        if emg_fc_thr is not None:
            _fc = row.get("fc_chg")
            if _num(_fc) and float(_fc) <= -emg_fc_thr:
                cut = True
        if emg_both:
            both = cut and opd
            return both, both
        return cut, opd
    # 価格で損切りする（2026年10月11日に追加）
    #   stop_loss     … 平均取得単価からこの率以上下がったら、全部売る（0.20 なら −20％）
    #   stop_cooldown … 損切りした銘柄を何か月買い直さないか（既定12か月）。
    #                   これがないと、値下がりで利回りが上がった銘柄を、売ったその月末に
    #                   買い直してしまい、損切りの意味がなくなる。
    # 判定は日々の終値で行い、最初に線を割った日の終値で売る（DAILY_PX があるとき）。
    # 月の途中の出来事なので、減配・業績急変の売りや「売らない Tier」より前に判定する。
    # DAILY_PX がないときは、月末の値で判定する（線を割っても月末に戻していれば売らない）。
    stop_loss = cfg.get("stop_loss")
    stop_cd = int(cfg.get("stop_cooldown", 12))
    stop_block: dict[str, int] = {}      # 銘柄 → この月（何か月目）になるまで買わない
    # 売って確定した損を、配当とも相殺するか（2026年10月11日に追加）。
    # 特定口座（源泉徴収あり）で配当を口座で受け取れば、同じ年の売却損と配当は自動で相殺され、
    # 確定申告すれば翌年以降に繰り越せる。以前の計算は損を売却益とだけ相殺していたため、
    # 損を出して売るルール（損切り）が不利に出る。
    # 以前の結果を再現できるよう、既定は従来どおり（相殺しない）。
    # 簡単のため、繰り越しの期限（3年）は見ていない（損を出すルールに少し甘い）。
    tax_div_offset = bool(cfg.get("tax_div_offset", False))
    # 上がった銘柄を売る形（2026年10月11日に追加）。「売らない Tier」より前に判定する。
    #   yield_exit   … 月末の利回り（予想配当÷株価）がこの値（％）を下回ったら全部売る。
    #                  株価が上がって割高になった銘柄を、利回りの高い銘柄に乗り換える形。
    #                  買いの足切り（3％）より低くしておけば、同じ銘柄をすぐ買い戻すことはない。
    #   tp_gain      … 月末の株価が平均取得単価からこの率以上上がったら売る（利確）
    #   tp_frac      … そのとき売る割合（1.0 なら全部。0.5 なら半分を100株単位で）。
    #                  一部だけ売ったときは、次はその売値からさらに tp_gain 上がったら売る。
    #                  100株しか持っていないなど、半分に分けられないときは売らない。
    #   act_cooldown … 全部売ったあと、何か月買い直さないか。利確で全部売ると、
    #                  まだ利回りが足切りより高い銘柄はその月末に買い戻してしまうため。
    act_yexit = cfg.get("yield_exit")
    act_tp = cfg.get("tp_gain")
    act_frac = float(cfg.get("tp_frac", 1.0))
    act_cd = int(cfg.get("act_cooldown", 0))
    # 売らずに動く形・税金を減らしながら動く形（2026年10月11日に追加）
    #   harvest … {"month": 12, "min_loss": 0.0}。その月末に、平均取得単価より min_loss 以上
    #             安くなっている銘柄を全部売り、同じ株数をすぐ買い戻す（年末の損出し）。
    #             持ち株は変わらないので、成績の差は「税金」と「売買の費用」だけになる。
    #             確定した損は売却益と、tax_div_offset があれば配当とも相殺する。
    #             日本には同じ銘柄の買い戻しを制限する決まりがないので、実際に使える手。
    #             実際は売りを先に約定させる（総平均法で、買いが先だと損が確定しないため）。
    #   topup   … True なら、その月の買いのあとに残った現金で、持ち株のうち
    #             目標の大きさ（新しく買うときの1銘柄の予算）をいちばん下回っている銘柄を買い増す。
    #             配当を毎月再投資する形。買いの条件（8条件・利回りの足切り・減配や業績急変でない）を
    #             満たす銘柄だけ。売らないので税金はかからない。
    harvest = cfg.get("harvest")
    hv_month = int(harvest.get("month", 12)) if harvest else 0
    hv_min = float(harvest.get("min_loss", 0.0)) if harvest else 0.0
    topup = bool(cfg.get("topup", False))
    # 配当を受け取る月を正しくする（2026年10月11日に追加）
    #   div_split … True なら、直近の通期の「中間配当÷年間配当」の割合で、中間と期末に分けて受け取る。
    #               期末だけの会社は期末に全額。割合が取れない会社は以前どおり半分ずつ。
    #               以前の結果を変えないため、既定は従来どおり（どの会社も半分ずつ）。
    div_split = bool(cfg.get("div_split", False))
    # 8条件のうち、外す条件（2026年10月11日に追加。例 ["pay"]。"all" なら利回り以外を全部外す）
    _sd = cfg.get("screen_drop") or []
    screen_drop = set(SCREEN_KEYS) if _sd == "all" else set(_sd)

    def screen_ok(row):
        """8条件（利回り以外）を通ったか。外す条件があれば、それ以外だけで判定する。"""
        if not screen_drop:
            return bool(row["screen_pass"])
        return all(bool(row["scr_" + k]) for k in SCREEN_KEYS if k not in screen_drop)

    def _int_share(row):
        """年間配当のうち中間で受け取る割合。取れなければ0.5。"""
        v = row.get("div_int_share")
        return float(v) if (v is not None and not pd.isna(v)) else 0.5

    # 配当の権利が確定する月の前に、買いの条件をゆるめる（2026年10月11日に追加）
    #   rec_relax … {"yield": 2.5, "miss": 1, "first": False}
    #     次の月末が配当の権利確定日（中間か期末）の銘柄に限って、
    #       yield … 利回りの足切りをこの値（％）までゆるめる（None ならゆるめない）
    #       miss  … 8条件のうち、この数までなら満たしていなくても買う（0 ならゆるめない）
    #       first … 同じ Tier の中で、こうした銘柄を先に買う（条件は変えない）
    #   月末にしか売買しないので、「権利確定月に入ってすぐ買う」形になる。
    #   権利確定月の月末の株価は、すでに配当の分だけ下がったあと（権利落ち）なので、
    #   その月末に買っても配当はもらえない。1か月前の月末に買えば、翌月末に配当を受け取る。
    rec_relax = cfg.get("rec_relax")
    rec_yield = rec_relax.get("yield") if rec_relax else None
    rec_miss = int(rec_relax.get("miss", 0)) if rec_relax else 0
    rec_first = bool(rec_relax.get("first", False)) if rec_relax else False

    def _rec_in(row, month_):
        """その月の月末が配当の権利確定日か（配当を受け取る割合が0の月は除く）。"""
        fm_, im_ = int(row.get("fiscal_month", 0) or 0), int(row.get("interim_month", 0) or 0)
        s_ = _int_share(row) if div_split else 0.5
        return (fm_ == month_ and s_ < 1.0) or (im_ == month_ and s_ > 0.0)

    def _rec_next(row, dt_):
        """次の月末が配当の権利確定日か（配当を受け取る割合が0の月は除く）。"""
        return _rec_in(row, pd.Timestamp(dt_).month % 12 + 1)

    # 2回目の候補（2026年10月11日に追加）
    #   max_yield … 利回りがこの値（％）を超える銘柄は買わない。高すぎる利回りは、
    #               株価が先に下がって減配を織り込みにいっている「罠」のことが多いため
    #   ex_first  … 同じ Tier の中で、この月末に配当の権利が確定した銘柄（＝権利落ちの直後）を先に買う。
    #               権利確定月の前に買う形の逆。配当にすぐ税金がかからず、権利落ちで下がった値段で買える
    max_yield = cfg.get("max_yield")
    ex_first = bool(cfg.get("ex_first", False))
    # 市況の合図が切り替わった月に、持ち株を入れ替えるか
    sell_on_flip = bool(cfg.get("sell_on_flip", False))
    _rng_shuffle = np.random.default_rng(cfg.get("seed", 0))
    # 市場全体が割安なときに厚く、割高なときに薄く買う。
    # 「暴落を待つ」戦略は、待っている間の取り逃がしが本体なので、
    # 効いているかどうかは全期間で確かめる必要がある。
    regime = cfg.get("regime_scale")      # 例 {"cheap":1.5, "rich":0.5}
    # 同じ業種を何銘柄まで持つか。
    # 業種が偏ると、その業種の逆風で一斉に沈む。分散の実効性を上げるための制約。
    max_sector = cfg.get("max_per_sector")
    # 総資産のうち、常に現金で残しておく割合。下落の受け止めに使う。
    cash_floor = cfg.get("cash_floor", 0.0)
    trail_arm = cfg.get("trail_arm")                # この率まで上がったら見張り開始
    trail = cfg.get("trail")                        # 高値からこの率下げたら売る
    rotate = cfg.get("rotate")
    # 質（Tier）で入れ替える。
    #   min_gap  … Tierが何段階上なら入れ替えるか（1ならB→A、2ならB→S）
    #   only_win … 含み益が出ている銘柄だけ手放す（損切りしない）
    #   max_year … 1年あたりの入れ替え回数の上限
    swap_tier = cfg.get("swap_tier")
    n_tr = len(entry)

    # 売買にかかる費用。
    #   slip … 約定のずれ。成行で買えば少し高く、売れば少し安くなる。
    #   fee  … 手数料。
    #   tax  … 譲渡益と配当への課税。損は繰り越して相殺する（損益通算）。
    # 資金を何か月かけて入れるか。
    # 0 なら初日に全額が使える。バックテストの初日は
    # 「その時点で条件を満たす銘柄をまとめて買う」状態になりやすく、
    # 結果が初日の一括購入に支配されてしまう。
    # ramp を指定すると、毎月 1/ramp ずつしか使えないようにする。
    ramp = max(0, int(ramp))

    slip = slip_bps / 10000.0

    if cfg.get('slip_bps') is not None:

        slip = cfg['slip_bps'] / 10000.0
    fee = fee_bps / 10000.0
    loss_pool = 0.0        # 相殺できる損失の残り

    dates = sorted(panel["date"].unique())
    cash = capital
    # code -> {"units": n, "lots": [(株数, 取得単価), ...]}
    pos: dict[str, dict] = {}
    curve, trades = [], []
    diag = {"adjusted": 0, "changed": 0, "full_slots": 0, "months": 0,
            "opened": 0, "closed": 0, "hold_months": [], "rotations": 0,
            # 売却して確定した損益（税引後）。含み益とは別に集計する。
            "realized": 0.0,
            # Tierごとに「建てた数」と「どこまで含み益が伸びたか」を記録する。
            # 利確ラインを変えても結果が動かないとき、
            # そもそもその銘柄を持っていないのか、
            # 持っていても伸びていないのかを切り分けるため。
            "tier_opened": {}, "tier_peak": []}
    opened_at: dict[str, int] = {}

    def _sell_shares(code, st, want, px, dt):
        """古い建玉から want 株を px で売る（税金・損の繰り越し・費用も計上）。売れた株数を返す。"""
        nonlocal cash, loss_pool
        left, newlots = want, []
        for sh, pr in st["lots"]:
            if left <= 0:
                newlots.append((sh, pr))
                continue
            take = min(sh, left)
            eff = px * (1 - slip) * (1 - fee)
            cash += take * eff
            g = (eff - pr) * take
            diag["realized"] += g
            if tax_rate > 0:
                if g > 0:
                    taxable = max(0.0, g - loss_pool)
                    loss_pool = max(0.0, loss_pool - g)
                    t = taxable * tax_rate
                    cash -= t
                    diag["tax"] = diag.get("tax", 0.0) + t
                else:
                    loss_pool += -g
            diag["fee"] = diag.get("fee", 0.0) + take * px * (slip + fee)
            trades.append({"code": code, "date": dt, "side": "sell",
                           "price": eff, "shares": take})
            left -= take
            if sh - take > 0:
                newlots.append((sh - take, pr))
        st["lots"] = newlots
        return want - left

    def _close(code, st, mi, label):
        """全部売った銘柄の後始末（到達益の記録・保有月数・建玉の明細）。"""
        if st.get("sh_sum"):
            avg = st["cost_sum"] / st["sh_sum"]
            g_ = st.get("peak", avg) / avg - 1
            diag["tier_peak"].append((st.get("tier", "B"), g_))
            for e_ in reversed(diag.get("trade_log", [])):
                if e_["code"] == code and e_["result"] == "保有中":
                    e_["peak_gain"] = g_
                    e_["result"] = label
                    break
        del pos[code]
        diag["closed"] += 1
        if code in opened_at:
            diag["hold_months"].append(mi - opened_at.pop(code))

    # 市場全体がどれだけ割安か。全銘柄の利回り中央値を、
    # 過去3年の平均と比べて何σ離れているかで測る。
    # 当日を含めると未来の情報が混ざるので、1期ずらしてから平均を取る。
    mkt = panel.groupby("date")["yield"].median()
    _past = mkt.shift(1)
    mkt_z = (mkt - _past.rolling(36, min_periods=12).mean()) / \
            _past.rolling(36, min_periods=12).std()

    def value_of(day):
        v = cash
        for c, st in pos.items():
            if c in day.index:
                v += sum(sh for sh, _ in st["lots"]) * day.loc[c, "price"]
        return v

    prev_day = None
    for mi, dt in enumerate(dates):
        day = panel[panel["date"] == dt].set_index("code")
        day = day.assign(mkt_yield_z=float(mkt_z.get(dt, 0) or 0))
        _dt_ns = pd.Timestamp(dt).value        # 損切りの判定で、日々の終値と比べるため

        _div_pending = []     # 権利落ち日以降にうねりで売った分（その月の配当は受け取る）
        # ── うねりで日々売り買い（9回目）──
        #   前の月末の翌日から今月末までの営業日を順に見る。
        #   売り：A・B の持ち株が、うねりの中で売り値に届いた日に、その値段で全部売る（取得単価より上のときだけ）
        #   買い：前の月末の時点で買える A・B の候補（8条件・利回り・緊急撤退）が、
        #         うねりの中で買い値に届いた日に、その値段で買う。売った日のうちには買い直さない。
        #   1銘柄の予算は前の月末の総資産で決める。取引の日付は、集計のためその月末で記録する（day に実際の日）。
        if swing_daily and mi > 0 and prev_day is not None:
            _cal = SWING_DAILY["_cal"]
            _lo_ns = pd.Timestamp(dates[mi - 1]).value
            _days = _cal[(_cal > _lo_ns) & (_cal <= _dt_ns)]
            _mg = float(swing_daily.get("min_gain", 0.0))
            _total0 = curve[-1]["value"] if curve else capital
            _reserved = capital * (1.0 - min(1.0, (mi + 1) / ramp)) if ramp > 0 else 0.0
            _ordt = {"S": 0, "A": 1, "B": 2}
            _avg_w0 = sum(tw.values()) / len(tw)
            _elig = []
            for c_, r_ in prev_day.iterrows():
                tg_ = r_.get("tier", "B")
                if tg_ not in _sw_tiers or c_ not in _swd:
                    continue
                if use_screen and not screen_ok(r_):
                    continue
                if min_yield > 0 and r_.get("yield", 0) < min_yield:
                    continue
                if require_col and not (isinstance(r_.get(require_col), (bool, np.bool_))
                                        and bool(r_.get(require_col))):
                    continue        # 11回目：増配の実績がある銘柄だけ買う
                if code_half is not None and code_half_of(c_) != int(code_half):
                    continue        # 12回目：決めた半分の銘柄だけ買う
                if skip_emg_buy:
                    if emg_actual:
                        _cu, _op = emg_flags(r_)
                    else:
                        _cu, _op = bool(r_.get("dps_cut")), bool(r_.get("op_drop"))
                    if _cu or _op:
                        continue
                _elig.append((_ordt.get(tg_, 9), -float(level(r_)), c_, tg_))
            _elig.sort()
            _slc = {}

            def _day_of(c_, d_):
                if c_ not in _slc:
                    arr_ = _swd.get(c_)
                    if arr_ is None:
                        _slc[c_] = {}
                    else:
                        a_ = int(np.searchsorted(arr_[0], _lo_ns, side="right"))
                        b_ = int(np.searchsorted(arr_[0], _dt_ns, side="right"))
                        # 水平線（10回目）は高値・安値も持っている。売りは高値、買いは安値で届いたかを見る（指値）。
                        # それ以外は終値で見る（9回目と同じ）。
                        _hl = len(arr_) >= 7
                        _slc[c_] = {int(arr_[0][i_]): (float(arr_[1][i_]), float(arr_[2][i_]),
                                                       float(arr_[3][i_]), bool(arr_[4][i_]),
                                                       float(arr_[5][i_]) if _hl else float(arr_[1][i_]),
                                                       float(arr_[6][i_]) if _hl else float(arr_[1][i_]))
                                    for i_ in range(a_, b_)}
                return _slc[c_].get(d_)

            _sold_on = {}
            # 配当の権利（月末が権利確定日。受け渡しは2営業日後なので、月の最後から3営業日目が権利付き最終日）
            #   最後の2営業日（権利落ち日と月末）に買った分は、その月の配当を受け取らない
            #   最後の2営業日に売った分は、その月の配当を受け取る（権利付き最終日まで持っていたため）
            _ex_ns = int(_days[-2]) if len(_days) >= 2 else None
            for d_ns in _days:
                d_ns = int(d_ns)
                for c_ in list(pos.keys()):
                    st_ = pos[c_]
                    tg_ = prev_day.loc[c_].get("tier", "B") if c_ in prev_day.index else st_.get("tier", "B")
                    if tg_ not in _sw_tiers or not st_["lots"]:
                        continue
                    v_ = _day_of(c_, d_ns)
                    if v_ is None:
                        continue
                    cl_, sp_, bp_, ok_, hi_, lo_ = v_
                    if not ok_:
                        continue
                    shs_ = sum(sh for sh, _ in st_["lots"])
                    avg_ = sum(sh * pr for sh, pr in st_["lots"]) / shs_ if shs_ else 0.0
                    # 売ってよい日（線の中・取得単価より上）を数える。ランダムな日の比較で回数をそろえるため
                    if avg_ > 0 and cl_ > avg_ * (1 + _mg):
                        diag["sw_elig_sell"] = diag.get("sw_elig_sell", 0) + 1
                    if _plc:
                        if _rng.random() >= float(_plc["p_sell"]):
                            continue
                        sp_ = lo_ + _rng.random() * (hi_ - lo_)     # その日の安値〜高値のランダムな値段
                    elif not (hi_ >= sp_):
                        continue
                    if not (avg_ > 0 and sp_ > avg_ * (1 + _mg)):
                        continue
                    _n0 = len(trades)
                    if _ex_ns is not None and d_ns >= _ex_ns:
                        _div_pending.append((c_, shs_))     # 権利付き最終日まで持っていたので、配当は受け取る
                    _sell_shares(c_, st_, shs_, sp_, dt)
                    for tr_ in trades[_n0:]:
                        tr_["day"] = pd.Timestamp(d_ns)
                    diag["range_sells"] = diag.get("range_sells", 0) + 1
                    diag["swing_gain_sum"] = diag.get("swing_gain_sum", 0.0) + (sp_ * (1 - slip) * (1 - fee) / avg_ - 1)
                    diag.setdefault("swing_sales", []).append((c_, d_ns, sp_ * (1 - slip) * (1 - fee) / avg_ - 1, sp_))
                    swing_sold[c_] = mi
                    _sold_on[c_] = d_ns
                    _close(c_, st_, mi, "うねりで売却")
                for _o, _np, c_, tg_ in _elig:
                    if c_ in pos or _sold_on.get(c_) == d_ns:
                        continue
                    if len(pos) >= max_names:
                        break
                    v_ = _day_of(c_, d_ns)
                    if v_ is None:
                        continue
                    cl_, sp_, bp_, ok_, hi_, lo_ = v_
                    if not ok_:
                        continue
                    diag["sw_elig_buy"] = diag.get("sw_elig_buy", 0) + 1     # 買ってよい日（線の中・まだ持っていない）
                    if _plc:
                        if _rng.random() >= float(_plc["p_buy"]):
                            continue
                        bp_ = lo_ + _rng.random() * (hi_ - lo_)     # その日の安値〜高値のランダムな値段
                    elif not (lo_ <= bp_):
                        continue
                    if weighted:
                        unit_ = _total0 / target_n * (tw.get(tg_, 1.0) / _avg_w0) / n_tr
                    else:
                        unit_ = _total0 / max_names / n_tr
                    usable_ = cash - _reserved
                    if usable_ < unit_ or unit_ < bp_:
                        continue
                    sh_ = int(unit_ // bp_ // 100) * 100 if tier_budget else int(unit_ // bp_)
                    if sh_ <= 0:
                        continue
                    be_ = bp_ * (1 + slip) * (1 + fee)
                    cash -= sh_ * be_
                    diag["fee"] = diag.get("fee", 0.0) + sh_ * bp_ * (slip + fee)
                    pos[c_] = {"units": 1, "lots": [(sh_, be_)], "peak": bp_, "tier": tg_,
                               "cost_sum": sh_ * be_, "sh_sum": sh_, "origin": "swing"}
                    if _ex_ns is not None and d_ns >= _ex_ns:
                        pos[c_]["no_div_mi"] = mi        # 権利落ちのあとに買ったので、この月の配当は受け取らない
                    diag["opened"] += 1
                    opened_at[c_] = mi
                    diag["tier_opened"][tg_] = diag["tier_opened"].get(tg_, 0) + 1
                    diag["swing_buys"] = diag.get("swing_buys", 0) + 1
                    trades.append({"code": c_, "date": dt, "side": "buy", "price": bp_,
                                   "shares": sh_, "day": pd.Timestamp(d_ns)})

        # ── 売り ──
        for code in list(pos.keys()):
            if code not in day.index:
                continue
            row = day.loc[code]
            st = pos[code]
            price = row["price"]
            st["peak"] = max(st.get("peak", price), price)

            # ── 価格で損切り ──
            # 前回の判定（買った月末、または前の月末）より後の日々の終値を順に見て、
            # 平均取得単価 ×（1 − 損切り率）以下で引けた最初の日に、その日の終値で全部売る。
            if stop_loss and st["lots"]:
                _shs = sum(sh for sh, _ in st["lots"])
                _avg = sum(sh * pr for sh, pr in st["lots"]) / _shs if _shs else 0.0
                _line = _avg * (1 - stop_loss)
                _hit = None                      # 売った値段（線を割った日の終値）
                _arr = DAILY_PX.get(code) if DAILY_PX is not None else None
                if _arr is not None:
                    _ds, _cs = _arr
                    _lo = int(np.searchsorted(_ds, st.get("chk_from", _dt_ns), side="right"))
                    _hi = int(np.searchsorted(_ds, _dt_ns, side="right"))
                    if _hi > _lo:
                        _below = np.flatnonzero(_cs[_lo:_hi] <= _line)
                        if _below.size:
                            _hit = float(_cs[_lo + int(_below[0])])
                elif price <= _line:
                    _hit = float(price)
                st["chk_from"] = _dt_ns
                if _hit is not None and _avg > 0:
                    eff = _hit * (1 - slip) * (1 - fee)
                    for sh, pr in st["lots"]:
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
                        diag["fee"] = diag.get("fee", 0.0) + sh * _hit * (slip + fee)
                        diag["realized"] += gain
                        trades.append({"code": code, "date": dt, "side": "sell",
                                       "price": eff, "shares": sh})
                    diag["stop_exits"] = diag.get("stop_exits", 0) + 1
                    diag.setdefault("stop_log", []).append(
                        {"code": code, "date": dt, "px": _hit, "loss": eff / _avg - 1})
                    stop_block[code] = mi + stop_cd
                    if st.get("sh_sum"):
                        avg = st["cost_sum"] / st["sh_sum"]
                        g_ = st.get("peak", avg) / avg - 1
                        diag["tier_peak"].append((st.get("tier", "B"), g_))
                        for e_ in reversed(diag.get("trade_log", [])):
                            if e_["code"] == code and e_["result"] == "保有中":
                                e_["peak_gain"] = g_
                                e_["result"] = "損切り"
                                break
                    del pos[code]
                    diag["closed"] += 1
                    if code in opened_at:
                        diag["hold_months"].append(mi - opened_at.pop(code))
                    continue

            # 減配は「売らない Tier」でも例外として手放す
            if emg_actual:
                _cut, _opd = emg_flags(row)
                _emg = bool(_cut)
                if exit_on_op and _opd:
                    _emg = True
            else:
                _emg = bool(row.get("dps_cut"))
                if exit_on_op and bool(row.get("op_drop")):
                    _emg = True
            # 市況の合図が切り替わった月だけ、新しい順位で上位から外れた銘柄を入れ替える
            if sell_on_flip and not _emg and bool(row.get("switch_flip")) \
                    and level(row) < min(entry_for(row)):
                _emg = True
                diag["flip_exits"] = diag.get("flip_exits", 0) + 1
            if exit_on_cut and _emg and st["lots"]:
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
                    diag["realized"] += gain
                    trades.append({"code": code, "date": dt, "side": "sell",
                                   "price": eff, "shares": sh})
                diag["cut_exits"] = diag.get("cut_exits", 0) + 1
                # 11回目：どう買った銘柄で緊急売りになったか（線の下で買った／月末に買った／最初の月に買った）と、
                # 売ったときの損益（約定のずれ込み後・税引前、取得単価に対して）
                _org = st.get("origin", "month")
                _co = diag.setdefault("cut_by_origin", {})
                _co[_org] = _co.get(_org, 0) + 1
                _shs_c = sum(sh for sh, _ in st["lots"])
                if _shs_c > 0:
                    _avg_c = sum(sh * pr for sh, pr in st["lots"]) / _shs_c
                    if _avg_c > 0:
                        diag.setdefault("cut_ret", []).append(price * (1 - slip) * (1 - fee) / _avg_c - 1)
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

            # ── 上がった銘柄を売る（利回り割れ・利確）──
            # 「売らない Tier」の判定より前に置く（後ろだと一度も働かない）。
            if (act_yexit is not None or act_tp is not None) and st["lots"]:
                _shs = sum(sh for sh, _ in st["lots"])
                _avg = sum(sh * pr for sh, pr in st["lots"]) / _shs if _shs else 0.0
                _y = row.get("yield")
                _why = None
                if act_yexit is not None and _y is not None and not pd.isna(_y) \
                        and float(_y) < float(act_yexit):
                    _why, _want = "利回り割れ", _shs
                elif act_tp is not None and _avg > 0:
                    _base = st.get("tp_base") or _avg
                    if price >= _base * (1 + float(act_tp)):
                        if act_frac >= 1.0:
                            _why, _want = "利確", _shs
                        else:
                            _want = int(_shs * act_frac // 100) * 100
                            if 0 < _want < _shs:
                                _why = "一部利確"
                            else:
                                diag["act_skip"] = diag.get("act_skip", 0) + 1
                if _why is not None:
                    _sold = _sell_shares(code, st, _want, price, dt)
                    diag.setdefault("act_log", []).append(
                        {"code": code, "date": dt, "px": float(price),
                         "gain": (price * (1 - slip) * (1 - fee)) / _avg - 1 if _avg else 0.0,
                         "kind": _why, "shares": _sold})
                    if _why == "一部利確":
                        diag["act_partial"] = diag.get("act_partial", 0) + 1
                        st["tp_base"] = float(price)
                    if not st["lots"]:
                        diag["act_exits"] = diag.get("act_exits", 0) + 1
                        if act_cd > 0:
                            stop_block[code] = mi + act_cd
                        _close(code, st, mi, _why)
                        continue

            # 高値の記録は、売る売らないに関わらず必ず更新する。
            # ここより後ろに置くと、売らない Tier は買った日の値のまま止まり、
            # 到達益が常に0近くになってしまう。
            st["peak"] = max(st.get("peak", price), price)

            if hold_tiers and row.get("tier", "B") in hold_tiers:
                continue          # この Tier は売らない

            # ── レンジの上限で降りる ──
            # 高配当のルールで買った銘柄が、たまたまレンジの中にいて
            # 上限まで来たら、そこで一度降りる。下がればまた買い直せる。
            tp = row.get(range_exit.get("top_col", "box_top_px")) if range_exit else None
            if range_exit and st["lots"] and tp is not None and not pd.isna(tp):
                sell_px = float(tp)
                shs = sum(sh for sh, _ in st["lots"])
                cost0 = sum(sh * pr for sh, pr in st["lots"]) / shs if shs else 0
                # 取得より上でなければ降りない（安値で投げない）
                if sell_px > cost0 * (1 + range_exit.get("min_gain", 0.0)):
                    frac = range_exit.get("fraction", 1.0)
                    want = int(shs * frac // 100) * 100 if frac < 1.0 else shs
                    left, newlots = want, []
                    for sh, pr in st["lots"]:
                        if left <= 0:
                            newlots.append((sh, pr)); continue
                        take = min(sh, left)
                        eff = sell_px * (1 - slip) * (1 - fee)
                        cash += take * eff
                        g = (eff - pr) * take
                        diag["realized"] += g
                        if tax_rate > 0:
                            if g > 0:
                                taxable = max(0.0, g - loss_pool)
                                loss_pool = max(0.0, loss_pool - g)
                                t = taxable * tax_rate
                                cash -= t
                                diag["tax"] = diag.get("tax", 0.0) + t
                            else:
                                loss_pool += -g
                        diag["fee"] = diag.get("fee", 0.0) + take * sell_px * (slip + fee)
                        trades.append({"code": code, "date": dt, "side": "sell",
                                       "price": eff, "shares": take})
                        left -= take
                        if sh - take > 0:
                            newlots.append((sh - take, pr))
                    if want > 0 and left < want:
                        diag["range_sells"] = diag.get("range_sells", 0) + 1
                        swing_sold[code] = mi        # 同じ月のうちに買い直さない（8回目）
                        st["lots"] = newlots
                        if not st["lots"]:
                            del pos[code]
                            diag["closed"] += 1
                            if code in opened_at:
                                diag["hold_months"].append(mi - opened_at.pop(code))
                            continue
                        st["units"] = max(1, len(st["lots"]))

            # ── 部分利確 ──
            if partial and st["lots"]:
                shs = sum(sh for sh, _ in st["lots"])
                cost0 = sum(sh * pr for sh, pr in st["lots"]) / shs if shs else 0
                base = st.get("last_partial") or cost0
                if base > 0 and price >= base * (1 + partial["threshold"]):
                    want = int(shs * partial.get("fraction", 0.5) // 100) * 100
                    left = want
                    newlots = []
                    for sh, pr in st["lots"]:
                        if left <= 0:
                            newlots.append((sh, pr))
                            continue
                        take = min(sh, left)
                        eff = price * (1 - slip) * (1 - fee)
                        cash += take * eff
                        g = (eff - pr) * take
                        diag["realized"] += g
                        if tax_rate > 0:
                            if g > 0:
                                taxable = max(0.0, g - loss_pool)
                                loss_pool = max(0.0, loss_pool - g)
                                t = taxable * tax_rate
                                cash -= t
                                diag["tax"] = diag.get("tax", 0.0) + t
                            else:
                                loss_pool += -g
                        diag["fee"] = diag.get("fee", 0.0) + take * price * (slip + fee)
                        trades.append({"code": code, "date": dt, "side": "sell",
                                       "price": eff, "shares": take})
                        left -= take
                        if sh - take > 0:
                            newlots.append((sh - take, pr))
                    if want > 0 and left < want:
                        st["lots"] = newlots
                        st["last_partial"] = price
                        diag["partial_sells"] = diag.get("partial_sells", 0) + 1
                        if not st["lots"]:
                            st["units"] = 0
                        else:
                            st["units"] = max(1, len(st["lots"]))

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
                        diag["realized"] += gain
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
                    diag["realized"] += gain
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

        # ── 年末の損出し ──
        # 含み損の銘柄を売って損を確定させ、同じ株数をその場で買い戻す。持ち株は変わらない。
        if harvest and pd.Timestamp(dt).month == hv_month:
            for code in list(pos.keys()):
                if code not in day.index:
                    continue
                st = pos[code]
                if not st["lots"]:
                    continue
                _px = float(day.loc[code, "price"])
                _shs = sum(sh for sh, _ in st["lots"])
                _avg = sum(sh * pr for sh, pr in st["lots"]) / _shs if _shs else 0.0
                if _avg <= 0 or _px >= _avg * (1 - hv_min):
                    continue
                # 特定口座の取得単価は平均（総平均法）。建玉を1つにまとめてから売る
                st["lots"] = [(_shs, _avg)]
                _loss = (_avg - _px * (1 - slip) * (1 - fee)) * _shs
                _sell_shares(code, st, _shs, _px, dt)
                _buy = _px * (1 + slip) * (1 + fee)
                cash -= _shs * _buy
                diag["fee"] = diag.get("fee", 0.0) + _shs * _px * (slip + fee)
                st["lots"] = [(_shs, _buy)]
                trades.append({"code": code, "date": dt, "side": "buy",
                               "price": _px, "shares": _shs})
                diag["harvests"] = diag.get("harvests", 0) + 1
                diag["harvest_loss"] = diag.get("harvest_loss", 0.0) + _loss
                diag.setdefault("harvest_log", []).append(
                    {"code": code, "date": dt, "loss": _loss})

        # ── 買い候補 ──
        diag["months"] += 1
        cands = []
        _recn = {}              # 次の月末に配当の権利が確定する銘柄（条件をゆるめる形でだけ使う）
        _relaxed = set()        # ゆるめた条件でだけ候補に入った銘柄
        for code, row in day.iterrows():
            _rn = _rec_next(row, dt) if rec_relax else False
            if rec_relax:
                _recn[code] = _rn
            if use_screen:
                if "screen_pass" not in row.index:
                    raise RuntimeError("screen_pass の列がありません。財務の項目が足されていません。")
                if not screen_ok(row):
                    _sf = row.get("screen_fail")
                    if not (_rn and rec_miss > 0 and _sf is not None and not pd.isna(_sf)
                            and int(_sf) <= rec_miss):
                        continue
                    _relaxed.add(code)
            if min_yield > 0 and row.get("yield", 0) < min_yield:
                if not (_rn and rec_yield is not None and row.get("yield", 0) >= float(rec_yield)):
                    continue        # 利回りが低すぎる銘柄は最初から除く
                _relaxed.add(code)
            if stop_block and mi < stop_block.get(code, -1):
                continue        # 損切りしてから決めた月数が過ぎるまでは買い直さない
            if max_yield is not None and row.get("yield", 0) > float(max_yield):
                diag["max_yield_skip"] = diag.get("max_yield_skip", 0) + 1
                continue        # 利回りが高すぎる銘柄は買わない
            if code_half is not None and code_half_of(code) != int(code_half):
                continue        # 12回目：決めた半分の銘柄だけ買う
            if require_col and row.get(require_col) is not True and not (
                    isinstance(row.get(require_col), (bool, np.bool_)) and bool(row.get(require_col))):
                diag["req_skip"] = diag.get("req_skip", 0) + 1
                continue        # 11回目：増配の実績がある銘柄だけ買う
            if tiers_allowed is not None and row.get("tier", "B") not in tiers_allowed:
                continue        # この Tier は新しく買わない
            if skip_emg_buy:
                if emg_actual:
                    _cut, _opd = emg_flags(row)
                else:
                    _cut, _opd = bool(row.get("dps_cut")), bool(row.get("op_drop"))
                if _cut or _opd:
                    continue    # 本番と同じく、減配・業績急変の銘柄は買わない
            if swing_daily and row.get("tier", "B") in _sw_tiers and code not in pos:
                if swing_sold.get(code) == mi:
                    continue    # うねりで売った月のうちは、月末に買い直さない
                if _box_ok_at(code, _dt_ns):
                    diag["swing_wait"] = diag.get("swing_wait", 0) + 1
                    continue    # うねりの中にいる。買い値に届く日を待つ（日々の売り買いで買う）
            if swing_entry and row.get("tier", "B") in swing_entry.get("tiers", ["A", "B"]):
                if swing_sold.get(code) == mi:
                    continue    # うねりで売った月のうちは買い直さない
                _t = swing_entry.get("tag", "m")
                _bot = row.get(f"box_bot_{_t}")
                _inb = bool(row.get(f"in_box_{_t}", False))
                if _bot is not None and not pd.isna(_bot) and code not in pos:
                    row = row.copy()
                    row["price"] = float(_bot)       # 下の線で買う（指値）
                    diag["swing_buys"] = diag.get("swing_buys", 0) + 1
                elif _inb and code not in pos:
                    diag["swing_wait"] = diag.get("swing_wait", 0) + 1
                    continue    # うねりの中にいる。下の線に届くまで待つ
            if skip_drop is not None:
                _pc, _dc = row.get("px_chg12"), row.get("dps_chg12")
                if (_pc is not None and not pd.isna(_pc) and float(_pc) <= -float(skip_drop)
                        and not (_dc is not None and not pd.isna(_dc) and float(_dc) > 0)):
                    diag["drop_skip"] = diag.get("drop_skip", 0) + 1
                    continue    # 株価の下落だけで利回りが上がった銘柄は買わない
            p = level(row)
            ent = entry_for(row)
            need = [required_pct(e, row, dyn) if measure == "pct" else e
                    for e in ent]
            want = sum(1 for nd in need if p >= nd)
            if dyn and measure == "pct" and need != list(map(float, ent)):
                diag["adjusted"] += 1
                if want != sum(1 for e in ent if p >= e):
                    diag["changed"] += 1
            have = 0 if add_on else pos.get(code, {}).get("units", 0)
            if want > have:
                cands.append((p, code, want - have, row["price"],
                              row.get("tier", "B")))
        # 並べ替え方。
        #   pct  … 割安な順（既定）。ただしS銘柄は割安度が低く出るため、
        #          Bに枠を先取りされてSが永久に買えなくなる。
        #   tier … Tier順（S→A→B）。実運用の portfolio_engine と同じ優先順位。
        if cfg.get("priority") == "tier":
            _ord = {"S": 0, "A": 1, "B": 2}
            if rec_first:
                # 同じ Tier の中で、次の月末に配当の権利が確定する銘柄を先にする
                cands.sort(key=lambda x: (_ord.get(x[4], 9), 0 if _recn.get(x[1]) else 1, -x[0]))
            elif ex_first:
                # 同じ Tier の中で、この月末に配当の権利が確定した銘柄（権利落ちの直後）を先にする
                _mth = pd.Timestamp(dt).month
                _exf = {c: _rec_in(day.loc[c], _mth) for _, c, _, _, _ in cands}
                cands.sort(key=lambda x: (_ord.get(x[4], 9), 0 if _exf.get(x[1]) else 1, -x[0]))
            else:
                cands.sort(key=lambda x: (_ord.get(x[4], 9), -x[0]))
        elif cfg.get("priority") == "code":
            # 銘柄コード順。判断を一切入れない並べ方。
            # これと差がなければ「順番を考えること」に価値がない。
            cands.sort(key=lambda x: x[1])
        else:
            cands.sort(key=lambda x: -x[0])

        # でたらめに並べる（運の幅を測るため）。
        #   all  … 候補の中から完全にでたらめに選ぶ
        #   tier … Tier順（S→A→B）は守り、同じTierの中だけでたらめ
        _shuf = cfg.get("shuffle")
        if _shuf and cands:
            _rng_shuffle.shuffle(cands)
            if _shuf == "tier":
                _ord = {"S": 0, "A": 1, "B": 2}
                cands.sort(key=lambda x: _ord.get(x[4], 9))   # 並べ替えは安定なので、Tier内はでたらめのまま

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
                        _g = (pr - _c) * sh
                        diag["realized"] += _g
                        # 入れ替えでも売却益には課税される。
                        # ここが抜けていると、入れ替えが不当に有利に出る。
                        if tax_rate > 0:
                            if _g > 0:
                                _tx = max(0.0, _g - loss_pool)
                                loss_pool = max(0.0, loss_pool - _g)
                                _t2 = _tx * tax_rate
                                cash -= _t2
                                diag["tax"] = diag.get("tax", 0.0) + _t2
                            else:
                                loss_pool += -_g
                        diag["fee"] = diag.get("fee", 0.0) + sh * pr0 * (slip + fee)
                        trades.append({"code": worst_c, "date": dt, "side": "sell",
                                       "price": pr, "shares": sh})
                    diag["closed"] += 1
                    diag["rotations"] += 1
                    if worst_c in opened_at:
                        diag["hold_months"].append(mi - opened_at.pop(worst_c))

        # ── 質で入れ替える ──
        # いま持っている中でいちばん質の低い銘柄を、
        # より質の高い買い候補と入れ替える。
        # 売却益に課税されるので、それを取り戻せるかが問われる。
        if swap_tier and cands:
            _ord2 = {"S": 0, "A": 1, "B": 2}
            # 新規で買えるうちは入れ替えない、という条件。
            # 資金が潤沢なら買い増しで質を上げられるので、
            # わざわざ税金を払って入れ替える必要がない。
            if swap_tier.get("only_when_stuck"):
                _p0, _c0, _a0, _px0b, _t0 = cands[0]
                if tier_budget:
                    _need = TIER_BUDGET.get(_t0, TIER_BUDGET["B"]) / n_tr
                else:
                    _need = value_of(day) / max_names / n_tr
                _stuck = (len(pos) >= max_names) or (cash < _need)
                if not _stuck:
                    diag["swap_skip_rich"] = diag.get("swap_skip_rich", 0) + 1
                    cands = cands      # 何もしない
                    swap_ok_now = False
                else:
                    swap_ok_now = True
            else:
                swap_ok_now = True
            _cap = swap_tier.get("max_year", 4) * max(mi / 12.0, 0.1)
            if swap_ok_now and diag.get("tier_swaps", 0) < _cap:
                # 買える候補のうち、いちばん質が高いもの
                best = None
                for _p, _c, _a, _px, _t in cands:
                    if _c in pos:
                        continue
                    if best is None or _ord2.get(_t, 9) < _ord2.get(best[2], 9):
                        best = (_p, _c, _t, _px)
                # 保有のうち、いちばん質が低いもの
                worst = None
                for _c in pos:
                    if _c not in day.index:
                        continue
                    _t = day.loc[_c].get("tier", "B")
                    if worst is None or _ord2.get(_t, 9) > _ord2.get(worst[1], 9):
                        worst = (_c, _t)
                if best and worst:
                    _gap = _ord2.get(worst[1], 9) - _ord2.get(best[2], 9)
                    if _gap >= swap_tier.get("min_gap", 1):
                        _st = pos[worst[0]]
                        _shs = sum(sh for sh, _ in _st["lots"])
                        _avg = (sum(sh * pr for sh, pr in _st["lots"]) / _shs
                                if _shs else 0)
                        _px0 = day.loc[worst[0], "price"]
                        _win = _px0 > _avg
                        # 権利確定月までの月数。近いほど、売ると配当を逃す。
                        _fm = int(day.loc[worst[0]].get("fiscal_month", 0) or 0)
                        _im = int(day.loc[worst[0]].get("interim_month", 0) or 0)
                        _cm = pd.Timestamp(dt).month
                        _ms = [((m - _cm) % 12) for m in (_fm, _im) if m]
                        _near = min(_ms) if _ms else 99

                        _gain = (_px0 / _avg - 1) if _avg > 0 else 0.0
                        _lvl = level(day.loc[worst[0]])

                        _ok = True
                        # 含み益がこの率以上あること（少額の利確を避ける）
                        _mg = swap_tier.get("min_gain")
                        if _mg is not None and _gain < _mg:
                            _ok = False
                            diag["swap_skip_gain"] = diag.get("swap_skip_gain", 0) + 1
                        # 利回りが下がっていること（分位がこの値以下＝割高側）
                        _mp = swap_tier.get("max_pct")
                        if _mp is not None and _lvl > _mp:
                            _ok = False
                            diag["swap_skip_pct"] = diag.get("swap_skip_pct", 0) + 1
                        if swap_tier.get("only_win") and not _win:
                            _ok = False          # 含み益のときだけ
                        if swap_tier.get("only_loss") and _win:
                            _ok = False          # 含み損のときだけ（税金がかからない）
                        if _ok is False:
                            pass
                        _avoid = swap_tier.get("avoid_div_months", 0)
                        if _avoid and _near <= _avoid:
                            _ok = False          # 権利月が近いので見送る
                            diag["swap_skipped_div"] = diag.get("swap_skipped_div", 0) + 1
                        if _ok:
                            _pr = _px0 * (1 - slip) * (1 - fee)
                            for sh, _c0 in _st["lots"]:
                                cash += sh * _pr
                                _g = (_pr - _c0) * sh
                                diag["realized"] += _g
                                if tax_rate > 0:
                                    if _g > 0:
                                        _tx = max(0.0, _g - loss_pool)
                                        loss_pool = max(0.0, loss_pool - _g)
                                        _t2 = _tx * tax_rate
                                        cash -= _t2
                                        diag["tax"] = diag.get("tax", 0.0) + _t2
                                    else:
                                        loss_pool += -_g
                                diag["fee"] = diag.get("fee", 0.0) + sh * _px0 * (slip + fee)
                                trades.append({"code": worst[0], "date": dt,
                                               "side": "sell", "price": _pr,
                                               "shares": sh})
                            del pos[worst[0]]
                            diag["closed"] += 1
                            diag["tier_swaps"] = diag.get("tier_swaps", 0) + 1
                            if worst[0] in opened_at:
                                diag["hold_months"].append(mi - opened_at.pop(worst[0]))

        if len(pos) >= max_names:
            diag["full_slots"] += 1

        # ── 配当（権利月に 年間DPS ÷ 2 を受け取る）──
        if dividends:
            _div_items = [(code, sum(sh for sh, _ in st["lots"])) for code, st in pos.items()
                          if st.get("no_div_mi") != mi]
            _div_items += _div_pending
            for code, held in _div_items:
                if code not in day.index:
                    continue
                row = day.loc[code]
                mth = pd.Timestamp(dt).month
                for key in ("fiscal_month", "interim_month"):
                    if int(row.get(key, 0) or 0) == mth and row["dps"] > 0:
                        if div_split:
                            # 中間と期末の実際の割合で受け取る（期末だけの会社は中間に0）
                            _fr = _int_share(row)
                            _fr = _fr if key == "interim_month" else 1.0 - _fr
                            if _fr <= 0:
                                continue
                            gross = row["dps"] * _fr * held
                        else:
                            gross = row["dps"] * 0.5 * held
                        if tax_div_offset and tax_rate > 0 and loss_pool > 0:
                            # 確定した損の残りと相殺してから課税する
                            _taxable = max(0.0, gross - loss_pool)
                            loss_pool = max(0.0, loss_pool - gross)
                            net = gross - _taxable * tax_rate
                        else:
                            net = gross * (1 - tax_rate)
                        cash += net
                        diag["dividend"] = diag.get("dividend", 0) + net
                        diag["tax"] = diag.get("tax", 0.0) + (gross - net)

        # ── 買い ──
        total = value_of(day)
        # まだ解放されていない資金は使えない
        usable = cash
        if cash_floor > 0:
            usable = max(0.0, usable - total * cash_floor)
        if ramp > 0:
            released = min(1.0, (mi + 1) / ramp)
            reserved = capital * (1.0 - released)
            # 現金の下限（cash_floor）があれば、そちらの制限も残す（2026年10月11日に修正）。
            # 以前は資金投入を分けた条件で現金の下限が上書きされ、一度も効いていなかった。
            # 現金の下限がないルールでは、以前と同じ値になる。
            usable = max(0.0, min(usable, cash - reserved))
        # いま何をどの業種で持っているか
        sec_count = {}
        if max_sector:
            for c in pos:
                if c in day.index:
                    sc = day.loc[c].get("sector", "")
                    sec_count[sc] = sec_count.get(sc, 0) + 1

        for p, code, add, price, _tier in cands:
            if len(pos) >= max_names and code not in pos:
                continue
            if add_on and code in pos:
                # 買い増しは回数を制限する
                if len(pos[code].get("lots", [])) >= add_on_max:
                    continue
            if max_sector and code not in pos:
                sc = day.loc[code].get("sector", "")
                if sec_count.get(sc, 0) >= max_sector:
                    diag["sector_blocked"] = diag.get("sector_blocked", 0) + 1
                    continue
            # Tier別予算か、等金額か
            # 外部要因が逆風の業種を避ける／薄くする
            f_scale = 1.0
            if factor_rule:
                _sec = day.loc[code].get("sector", "")
                for _fac, _secs in SECTOR_FACTOR.items():
                    if _sec not in _secs:
                        continue
                    _t = float(day.iloc[0].get(f"trend_{_fac}", 0) or 0)
                    if _t < 0:      # 逆風
                        mode = factor_rule.get("mode", "skip")
                        if mode == "skip":
                            f_scale = 0.0
                        else:
                            f_scale = min(f_scale, factor_rule.get("thin", 0.5))
                        diag["factor_blocked"] = diag.get("factor_blocked", 0) + 1
                        break
            if f_scale <= 0:
                continue

            nt = len(entry_for(day.loc[code])) if entry_by_tier else n_tr
            rs = 1.0
            if regime:
                if regime.get("use") == "topix":
                    # TOPIX が高値から何％下げているかで判断する
                    v = day.iloc[0].get("mkt_dd")
                    v = float(v) if v is not None and not pd.isna(v) else 0.0
                    if v <= regime.get("cheap_dd", -15.0):
                        rs = regime.get("cheap", 1.0)
                    elif v >= regime.get("rich_dd", -3.0):
                        rs = regime.get("rich", 1.0)
                else:
                    z = float(day.iloc[0].get("mkt_yield_z", 0) or 0)
                    if z >= regime.get("cheap_z", 0.5):
                        rs = regime.get("cheap", 1.0)
                    elif z <= regime.get("rich_z", -0.5):
                        rs = regime.get("rich", 1.0)
                diag.setdefault("regime_months", {})
                k = "割安" if rs > 1 else ("割高" if rs < 1 else "普通")
                diag["regime_months"][k] = diag["regime_months"].get(k, 0) + 1
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
            unit_size *= rs * f_scale
            for _ in range(add):
                if usable < unit_size or unit_size < price:
                    break
                # 100株単位（実運用に合わせる）
                sh = int(unit_size // price // 100) * 100 if tier_budget \
                     else int(unit_size // price)
                if sh <= 0:
                    break
                buy_eff = price * (1 + slip) * (1 + fee)
                cash -= sh * buy_eff
                usable -= sh * buy_eff
                diag["fee"] = diag.get("fee", 0.0) + sh * price * (slip + fee)
                st = pos.setdefault(code, {"units": 0, "lots": [], "peak": price})
                if st["units"] == 0:
                    diag["opened"] += 1
                    opened_at[code] = mi
                    st["origin"] = "first" if mi == 0 else "month"   # 11回目：緊急売りの内訳に使う
                    if stop_loss:
                        st["chk_from"] = _dt_ns      # 損切りの判定は、買った月末の翌日から
                    if rec_relax:
                        if _recn.get(code):
                            diag["rec_buys"] = diag.get("rec_buys", 0) + 1
                        if code in _relaxed:
                            diag["rec_relaxed"] = diag.get("rec_relaxed", 0) + 1
                    if max_sector:
                        _sc = day.loc[code].get("sector", "")
                        sec_count[_sc] = sec_count.get(_sc, 0) + 1
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

        # ── 配当で買い増す ──
        # 新しく買える銘柄を買ったあと、残った現金で、目標の大きさをいちばん下回っている
        # 持ち株を買い増す。目標を超えない範囲で、買える単位（実運用なら100株）ずつ。
        if topup and usable > 0 and pos:
            _lot = 100 if tier_budget else 1
            _avg_w = sum(tw.values()) / len(tw)
            while True:
                _best = None
                for code, st in pos.items():
                    if code not in day.index or not st["lots"]:
                        continue
                    row = day.loc[code]
                    if use_screen and not screen_ok(row):
                        continue
                    if min_yield > 0 and row.get("yield", 0) < min_yield:
                        continue
                    if skip_emg_buy:
                        if emg_actual:
                            _c, _o = emg_flags(row)
                        else:
                            _c, _o = bool(row.get("dps_cut")), bool(row.get("op_drop"))
                        if _c or _o:
                            continue
                    if stop_block and mi < stop_block.get(code, -1):
                        continue
                    _px = float(row["price"])
                    _buy = _px * (1 + slip) * (1 + fee)
                    if usable < _lot * _buy:
                        continue
                    _tier = row.get("tier", "B")
                    if weighted:
                        _tgt = total / target_n * (tw.get(_tier, 1.0) / _avg_w)
                    elif tier_budget:
                        _tgt = TIER_BUDGET.get(_tier, TIER_BUDGET["B"])
                    else:
                        _tgt = total / max_names
                    _val = sum(sh for sh, _ in st["lots"]) * _px
                    if _tgt - _val < _lot * _px:
                        continue
                    _ratio = _val / _tgt if _tgt > 0 else 9.9
                    if _best is None or _ratio < _best[0]:
                        _best = (_ratio, code, _px, _buy, _tgt - _val)
                if _best is None:
                    break
                _, code, _px, _buy, _room = _best
                _n = int(min(_room // (_lot * _px), usable // (_lot * _buy)))
                if _n <= 0:
                    break
                sh = _n * _lot
                cash -= sh * _buy
                usable -= sh * _buy
                diag["fee"] = diag.get("fee", 0.0) + sh * _px * (slip + fee)
                st = pos[code]
                # 特定口座と同じく、取得単価は平均でまとめる（建玉は1つのまま）
                _s0 = sum(s for s, _ in st["lots"])
                _c0 = sum(s * p for s, p in st["lots"])
                st["lots"] = [(_s0 + sh, (_c0 + sh * _buy) / (_s0 + sh))]
                st["cost_sum"] = st.get("cost_sum", 0.0) + sh * _buy
                st["sh_sum"] = st.get("sh_sum", 0) + sh
                trades.append({"code": code, "date": dt, "side": "buy",
                               "price": _px, "shares": sh})
                diag["topups"] = diag.get("topups", 0) + 1

        curve.append({"date": dt, "value": value_of(day),
                      "cash": cash, "names": len(pos)})
        prev_day = day

    # 期末に残っている含み損益（まだ確定していない分）
    unreal = 0.0
    last_day = panel[panel["date"] == dates[-1]].set_index("code") if dates else None
    for code, st in pos.items():
        if last_day is not None and code in last_day.index and st.get("sh_sum"):
            avg = st["cost_sum"] / st["sh_sum"]
            unreal += (last_day.loc[code, "price"] - avg) * st["sh_sum"]
    diag["unrealized"] = unreal

    # 最後に全部売ったときの手取り（10回目。2026年10月12日に追加）
    #   評価額は含み益に税がかかっていない。売る形は売るたびに税を払うので、そのままだと
    #   持ち続ける形が有利に見える。最後の月末の終値で全部売り、約定のずれと税を払った後の額も出しておく。
    #   税は、売却益の合計から繰り越した損（loss_pool）を引いた分にかける（特定口座の損益通算と同じ考え方）。
    liq_value = cash
    liq_gain = 0.0
    for code, st in pos.items():
        if last_day is None or code not in last_day.index or not st["lots"]:
            continue        # 評価額（value_of）と同じく、最後の月にいない銘柄は数えない
        eff_ = float(last_day.loc[code, "price"]) * (1 - slip) * (1 - fee)
        for sh, pr in st["lots"]:
            liq_value += sh * eff_
            liq_gain += (eff_ - pr) * sh
    liq_tax = tax_rate * max(0.0, liq_gain - loss_pool) if tax_rate > 0 else 0.0
    liq_value -= liq_tax
    diag["liq_tax"] = liq_tax

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
            "capital": capital, "diag": diag, "liq_value": liq_value}


def metrics(res: dict) -> dict:
    c = res["curve"]
    if c.empty:
        return {}
    v = c["value"].to_numpy()
    cap = res["capital"]
    yrs = max((c["date"].iloc[-1] - c["date"].iloc[0]).days / 365.25, 0.5)
    total = v[-1] / cap - 1
    cagr = (v[-1] / cap) ** (1 / yrs) - 1
    # 最後に全部売って、約定のずれと税を払った後の年率（10回目）。無ければ評価額の年率と同じにする
    lv = res.get("liq_value")
    if lv is not None and lv > 0:
        liq_total, liq_cagr = lv / cap - 1, (lv / cap) ** (1 / yrs) - 1
    else:
        liq_total, liq_cagr = total, cagr
    # 現金で置いていた割合（月末の平均）。最初の12か月（買い付けの時期）は除く
    if "cash" in c.columns and len(c) > 12:
        _cv = (c["cash"] / c["value"].where(c["value"] > 0)).iloc[12:]
        cash_share = float(_cv.mean() * 100) if _cv.notna().any() else float("nan")
    else:
        cash_share = float("nan")
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
            "確定損益": d.get("realized", 0.0) + d.get("dividend", 0.0),
            "売却益": d.get("realized", 0.0), "配当": d.get("dividend", 0.0),
            "含み損益": d.get("unrealized", 0.0),
            "決済率": exit_rate, "平均保有月数": avg_hold,
            "入れ替え": d.get("rotations", 0),
            "清算後総リターン": liq_total * 100, "清算後年率": liq_cagr * 100,
            "現金比率": cash_share}


# ── 比較の基準：対象銘柄を等金額で買って持ち続けた場合 ──
# ルールが本当に価値を生んでいるのか、それとも相場が上がっただけなのか。
# 銘柄選択も売買判断も一切しない場合の成績を出して並べる。

# ══════════════════════════════════════════
# レンジ（ボックス）売買の検証
#
#   目視でやっている「上下の線に挟まれた動き」を機械の判定に置き換える。
#   ・過去 N 営業日の終値の高値と安値でレンジの上下を決める
#   ・幅が狭すぎ／広すぎるものは「レンジではない」として除く
#   ・下限に近づいたら買い、上限に近づいたら売る
#
#   日次で判定する。月次だと、ひと月のあいだの往復を取りこぼすため。
# ══════════════════════════════════════════
def build_daily(store: dict, years: int) -> dict:
    """銘柄ごとの日次終値を用意する（レンジ判定に使う）。"""
    out = {}
    end = None
    for code, rows in store["quotes"].items():
        px = quotes_to_df(rows)
        if px.empty or len(px) < 200:
            continue
        px = px.set_index("date")["close"]
        out[code] = px
        end = px.index[-1] if end is None else max(end, px.index[-1])
    if end is None:
        return {}
    start = end - pd.DateOffset(years=years)
    return {c: s[s.index >= start] for c, s in out.items()
            if len(s[s.index >= start]) > 60}


def box_bounds(px: pd.Series, win: int, w_min: float, w_max: float):
    """各日について、その日までの過去 win 日でレンジの上下と成否を返す。

    当日を含めると未来の情報が混ざるので、必ず1日ずらしてから計算する。
    """
    prev = px.shift(1)
    hi = prev.rolling(win, min_periods=win).max()
    lo = prev.rolling(win, min_periods=win).min()
    width = (hi - lo) / lo
    ok = (width >= w_min) & (width <= w_max)
    return hi, lo, ok


def simulate_range(store: dict, panel: pd.DataFrame, cfg: dict,
                   capital: float, max_names: int,
                   slip_bps: float = 0.0, fee_bps: float = 0.0,
                   tax_rate: float = 0.0, min_yield: float = 0.0,
                   years: int = 7, d_from=None, d_to=None,
                   dividends: bool = False, daily: dict | None = None,
                   bounds: dict | None = None) -> dict:
    """レンジの下限で買い、上限で売る。日次で判定する。

    cfg の項目
      win        … レンジを測る日数（既定60営業日）
      w_min/w_max… レンジとみなす値幅の範囲（0.08〜0.20 なら 8〜20％）
      buy_at     … 下限からどこまで近づいたら買うか（0.02 なら下限+2％以内）
      sell_at    … 上限からどこまで近づいたら売るか
      stop       … 下限をどれだけ割ったら諦めるか（None なら諦めない）
      gain_only  … True なら、上限で売るのは取得単価より上のときだけ（安値で投げない）
      screen     … True なら、8条件を通り利回りが足切り以上の銘柄だけを対象にする

    2026年10月10日に直したこと
      ・screen：以前は利回りの足切りしか見ていなかった。8条件（screen_pass）も見る。
      ・その月に買ってよい銘柄は「前の月末」の値で決める。以前はその月の月末の値を
        月初から使っていた（月末の利回りは月の途中では分からない＝先の情報）。
      ・パネルにない月（検証期間の前）は、何でも買えてしまっていた。買わないようにした。
      ・dividends：本体のシミュレーションと同じく、権利月の月末に 年間DPS÷2 を受け取る。
        損切りしない形ほど長く持つので、配当を入れないと不利に出る。
      ・d_from／d_to：期間を区切って回せるようにした（窓をずらす検証のため）。
      ・daily／bounds：同じ設定で何度も回すとき、日次の株価とレンジの計算を使い回す。
    """
    win = cfg.get("win", 60)
    w_min = cfg.get("w_min", 0.08)
    w_max = cfg.get("w_max", 0.20)
    buy_at = cfg.get("buy_at", 0.02)
    sell_at = cfg.get("sell_at", 0.02)
    stop = cfg.get("stop")
    gain_only = bool(cfg.get("gain_only", False))
    use_screen = cfg.get("screen", True)

    slip = slip_bps / 10000.0
    fee = fee_bps / 10000.0

    if daily is None:
        daily = build_daily(store, years)
    if not daily:
        return {}

    # 買ってよい銘柄（月ごと）。8条件を通り、利回りが足切り以上。
    ok_by_month = {}
    div_info = {}
    if not panel.empty:
        if use_screen:
            q = panel
            if "screen_pass" in q.columns:
                q = q[q["screen_pass"].fillna(False).astype(bool)]
            if min_yield > 0:
                q = q[q["yield"] >= min_yield]
            for d, g in q.groupby("date"):
                ok_by_month[pd.Timestamp(d).to_period("M")] = set(g["code"])
        if dividends:
            cols = [c for c in ("code", "date", "dps", "fiscal_month", "interim_month")
                    if c in panel.columns]
            if len(cols) == 5:
                for r_ in panel[cols].itertuples(index=False):
                    div_info[(r_.code, pd.Timestamp(r_.date).to_period("M"))] = \
                        (r_.dps, r_.fiscal_month, r_.interim_month)

    # レンジの上下をあらかじめ全銘柄ぶん計算しておく
    if bounds is None:
        bounds = {}
        for code, px in daily.items():
            hi, lo, ok = box_bounds(px, win, w_min, w_max)
            bounds[code] = (hi, lo, ok)

    dates = sorted({d for px in daily.values() for d in px.index})
    if d_from is not None:
        dates = [d for d in dates if d >= pd.Timestamp(d_from)]
    if d_to is not None:
        dates = [d for d in dates if d <= pd.Timestamp(d_to)]
    if not dates:
        return {}

    cash = capital
    pos = {}            # code -> {"sh":株数, "cost":取得単価}
    loss_pool = 0.0
    curve, trades = [], []
    diag = {"buys": 0, "sells": 0, "stops": 0, "tax": 0.0, "fee": 0.0,
            "realized": 0.0, "hold_days": [], "boxes_seen": 0, "dividend": 0.0}

    for i_, dt in enumerate(dates):
        mp = pd.Timestamp(dt).to_period("M")
        # その月に買ってよい銘柄は、前の月末の判定で決める
        allowed = ok_by_month.get(mp - 1, set()) if use_screen else None

        # ── 売り ──
        for code in list(pos):
            px = daily.get(code)
            if px is None or dt not in px.index:
                continue
            price = float(px.loc[dt])
            hi, lo, ok = bounds[code]
            if dt not in hi.index or pd.isna(hi.loc[dt]):
                continue
            h, l = float(hi.loc[dt]), float(lo.loc[dt])
            st = pos[code]

            hit_top = price >= h * (1 - sell_at)
            if hit_top and gain_only and \
                    price * (1 - slip) * (1 - fee) <= st["cost"]:
                hit_top = False       # 取得単価を下回るなら上限でも売らない
            hit_stop = stop is not None and price <= l * (1 - stop)
            if not (hit_top or hit_stop):
                continue

            eff = price * (1 - slip) * (1 - fee)
            gain = (eff - st["cost"]) * st["sh"]
            cash += st["sh"] * eff
            diag["realized"] += gain
            if tax_rate > 0:
                if gain > 0:
                    taxable = max(0.0, gain - loss_pool)
                    loss_pool = max(0.0, loss_pool - gain)
                    t = taxable * tax_rate
                    cash -= t
                    diag["tax"] += t
                else:
                    loss_pool += -gain
            diag["fee"] += st["sh"] * price * (slip + fee)
            diag["sells"] += 1
            if hit_stop:
                diag["stops"] += 1
            diag["hold_days"].append((pd.Timestamp(dt) - st["at"]).days)
            trades.append({"code": code, "date": dt, "side": "sell",
                           "price": eff, "shares": st["sh"],
                           "reason": "stop" if hit_stop else "top"})
            del pos[code]

        # ── 買い ──
        if len(pos) < max_names:
            cands = []
            for code, px in daily.items():
                if code in pos or dt not in px.index:
                    continue
                if allowed is not None and code not in allowed:
                    continue
                hi, lo, ok = bounds[code]
                if dt not in ok.index or not bool(ok.loc[dt]):
                    continue
                price = float(px.loc[dt])
                l, h = float(lo.loc[dt]), float(hi.loc[dt])
                if l <= 0:
                    continue
                # 下限にどれだけ近いか。近いほど優先する。
                near = (price - l) / l
                if near <= buy_at:
                    cands.append((near, code, price))
            cands.sort()
            diag["boxes_seen"] += len(cands)

            total = cash + sum(float(daily[c].loc[dt]) * s["sh"]
                               for c, s in pos.items() if dt in daily[c].index)
            unit = total / max_names
            for _, code, price in cands:
                if len(pos) >= max_names:
                    break
                sh = int(unit // price // 100) * 100
                if sh <= 0:
                    continue
                buy_eff = price * (1 + slip) * (1 + fee)
                if cash < sh * buy_eff:
                    continue
                cash -= sh * buy_eff
                diag["fee"] += sh * price * (slip + fee)
                diag["buys"] += 1
                pos[code] = {"sh": sh, "cost": buy_eff, "at": pd.Timestamp(dt)}
                trades.append({"code": code, "date": dt, "side": "buy",
                               "price": buy_eff, "shares": sh})

        # ── 配当（権利月の月末に 年間DPS÷2。本体のシミュレーションと同じ扱い）──
        if dividends and div_info:
            last_of_month = (i_ + 1 == len(dates)) or \
                (pd.Timestamp(dates[i_ + 1]).to_period("M") != mp)
            if last_of_month:
                mth = pd.Timestamp(dt).month
                for code, st in pos.items():
                    info = div_info.get((code, mp))
                    if not info:
                        continue
                    dps, fm, im = info
                    try:
                        fm, im = int(fm or 0), int(im or 0)
                    except (TypeError, ValueError):
                        continue
                    if dps is not None and not pd.isna(dps) and dps > 0 and mth in (fm, im):
                        gross = float(dps) * 0.5 * st["sh"]
                        net = gross * (1 - tax_rate)
                        cash += net
                        diag["dividend"] += net
                        diag["tax"] += gross - net

        val = cash + sum(float(daily[c].loc[dt]) * s["sh"]
                         for c, s in pos.items() if dt in daily[c].index)
        curve.append({"date": dt, "value": val, "names": len(pos)})

    if not curve:
        return {}
    # 評価額は日次で作ったが、比較相手が月次なので月末だけを取り出す。
    # そうしないとシャープの計算（√12倍）がかみ合わない。
    eq = pd.DataFrame(curve)
    eq["date"] = pd.to_datetime(eq["date"])
    mm = eq.set_index("date").resample("ME").last().dropna()
    curve_m = pd.DataFrame({"date": mm.index, "value": mm["value"].to_numpy(),
                            "names": mm["names"].to_numpy()})

    diag["unrealized"] = sum(
        (float(daily[c].iloc[-1]) - s["cost"]) * s["sh"] for c, s in pos.items())
    diag["opened"] = diag["buys"]
    diag["closed"] = diag["sells"]
    diag["hold_months"] = [d / 30.0 for d in diag["hold_days"]]
    diag["months"] = len(curve_m)
    diag["full_slots"] = 0
    return {"curve": curve_m, "trades": trades, "diag": diag,
            "capital": capital}


# ══════════════════════════════════════════
# 手法の探索
#
#   いまのルールの設定をいじるのではなく、
#   まったく別の考え方の手法を並べて比べる。
#
#   大事なのは選び方。
#   同じ7年のデータで何十通りも試して「いちばん良かったもの」を選ぶと、
#   実力ではなく偶然で選んだものになる。
#   そこで期間を前半と後半に分け、
#     ・前半のデータだけで順位をつける
#     ・後半（選ぶときに一切見ていない期間）で答え合わせをする
#   前半で上位だったものが後半でも上位なら、偶然ではない可能性が高い。
#
#   ここでの比較は、すべて同じ簡易エンジンで行う。
#   Tier別の予算や建玉の管理は入っていないので、
#   本番ルールの数字とは一致しない。手法どうしの優劣を見るためのもの。
# ══════════════════════════════════════════
def _mats(panel: pd.DataFrame) -> dict:
    """比較に使う行列（月 × 銘柄）をまとめて作る。"""
    px = panel.pivot_table(index="date", columns="code", values="price")
    yl = panel.pivot_table(index="date", columns="code", values="yield")
    dg = panel.pivot_table(index="date", columns="code", values="dps_growth")
    po = panel.pivot_table(index="date", columns="code", values="pct_own")
    cut = panel.pivot_table(index="date", columns="code", values="dps_cut")
    px = px.sort_index()
    ret = px.pct_change()
    # 過去12か月の値動き（前月までの情報だけを使う）
    mom = (px / px.shift(12) - 1).shift(1)
    vol = ret.rolling(12, min_periods=6).std().shift(1)
    # 市場全体の動き。等金額で全銘柄を持ったときの指数。
    mret = ret.mean(axis=1).fillna(0)
    idx = (1 + mret).cumprod()
    mkt = (idx / idx.shift(12) - 1).shift(1)
    # 以下は高配当と関係のない指標。すべて前月までの情報だけを使う。
    hi52 = (px / px.rolling(12, min_periods=6).max()).shift(1)       # 1年の高値にどれだけ近いか
    rev1 = (-(px / px.shift(1) - 1)).shift(1)                        # 先月下げた順
    mom121 = (px.shift(1) / px.shift(12) - 1).shift(1)               # 直近1か月を除いた12か月の上昇率
    cov = ret.rolling(24, min_periods=12).cov(mret)
    beta = (cov.div(mret.rolling(24, min_periods=12).var(), axis=0)).shift(1)  # 市場との連動の強さ
    trend_on = (idx > idx.rolling(10, min_periods=10).mean()).shift(1).fillna(True)  # 市場が10か月平均より上
    mvol = mret.rolling(3, min_periods=2).std().shift(1)          # 市場の荒れ具合（直近3か月）
    mvol_hi = (mvol > mvol.rolling(24, min_periods=12).median()).fillna(False)
    out = {"px": px, "yield": yl, "dps_growth": dg, "pct_own": po,
           "cut": cut.fillna(False).astype(bool), "ret": ret,
           "mom": mom, "vol": vol, "mkt": mkt,
           "hi52": hi52, "rev1": rev1, "mom121": mom121, "beta": beta,
           "trend_on": trend_on, "mvol_hi": mvol_hi}
    # 財務を使う指標（列があるときだけ）
    if "eps" in panel.columns:
        f = {c: panel.pivot_table(index="date", columns="code", values=c)
             for c in ("eps", "bps", "np", "eq", "sh", "np_prev")}
        eps, bps = f["eps"].where(f["eps"] > 0), f["bps"].where(f["bps"] > 0)
        out["per"] = (px / eps).shift(1)                       # 低いほど割安
        out["pbr"] = (px / bps).shift(1)
        out["roe"] = (f["np"] / f["eq"].where(f["eq"] > 0)).shift(1)   # 高いほど稼ぐ力
        out["npg"] = ((f["np"] - f["np_prev"]) / f["np_prev"].abs().where(f["np_prev"].abs() > 0)).shift(1)
        out["size"] = (px * f["sh"]).shift(1)                  # 時価総額
    return out


def _score(m: dict, kind: str, d) -> pd.Series:
    """その月の銘柄ごとの点数。大きいほど買いたい。"""
    if kind == "yield":          # 利回りが高い順
        return m["yield"].loc[d]
    if kind == "pct_own":        # その銘柄の過去と比べて安い順
        return m["pct_own"].loc[d]
    if kind == "growth":         # 増配率が高い順
        return m["dps_growth"].loc[d]
    if kind == "mom":            # 過去12か月で上がった順
        return m["mom"].loc[d]
    if kind == "rev":            # 過去12か月で下がった順
        return -m["mom"].loc[d]
    if kind == "lowvol":         # 値動きが小さい順
        return -m["vol"].loc[d]
    if kind == "yield_lowvol":   # 高配当 かつ 値動きが小さい
        # 片方が欠けている銘柄は選ばれないよう、順位づけの前に揃える
        return (m["yield"].loc[d].rank(pct=True)
                + (-m["vol"].loc[d]).rank(pct=True))
    if kind == "yield_mom":      # 高配当 かつ 上がっている
        return (m["yield"].loc[d].rank(pct=True)
                + m["mom"].loc[d].rank(pct=True))
    if kind == "yield_growth":   # 高配当 かつ 増配している
        return (m["yield"].loc[d].rank(pct=True)
                + m["dps_growth"].loc[d].rank(pct=True))
    if kind == "hi52":           # 1年の高値に近い順
        return m["hi52"].loc[d]
    if kind == "rev1":           # 先月下げた順（短期の反発を狙う）
        return m["rev1"].loc[d]
    if kind == "mom121":         # 直近1か月を除いた12か月の上昇率
        return m["mom121"].loc[d]
    if kind == "lowbeta":        # 市場と連動しにくい順
        return -m["beta"].loc[d]
    if kind == "lowper":         # 利益に比べて株価が安い順
        return -m["per"].loc[d]
    if kind == "lowpbr":         # 資産に比べて株価が安い順
        return -m["pbr"].loc[d]
    if kind == "roe":            # 稼ぐ力が高い順
        return m["roe"].loc[d]
    if kind == "npg":            # 利益が伸びている順
        return m["npg"].loc[d]
    if kind == "small":          # 時価総額が小さい順
        return -m["size"].loc[d]
    if kind == "magic":          # 高ROE かつ 低PER（マジックフォーミュラ）
        return m["roe"].loc[d].rank(pct=True) + (-m["per"].loc[d]).rank(pct=True)
    if kind == "value_quality":  # 低PBR かつ 高ROE
        return (-m["pbr"].loc[d]).rank(pct=True) + m["roe"].loc[d].rank(pct=True)
    if kind == "growth_mom":     # 利益が伸びていて、株価も上がっている
        return m["npg"].loc[d].rank(pct=True) + m["mom121"].loc[d].rank(pct=True)
    raise ValueError(kind)


def run_strategy(m: dict, cfg: dict, dates, capital: float,
                 n: int, min_yield: float, tax: float, slip: float) -> dict:
    """ひとつの手法を月次で回す。

    cfg の項目
      score     … 銘柄の選び方（上の _score の種類）
      rebalance … 何か月ごとに入れ替えるか。0 なら入れ替えない（買ったら持ち続ける）
      regime    … 市場の状態で選び方を変える {"up": "mom", "flat": "yield", ...}
      exit_cut  … 減配したら手放すか
    """
    cash, pos = capital, {}          # pos: code -> {"sh": 株数, "cost": 取得単価}
    loss_pool, tax_paid, trades = 0.0, 0.0, 0
    curve = []
    reb = cfg.get("rebalance", 0)
    _rg = {}                         # 状況の判定の記憶（確認期間に使う）

    def price(code, d):
        v = m["px"].at[d, code] if code in m["px"].columns else np.nan
        return v if pd.notna(v) else None

    for i, d in enumerate(dates):
        # ── 減配したら手放す ──
        if cfg.get("exit_cut"):
            for c in list(pos):
                if c in m["cut"].columns and bool(m["cut"].at[d, c]):
                    p = price(c, d)
                    if p is None:
                        continue
                    eff = p * (1 - slip)
                    g = (eff - pos[c]["cost"]) * pos[c]["sh"]
                    cash += pos[c]["sh"] * eff
                    if g > 0:
                        t = max(0.0, g - loss_pool) * tax
                        loss_pool = max(0.0, loss_pool - g)
                        cash -= t
                        tax_paid += t
                    else:
                        loss_pool += -g
                    del pos[c]
                    trades += 1

        # ── 持つ時期を変える手法：持たない月は全部売って現金にする ──
        timing = cfg.get("timing")
        if timing:
            on = True
            if timing == "trend":      # 市場が10か月平均を下回ったら持たない
                on = bool(m["trend_on"].loc[d]) if d in m["trend_on"].index else True
            elif timing == "season":   # 5〜10月は持たない（セル・イン・メイ）
                on = pd.Timestamp(d).month not in (5, 6, 7, 8, 9, 10)
            if not on:
                for c in list(pos):
                    p = price(c, d)
                    if p is None:
                        continue
                    eff = p * (1 - slip)
                    g = (eff - pos[c]["cost"]) * pos[c]["sh"]
                    cash += pos[c]["sh"] * eff
                    if g > 0:
                        t = max(0.0, g - loss_pool) * tax
                        loss_pool = max(0.0, loss_pool - g)
                        cash -= t
                        tax_paid += t
                    else:
                        loss_pool += -g
                    del pos[c]
                    trades += 1
                curve.append({"date": d, "value": cash})
                continue

        # ── 買う銘柄を決める ──
        # 何も持っていないときは、入れ替えの月でなくても買う。
        # そうしないと、年1回の入れ替えが毎回「持たない月」に当たった場合に
        # 一度も買えないまま終わってしまう。
        do_reb = (reb > 0 and i % reb == 0) or (not pos)
        if do_reb:
            kind = cfg.get("score", "yield")
            if cfg.get("regime"):
                if cfg.get("regime_by") == "vol":
                    hi_ = bool(m["mvol_hi"].loc[d]) if d in m["mvol_hi"].index else False
                    st = "rough" if hi_ else "calm"
                else:
                    mk = m["mkt"].loc[d] if d in m["mkt"].index else np.nan
                    st = ("up" if pd.notna(mk) and mk > 0.05 else
                          "down" if pd.notna(mk) and mk < -0.05 else "flat")
                # 同じ状況が続いた月数を数え、決めた回数だけ続いたときに切り替える
                need = cfg.get("regime_confirm", 1)
                if st == _rg.get("pending"):
                    _rg["count"] = _rg.get("count", 0) + 1
                else:
                    _rg["pending"], _rg["count"] = st, 1
                if _rg["count"] >= need or _rg.get("cur") is None:
                    _rg["cur"] = st
                kind = cfg["regime"].get(_rg["cur"], kind)
                if kind == "cash":
                    # 現金にする：全部売る
                    for c in list(pos):
                        p = price(c, d)
                        if p is None:
                            continue
                        eff = p * (1 - slip)
                        g = (eff - pos[c]["cost"]) * pos[c]["sh"]
                        cash += pos[c]["sh"] * eff
                        if g > 0:
                            t = max(0.0, g - loss_pool) * tax
                            loss_pool = max(0.0, loss_pool - g)
                            cash -= t
                            tax_paid += t
                        else:
                            loss_pool += -g
                        del pos[c]
                        trades += 1
                    curve.append({"date": d, "value": cash})
                    continue
            sc = _score(m, kind, d)
            ok = m["yield"].loc[d] >= min_yield
            sc = sc[ok & m["px"].loc[d].notna()].dropna()
            want = list(sc.sort_values(ascending=False).head(n).index)

            # 入れ替える手法は、外れた銘柄を売る
            if reb > 0:
                for c in list(pos):
                    if c in want:
                        continue
                    p = price(c, d)
                    if p is None:
                        continue
                    eff = p * (1 - slip)
                    g = (eff - pos[c]["cost"]) * pos[c]["sh"]
                    cash += pos[c]["sh"] * eff
                    if g > 0:
                        t = max(0.0, g - loss_pool) * tax
                        loss_pool = max(0.0, loss_pool - g)
                        cash -= t
                        tax_paid += t
                    else:
                        loss_pool += -g
                    del pos[c]
                    trades += 1

            # 空いている枠を買う
            total = cash + sum(pos[c]["sh"] * (price(c, d) or pos[c]["cost"])
                               for c in pos)
            unit = total / n
            for c in want:
                if c in pos or len(pos) >= n:
                    continue
                p = price(c, d)
                if not p or p <= 0:
                    continue
                sh = int(unit // p // 100) * 100
                if sh <= 0:
                    continue
                eff = p * (1 + slip)
                if sh * eff > cash:
                    sh = int(cash // eff // 100) * 100
                    if sh <= 0:
                        continue
                cash -= sh * eff
                pos[c] = {"sh": sh, "cost": eff}
                trades += 1

        # ── 評価と配当 ──
        val = cash
        div = 0.0
        for c, st in pos.items():
            p = price(c, d) or st["cost"]
            val += st["sh"] * p
            y = m["yield"].at[d, c] if c in m["yield"].columns else np.nan
            if pd.notna(y):
                div += st["sh"] * p * (y / 100) / 12
        cash += div * (1 - tax)
        curve.append({"date": d, "value": val + div * (1 - tax)})

    eq = pd.DataFrame(curve)
    v = eq["value"].to_numpy()
    yrs = max((dates[-1] - dates[0]).days / 365.25, 0.5)
    dd = float((1 - v / np.maximum.accumulate(v)).max())
    r = pd.Series(v).pct_change().dropna()
    return {"年率": (v[-1] / capital) ** (1 / yrs) - 1,
            "最大下落": dd, "シャープ": float(r.mean() / r.std() * np.sqrt(12))
            if r.std() > 0 else 0.0, "売買": trades, "税金": tax_paid,
            "curve": eq}




# ══════════════════════════════════════════
# 組み合わせ vs 使い分け
#
#   「どんな状況でも一定の成果」を目指すとき、方法は2つある。
#     A. 使い分け … 状況を読んで、そのとき良さそうな手法に乗り換える
#     B. 同時に持つ … 動きの違う手法を最初から一緒に持ち、片方の弱い時期を補う
#   これまでの検証では A は16回試して16回とも成績を下げた。
#   ここでは A の最も丁寧な形（確認期間つき）と B を同じ土俵で比べる。
#
#   見るのは平均ではなく「悪いとき」。最悪の年、マイナスの年の数、最大下落。
# ══════════════════════════════════════════
SLEEVES = {
    "高配当": {"score": "yield", "rebalance": 0, "min_yield": 4.0,
              "desc": "利回り4％以上を、高い順に買って持ち続ける（本番に近い）"},
    "モメンタム": {"score": "mom121", "rebalance": 1, "min_yield": 0.0,
                  "desc": "12か月の上昇率が高い順（高配当と逆の時期に強い）"},
    "低ボラ": {"score": "lowvol", "rebalance": 12, "min_yield": 0.0,
              "desc": "値動きの小さい順（守り）"},
    "現金": None,
}

PORTFOLIOS = [
    # ── 1本だけ ──
    ("高配当だけ", {"mix": {"高配当": 1.0}}, "いまの本番に近い形。比較の基準"),
    ("モメンタムだけ", {"mix": {"モメンタム": 1.0}}, ""),
    ("低ボラだけ", {"mix": {"低ボラ": 1.0}}, ""),
    # ── 同時に持つ（別々の口座で持つ形。互いに触らない） ──
    ("高配当70＋モメンタム30", {"mix": {"高配当": 0.7, "モメンタム": 0.3}}, "動きの逆な2つを持つ"),
    ("高配当50＋モメンタム50", {"mix": {"高配当": 0.5, "モメンタム": 0.5}}, ""),
    ("高配当50＋低ボラ50", {"mix": {"高配当": 0.5, "低ボラ": 0.5}}, "守りを厚く"),
    ("高配当60＋モメンタム20＋低ボラ20", {"mix": {"高配当": 0.6, "モメンタム": 0.2, "低ボラ": 0.2}}, "3つに分ける"),
    ("高配当70＋現金30", {"mix": {"高配当": 0.7, "現金": 0.3}}, "現金を残す"),
    # ── 同時に持つ＋年1回、元の比率に戻す ──
    ("高配当50＋モメンタム50・年1回戻す", {"mix": {"高配当": 0.5, "モメンタム": 0.5}, "rebalance_years": 1},
     "増えた側を売って減った側を買い足す。状況は読まない"),
    # ── 使い分け（状況を読んで乗り換える。2か月続いたら切り替え） ──
    ("使い分け：上げはモメンタム・それ以外は高配当",
     {"switch": {"regime": {"up": "mom121", "flat": "yield", "down": "yield"},
                 "rebalance": 3, "regime_confirm": 2}, "min_yield": 0.0},
     "市場の12か月の動きで判定"),
    ("使い分け：荒れたら低ボラ・落ち着けば高配当",
     {"switch": {"regime": {"rough": "lowvol", "calm": "yield"}, "regime_by": "vol",
                 "rebalance": 3, "regime_confirm": 2}, "min_yield": 0.0},
     "市場の荒れ具合で判定"),
    ("使い分け：上げはモメンタム・下げは現金",
     {"switch": {"regime": {"up": "mom121", "flat": "yield", "down": "cash"},
                 "rebalance": 3, "regime_confirm": 2}, "min_yield": 0.0},
     "下げ相場は全部売って現金"),
]


def _curve_stats(curve: pd.DataFrame, capital: float) -> dict:
    v = curve["value"].to_numpy()
    yrs = max((curve["date"].iloc[-1] - curve["date"].iloc[0]).days / 365.25, 0.5)
    dd = float((1 - v / np.maximum.accumulate(v)).max())
    yearly = {}
    c = curve.set_index("date")["value"]
    ye = c.resample("YE").last()
    prev = capital
    for d_, val in ye.items():
        yearly[d_.year] = val / prev - 1
        prev = val
    return {"年率": (v[-1] / capital) ** (1 / yrs) - 1, "最大下落": dd,
            "年ごと": yearly}


def run_portfolio(m: dict, spec: dict, dates, capital: float, n: int,
                  tax: float, slip: float) -> dict:
    """組み合わせ、または使い分けを回す。"""
    if "switch" in spec:
        cfg = dict(spec["switch"])
        out = run_strategy(m, cfg, dates, capital, n, spec.get("min_yield", 0.0),
                           tax, slip)
        st = _curve_stats(out["curve"], capital)
        st["売買"] = out["売買"]
        return st

    mix = spec["mix"]
    curves, trades = [], 0
    for name, w in mix.items():
        if name == "現金":
            curves.append(pd.Series(capital * w, index=dates))
            continue
        cfg = dict(SLEEVES[name])
        my = cfg.pop("min_yield", 0.0)
        cfg.pop("desc", None)
        out = run_strategy(m, cfg, dates, capital * w, max(3, int(round(n * w))),
                           my, tax, slip)
        curves.append(out["curve"].set_index("date")["value"])
        trades += out["売買"]

    total = pd.concat(curves, axis=1).sum(axis=1)

    # 年1回、元の比率に戻す（増えた側を売り、減った側を買う）。
    # 各手法の月次の増減率を使って近似する。売った分の利益に税金、出し入れにずれ。
    if spec.get("rebalance_years"):
        rets = [c.pct_change().fillna(0) for c in curves]
        ws = list(mix.values())
        vals = [capital * w for w in ws]
        basis = list(vals)
        out_v, extra_tr = [], 0
        every = spec["rebalance_years"] * 12
        for i, d in enumerate(dates):
            vals = [v * (1 + r.loc[d]) for v, r in zip(vals, rets)]
            tot = sum(vals)
            if i > 0 and i % every == 0:
                tgt = [tot * w for w in ws]
                for k in range(len(vals)):
                    delta = tgt[k] - vals[k]
                    if delta < 0:            # 売る側：ずれと税金
                        sold = -delta
                        gain_ratio = max(0.0, 1 - basis[k] / vals[k]) if vals[k] > 0 else 0
                        cost = sold * slip + sold * gain_ratio * tax
                        tot -= cost
                        basis[k] *= (vals[k] - sold) / vals[k] if vals[k] > 0 else 1
                    else:                    # 買う側：ずれ
                        tot -= delta * slip
                        basis[k] += delta
                    extra_tr += 1
                vals = [tot * w for w in ws]
            out_v.append(tot)
        total = pd.Series(out_v, index=dates)
        trades += extra_tr

    curve = pd.DataFrame({"date": total.index, "value": total.to_numpy()})
    st = _curve_stats(curve, capital)
    st["売買"] = trades
    return st




# ══════════════════════════════════════════
# 日本の長い歴史で確かめる（1990年〜）
#
#   J-Quants は9年しか遡れず、その9年は割安株に極端に有利な時期だった。
#   ケネス・フレンチ教授（ダートマス大学）が公開している日本のファクターの月次成績で、
#   「割安株に傾ける」「上がっている株に傾ける」が35年間でどうだったかを見る。
#     HML … 割安株（PBRが低い）− 割高株。いまのルールの考え方に近い
#     WML … 上がっている株 − 下がっている株（モメンタム）
#     RMW … 稼ぐ力の高い会社 − 低い会社（取れれば）
#   どれも「差」なので、ドル建てでも為替の影響はほぼ打ち消される。
# ══════════════════════════════════════════
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def _french(name: str) -> pd.DataFrame | None:
    """フレンチ教授のデータを取得し、月次の表だけを取り出す（％→小数）。"""
    import io
    import zipfile
    url = FRENCH + name + "_CSV.zip"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        txt = z.read(z.namelist()[0]).decode("latin-1")
    except Exception as e:
        log.warning("%s を取得できませんでした（%s）", name, e)
        return None
    lines = txt.splitlines()
    head, rows = None, []
    for ln in lines:
        cells = [c.strip() for c in ln.split(",")]
        if head is None:
            if len(cells) >= 2 and cells[0] == "" and any(c for c in cells[1:]):
                head = cells[1:]
            continue
        if re.fullmatch(r"\d{6}", cells[0] or ""):
            try:
                rows.append([cells[0]] + [float(x) for x in cells[1:1 + len(head)]])
            except ValueError:
                continue
        elif rows:
            break            # 月次の表が終わった（このあと年次の表が続く）
    if not rows:
        log.warning("%s の中身を読めませんでした。先頭の行：\n%s", name, "\n".join(lines[:8]))
        return None
    df = pd.DataFrame(rows, columns=["ym"] + head)
    df["date"] = pd.to_datetime(df["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date").drop(columns=["ym"]) / 100.0
    log.info("  %s … %s 〜 %s（%d か月）列：%s", name, df.index.min().date(),
             df.index.max().date(), len(df), "・".join(head))
    return df


def _streaks(r: pd.Series) -> dict:
    """累積・最大下落・前の山を回復するまでの最長の月数・年ごとの成績。"""
    v = (1 + r).cumprod()
    peak = v.cummax()
    dd = 1 - v / peak
    under, longest = 0, 0
    for x in (v < peak).to_numpy():
        under = under + 1 if x else 0
        longest = max(longest, under)
    yr = (1 + r).groupby(r.index.year).prod() - 1
    yrs = len(r) / 12
    return {"年率": v.iloc[-1] ** (1 / yrs) - 1, "最大下落": dd.max(),
            "回復まで最長": longest, "年ごと": yr,
            "負けた年": int((yr < 0).sum()), "年数": len(yr)}


SWITCH_SIG = None   # 割安寄り（True）か稼ぐ力寄り（False）か。月末の日付ごと


def french_switch_signal(months: int = 36, lag: int = 2) -> pd.Series | None:
    """過去 months か月、割安（HML）が稼ぐ力（RMW）を上回っていれば True。
    データの公開には1〜2か月の遅れがあるので、lag か月前までの情報で判断する。"""
    f5 = _french("Japan_5_Factors")
    if f5 is None or "HML" not in f5.columns or "RMW" not in f5.columns:
        return None
    spread = f5["HML"] - f5["RMW"]
    sig = (spread.rolling(months).sum().shift(lag) > 0)
    sig = sig[spread.rolling(months).count().shift(lag) >= months]
    return sig


def history_test() -> int:
    log.info("日本の長い歴史のデータを取得します（ケネス・フレンチ教授のデータライブラリ）…")
    f3 = _french("Japan_3_Factors")
    mom = _french("Japan_Mom_Factor")
    f5 = _french("Japan_5_Factors")
    if f3 is None or "HML" not in f3.columns:
        sys.exit("日本の割安株のデータ（HML）が取れませんでした。ログの先頭の行を見せてください。")
    ser = {"割安株 − 割高株（HML）": f3["HML"]}
    if mom is not None:
        mc = [c for c in mom.columns if c.upper() in ("WML", "MOM", "UMD")] or list(mom.columns[:1])
        ser["上がっている株 − 下がっている株（モメンタム）"] = mom[mc[0]]
    if f5 is not None and "RMW" in f5.columns:
        ser["稼ぐ力の高い会社 − 低い会社（RMW）"] = f5["RMW"]
    df = pd.DataFrame(ser).dropna()
    if "上がっている株 − 下がっている株（モメンタム）" in df.columns:
        df["割安50＋モメンタム50"] = (df["割安株 − 割高株（HML）"] * 0.5
                                   + df["上がっている株 − 下がっている株（モメンタム）"] * 0.5)
    if "稼ぐ力の高い会社 − 低い会社（RMW）" in df.columns:
        H, Rm = df["割安株 − 割高株（HML）"], df["稼ぐ力の高い会社 − 低い会社（RMW）"]
        for wv in (0.7, 0.5, 0.3):
            df[f"割安{int(wv*100)}＋稼ぐ力{int((1-wv)*100)}"] = H * wv + Rm * (1 - wv)
        if f5 is not None and "CMA" in f5.columns:
            C = f5["CMA"].reindex(df.index)
            df["投資を控えめにする会社 − 積極的な会社（CMA）"] = C
            df["割安＋稼ぐ力＋控えめ（3分の1ずつ）"] = (H + Rm + C) / 3
    a, b = df.index.min(), df.index.max()
    print(f"\n■ 日本の長い歴史で確かめる（{a.date()} 〜 {b.date()}・{len(df)}か月）\n")
    print("  どれも「Aの株を買い、Bの株を売った」ときの差の成績です。")
    print("  ＋なら A が B を上回った。ドル建てですが、差なので為替の影響はほぼ打ち消されます。\n")

    st = {c: _streaks(df[c]) for c in df.columns}
    print(f"{'':<34}{'年率':>7}{'最大下落':>9}{'回復まで最長':>12}{'負けた年':>9}")
    print("-" * 74)
    for c, x in st.items():
        print(f"{c[:32]:<34}{x['年率'] * 100:>+6.1f}%{x['最大下落'] * 100:>8.1f}%"
              f"{x['回復まで最長']:>9}か月{x['負けた年']:>5}/{x['年数']}")

    periods = [("バブル崩壊（1990〜1999）", "1990", "1999"),
               ("ITバブルと回復（2000〜2007）", "2000", "2007"),
               ("リーマン・円高（2008〜2012）", "2008", "2012"),
               ("アベノミクス・低金利（2013〜2019）", "2013", "2019"),
               ("コロナ・金利上昇（2020〜）", "2020", "2100")]
    print("\n■ 時代ごとの年率\n")
    print(f"{'':<34}" + "".join(f"{p[0][:12]:>14}" for p in periods))
    print("-" * (34 + 14 * len(periods)))
    for c in df.columns:
        cells = []
        for _n, y0, y1 in periods:
            sub = df[c][(df.index.year >= int(y0)) & (df.index.year <= int(y1))]
            cells.append(f"{((1 + sub).prod() ** (12 / len(sub)) - 1) * 100:>+13.1f}%"
                         if len(sub) >= 12 else f"{'—':>14}")
        print(f"{c[:32]:<34}" + "".join(cells))
    print("\n  列の見出し：" + " ／ ".join(p[0] for p in periods))

    # どんな状況でも、に最も近いのはどれか（基準は先に決めておく）
    #   ① 5つの時代すべてでプラス
    #   ② そのうえで、回復までの最長がいちばん短い
    print("\n■ 5つの時代すべてでプラスだったもの（回復までの最長が短い順）\n")
    ok = []
    for c in df.columns:
        eras = []
        for _n, y0, y1 in periods:
            sub = df[c][(df.index.year >= int(y0)) & (df.index.year <= int(y1))]
            if len(sub) >= 12:
                eras.append((1 + sub).prod() ** (12 / len(sub)) - 1)
        if eras and min(eras) > 0:
            ok.append((st[c]["回復まで最長"], c, min(eras)))
    if not ok:
        print("  ありませんでした。")
    for rec, c, mn in sorted(ok):
        x = st[c]
        print(f"  {c[:30]:<32} 回復まで最長 {rec:>3}か月 ／ 年率 {x['年率']*100:+.1f}% ／ "
              f"最大下落 {x['最大下落']*100:.1f}% ／ いちばん悪い時代 {mn*100:+.1f}%")

    if "稼ぐ力の高い会社 − 低い会社（RMW）" in df.columns:
        cor2 = df["割安株 − 割高株（HML）"].corr(df["稼ぐ力の高い会社 − 低い会社（RMW）"])
        print(f"\n  割安と稼ぐ力の連動 … {cor2:+.2f}")
    if "上がっている株 − 下がっている株（モメンタム）" in df.columns:
        cor = df["割安株 − 割高株（HML）"].corr(df["上がっている株 − 下がっている株（モメンタム）"])
        print(f"\n  割安とモメンタムの連動 … {cor:+.2f}（マイナスなら、片方が負ける月にもう片方が勝ちやすい）")

    # ══════════════════════════════════════════
    # 市況に合わせて、割安と稼ぐ力の割合を変える（35年）
    #   割安寄り＝割安70＋稼ぐ力30、稼ぐ力寄り＝割安30＋稼ぐ力70。
    #   判断には前の月までの情報だけを使う。
    #   切り替えるたびに入れ替えの費用がかかるとして、割合が変わった月に
    #   変わった割合 × 0.5％ を差し引く（売らずに新しい買いだけで寄せるなら、実際はもっと小さい）。
    # ══════════════════════════════════════════
    if "稼ぐ力の高い会社 − 低い会社（RMW）" in df.columns:
        H = df["割安株 − 割高株（HML）"]
        Rm = df["稼ぐ力の高い会社 − 低い会社（RMW）"]
        spread = H - Rm
        sig = {}
        sig["因子の勢い（12か月）"] = spread.rolling(12).sum().shift(1) > 0
        sig["因子の勢い（36か月）"] = spread.rolling(36).sum().shift(1) > 0
        try:
            mk = fetch_markets()
        except Exception as e:
            mk = {}
            log.warning("米金利・ドル円を取得できませんでした（%s）", e)
        if "us10y" in mk:
            r10 = mk["us10y"].resample("ME").last().reindex(df.index, method="ffill")
            sig["米金利の向き（12か月）"] = (r10 - r10.shift(12)).shift(1) > 0
        if "usdjpy" in mk:
            fx = mk["usdjpy"].resample("ME").last().reindex(df.index, method="ffill")
            sig["円安・円高の向き（12か月）"] = (fx / fx.shift(12) - 1).shift(1) > 0

        def _mix(w: pd.Series) -> pd.Series:
            w = w.astype(float)
            cost = w.diff().abs().fillna(0) * 0.005
            return w * H + (1 - w) * Rm - cost

        cand = {"固定：割安100（いまの本番の考え方）": H,
                "固定：割安70＋稼ぐ力30": H * 0.7 + Rm * 0.3,
                "固定：割安50＋稼ぐ力50": H * 0.5 + Rm * 0.5}
        for nm_, sg in sig.items():
            w = sg.map({True: 0.7, False: 0.3}).where(sg.notna(), 0.5)
            cand["切り替え：" + nm_] = _mix(w)
        cdf = pd.DataFrame(cand).dropna()
        mid = cdf.index[len(cdf) // 2]

        def _ann(x):
            return (1 + x).prod() ** (12 / len(x)) - 1 if len(x) >= 12 else float("nan")

        print(f"\n■ 市況に合わせて割合を変える（{cdf.index.min().date()} 〜 {cdf.index.max().date()}）\n")
        print(f"  前半 〜{mid.date()} ／ 後半 {mid.date()}〜。判断には前の月までの情報だけを使う。\n")
        print(f"{'':<36}{'年率':>7}{'前半':>7}{'後半':>7}{'最大下落':>9}{'回復まで最長':>12}{'切替':>6}")
        print("-" * 86)
        res = {}
        for c in cdf.columns:
            x = _streaks(cdf[c])
            a_, b_ = _ann(cdf[c][cdf.index < mid]), _ann(cdf[c][cdf.index >= mid])
            nsw = "—"
            if c.startswith("切り替え："):
                sg = sig[c.replace("切り替え：", "")].reindex(cdf.index)
                nsw = str(int((sg.astype(float).diff().abs() > 0).sum()))
            res[c] = (x["年率"], a_, b_, x["最大下落"], x["回復まで最長"])
            print(f"{c[:34]:<36}{x['年率']*100:>+6.1f}%{a_*100:>+6.1f}%{b_*100:>+6.1f}%"
                  f"{x['最大下落']*100:>8.1f}%{x['回復まで最長']:>9}か月{nsw:>6}")

        print("\n  時代ごとの年率\n")
        print(f"{'':<36}" + "".join(f"{p[0][:10]:>12}" for p in periods))
        print("-" * (36 + 12 * len(periods)))
        for c in cdf.columns:
            cells = []
            for _n, y0, y1 in periods:
                sub = cdf[c][(cdf.index.year >= int(y0)) & (cdf.index.year <= int(y1))]
                v = _ann(sub)
                cells.append(f"{v*100:>+11.1f}%" if v == v else f"{'—':>12}")
            print(f"{c[:34]:<36}" + "".join(cells))

        base = res["固定：割安70＋稼ぐ力30"]
        print("\n【判定】基準＝固定の割安70＋稼ぐ力30。3つとも満たせば採用\n")
        print("  ① 35年の年率で上回る ② 回復までの最長が長くならない ③ 前半・後半の両方で上回る\n")
        for c, v in res.items():
            if not c.startswith("切り替え："):
                continue
            ok = [v[0] > base[0], v[4] <= base[4], v[1] > base[1] and v[2] > base[2]]
            mark = "◎ 採用の条件を満たす" if all(ok) else f"× 満たさない（{'・'.join(n for n, o in zip(['①', '②', '③'], ok) if not o)}）"
            print(f"  {c[5:][:24]:<26} {mark}")
        print("\n  ※ ここでの数字は市場全体の「割安株と割高株の差」「稼ぐ力の差」で、")
        print("    いまのルールそのものではありません。考え方として効くかどうかの確認です。")

    print("\n■ 年ごとの成績（割安株 − 割高株）\n")
    yr = st["割安株 − 割高株（HML）"]["年ごと"]
    line = []
    for y, v in yr.items():
        line.append(f"{y}:{v * 100:+.0f}%")
        if len(line) == 8:
            print("  " + "  ".join(line)); line = []
    if line:
        print("  " + "  ".join(line))

    print("\n【読み方】\n")
    print("  ・いまのルール（低PBR順）は「割安株に傾ける」考え方。HML が負けている時代には、")
    print("    いまのルールも市場平均に負けやすいと考えられる。")
    print("  ・「回復まで最長」は、前の山を取り戻すまでにかかった最長の月数。")
    print("    この長さの逆風に耐えられるかが、実際に続けられるかどうかを決める。")
    print("  ・組み合わせの行は、2つを半分ずつ持った場合。最大下落と回復までの長さが")
    print("    片方だけより短くなっていれば、「同時に持つ」ことに長い歴史の裏づけがある。")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / "japan_factors.csv", encoding="utf-8-sig")
    print("\n書き出しました: data/japan_factors.csv")
    return 0


# 比べる手法。設定をいじったものではなく、考え方が違うものを並べる。
EXPLORE_FUND = [
    ("低PER（利益に比べて安い）", {"score": "lowper", "rebalance": 12},
     "割安株の代表。配当は見ない"),
    ("低PBR（資産に比べて安い）", {"score": "lowpbr", "rebalance": 12},
     "もう一つの割安株。日本で長く効くとされた"),
    ("高ROE（稼ぐ力）", {"score": "roe", "rebalance": 12},
     "質の高い会社を買う。割安かどうかは見ない"),
    ("利益の伸び（成長株）", {"score": "npg", "rebalance": 12},
     "純利益の伸び率が高い順。高配当の反対側"),
    ("小型株", {"score": "small", "rebalance": 12},
     "時価総額の小さい順。小型株効果"),
    ("マジックフォーミュラ（高ROE×低PER）", {"score": "magic", "rebalance": 12},
     "グリーンブラットの有名な手法"),
    ("低PBR×高ROE", {"score": "value_quality", "rebalance": 12},
     "安くて質の高い会社"),
    ("成長×モメンタム", {"score": "growth_mom", "rebalance": 3},
     "利益が伸びていて株価も上がっている会社"),
]


EXPLORE_OTHER = [
    ("1年の高値に近い順", {"score": "hi52", "rebalance": 1},
     "高値を更新しそうな銘柄を買う。「高値は更新されやすい」という経験則"),
    ("先月下げた順（短期の反発）", {"score": "rev1", "rebalance": 1},
     "先月大きく下げた銘柄を買い、翌月の戻りを狙う"),
    ("12か月の上昇率（直近1か月を除く）", {"score": "mom121", "rebalance": 1},
     "学術的に最もよく知られたモメンタムの形。直近の揺り戻しを避ける"),
    ("市場と連動しにくい順（低ベータ）", {"score": "lowbeta", "rebalance": 12},
     "相場全体が動いても動きにくい銘柄を持つ"),
    ("モメンタム＋下げ相場は現金", {"score": "mom121", "rebalance": 1, "timing": "trend"},
     "市場が10か月平均を下回ったら全部売って現金で待つ"),
    ("値動きが小さい順＋下げ相場は現金", {"score": "lowvol", "rebalance": 12, "timing": "trend"},
     "低ボラの銘柄を持ち、市場が崩れたら現金に逃げる"),
    ("1年の高値に近い順＋下げ相場は現金", {"score": "hi52", "rebalance": 1, "timing": "trend"},
     "高値圏の銘柄を買い、市場が崩れたら現金"),
    ("モメンタム＋夏は持たない", {"score": "mom121", "rebalance": 1, "timing": "season"},
     "5〜10月は現金。「5月に売って逃げろ」という格言"),
    ("値動きが小さい順＋夏は持たない", {"score": "lowvol", "rebalance": 12, "timing": "season"},
     "低ボラの銘柄を11〜4月だけ持つ"),
    ("市況で切り替え：上げはモメンタム・下げは低ベータ", {"rebalance": 3,
     "regime": {"up": "mom121", "flat": "lowvol", "down": "lowbeta"}},
     "市場の勢いで、持つ銘柄の性格を変える"),
]


EXPLORE = [
    ("いまのルールに近い形", {"score": "pct_own", "rebalance": 0, "exit_cut": True},
     "その銘柄の過去3年と比べて安いものを買い、減配するまで売らない"),
    ("ダウの犬（年1回）", {"score": "yield", "rebalance": 12},
     "利回りが高い順に買い、1年ごとに入れ替える。有名な古典的手法"),
    ("高配当・毎月入れ替え", {"score": "yield", "rebalance": 1},
     "利回りが高い順に買い、毎月見直す"),
    ("高配当・買って放置", {"score": "yield", "rebalance": 0},
     "利回りが高い順に買い、あとは何もしない"),
    ("増配率が高い順", {"score": "growth", "rebalance": 12},
     "配当を増やしている会社を買う。利回りの高さは見ない"),
    ("上がっている順（モメンタム）", {"score": "mom", "rebalance": 1},
     "過去12か月で上がった銘柄を買う。高配当戦略とは逆の発想"),
    ("下がっている順（逆張り）", {"score": "rev", "rebalance": 1},
     "過去12か月で下がった銘柄を買う"),
    ("値動きが小さい順", {"score": "lowvol", "rebalance": 12},
     "値動きの小さい銘柄を買う。世界的に効くとされる手法"),
    ("高配当 × 値動きが小さい", {"score": "yield_lowvol", "rebalance": 12},
     "2つの条件を組み合わせる"),
    ("高配当 × 上がっている", {"score": "yield_mom", "rebalance": 12},
     "高配当のなかで、勢いのあるものを買う"),
    ("高配当 × 増配している", {"score": "yield_growth", "rebalance": 12},
     "高配当のなかで、配当を増やしているものを買う"),
    ("市況で切り替え：上昇はモメンタム", {"rebalance": 3,
     "regime": {"up": "mom", "flat": "yield", "down": "yield"}},
     "市場が上げているときはモメンタム、それ以外は高配当"),
    ("市況で切り替え：上昇は放置", {"rebalance": 3,
     "regime": {"up": "pct_own", "flat": "yield", "down": "yield"}},
     "市場が上げているときは割安買い、それ以外は高配当"),
    ("市況で切り替え：下落だけ低ボラ", {"rebalance": 3,
     "regime": {"up": "yield", "flat": "yield", "down": "lowvol"}},
     "市場が下げているときだけ、値動きの小さい銘柄に逃げる"),
]


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
    gain_last = 0.0       # 最後の月の値上がり益（最後に全部売ったときの税の計算に使う）
    for dt in dts:
        day = pn[pn["date"] == dt].set_index("code")
        v = 0.0
        g_px = 0.0
        for c in codes:
            if c in day.index and base_px.get(c):
                v += w * day.loc[c, "price"] / base_px[c]
                g_px += w * (day.loc[c, "price"] / base_px[c] - 1.0)
                m = pd.Timestamp(dt).month
                for key in ("fiscal_month", "interim_month"):
                    if int(day.loc[c].get(key, 0) or 0) == m and day.loc[c, "dps"] > 0:
                        g = w * (day.loc[c, "dps"] * 0.5) / base_px[c]
                        div_total += g * (1 - tax_rate)
            else:
                v += w        # 上場廃止などは取得時の値で据え置く
        vals.append(v + div_total)
        gain_last = g_px
    arr = np.array(vals)
    yrs = max((pd.Timestamp(dts[-1]) - pd.Timestamp(dts[0])).days / 365.25, 0.5)
    dd = float((1 - arr / np.maximum.accumulate(arr)).max())
    r = pd.Series(arr).pct_change().dropna()
    # 最後に全部売ったときの手取り（10回目）。値上がり益に税をかける（配当は受け取ったときに課税済み）
    liq_last = arr[-1] - tax_rate * max(0.0, gain_last)
    return {"curve": pd.DataFrame({"date": pd.to_datetime(dts),
                                  "value": arr * 1.0}),
            "総リターン": (arr[-1] - 1) * 100,
            "年率": (arr[-1] ** (1 / yrs) - 1) * 100,
            "清算後年率": ((liq_last ** (1 / yrs) - 1) * 100) if liq_last > 0 else float("nan"),
            "最大下落": dd * 100,
            "シャープ": float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0.0}


# ══════════════════════════════════════════
def main() -> int:
    # 結果（print）と進み具合（log）の順番を、ファイルに書き出したときも崩さない。
    # 書き出し先がファイルだと print はためてから書かれるため、
    # 「続けて総当たりを回します」が表の途中に割り込むことがあった（2026年10月11日）。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
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
    ap.add_argument("--random-test", action="store_true",
                    help="同じ候補の中からでたらめに選んだ場合と比べ、"
                         "いまのルールの優位が運ではないかを確かめる")
    ap.add_argument("--random-n", type=int, default=20,
                    help="でたらめに選ぶ回数（既定20。期間6通り×3種類なので、20回で360回まわる）")
    ap.add_argument("--blend", action="store_true",
                    help="動きの違う手法を同時に持つ形と、状況で使い分ける形を、"
                         "同じ土俵で比べる。悪いとき（最悪の年・最大下落）を見る")
    ap.add_argument("--explore", action="store_true",
                    help="別の考え方の手法を並べて比べる。"
                         "前半のデータで順位をつけ、後半で答え合わせをする")
    ap.add_argument("--explore-names", type=int, default=15,
                    help="探索で同時に持つ銘柄数（既定15）")
    ap.add_argument("--after-event", action="store_true",
                    help="減配・業績急変のあと、株価がどうなったかを調べる")
    ap.add_argument("--factor-test", action="store_true",
                    help="外部要因（金利・原油・為替）が逆風の業種を"
                         "避ける形を検証する")
    ap.add_argument("--factor-win", type=int, default=6,
                    help="何か月前と比べて上向き／下向きを判定するか")
    ap.add_argument("--fx-test", action="store_true",
                    help="ドル円との関係を調べる。成績のどれだけが"
                         "円安に支えられていたかを切り分ける")
    ap.add_argument("--range-swing", action="store_true",
                    help="高配当で買った銘柄を、レンジの上限で一度降りる形を検証する。"
                         "土台は売らない戦略のまま、その上に往復を乗せる")
    ap.add_argument("--range-test", action="store_true",
                    help="レンジ（ボックス）売買を検証する。"
                         "下限で買い・上限で売る形を日次で回し、"
                         "「売らない」戦略と同じ土俵で比べる")
    ap.add_argument("--range-opts", default="60,0.08,0.20,0.02,0.02,0",
                    help="レンジの設定をまとめて指定する。"
                         "「日数,幅の下限,幅の上限,買い幅,売り幅,諦める幅」の順。"
                         "例: 60,0.08,0.20,0.02,0.02,0")
    ap.add_argument("--grid", action="store_true",
                    help="条件を総当たりで組み合わせて検証し、"
                         "どの条件でも成り立つ結論だけを取り出す")
    ap.add_argument("--grid-ramp", default="0,12",
                    help="総当たりで試す資金投入の月数（カンマ区切り）")
    ap.add_argument("--grid-min-yield", default="0,3",
                    help="総当たりで試す利回り足切り（カンマ区切り）")
    ap.add_argument("--grid-lookback", default="24",
                    help="総当たりで試す分位の月数（カンマ区切り）。例: 24,60")
    ap.add_argument("--grid-max-names", default="20",
                    help="総当たりで試す保有上限（カンマ区切り）。例: 20,30")
    ap.add_argument("--grid-window", type=int, default=4,
                    help="総当たりで使う窓の年数")
    ap.add_argument("--ramp", type=int, default=0,
                    help="資金を何か月かけて入れるか。0なら初日に全額。"
                         "12なら毎月1/12ずつ。初日の一括購入に結果が"
                         "支配されるのを防ぐ")
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
    # 入力欄が上限（25個）に達しているので、新しいモードは
    # 「比較するルール」の欄に特別な言葉を入れて動かす。
    if (args.only or "").strip() == "random_test":
        # 本番どおり（配当の時期を正確に）と、SとAだけ買う形を、でたらめに選んだ場合と比べる
        # （2026年10月12日に、いまの本番に合わせて作り直した）
        args.random_test, args.only = True, "base_div_scr,tier_sa_scr"
    # 日本の長い歴史（J-Quants のデータは使わないので、ここで抜ける）
    if (args.only or "").strip() == "history_test":
        return history_test()
    args.stress_test = False
    if (args.only or "").strip() == "stress_test":
        # 後出しなし（その時点の首位級だけで Tier）と、本番どおりで回す（2026年10月12日に作り直し）
        args.stress_test, args.only = True, "tier_pit_lead_scr,base_div_scr"
    # 10回目（2026年10月12日に追加）：手で引くような水平線でうねり取り（1年・2年・3年の線）と、
    # 9回目の日ごとのうねり取り（中くらい）を、全部持ち続ける形と比べる。
    # 年率は「最後に全部売って、約定のずれと税を払った後」（清算後）で判定し、評価額の年率も並べる。
    args.line_test = False
    args.liq_metric = False
    if (args.only or "").strip() == "line_test":
        args.line_test, args.liq_metric = True, True
        args.only = "tier_pit_mix_scr,line_1y_pit,line_2y_pit,line_3y_pit,swing_d_m_pit"
    # 11回目（2026年10月12日に追加）：増配の実績がある銘柄だけで、全部持ち続ける形と、1年・2年・3年線のうねり取り。
    # 比べる相手は「増配の実績だけ・全部持ち続ける」と、今の本番（首位級だけその時点で判定）。年率は清算後で判定する。
    args.safe_test = False
    if (args.only or "").strip() == "safe_test":
        args.safe_test, args.liq_metric = True, True
        args.only = "tier_pit_mix_scr,safe_hold_pit,safe_line_1y_pit,safe_line_2y_pit,safe_line_3y_pit"
    # 12回目（2026年10月12日に追加）：「増配の実績だけ・2年線」を崩しにいく検証（反証テスト）。
    #   ランダムな日の売り買いとの比較・銘柄を半分に分ける・重ならない前半と後半・線の設定を少し変える・業種別の中身。
    args.robust_test = False
    if (args.only or "").strip() == "robust_test":
        args.robust_test, args.liq_metric = True, True
        args.only = ("tier_pit_mix_scr,safe_hold_pit,safe_line_2y_pit,safe_hold_a,safe_2y_a,safe_hold_b,safe_2y_b,"
                     "safe_2y_t15,safe_2y_t25,safe_2y_i05,safe_2y_i15")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 取得が途中で止まったデータ（complete が False）は、取り直しの指定に関係なく続きから取る。
    # 取り終えたデータ（complete が True、または古い形で印が無いもの）は、そのまま使う。
    store = None
    if CACHE.exists():
        store = pd.read_pickle(CACHE)
        if store.get("complete") is False:
            log.info("前回の株価の取得が途中で止まっています。続きから取ります。")
        elif args.refetch:
            store = None              # 取り終えたデータを捨てて、最初から取り直す
        else:
            log.info("キャッシュから読み込みます（取り直すなら --refetch）")
    if store is None or store.get("complete") is False:
        key = os.environ.get("J_QUANTS_API_KEY")
        if not key:
            log.error("J_QUANTS_API_KEY が設定されていません")
            return 2
        store = fetch_all(JQ(key), args.years, args.limit,
                          scale_filter=(args.universe == "core"), store=store)
        store.setdefault("universe_mode", args.universe)
        save_store(store)
        log.info("キャッシュに保存しました: %s", CACHE)
        if store.get("complete") is False:
            print_unfinished(store, compute_only=False)
            return EXIT_UNFINISHED
        if _elapsed_min() > COMPUTE_START_LIMIT_MIN:
            print_unfinished(store, compute_only=True)
            return EXIT_UNFINISHED

    # 未指定のときだけ既定値を入れる。
    # ここで無条件に上書きすると、実運用モードで銘柄数を変えても効かなくなる。
    if args.capital <= 0:
        args.capital = 10_000_000 if args.realistic else 3_000_000
    if args.max_names <= 0:
        args.max_names = 20 if args.realistic else 15
    # キャッシュに TOPIX が無ければ、指数だけ取り直す（数秒）。
    if store.get("topix") is None or (isinstance(store.get("topix"), pd.DataFrame)
                                      and store["topix"].empty):
        key = os.environ.get("J_QUANTS_API_KEY")
        if key:
            try:
                store["topix"] = fetch_topix(JQ(key))
                pd.to_pickle(store, CACHE)
            except Exception as e:
                log.warning("TOPIX の取得に失敗: %s", e)

    # 窓をずらす検証では、データ全体を使わないと窓を作れない。
    # 「検証する年数」で先に切ってしまうと窓が1つしかできないため上書きする。
    if args.grid and args.years < args.grid_window + 2:
        log.info("総当たり検証のため、対象期間を最大まで広げます（%d年→9年）",
                 args.years)
        args.years = 9

    if args.walk > 0 and args.years < args.walk + 2:
        log.info("窓をずらす検証のため、対象期間を最大まで広げます（%d年→9年）",
                 args.years)
        args.years = 9

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

    if args.ramp > 0:
        log.info("資金を %dか月かけて入れます（初日の一括購入を避けます）", args.ramp)

    log.info("パネルを作成中…")
    panel = build_panel(store, args.years, args.lookback)

    # 選んだルールに PBR / PER / 時価総額 で選ぶものがあれば、財務の項目を足す。
    # 総当たりなどでパネルを作り直すときも、必ず同じように足すこと
    # （足し忘れると全銘柄の順位が空になり、一度も買わずに0％になる）。
    _sel_names = [x.strip() for x in (args.only or "").split(",") if x.strip()]
    _need_fund = any(VARIANTS.get(n_, {}).get("measure") in
                     ("pbr", "per", "size", "pbr_sec", "value",
                      "pbrroe70", "pbrroe50", "pbrroe30", "switch36")
                     or VARIANTS.get(n_, {}).get("screen")
                     or VARIANTS.get(n_, {}).get("tier_col")
                     for n_ in _sel_names)
    # 後出しなしで判定した Tier を使うルールがあれば、その列を足す（2026年10月12日に追加）
    _need_pit = any(VARIANTS.get(n_, {}).get("tier_col") for n_ in _sel_names)
    # 利回りが上がった理由を使うルールがあれば、その列を足す（2026年10月12日に追加）
    _need_reason = any(VARIANTS.get(n_, {}).get("skip_price_drop") is not None for n_ in _sel_names)
    # 増配の実績で Tier を作るルールがあれば、その列を足す（2026年10月12日に追加）
    _need_grow = any(VARIANTS.get(n_, {}).get("tier_col") == "tier_pit_grow"
                     or VARIANTS.get(n_, {}).get("require_col") == "grow_pit" for n_ in _sel_names)

    # レンジの上限で降りるルール（range_exit）を選んだら、「レンジ往復を検証」の欄が false でも
    # レンジの列を足す。足さないと、そのルールは一度も降りず、土台とまったく同じ成績になり、
    # 「効果なし」と見分けがつかない（2026年10月9日に追加）。
    _need_range = bool(args.range_swing) or any(
        VARIANTS.get(n_, {}).get("range_exit") and not VARIANTS[n_]["range_exit"].get("top_col")
        for n_ in _sel_names)
    # うねり（箱）の列（8回目。2026年10月12日に追加）
    _box_tags = sorted({VARIANTS[n_]["swing_entry"].get("tag", "m") for n_ in _sel_names
                        if VARIANTS.get(n_, {}).get("swing_entry")})

    def _panel_for(lb_):
        pn_ = build_panel(store, args.years, lb_)
        if _need_fund and not pn_.empty:
            pn_ = add_fund_columns(store, pn_)
        if _need_pit and not pn_.empty:
            pn_ = add_pit_tiers(pn_)
        if _need_grow and not pn_.empty:
            pn_ = add_grow_tiers(store, pn_)
        if _need_reason and not pn_.empty:
            pn_ = add_yield_reason_columns(store, pn_)
        if _box_tags and not pn_.empty:
            pn_ = add_box_columns(store, pn_, _box_tags)
        # 総当たりで分位の月数を変えてパネルを作り直すときも、レンジの列を忘れずに足す
        if _need_range and not pn_.empty:
            pn_ = add_range_columns(store, pn_, int(_d[0]), _d[1], _d[2], _d[4])
        return pn_

    # 価格で損切りするルールがあれば、日々の終値を用意する。
    # 月末の値だけで判定すると、月の途中で線を割っても月末に戻していれば売らないことになり、
    # 実際の損切り（毎日の値動きで判断する）と食い違うため。
    global SWING_DAILY
    _sw_tags = sorted({VARIANTS[n_]["swing_daily"].get("tag", "m") for n_ in _sel_names
                       if VARIANTS.get(n_, {}).get("swing_daily")})
    if _sw_tags:
        log.info("うねりを日ごとに売り買いするルールがあるので、日々の上の線・下の線を用意します…")
        SWING_DAILY = build_swing_daily(store, _sw_tags)
        for _t in _sw_tags:
            _okr = [float(np.mean(a[4])) for a in SWING_DAILY[_t].values() if len(a[4])]
            log.info("  うねり（%s） … 日々のうちうねりの中にいた割合 平均 %.0f％",
                     DAILY_BOX_SETS[_t]["label"], (float(np.mean(_okr)) * 100) if _okr else 0.0)
    global DAILY_PX
    if any(VARIANTS.get(n_, {}).get("stop_loss") for n_ in _sel_names):
        DAILY_PX = build_daily_arrays(store)
        log.info("価格で損切りするルールがあるので、日々の終値を用意しました（%d銘柄）",
                 len(DAILY_PX))

    global SWITCH_SIG
    if any(VARIANTS.get(n_, {}).get("measure") == "switch36" for n_ in _sel_names):
        log.info("市況で切り替えるルールがあるので、フレンチ教授のデータを取得します…")
        SWITCH_SIG = french_switch_signal(36, 2)
        if SWITCH_SIG is None:
            sys.exit("切り替えの合図を作れませんでした（フレンチ教授のデータが取れない）。")
        _last = SWITCH_SIG.index.max().date()
        log.info("  合図：%s 時点で %s", _last, "割安寄り" if bool(SWITCH_SIG.iloc[-1]) else "稼ぐ力寄り")
    if _need_fund:
        log.info("PBR・PER・時価総額で選ぶルールがあるので、財務の項目を足します…")
        panel = add_fund_columns(store, panel)
    if _need_pit:
        log.info("Tier を後出しなしで判定し直すルールがあるので、その月末の時価総額で業界首位級を判定します…")
        panel = add_pit_tiers(panel)
    if _need_grow:
        log.info("増配の実績を使うルールがあるので、その時点までの年間配当の推移を調べます…")
        panel = add_grow_tiers(store, panel)
    if _need_reason:
        log.info("利回りが上がった理由を使うルールがあるので、12か月の株価と配当の変化を足します…")
        panel = add_yield_reason_columns(store, panel)
    if _box_tags:
        log.info("うねりで売り買いするルールがあるので、日々の終値から上の線と下の線を出します…")
        panel = add_box_columns(store, panel, _box_tags)

    # 外部要因を使う検証では、追い風か逆風かをパネルに足す
    if args.factor_test:
        _mk = fetch_markets()
        panel = add_factor_columns(panel, _mk, args.factor_win)
        for _f, _n in [("rate", "米金利"), ("oil", "原油"), ("fx", "ドル円")]:
            _c = panel.groupby("date")[f"trend_{_f}"].first()
            _up = int((_c > 0).sum())
            _dn = int((_c < 0).sum())
            log.info("  %s … 追い風 %dか月 / 逆風 %dか月（%dか月前と比較）",
                     _n, _up, _dn, args.factor_win)

    # レンジ往復の検証では、月の途中で上限に届いたかを見る必要がある
    _d = [60.0, 0.08, 0.20, 0.02, 0.02, 0.0]
    if _need_range:
        # 「/」で複数の設定が並んでいるとき（レンジ売買だけの検証用）は、最初の設定を使う
        for _i, _x in enumerate([x.strip() for x in
                                 str(args.range_opts).split("/")[0].split(",")][:6]):
            if _x:
                _f = float(_x)
                if _i >= 1 and _f >= 1.0:      # 1以上は百分率（「1」は1％。100％の買い幅はありえない）
                    _f = _f / 100.0
                _d[_i] = _f
        log.info("レンジ判定：%d営業日 ／ 値幅 %.0f〜%.0f％ ／ 上限−%.0f％で降りる",
                 int(_d[0]), _d[1] * 100, _d[2] * 100, _d[4] * 100)
        panel = add_range_columns(store, panel, int(_d[0]), _d[1], _d[2], _d[4])
        _n = panel["box_top_px"].notna().sum()
        _b = panel["in_box"].sum()
        log.info("  レンジの中にいた時点 %d件 ／ うち上限に届いた %d件（%.1f％）",
                 _b, _n, _n / max(_b, 1) * 100)
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
            pn = _panel_for(lb)
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
                                     tax_rate=args.tax / 100.0, ramp=args.ramp))
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

    # ══════════════════════════════════════════
    # 逆風が来たら何％沈むか（ストレステスト）
    #   2026年10月12日に作り直した。以前は低PBR順など、いまは使っていないルールで見ていた。
    #
    #   この8年は円安・金利上昇・原油高が重なった、高配当株に有利な期間で、本当の下げ相場はコロナだけ。
    #   その逆が来たときの下げを、外部要因への反応の強さから見積もる。
    #   月ごとの資産の増減を、ドル円・米金利・原油・米国株の月ごとの変化で説明する式を作り、
    #   そこに「逆風のシナリオ」を入れて1年間の増減を計算する（部品は stress_fit など）。
    #   あわせて、実際に逆風だった月（各要因が悪い側から1割の月）の平均の増減も並べる（式を使わない確かめ）。
    #   比べるのは、後出しなし（その時点の首位級だけで Tier）・本番どおり・全部買って放置。
    #   判定の基準は結果を見る前に決めたもの（下の【判定】に書いてある）。
    # ══════════════════════════════════════════
    if args.stress_test:
        def _dw(x):
            return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(x))

        def _lj(x, w):
            x = str(x)
            return x + " " * max(0, w - _dw(x))

        def _rj(x, w):
            x = str(x)
            return " " * max(0, w - _dw(x)) + x

        common = dict(tier_budget=True, dividends=True, slip_bps=args.slip_bps,
                      fee_bps=args.fee_bps, tax_rate=args.tax / 100.0)
        T_, B_, M_ = "後出しなし", "本番（後出しあり）", "全部買って放置"
        curves = {}
        for nm, key in ((T_, "tier_pit_lead_scr"), (B_, "base_div_scr")):
            curves[nm] = simulate(panel, VARIANTS[key], args.capital, 20, **common)["curve"]
        bh = buy_and_hold(panel, args.tax / 100.0)
        if bh is not None:
            curves[M_] = bh["curve"]

        mk = fetch_markets()
        need = [("usdjpy", "ドル円"), ("us10y", "米10年金利"), ("brent", "原油"), ("sp500", "米国株")]
        miss = [j for k, j in need if k not in mk]
        if miss:
            sys.exit("外部要因のデータが取れませんでした：" + "、".join(miss))
        dX = stress_factor_changes(mk)
        fits = {nm: stress_fit(c, dX) for nm, c in curves.items()}
        fits = {nm: f for nm, f in fits.items() if f is not None}
        if not fits:
            sys.exit("外部要因と重なる期間が短すぎて、式を作れませんでした。")
        nms = list(fits)
        jn = {"fx": "円安＋1％", "rate": "米金利＋1pt", "oil": "原油＋1％", "spx": "米国株＋1％"}
        f0 = fits[nms[0]]
        _per = f0["ret"].index.intersection(dX.index)
        print(f"\n■ 逆風が来たら何％沈むか（{_per.min().date()} 〜 {_per.max().date()} の"
              f"{f0['n']}か月の動きから推定）\n")
        print("　 比べるもの：後出しなし（その時点の首位級だけでTier）／本番どおり／全部買って放置")
        print(f"　 元本 {args.capital:,.0f}円 ／ 税率{args.tax:.3f}％ ／ 約定のずれ 片道{args.slip_bps:g}bps"
              " ／ 資金は初日に全額\n")

        W = 22
        print("【外部要因への反応の強さ】（その要因が動いたとき、1か月の資産が何％動くか）\n")
        hdr = _lj("", W) + "".join(_rj(jn[k], 12) for k in STRESS_FACTORS) + _rj("説明できる割合", 16)
        print(hdr)
        print("-" * _dw(hdr))
        for nm in nms:
            c_ = fits[nm]["coef"]
            print(_lj(nm, W) + "".join(_rj(f"{c_[i + 1] * (1 if k == 'rate' else 0.01) * 100:+.2f}%", 12)
                                       for i, k in enumerate(STRESS_FACTORS))
                  + _rj(f"{fits[nm]['r2'] * 100:.0f}%", 16))
        print("\n  米金利と米国株のデータは月平均なので、その月と翌月の変化の平均で、月末の値動きと時期をそろえている。")

        print("\n【シナリオ別：1年間の資産の増減（外部要因の影響だけ）】\n")
        hdr = _lj("シナリオ", 16) + "".join(_rj(nm, 20) for nm in nms)
        print(hdr)
        print("-" * _dw(hdr))
        for sname, _s, _d in STRESS_SCEN:
            print(_lj(sname, 16) + "".join(_rj(f"{fits[nm]['hit'][sname] * 100:+.1f}%", 20) for nm in nms))
        print("\n  中身：")
        for sname, _s, desc in STRESS_SCEN:
            print(f"    {sname} … {desc}")

        print("\n【参考：この期間のいつもの伸びも足した場合と、実際の最悪の12か月】\n")
        print(hdr)
        print("-" * _dw(hdr))
        for sname, _s, _d in STRESS_SCEN[:2]:
            print(_lj(sname, 16) + "".join(_rj(f"{fits[nm]['hit_all'][sname] * 100:+.1f}%", 20) for nm in nms))
        print(_lj("実際の最悪の12か月", 16) + "".join(_rj(f"{fits[nm]['worst12'] * 100:+.1f}%", 20) for nm in nms))

        bad = stress_bad_months({nm: fits[nm]["ret"] for nm in nms}, dX)
        lab_bad = {"fx": "円高が大きく進んだ月", "rate": "米金利が大きく下がった月",
                   "oil": "原油が大きく下げた月", "spx": "米国株が大きく下げた月"}
        print("\n【実際に逆風だった月の平均】（各要因の変化が悪い側から1割の月。式を使わない確かめ）\n")
        hdr2 = _lj("", 26) + _rj("月数", 6) + "".join(_rj(nm, 20) for nm in nms)
        if T_ in nms and M_ in nms:
            hdr2 += _rj("後出しなし−放置", 18)
        print(hdr2)
        print("-" * _dw(hdr2))
        emp = {}
        for k, n_, thr, avg in bad:
            line = _lj(lab_bad[k], 26) + _rj(f"{n_}", 6) + "".join(_rj(f"{avg[nm] * 100:+.1f}%", 20) for nm in nms)
            if T_ in nms and M_ in nms:
                emp[k] = avg[T_] - avg[M_]
                line += _rj(f"{emp[k] * 100:+.1f}pt", 18)
            print(line)
        print("\n  1か月あたりの平均の増減。月数が少ないので、向きを見る程度に使う。")

        # 判定（結果を見る前に決めた基準）
        print("\n【判定】（基準は結果を見る前に決めたもの）\n")
        if T_ in fits and M_ in fits:
            ft, fm = fits[T_], fits[M_]
            a_t, a_m = ft["hit"]["逆風が全部来る"], fm["hit"]["逆風が全部来る"]
            l_t = ft["hit"]["リーマン級"]
            print(f"  ① 逆風が全部来たとき（外部要因の影響だけ）の1年の増減 … 後出しなし {a_t * 100:+.1f}％"
                  f" ／ 全部買って放置 {a_m * 100:+.1f}％（リーマン級では後出しなし {l_t * 100:+.1f}％）")
            if ft["r2"] < 0.30:
                print(f"  ④ 説明できる割合が {ft['r2'] * 100:.0f}％ で30％未満 → この見積もりは参考にとどめ、"
                      "外向けの説明には使わない")
            else:
                d_ = a_t - a_m
                if d_ <= -0.05:
                    v_ = "市場全体より逆風に弱い（外向けにもそう書く）"
                elif d_ >= 0.05:
                    v_ = "市場全体より下げにくい"
                else:
                    v_ = "市場全体と同じくらい下がる"
                print(f"  ② 全部買って放置との差 {d_ * 100:+.1f}pt → {v_}")
                print("       （基準：5pt以上深ければ「市場全体より逆風に弱い」、5pt以内なら「同じくらい」、"
                      "5pt以上浅ければ「下げにくい」）")
                weak = []
                for sname, k in STRESS_SINGLE.items():
                    ds = ft["hit"][sname] - fm["hit"][sname]
                    agree = ""
                    if k in emp:
                        agree = "（実際の逆風の月も同じ向き）" if (ds < 0) == (emp[k] < 0) \
                            else "（実際の逆風の月とは逆の向き。確かではない）"
                    mark = "弱点" if ds <= -0.03 else ("強い" if ds >= 0.03 else "差は小さい")
                    if ds <= -0.03:
                        weak.append(sname.replace("だけ", ""))
                    print(f"  ③ {sname}：放置との差 {ds * 100:+.1f}pt → {mark}{agree}")
                print("       （基準：放置より3pt以上深いものを「弱点」として名指しする）")
                print("  ⑤ 外向けのリスクの説明 … 「円高・米金利低下・原油安・米国株安が同時に来ると、"
                      f"1年で{a_t * 100:+.0f}％程度」" + (f"、弱点は{'・'.join(weak)}" if weak else ""))
        print("  ⑥ 本番のルールはこの結果だけでは変えない（業種の上限や現金を残す形は、同じ期間の検証で成績を下げた）。"
              "資金をいくら入れるかの判断材料にする。")

        print("\n【読み方】\n")
        print("  ・「外部要因の影響だけ」は、この期間のいつもの伸びを0と置いた下げ。こちらを主に見る。")
        print("    この期間の伸びは追い風の中で出たもので、逆風の年にそのまま出る保証はない。")
        print("  ・式はこの期間の月ごとの動きから作っている。リーマン級のように、この期間になかった大きさの")
        print("    変化では、実際の下げはもっと深くなりやすい（暴落時は、ふだん関係の薄いものまで一緒に下がる）。")
        print("  ・「説明できる割合」が低いほど、外部要因以外で動いている部分が大きく、見積もりは当てにならない。")
        print("  ・円高と米国株安のように一緒に動きやすい要因は、単独の数字がぶれやすい。①（全部来る）を主に見る。")
        return 0

    # ══════════════════════════════════════════
    # 運の幅を測る（でたらめに選んだ場合との比較）
    #   2026年10月12日に作り直した。以前は低PBR順など、いまは使っていないルールで比べていた。
    #
    #   本番と同じ仕組みのまま「選ぶ順番」だけをでたらめにしたものを何十回も回し、
    #   本番がその分布のどこにいるかを見る。情報を3段階で抜いていく。
    #     tier … Tierの順番と予算は本番どおり。同じTierの中だけ、利回りの順をやめてでたらめに選ぶ
    #     all  … 候補（8条件・利回り3％以上・減配や業績急変なし）と予算は本番どおり。順番をすべてでたらめに
    #     univ … 条件をすべて外す。プライム全体から、本番と同じくらいの銘柄数を等金額ででたらめに選び、
    #            売らずに持ち続ける（配当で現金が貯まれば、またでたらめに買い足す）
    #   期間は総当たりと同じ6通り（4年の窓5つ＋全期間）。資金投入は0か月だけ。
    #   判定の基準は結果を見る前に決めたもの（下の【判定】に書いてある）。
    # ══════════════════════════════════════════
    if args.random_test:
        def _dw(x):
            return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(x))

        def _lj(x, w):
            x = str(x)
            return x + " " * max(0, w - _dw(x))

        def _rj(x, w):
            x = str(x)
            return " " * max(0, w - _dw(x)) + x

        K = max(5, args.random_n)
        base_key, sa_key = "base_div_scr", "tier_sa_scr"
        base_cfg = VARIANTS[base_key]
        tax_ = args.tax / 100.0
        common = dict(tier_budget=True, dividends=True, slip_bps=args.slip_bps,
                      fee_bps=args.fee_bps, tax_rate=tax_, ramp=0)

        # 窓（総当たりと同じ作り方：4年の窓を1年ずつずらしたもの＋全期間）
        wl = args.grid_window
        d0_, d1_ = panel["date"].min(), panel["date"].max()
        wins = []
        st_ = d1_ - pd.DateOffset(years=wl)
        while st_ >= d0_:
            wins.append((st_, d1_ if not wins else st_ + pd.DateOffset(years=wl)))
            st_ = st_ - pd.DateOffset(years=1)
        wins = [(a, a + pd.DateOffset(years=wl)) for a, _ in wins]
        wins.reverse()
        wins.append((d0_, d1_))

        fams = [("tier", "でたらめ（Tier内）"), ("all", "でたらめ（候補の中）"),
                ("univ", "でたらめ（プライム全体）")]
        log.info("運の幅を測ります：期間%d × でたらめ%d回 × %d通り ＝ %d回（ほかに本番とSとAだけ）",
                 len(wins), K, len(fams), len(wins) * K * len(fams))

        rows_r, refs, bhs, spans, nbs = [], {}, {}, [], {}
        for a, b in wins:
            sub = panel[(panel["date"] >= a) & (panel["date"] <= b)]
            if sub["date"].nunique() < 24:
                continue
            span = f"{a.date()}〜{b.date()}"
            spans.append(span)
            bh_ = buy_and_hold(sub, tax_)
            bhs[span] = bh_["年率"] if bh_ else float("nan")
            for key in (base_key, sa_key):
                r_ = simulate(sub, VARIANTS[key], args.capital, 20, **common)
                m_ = metrics(r_)
                refs[(span, key)] = {"年率": m_["年率"], "最大下落": m_["最大下落"],
                                     "銘柄数": m_["平均保有銘柄"],
                                     "緊急売り": r_.get("diag", {}).get("cut_exits", 0)}
            # プライム全体から選ぶときの銘柄数は、本番の平均の銘柄数にそろえる
            nb = max(5, int(round(refs[(span, base_key)]["銘柄数"])))
            nbs[span] = nb
            univ_cfg = {
                "label": "でたらめ（プライム全体）",
                "entry": [0], "exit": [], "priority": "tier", "cross": True, "screen": False,
                "budget_weighted": True, "target_names": nb,
                "tier_weight": {"S": 1.0, "A": 1.0, "B": 1.0},
                "min_yield_fixed": 0.0, "hold_tiers": ["S", "A", "B"],
                "exit_on_cut": False, "exit_on_op": False, "skip_emg_buy": False,
                "div_split": True, "shuffle": "all",
            }
            for fam, flbl in fams:
                for k in range(K):
                    if fam == "univ":
                        cfg = dict(univ_cfg, seed=1000 + k)
                    else:
                        cfg = dict(base_cfg, shuffle=fam, seed=1000 + k)
                    m_ = metrics(simulate(sub, cfg, args.capital, 20, **common))
                    if not m_:
                        continue
                    rows_r.append({"期間": span, "種類": flbl, "回": k + 1, "年率": m_["年率"],
                                   "最大下落": m_["最大下落"], "銘柄数": m_["平均保有銘柄"]})
                _v = [x["年率"] for x in rows_r if x["期間"] == span and x["種類"] == flbl]
                log.info("  %s ／ %s … 中央 %.1f％（本番 %.1f％）", span, flbl,
                         float(np.median(_v)) if _v else float("nan"),
                         refs[(span, base_key)]["年率"])

        if not rows_r:
            print("結果がありません。")
            return 1
        dfr = pd.DataFrame(rows_r)

        def _vals(span, flbl, col="年率"):
            return dfr[(dfr["期間"] == span) & (dfr["種類"] == flbl)][col].to_numpy()

        def _beat(span, flbl, val):
            v = _vals(span, flbl)
            return float((v < val).mean() * 100) if len(v) else float("nan")

        n_sp = len(spans)
        W = 26          # 期間の欄の幅
        print("\n■ 運の幅を測る（でたらめに選んだ場合との比較）\n")
        print(f"　 期間 {n_sp}通り ／ 資金投入0か月 ／ でたらめ 各{K}回")
        print(f"　 元本 {args.capital:,.0f}円 ／ 税率{args.tax:.3f}％ ／ 約定のずれ 片道{args.slip_bps:g}bps\n")
        print("【比べる相手】\n")
        print("  でたらめ（Tier内）       … Tierの順番と予算は本番どおり。同じTierの中だけ、利回りの順をやめて選ぶ")
        print("  でたらめ（候補の中）     … 候補（8条件・利回り3％以上・減配や業績急変なし）と予算は本番どおり。")
        print("                             Tierの順番も利回りの順もやめて選ぶ")
        print("  でたらめ（プライム全体） … 条件なし。プライム全体から、本番と同じくらいの銘柄数を等金額で選び、")
        print("                             売らずに持ち続ける")
        print()

        # 本番と、SとAだけ
        print("【本番とSとAだけ（資金投入0か月）】\n")
        hdr = (_lj("期間", W) + _rj("本番", 8) + _rj("SとAだけ", 10) + _rj("差", 8)
               + _rj("銘柄数（本番）", 16) + _rj("（SとA）", 10)
               + _rj("減配などの売り（本番）", 24) + _rj("（SとA）", 10))
        print(hdr)
        print("-" * _dw(hdr))
        acc = {"b": [], "s": [], "nb": [], "ns": [], "cb": [], "cs": []}
        for sp in spans:
            rb, rs = refs[(sp, base_key)], refs[(sp, sa_key)]
            print(_lj(sp, W) + _rj(f"{rb['年率']:.1f}%", 8) + _rj(f"{rs['年率']:.1f}%", 10)
                  + _rj(f"{rs['年率'] - rb['年率']:+.1f}pt", 8)
                  + _rj(f"{rb['銘柄数']:.1f}", 16) + _rj(f"{rs['銘柄数']:.1f}", 10)
                  + _rj(f"{rb['緊急売り']}回", 24) + _rj(f"{rs['緊急売り']}回", 10))
            acc["b"].append(rb["年率"]); acc["s"].append(rs["年率"])
            acc["nb"].append(rb["銘柄数"]); acc["ns"].append(rs["銘柄数"])
            acc["cb"].append(rb["緊急売り"]); acc["cs"].append(rs["緊急売り"])
        sa_diff = float(np.mean(acc["s"]) - np.mean(acc["b"]))
        print(_lj("平均", W) + _rj(f"{np.mean(acc['b']):.1f}%", 8) + _rj(f"{np.mean(acc['s']):.1f}%", 10)
              + _rj(f"{sa_diff:+.1f}pt", 8)
              + _rj(f"{np.mean(acc['nb']):.1f}", 16) + _rj(f"{np.mean(acc['ns']):.1f}", 10)
              + _rj(f"{np.mean(acc['cb']):.1f}回", 24) + _rj(f"{np.mean(acc['cs']):.1f}回", 10))
        print()
        _rb = [c / max(n, 1e-9) for c, n in zip(acc["cb"], acc["nb"])]
        _rs = [c / max(n, 1e-9) for c, n in zip(acc["cs"], acc["ns"])]
        print(f"  1銘柄あたりの減配などの売り … 本番 {np.mean(_rb):.2f}回 ／ SとAだけ {np.mean(_rs):.2f}回"
              "（銘柄数の違いをならした数字）")
        print()

        # でたらめの分布
        stats = {}
        for fam, flbl in fams:
            print(f"【{flbl}】\n")
            hdr = (_lj("期間", W) + _rj("本番", 8) + _rj("下位5％", 10) + _rj("中央", 8)
                   + _rj("上位5％", 10) + _rj("本番が上回った割合", 20))
            if fam == "univ":
                hdr += _rj("全部買って放置", 16)
            print(hdr)
            print("-" * _dw(hdr))
            beats, meds, p5s, p95s, above = [], [], [], [], 0
            for sp in spans:
                v = _vals(sp, flbl)
                if not len(v):
                    continue
                bv = refs[(sp, base_key)]["年率"]
                p5, md, p95 = (float(np.percentile(v, 5)), float(np.median(v)),
                               float(np.percentile(v, 95)))
                bt = _beat(sp, flbl, bv)
                beats.append(bt); meds.append(md); p5s.append(p5); p95s.append(p95)
                above += int(bv > md)
                line = (_lj(sp, W) + _rj(f"{bv:.1f}%", 8) + _rj(f"{p5:.1f}%", 10)
                        + _rj(f"{md:.1f}%", 8) + _rj(f"{p95:.1f}%", 10) + _rj(f"{bt:.0f}％", 20))
                if fam == "univ":
                    line += _rj(f"{bhs.get(sp, float('nan')):.1f}%", 16)
                print(line)
            line = (_lj("平均", W) + _rj(f"{np.mean(acc['b']):.1f}%", 8)
                    + _rj(f"{np.mean(p5s):.1f}%", 10) + _rj(f"{np.mean(meds):.1f}%", 8)
                    + _rj(f"{np.mean(p95s):.1f}%", 10) + _rj(f"{np.mean(beats):.0f}％", 20))
            if fam == "univ":
                line += _rj(f"{np.nanmean([bhs.get(sp, np.nan) for sp in spans]):.1f}%", 16)
            print(line)
            stats[fam] = {"beats": beats, "meds": meds, "p5": p5s, "p95": p95s, "above": above}
            if fam == "univ":
                _nn = [float(np.mean(_vals(sp, flbl, "銘柄数"))) for sp in spans if len(_vals(sp, flbl))]
                print(f"\n  プライム全体のでたらめの平均の銘柄数 … {np.mean(_nn):.1f}"
                      f"（本番 {np.mean(acc['nb']):.1f}）")
            print()

        # 最大下落
        print("【最大下落】（でたらめは中央）\n")
        hdr = (_lj("期間", W) + _rj("本番", 8) + _rj("SとAだけ", 10) + _rj("Tier内", 10)
               + _rj("候補の中", 10) + _rj("プライム全体", 14))
        print(hdr)
        print("-" * _dw(hdr))
        dd_acc = {k: [] for k in ("b", "s", "tier", "all", "univ")}
        for sp in spans:
            vals_ = [refs[(sp, base_key)]["最大下落"], refs[(sp, sa_key)]["最大下落"]]
            vals_ += [float(np.median(_vals(sp, fl, "最大下落"))) for _, fl in fams]
            for k_, x in zip(("b", "s", "tier", "all", "univ"), vals_):
                dd_acc[k_].append(x)
            print(_lj(sp, W) + _rj(f"{vals_[0]:.1f}%", 8) + _rj(f"{vals_[1]:.1f}%", 10)
                  + _rj(f"{vals_[2]:.1f}%", 10) + _rj(f"{vals_[3]:.1f}%", 10) + _rj(f"{vals_[4]:.1f}%", 14))
        print(_lj("平均", W) + _rj(f"{np.mean(dd_acc['b']):.1f}%", 8) + _rj(f"{np.mean(dd_acc['s']):.1f}%", 10)
              + _rj(f"{np.mean(dd_acc['tier']):.1f}%", 10) + _rj(f"{np.mean(dd_acc['all']):.1f}%", 10)
              + _rj(f"{np.mean(dd_acc['univ']):.1f}%", 14))
        print()

        # 判定（結果を見る前に決めた基準）
        print("【判定】（基準は結果を見る前に決めたもの）\n")
        need = max(1, n_sp - 1)          # 6期間なら5期間
        su = stats["univ"]
        n95 = sum(1 for x in su["beats"] if x >= 95.0)
        verdict = "運だけでは説明できない" if n95 >= need else "運の可能性を残す"
        print(f"  ① ルール全体の力 … 本番が「プライム全体」のでたらめの95％以上を上回った期間 {n95}/{n_sp}"
              f" → {verdict}")
        print(f"       （基準：{need}期間以上なら「運だけでは説明できない」）")
        n_sc = sum(1 for x, y in zip(stats["all"]["meds"], su["meds"]) if x > y)
        d_sc = float(np.mean(stats["all"]["meds"]) - np.mean(su["meds"]))
        verdict = "候補を絞るだけで上回っていた" if n_sc >= need else "絞るだけでは上回れなかった"
        print(f"  ② 候補を絞る力 … 「候補の中」のでたらめ（中央）が「プライム全体」（中央）を上回った期間"
              f" {n_sc}/{n_sp}、差の平均 {d_sc:+.1f}pt → {verdict}")
        print(f"       （基準：{need}期間以上なら「候補を絞るだけで上回っていた」）")
        for no, fam, what in (("③", "all", "順番の力（Tierの順番と利回りの順）"),
                              ("④", "tier", "利回りの順の力（同じTierの中）")):
            s_ = stats[fam]
            mb = float(np.mean(s_["beats"]))
            if mb >= 80 and s_["above"] >= need:
                verdict = "価値がある"
            elif 35 <= mb <= 65:
                verdict = "でたらめと変わらない（飾り）"
            else:
                verdict = "どちらとも言えない"
            print(f"  {no} {what} … 本番が上回った割合の平均 {mb:.0f}％、"
                  f"中央を上回った期間 {s_['above']}/{n_sp} → {verdict}")
        print("       （基準：平均80％以上かつ中央を上回った期間が"
              f"{need}以上なら「価値がある」、平均35〜65％なら「飾り」）")
        half = float(np.mean([b_ - a_ for a_, b_ in zip(stats["tier"]["p5"], stats["tier"]["p95"])])) / 2
        print(f"  ⑤ 運の幅 … 同じ候補・同じTierの中で別の銘柄を選ぶだけで、年率は ±{half:.1f}pt 振れる"
              "（Tier内のでたらめの下位5％〜上位5％の半分）")
        verdict = ("運の幅に収まる。運と区別できない" if abs(sa_diff) < half
                   else "運の幅を超える")
        print(f"       SとAだけ買う：本番との差 {sa_diff:+.1f}pt → {verdict}")
        print()
        print("【読み方】\n")
        print("  ・本番が上回った割合 … でたらめの各回のうち、本番より成績が悪かった回の割合。")
        print("  ・「プライム全体」に勝つのは、高配当で選ぶ考え方がこの期間に効いたということ。")
        print("    この期間は高配当・割安株に追い風が吹いた時期を含み、この先も同じだけ効く保証ではない。")
        print("  ・データに潰れた会社・上場廃止の会社が入っていないのは、でたらめの側も同じ。")
        print("    この比較そのものは公平だが、どちらの成績も実際より良く出ている。")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        dfr.to_csv(OUTDIR / "random_test.csv", index=False, encoding="utf-8-sig")
        return 0

    # ══════════════════════════════════════════
    # 組み合わせ vs 使い分け
    # ══════════════════════════════════════════
    if args.blend:
        m = _mats(panel)
        dates = list(m["px"].index)
        dates = [d for d in dates if d >= dates[0] + pd.DateOffset(months=12)]
        if len(dates) < 36:
            sys.exit("期間が短すぎます。3年以上のデータが必要です。")
        half = len(dates) // 2
        tr, te = dates[:half], dates[half:]
        cap, tax, slip, n = args.capital, args.tax / 100.0, args.slip_bps / 10000.0, 15

        log.info("組み合わせ vs 使い分け：%d通り", len(PORTFOLIOS))
        rows = []
        for name, spec, desc in PORTFOLIOS:
            try:
                full = run_portfolio(m, spec, dates, cap, n, tax, slip)
                a = run_portfolio(m, spec, tr, cap, n, tax, slip)
                b = run_portfolio(m, spec, te, cap, n, tax, slip)
            except Exception as e:
                log.warning("  %s は計算できませんでした（%s）", name, e)
                continue
            ys = full["年ごと"]
            yv = [v for y, v in ys.items() if y >= dates[0].year + 1]  # 最初の端数の年は除く
            rows.append({"名前": name, "説明": desc, "種類":
                         "使い分け" if "switch" in spec else
                         ("1本だけ" if len(spec["mix"]) == 1 else "同時に持つ"),
                         "全期間": full["年率"] * 100, "前半": a["年率"] * 100,
                         "後半": b["年率"] * 100, "最大下落": full["最大下落"] * 100,
                         "最悪の年": (min(yv) * 100) if yv else float("nan"),
                         "マイナスの年": sum(1 for v in yv if v < 0),
                         "年数": len(yv), "売買": full["売買"], "年ごと": ys})
            log.info("  %-30s 全期間 %5.1f%%  最悪の年 %+5.1f%%",
                     name[:30], rows[-1]["全期間"], rows[-1]["最悪の年"])
        df = pd.DataFrame(rows)

        print(f"\n■ 組み合わせ vs 使い分け（{dates[0].date()} 〜 {dates[-1].date()}"
              f" ／ 元本 {cap:,.0f}円・税{args.tax}%・ずれ{args.slip_bps}bps）\n")
        print("  「どんな状況でも一定」を見るため、平均ではなく悪いときの数字を並べています。\n")
        print(f"{'':<34}{'種類':<8}{'全期間':>7}{'前半':>7}{'後半':>7}"
              f"{'最悪の年':>9}{'赤字の年':>8}{'最大下落':>9}{'売買':>7}")
        print("-" * 100)
        for _, r in df.iterrows():
            print(f"{r['名前'][:32]:<34}{r['種類']:<8}{r['全期間']:>6.1f}%{r['前半']:>6.1f}%"
                  f"{r['後半']:>6.1f}%{r['最悪の年']:>+8.1f}%"
                  f"{int(r['マイナスの年']):>5}/{int(r['年数'])}{r['最大下落']:>8.1f}%"
                  f"{int(r['売買']):>7}")

        # 年ごと
        years = sorted({y for ys in df["年ごと"] for y in ys})
        years = [y for y in years if y >= dates[0].year + 1]
        print("\n■ 年ごとの成績\n")
        print(f"{'':<34}" + "".join(f"{y:>8}" for y in years))
        print("-" * (34 + 8 * len(years)))
        for _, r in df.iterrows():
            print(f"{r['名前'][:32]:<34}" + "".join(
                f"{r['年ごと'].get(y, float('nan')) * 100:>+7.1f}%" for y in years))

        # 判定（基準は先に決めておく）
        base = df[df["名前"] == "高配当だけ"].iloc[0]
        print("\n【判定】基準＝高配当だけ。"
              f"年率 {base['全期間']:.1f}% ／ 最悪の年 {base['最悪の年']:+.1f}% ／ "
              f"最大下落 {base['最大下落']:.1f}%\n")
        print("  採用の条件（事前に決めたもの）：")
        print("    最悪の年 または 最大下落 が 3pt 以上よくなり、かつ 年率の低下が 2pt 以内\n")
        for _, r in df.iterrows():
            if r["名前"] == "高配当だけ":
                continue
            dy = r["全期間"] - base["全期間"]
            dw = r["最悪の年"] - base["最悪の年"]
            dd = base["最大下落"] - r["最大下落"]
            ok = (dw >= 3 or dd >= 3) and dy >= -2
            tag = "◎ 条件を満たす" if ok else ("△ 悪いときは改善、年率は代償が大きい"
                                         if (dw >= 3 or dd >= 3) else "× 改善なし")
            print(f"  {tag:<22} {r['名前'][:32]:<34} 年率{dy:+.1f}pt ／ "
                  f"最悪の年{dw:+.1f}pt ／ 最大下落{dd:+.1f}pt")

        print("\n【読み方】")
        print("  ・「使い分け」は、状況が2か月続いてから切り替える丁寧な形にしています。")
        print("  ・「同時に持つ」は、別々の口座で持つ形。互いに触りません。")
        print("    「年1回戻す」だけ、増えた側を売って減った側を買い足します（税金とずれを計上）。")
        print("  ・7年の間に本当の下げ相場は1回（コロナ）しかありません。")
        print("    「どんな状況でも」は、このデータでは確かめきれません。")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["年ごと"]).to_csv(OUTDIR / "blend.csv", index=False,
                                          encoding="utf-8-sig")
        print("\n書き出しました: data/blend.csv")
        return 0

    # ══════════════════════════════════════════
    # 手法の探索
    # ══════════════════════════════════════════
    if args.explore:
        log.info("手法の探索：財務の項目を足しています…")
        panel = add_fund_columns(store, panel)
        log.info("手法の探索：行列を作成中…")
        try:
            m = _mats(panel)
        except Exception as e:
            log.exception("行列を作れませんでした")
            sys.exit(f"データの形が想定と違います：{e}")
        dates = list(m["px"].index)
        log.info("  月数 %d ／ 銘柄 %d", len(dates), m["px"].shape[1])
        # 増配率が全く無いと、その手法だけ動かないので知らせる
        if m["dps_growth"].notna().sum().sum() == 0:
            log.warning("増配率のデータがありません。その手法は0%%になります。")
        # 最初の12か月は、過去12か月の情報が揃わないので使わない
        dates = [d for d in dates if d >= dates[0] + pd.DateOffset(months=12)]
        if len(dates) < 36:
            sys.exit("期間が短すぎます。3年以上のデータが必要です。")
        half = len(dates) // 2
        tr, te = dates[:half], dates[half:]
        cap = args.capital
        tax = args.tax / 100.0
        slip = args.slip_bps / 10000.0
        n = args.explore_names
        my = args.min_yield or 0.0

        log.info("手法の探索：%d通り × 3期間（前半 %s〜%s ／ 後半 %s〜%s）",
                 len(EXPLORE) + len(EXPLORE_OTHER) + (len(EXPLORE_FUND) if "per" in m else 0),
                 tr[0].date(), tr[-1].date(),
                 te[0].date(), te[-1].date())

        rows = []
        _groups = [("高配当に関わる手法", EXPLORE),
                   ("高配当と関係のない手法", EXPLORE_OTHER),
                   ("財務を使う手法", EXPLORE_FUND if "per" in m else [])]
        for _g, _lst in _groups:
            for name, cfg, desc in _lst:
                r = {"手法": name, "説明": desc, "分類": _g}
                try:
                    for lbl, ds in (("前半", tr), ("後半", te), ("全期間", dates)):
                        out = run_strategy(m, cfg, ds, cap, n, my, tax, slip)
                        r[lbl] = out["年率"] * 100
                        if lbl == "全期間":
                            r["最大下落"] = out["最大下落"] * 100
                            r["売買"] = out["売買"]
                            r["税金"] = out["税金"]
                except Exception as e:
                    log.warning("  %s は計算できませんでした（%s）。飛ばします。", name, e)
                    continue
                rows.append(r)
                log.info("  %-28s 前半 %5.1f%% ／ 後半 %5.1f%%",
                         name[:28], r["前半"], r["後半"])
        if len(rows) < 2:
            sys.exit("計算できた手法が足りません。")

        df = pd.DataFrame(rows)
        df["前半順位"] = df["前半"].rank(ascending=False).astype(int)
        df["後半順位"] = df["後半"].rank(ascending=False).astype(int)

        print(f"\n■ 手法の探索（{dates[0].date()} 〜 {dates[-1].date()} ／ "
              f"{n}銘柄・税{args.tax}%・ずれ{args.slip_bps}bps）\n")
        print("  前半のデータだけで順位をつけ、後半で答え合わせをしています。")
        print("  後半は、順位をつけるときに一切見ていない期間です。\n")

        d2 = df.sort_values("前半", ascending=False)
        print(f"{'手法':<30}{'前半':>8}{'後半':>8}{'全期間':>9}"
              f"{'前半順位':>9}{'後半順位':>9}{'最大下落':>9}{'売買':>7}")
        print("-" * 92)
        for _, r in d2.iterrows():
            print(f"{r['手法'][:28]:<30}{r['前半']:>7.1f}%{r['後半']:>7.1f}%"
                  f"{r['全期間']:>8.1f}%{r['前半順位']:>9}{r['後半順位']:>9}"
                  f"{r['最大下落']:>8.1f}%{r['売買']:>7.0f}")

        # 前半の順位が、後半でどれだけ当たっているか
        # 順位どうしの相関（scipy を使わずに計算する）
        corr = df["前半"].rank().corr(df["後半"].rank())
        top = d2.iloc[0]
        print(f"\n【答え合わせ】\n")
        print(f"  前半で1位だった手法 … {top['手法']}")
        print(f"    後半では {int(top['後半順位'])}位 / {len(df)}通り"
              f"（年率 {top['後半']:.1f}%）")
        best_te = df.sort_values("後半", ascending=False).iloc[0]
        print(f"  後半で1位だった手法 … {best_te['手法']}"
              f"（前半は {int(best_te['前半順位'])}位）")
        print(f"\n  前半の順位と後半の順位の一致度 … {corr:+.2f}")
        if corr > 0.5:
            print("    → かなり一致しています。前半で良かった手法は、"
                  "後半でも良い傾向があります。")
        elif corr > 0.2:
            print("    → ゆるやかに一致しています。多少は参考になります。")
        elif corr > -0.2:
            print("    → ほとんど関係がありません。"
                  "**前半で良かった手法を選んでも、後半では役に立ちません。**")
        else:
            print("    → 逆の関係です。前半で良かった手法ほど、後半で悪くなっています。")

        for _g in df["分類"].unique():
            _x = df[df["分類"] == _g]
            if len(_x) >= 4:
                _c = _x["前半"].rank().corr(_x["後半"].rank())
                print(f"  {_g}だけで見た一致度 … {_c:+.2f}（{len(_x)}通り）")

        print("\n【手法の説明】\n")
        for _, r in d2.iterrows():
            print(f"  {r['手法']}")
            print(f"    {r['説明']}")

        print("\n【読み方】\n")
        print("  ・ここでの比較は、手法どうしの優劣を見るための簡易エンジンです。")
        print("    Tier別の予算などは入っていないので、本番ルールの数字とは一致しません。")
        print("  ・「全期間」でいちばん良かった手法を選ぶのは危険です。")
        print(f"    {len(df)}通りも試せば、偶然いちばん良いものが必ず出ます。")
        print("  ・見るべきは一致度です。これが低ければ、"
              "どの手法を選んでも将来の役には立ちません。")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTDIR / "explore.csv", index=False, encoding="utf-8-sig")
        print("\n書き出しました: data/explore.csv")
        return 0

    # ══════════════════════════════════════════
    # 減配・業績急変のあと、株価はどうなったか
    #   「握っておけば戻る」という感覚を、事実として確かめる。
    #   ただし上場廃止になった会社はデータに存在しないため、
    #   ここに出る数字は「生き残った会社だけ」のものである。
    # ══════════════════════════════════════════
    if args.after_event:
        # ── 減配・業績急変のあと、株価はどうなったか（2026年10月11日に作り直し）──
        # 以前の集計は、減配の印が立っている月を毎月1件と数えていたため、
        # 1回の減配が最大12件に数えられていた。また市場全体の動きと比べていなかった。
        #   ・出来事は本番の緊急撤退と同じ判定（実績の年間配当−10％以上／営業利益−20％以上）
        #   ・印が立った最初の月だけを1件と数える
        #   ・前の月に「保有候補」（8条件を通り、利回りが足切り以上）だった銘柄を主に見る
        #   ・同じ期間の市場（パネル全銘柄の等金額平均）と比べた差を出す
        if "op_cut20" not in panel.columns or "screen_pass" not in panel.columns \
                or "div_chg" not in panel.columns:
            log.info("財務の項目を足します…")
            panel = add_fund_columns(store, panel)
        my_ = args.min_yield if args.min_yield > 0 else 3.0
        pv = panel.pivot_table(index="date", columns="code", values="price").sort_index()
        dts = list(pv.index)
        pos_of = {d: i for i, d in enumerate(dts)}
        arr = pv.to_numpy(dtype=float)
        col_of = {c: j for j, c in enumerate(pv.columns)}
        hz = (6, 12, 24, 36)
        mkt = {}
        for i0 in range(len(dts)):
            for m in hz:
                if i0 + m < len(dts):
                    a0, a1 = arr[i0], arr[i0 + m]
                    ok_ = np.isfinite(a0) & np.isfinite(a1) & (a0 > 0)
                    mkt[(i0, m)] = float(np.mean(a1[ok_] / a0[ok_] - 1) * 100) if ok_.any() else np.nan

        ev = []
        for code, g in panel.groupby("code"):
            g = g.sort_values("date").reset_index(drop=True)
            cut = g["div_cut10"].fillna(0).astype(float) > 0.5
            opd = g["op_cut20"].fillna(0).astype(float) > 0.5
            # 予想の減配：その月末の予想の年間配当が、直近の実績より10％以上低い
            fcc = pd.to_numeric(g["fc_chg"], errors="coerce").fillna(0) <= -0.10
            for kind, flag in (("減配", cut), ("業績急変", opd), ("予想の減配", fcc)):
                start = flag & ~flag.shift(1, fill_value=False)
                for i in np.flatnonzero(start.to_numpy()):
                    if i == 0:
                        continue          # 前の月が無いので、保有候補だったか分からない
                    r = g.loc[i]
                    p0 = r["price"]
                    if not p0 or p0 <= 0 or r["date"] not in pos_of:
                        continue
                    prev = g.loc[i - 1]
                    elig = bool(prev.get("screen_pass")) and (prev.get("yield") or 0) >= my_
                    i0, j = pos_of[r["date"]], col_of[code]
                    rec = {"code": code, "name": r.get("name", code), "date": r["date"],
                           "kind": kind, "px": p0, "候補": elig}
                    # 落ち方（2026年10月11日に追加）
                    #   深さ … 減配なら実績の配当の変化率、業績急変なら営業利益の変化率、
                    #          予想の減配なら予想と実績の差
                    #   重なり … もう一方の印もその月に立っていたか
                    #   予想で先行 … 実績の減配の印が立つ前の15か月で、予想の減配が先に出ていたら、何か月前か
                    if kind == "減配":
                        _d = r.get("div_chg"); rec["重なり"] = bool(opd.iloc[i])
                        _lo = max(0, i - 15)
                        _seg = fcc.iloc[_lo:i].to_numpy()
                        rec["予想で先行"] = float(i - (_lo + int(np.flatnonzero(_seg)[0]))) \
                            if _seg.any() else np.nan
                    elif kind == "業績急変":
                        _d = r.get("op_chg"); rec["重なり"] = bool(cut.iloc[i])
                        rec["予想で先行"] = np.nan
                    else:
                        _d = r.get("fc_chg"); rec["重なり"] = bool(opd.iloc[i])
                        rec["予想で先行"] = np.nan
                    rec["深さ"] = float(_d) if (_d is not None and not pd.isna(_d)) else np.nan
                    for m in hz:
                        if i0 + m < len(dts) and np.isfinite(arr[i0 + m, j]):
                            ch = (arr[i0 + m, j] / p0 - 1) * 100
                            rec[f"m{m}"] = ch
                            rec[f"x{m}"] = ch - mkt.get((i0, m), np.nan)
                        else:
                            rec[f"m{m}"] = rec[f"x{m}"] = np.nan
                    seg = arr[i0:min(i0 + 13, len(dts)), j]
                    seg = seg[np.isfinite(seg)]
                    rec["dd"] = (seg.min() / p0 - 1) * 100 if len(seg) else np.nan
                    ev.append(rec)
        if not ev:
            print("該当する出来事が見つかりませんでした。")
            return 1
        ed = pd.DataFrame(ev)
        print(f"\n■ 減配・業績急変のあと、株価はどうなったか"
              f"（{panel['date'].min().date()} 〜 {panel['date'].max().date()}）\n")
        print("　 減配 … 実績の年間配当が前の通期より10％以上減った（本番の緊急撤退と同じ判定）")
        print("　 業績急変 … 営業利益が前の通期より20％以上減った・赤字転落を含む（同じ）")
        print("　 予想の減配 … 予想の年間配当が直近の実績より10％以上低くなった（本番は見ていない判定）")
        print("　 印が立った最初の月を1件と数えます。市場との差は、同じ期間のパネル全銘柄の")
        print("　 等金額平均の値動きを引いたもの。プラスなら市場より上、マイナスなら下。\n")
        for grp, sel in ((f"前の月に保有候補だった銘柄（8条件・利回り{my_:g}％以上）", ed["候補"]),
                         ("全銘柄（参考）", pd.Series(True, index=ed.index))):
            sub_e = ed[sel]
            print(f"━━ {grp} ━━\n")
            for kind in ("減配", "業績急変", "予想の減配"):
                g = sub_e[sub_e["kind"] == kind]
                print(f"【{kind}】{len(g)}件（{g['code'].nunique()}社）\n")
                if len(g) < 5:
                    print("  件数が少ないので集計しません。\n")
                    continue
                print(f"{'その後':<8}{'平均':>9}{'中央値':>9}{'プラスの割合':>12}"
                      f"{'市場との差（平均）':>16}{'市場に勝った割合':>14}{'件数':>6}")
                print("-" * 80)
                for m, lbl in ((6, "6か月後"), (12, "1年後"), (24, "2年後"), (36, "3年後")):
                    v, x = g[f"m{m}"].dropna(), g[f"x{m}"].dropna()
                    if len(v) < 5:
                        continue
                    print(f"{lbl:<8}{v.mean():>+8.1f}%{v.median():>+8.1f}%"
                          f"{(v > 0).mean() * 100:>11.0f}%{x.mean():>+15.1f}pt"
                          f"{(x > 0).mean() * 100:>13.0f}%{len(v):>6}")
                dd = g["dd"].dropna()
                if len(dd):
                    print(f"\n  その後1年のうちの最大の下げ … 平均 {dd.mean():.1f}% ／ "
                          f"中央値 {dd.median():.1f}% ／ 最悪 {dd.min():.1f}%")
                print()

        print("【読み方】\n")
        print("  ・市場との差がマイナスなら、その銘柄を持ち続けるより、売って別の銘柄に")
        print("    乗り換えた方が良かったことになります（税金と手数料は含みません）。")
        print("  ・値動きだけで、配当は含みません。減配した銘柄は配当も減っているので、")
        print("    配当まで含めると、市場との差はここより少し悪くなります。")
        print("  ・ここに出ているのは「いま上場している会社」だけです。")
        print("    業績が崩れて上場廃止になった会社は含まれていないので、")
        print("    実際の数字は、ここに出るものより悪いはずです。")

        # ── 落ち方で分ける（2026年10月11日に追加）──
        # 「深い減配ほど戻らない」「減配と減益が重なると戻らない」が本当なら、
        # 売る・売らないを落ち方で変える意味がある。区分ごとに、12か月後の市場との差で見る。
        def _bucket(kind, d):
            if pd.isna(d):
                return "不明"
            if kind == "業績急変":
                return ("赤字転落" if d <= -0.999 else
                        "−50〜100％（黒字）" if d <= -0.50 else "−20〜50％")
            return ("−50％以上・無配" if d <= -0.50 else
                    "−25〜50％" if d <= -0.25 else "−10〜25％")

        def _verdict(x12, n):
            if n < 20:
                return "件数不足"
            med = float(np.median(x12))
            return ("売る方がよかった" if med <= -5.0 else
                    "持つ方がよかった" if med >= 0.0 else "差が小さい")

        def _table(title, rows_):
            # rows_: [(区分名, DataFrame)]
            print(f"【{title}】\n")
            print(f"{'区分':<20}{'件数':>5}{'1年後：市場との差 中央値':>22}{'平均':>8}{'市場に勝った':>10}"
                  f"{'2年後 中央値':>12}{'1年の最大下げ':>12}  判定")
            print("-" * 104)
            for lab_, g_ in rows_:
                x12 = g_["x12"].dropna()
                x24 = g_["x24"].dropna()
                dd_ = g_["dd"].dropna()
                if len(x12) == 0:
                    print(f"{lab_:<20}{len(g_):>5}{'—':>22}{'—':>8}{'—':>10}{'—':>12}{'—':>12}  件数不足")
                    continue
                print(f"{lab_:<20}{len(g_):>5}{x12.median():>+21.1f}pt{x12.mean():>+7.1f}pt"
                      f"{(x12 > 0).mean() * 100:>9.0f}%"
                      f"{(x24.median() if len(x24) else float('nan')):>+11.1f}pt"
                      f"{(dd_.median() if len(dd_) else float('nan')):>11.1f}%"
                      f"  {_verdict(x12, len(x12))}")
            print()

        ec = ed[ed["候補"]]
        print(f"\n■ 落ち方で分ける（前の月に保有候補だった銘柄・{len(ec)}件）\n")
        print("　 判定は「1年後の市場との差の中央値」で決めます（結果を見る前に決めた基準）。")
        print("　 　−5pt以下 … 売る方がよかった ／ 0pt以上 … 持つ方がよかった ／ その間 … 差が小さい")
        print("　 　20件に満たない区分は判定しません（運と区別がつかないため）。\n")
        for kind, ttl in (("減配", "減配の深さ（実績の年間配当の減り方）"),
                          ("業績急変", "業績の落ち方（営業利益の減り方）"),
                          ("予想の減配", "予想の減配の深さ（予想が実績をどれだけ下回ったか）")):
            g = ec[ec["kind"] == kind].copy()
            if g.empty:
                continue
            g["区分"] = [_bucket(kind, d) for d in g["深さ"]]
            order = (["−20〜50％", "−50〜100％（黒字）", "赤字転落", "不明"] if kind == "業績急変"
                     else ["−10〜25％", "−25〜50％", "−50％以上・無配", "不明"])
            _table(ttl, [(b, g[g["区分"] == b]) for b in order if (g["区分"] == b).any()]
                   + [("（全体）", g)])
        for kind, other in (("減配", "業績急変"), ("業績急変", "減配")):
            g = ec[ec["kind"] == kind]
            if g.empty:
                continue
            _table(f"{kind}と{other}の重なり",
                   [(f"{other}も同時", g[g["重なり"]]), (f"{kind}だけ", g[~g["重なり"]])])
        # 予想の減配が、実績の減配をどれだけ先取りしていたか
        g = ec[ec["kind"] == "減配"]
        if len(g):
            lead = g["予想で先行"].dropna()
            print("【実績の減配は、予想の段階で分かっていたか】\n")
            print(f"  実績の減配 {len(g)}件のうち、その前の15か月に予想の減配（−10％以上）が出ていたのは "
                  f"{len(lead)}件（{len(lead) / len(g) * 100:.0f}％）。")
            if len(lead):
                print(f"  先に出ていた場合、実績の印より中央値で {lead.median():.0f}か月早い"
                      f"（最短 {lead.min():.0f}か月・最長 {lead.max():.0f}か月）。")
            print("  ※ 無配の予想は、このツールでは予想の欄に出ない（実績の配当で代用する）ため、数えられていません。\n")
        print("【読み方】\n")
        print("  ・「売る方がよかった」の区分だけ売り、「持つ方がよかった」の区分は持ち続ける、という")
        print("    分け方に意味があるかは、このあとの総当たり（同じ実行で続けて回ります）で確かめます。")
        print("  ・区分の件数が少ないほど、1〜2社の値動きで中央値が動きます。件数も見てください。")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        ed.to_csv(OUTDIR / "after_event.csv", index=False, encoding="utf-8-sig")
        print("\n書き出しました: data/after_event.csv")
        if not args.grid:
            return 0
        log.info("続けて総当たりを回します")

    # ══════════════════════════════════════════
    # 為替（ドル円）との関係を調べる
    #   2020〜2026年は円安が大きく進んだ時期。
    #   高配当バリュー株（商社・銀行・輸出）はその恩恵を受けている可能性がある。
    #   成績のどれだけが円安由来かを切り分ける。
    # ══════════════════════════════════════════
    if args.fx_test:
        fx = fetch_fx()
        if fx.empty:
            print("ドル円を取得できませんでした。")
            return 1

        fxm = fx.set_index("date")["usdjpy"].resample("ME").last().dropna()
        nm = [x.strip() for x in args.only.split(",") if x.strip()] or \
             ["live15_holdS_cut", "live15_holdall", "current_live"]
        nm = [n for n in nm if n in VARIANTS]

        runs = []
        for n_ in nm:
            r = simulate(panel, VARIANTS[n_], args.capital, args.max_names,
                         tier_budget=args.realistic, dividends=args.realistic,
                         slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                         tax_rate=args.tax / 100.0, ramp=args.ramp,
                         min_yield_override=args.min_yield or None)
            if r.get("curve") is not None and not r["curve"].empty:
                runs.append((VARIANTS[n_]["label"], r))
        bh = buy_and_hold(panel, args.tax / 100.0)
        if bh and bh.get("curve") is not None:
            runs.append(("（基準）全部買って放置", bh))

        d0, d1 = panel["date"].min(), panel["date"].max()
        f0 = float(fxm[fxm.index <= d0].iloc[-1]) if (fxm.index <= d0).any() \
            else float(fxm.iloc[0])
        f1 = float(fxm[fxm.index <= d1].iloc[-1]) if (fxm.index <= d1).any() \
            else float(fxm.iloc[-1])
        yrs_ = max((d1 - d0).days / 365.25, 0.5)

        print(f"\n■ 為替（ドル円）との関係"
              f"（{d0.date()} 〜 {d1.date()}）\n")
        print(f"  期初 {f0:.1f}円 → 期末 {f1:.1f}円 "
              f"（{(f1 / f0 - 1) * 100:+.1f}％ / 年率 "
              f"{((f1 / f0) ** (1 / yrs_) - 1) * 100:+.1f}％）")

        gaps = {}      # 要因名 -> [ルールの差, 基準の差]

        # ── 月次の連動を見る ──
        print("\n【月ごとの動きが、どれだけ為替と連動しているか】\n")
        print(f"{'ルール':<32}{'相関':>8}{'円安月の平均':>14}{'円高月の平均':>14}{'差':>9}")
        print("-" * 80)
        fx_ret = fxm.pct_change().dropna()
        fx_ret = fx_ret[(fx_ret.index >= d0) & (fx_ret.index <= d1)]
        for lab, r in runs:
            c = r["curve"].set_index("date")["value"]
            pr = c.pct_change().dropna()
            j = pd.concat([pr.rename("p"), fx_ret.rename("f")],
                          axis=1, sort=True).dropna()
            if len(j) < 12:
                continue
            corr = j["p"].corr(j["f"])
            up = j[j["f"] > 0]["p"].mean() * 100
            dn = j[j["f"] <= 0]["p"].mean() * 100
            key_ = "（基準）" if lab.startswith("（基準）") else lab
            gaps.setdefault("ドル円（円安）", {})[key_] = up - dn
            print(f"{lab[:30]:<32}{corr:>8.2f}{up:>13.2f}%{dn:>13.2f}%"
                  f"{up - dn:>+8.2f}pt")

        print("\n  相関 … +1に近いほど為替と同じ動き、0なら無関係、−1なら逆。")
        print("  円安月／円高月 … その月の資産の増え方の平均。")
        print("  差が大きいほど、成績が円安に支えられていたことになる。")

        # ── 円安の期間と、そうでない期間で分ける ──
        print("\n【円安が進んだ期間と、そうでない期間で分ける】\n")
        # 12か月前と比べて円安かどうかで月を分類する
        # 検証期間だけに絞る（絞らないと1971年からの全期間を数えてしまう）
        fx_yoy = (fxm / fxm.shift(12) - 1).dropna()
        fx_yoy = fx_yoy[(fx_yoy.index >= d0) & (fx_yoy.index <= d1)]
        weak = set(fx_yoy[fx_yoy > 0.05].index)      # 1年で5％以上の円安
        strong = set(fx_yoy[fx_yoy < -0.05].index)   # 1年で5％以上の円高
        print(f"  円安の月 {len(weak)}か月 ／ 円高の月 {len(strong)}か月 ／ "
              f"その他 {len(fx_yoy) - len(weak) - len(strong)}か月\n")
        print(f"{'ルール':<32}{'円安期の年率':>14}{'円高・横ばい期の年率':>20}{'差':>10}")
        print("-" * 80)
        for lab, r in runs:
            c = r["curve"].set_index("date")["value"]
            pr = c.pct_change().dropna()
            w = pr[pr.index.isin(weak)]
            o = pr[~pr.index.isin(weak)]
            if len(w) < 6 or len(o) < 6:
                continue
            wa = ((1 + w.mean()) ** 12 - 1) * 100
            oa = ((1 + o.mean()) ** 12 - 1) * 100
            print(f"{lab[:30]:<32}{wa:>13.1f}%{oa:>19.1f}%{wa - oa:>+9.1f}pt")

        print("\n  円安期 … 1年前と比べて5％以上の円安だった月。")
        print("  差が大きいほど、円安局面でだけ成績が出ていたことになる。")
        print("  ※ 月ごとの平均を年率に直しているため、"
              "通常の年率とは一致しません。")

        # ── 他の市場との関係 ──
        mk = fetch_markets()
        _tgt = [("sp500", "S&P500", "米国株の影響を切り分ける"),
                ("vix", "VIX（恐怖指数）",
                 "市場が不安なときに上がる。暴落局面の代用")]
        _tgt += [(k, n, nt) for k, (n, nt, _u) in OTHER_SRC.items()]
        for key, jname, note in _tgt:
            sr = mk.get(key)
            if sr is None or sr.empty:
                continue
            m = sr.resample("ME").last().dropna()
            m = m[(m.index >= d0) & (m.index <= d1)]
            if len(m) < 18:
                log.warning("%s は検証期間の重なりが %dか月しかないため省略します",
                            jname, len(m))
                continue

            print(f"\n■ {jname} との関係（{note}）\n")
            print(f"  期初 {m.iloc[0]:.1f} → 期末 {m.iloc[-1]:.1f}"
                  f"（{(m.iloc[-1] / m.iloc[0] - 1) * 100:+.1f}％）")

            if key == "vix":
                # 恐怖指数が高い月＝市場が不安な月
                hi = set(m[m >= m.median()].index)
                l1, l2 = "不安が強い月", "落ち着いた月"
            else:
                ret = m.pct_change().dropna()
                hi = set(ret[ret > 0].index)
                l1, l2 = f"{jname}が上げた月", "下げた月"

            print(f"\n{'ルール':<32}{'相関':>8}{l1:>15}{l2:>15}{'差':>10}")
            print("-" * 82)
            base_gap = None
            ref = m.pct_change().dropna()
            for lab, r in runs:
                c = r["curve"].set_index("date")["value"]
                pr = c.pct_change().dropna()
                j = pd.concat([pr.rename("p"), ref.rename("f")],
                              axis=1, sort=True).dropna()
                if len(j) < 12:
                    continue
                corr = j["p"].corr(j["f"])
                a = pr[pr.index.isin(hi)].mean() * 100
                b = pr[~pr.index.isin(hi)].mean() * 100
                if lab.startswith("（基準）"):
                    base_gap = a - b
                elif not lab.startswith("（基準）"):
                    gaps.setdefault(jname, {})[lab] = a - b
                print(f"{lab[:30]:<32}{corr:>8.2f}{a:>14.2f}%{b:>14.2f}%"
                      f"{a - b:>+9.2f}pt")
            if base_gap is not None:
                gaps.setdefault(jname, {})["（基準）"] = base_gap

            if key == "vix" and base_gap is not None:
                print("\n  VIXが高い＝暴落や不安の局面。")
                print("  そこでの成績が基準より良ければ、"
                      "「暴落に強い」という主張の裏づけになる。")

        # ── 外部要因のまとめ ──
        if gaps:
            print("\n■ 外部要因への依存（基準を引いた値）\n")
            labs = [l for l, _ in runs if not l.startswith("（基準）")]
            facs = list(gaps.keys())
            hdr = f"{'ルール':<30}" + "".join(f"{f[:8]:>11}" for f in facs)
            print(hdr)
            print("-" * min(len(hdr) + 10, 130))
            tot = {}
            for lab in labs:
                line = f"{lab[:28]:<30}"
                ssum = 0.0
                for f in facs:
                    g = gaps.get(f, {})
                    if lab in g and "（基準）" in g:
                        d_ = g[lab] - g["（基準）"]
                        ssum += d_
                        line += f"{d_:>+10.2f}"
                    else:
                        line += f"{'—':>11}"
                tot[lab] = ssum
                print(line)
            print("\n  各数字＝そのルールの差 − 基準の差。")
            print("  プラスが大きいほど、その要因に市場平均以上に依存している。\n")
            print(f"{'ルール':<30}{'依存の合計':>14}")
            print("-" * 46)
            for lab, v in sorted(tot.items(), key=lambda x: x[1]):
                print(f"{lab[:28]:<30}{v:>+13.2f}pt")
            print("\n  合計が小さいほど、外部要因に振り回されにくい。")

        print("\n【読み方】\n")
        print("  ルールと基準の「差」を比べてください。")
        print("  両方とも同じだけ円安に支えられているなら、"
              "それは相場全体の話であって、")
        print("  このルール固有の弱点ではありません。")
        print("  ルールだけ差が大きいなら、"
              "円安に依存した銘柄に偏っていることになります。")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        fx.to_csv(OUTDIR / "usdjpy.csv", index=False, encoding="utf-8-sig")
        for k, sr in mk.items():
            sr.to_frame(k).to_csv(OUTDIR / f"{k}.csv", encoding="utf-8-sig")
        print("\n書き出しました: data/usdjpy.csv" +
              "".join(f", data/{k}.csv" for k in mk))
        return 0

    # ══════════════════════════════════════════
    # レンジ売買の検証
    #   目視でやっている「上下に挟まれた動き」を機械の判定に置き換え、
    #   「売らない」戦略と同じ費用条件で比べる。
    # ══════════════════════════════════════════
    if args.range_test:
        # ── レンジ売買だけの検証 ──
        # 「レンジの設定」の欄：「日数,幅の下限,幅の上限,買い幅,売り幅,諦める幅,単価より上でだけ売る」
        #   ・「/」で区切れば、いくつもの設定を一度に比べられる（2026年10月10日に追加）
        #   ・7番目は 1 なら、上限で売るのは取得単価より上のときだけ（安値で投げない）。省けば 0
        #   ・2〜6番目の割合は、1以上なら百分率とみなす（8 と書けば 0.08、1 と書けば 0.01）
        # 比べる相手は「比較するルール」の欄のルール（空欄なら9月以前の古いルール）。
        # 「この年数の窓を1年ずつずらして何度も検証」に数字を入れると、窓ごとに比べる。
        cfgs = []
        for part in str(args.range_opts).split("/"):
            part = part.strip()
            if not part:
                continue
            _d = [60.0, 0.08, 0.20, 0.02, 0.02, 0.0, 0.0]
            _v = [x.strip() for x in part.split(",")]
            for _i, _x in enumerate(_v[:7]):
                if _x:
                    try:
                        _f = float(_x)
                    except ValueError:
                        sys.exit(f"レンジの設定「{part}」を読めません。"
                                 "「60,0.08,0.20,0.02,0.02,0」の形で指定してください。")
                    if 1 <= _i <= 5 and _f >= 1.0:
                        _f = _f / 100.0      # 8 と書かれたら 0.08、1 と書かれたら 0.01
                    _d[_i] = _f
            win_ = int(_d[0])
            wmin, wmax, b_at, s_at, _st = _d[1], _d[2], _d[3], _d[4], _d[5]
            stop = _st if _st > 0 else None
            g_only = _d[6] >= 0.5
            label = f"{win_}日・幅{wmin * 100:g}〜{wmax * 100:g}％"
            if stop:
                label += f"・撤退{stop * 100:g}％"
            cfgs.append({"win": win_, "w_min": wmin, "w_max": wmax,
                         "buy_at": b_at, "sell_at": s_at, "stop": stop,
                         "gain_only": g_only, "screen": True, "label": label})
            log.info("レンジ売買：%s（下限+%g％で買い・上限−%g％で売り・%s・%s）",
                     label, b_at * 100, s_at * 100,
                     f"下限を{stop * 100:g}％割ったら撤退" if stop else "撤退なし",
                     "単価より上でだけ売る" if g_only else "上限なら売る")
        if not cfgs:
            sys.exit("レンジの設定が空です。")
        if len({c["label"] for c in cfgs}) != len(cfgs):
            # 日数と値幅が同じで、買い幅・売り幅だけ違う設定は名前が重なるので番号を付ける
            for i_, c in enumerate(cfgs, 1):
                c["label"] = f"{i_}:{c['label']}"
        log.info("  対象は8条件を通り、前の月末の利回りが %.1f％ 以上の銘柄", args.min_yield)

        comp = [x.strip() for x in (args.only or "").split(",")
                if x.strip() and x.strip() in VARIANTS]
        if not comp:
            comp = [n_ for n_ in ("live15_holdS_cut", "live15_holdall", "current_live")
                    if n_ in VARIANTS]
        daily_ = build_daily(store, args.years)
        bounds_ = [{c_: box_bounds(px_, c["win"], c["w_min"], c["w_max"])
                    for c_, px_ in daily_.items()} for c in cfgs]

        def _run_range(i_, a=None, b=None):
            return simulate_range(store, panel, cfgs[i_], args.capital, args.max_names,
                                  slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                                  tax_rate=args.tax / 100.0, min_yield=args.min_yield,
                                  years=args.years, d_from=a, d_to=b,
                                  dividends=args.realistic, daily=daily_,
                                  bounds=bounds_[i_])

        def _run_comp(n_, pn_):
            return simulate(pn_, VARIANTS[n_], args.capital, args.max_names,
                            tier_budget=args.realistic, dividends=args.realistic,
                            slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                            tax_rate=args.tax / 100.0,
                            min_yield_override=args.min_yield or None)

        d0, d1 = panel["date"].min(), panel["date"].max()
        if args.walk > 0:
            wins = []
            st_ = d1 - pd.DateOffset(years=args.walk)
            while st_ >= d0:
                wins.append((st_, st_ + pd.DateOffset(years=args.walk)))
                st_ = st_ - pd.DateOffset(years=1)
            wins.reverse()
        else:
            wins = []
        wins.append((d0, d1))            # 全期間も1行として加える
        log.info("レンジ売買 %d通り × 比べる相手 %d × 期間 %d", len(cfgs), len(comp), len(wins))

        cols = [c["label"] for c in cfgs] + [VARIANTS[n_]["label"] for n_ in comp] \
            + ["（基準）全部買って放置"]
        rows_cagr, rows_dd, full_runs = [], [], {}
        for a, b in wins:
            sub = panel[(panel["date"] >= a) & (panel["date"] <= b)]
            if sub["date"].nunique() < max(args.walk, 1) * 10:
                continue
            span = "全期間" if (a, b) == (d0, d1) else f"{a.date()}〜{b.date()}"
            rc, rd = {"窓": span}, {"窓": span}
            for i_, c in enumerate(cfgs):
                r_ = _run_range(i_, a, b)
                m_ = metrics(r_) if r_ else {}
                rc[c["label"]] = m_.get("年率", float("nan"))
                rd[c["label"]] = m_.get("最大下落", float("nan"))
                if span == "全期間" and r_:
                    full_runs[c["label"]] = (m_, r_)
            for n_ in comp:
                r_ = _run_comp(n_, sub)
                m_ = metrics(r_) if r_ else {}
                rc[VARIANTS[n_]["label"]] = m_.get("年率", float("nan"))
                rd[VARIANTS[n_]["label"]] = m_.get("最大下落", float("nan"))
                if span == "全期間" and r_:
                    full_runs[VARIANTS[n_]["label"]] = (m_, r_)
            bh_ = buy_and_hold(sub, args.tax / 100.0)
            rc["（基準）全部買って放置"] = bh_["年率"] if bh_ else float("nan")
            rd["（基準）全部買って放置"] = bh_["最大下落"] if bh_ else float("nan")
            rows_cagr.append(rc)
            rows_dd.append(rd)
            log.info("  %s 完了", span)

        if not rows_cagr:
            print("結果がありません。")
            return 1

        print(f"\n■ レンジ売買の比較（{d0.date()} 〜 {d1.date()} / 元本 {args.capital:,.0f}円"
              f"{' ／ 実運用条件・配当あり' if args.realistic else ''}）\n")
        print("　 レンジ売買の対象は、8条件を通り、前の月末の利回りが "
              f"{args.min_yield:.1f}％ 以上の銘柄。最大 {args.max_names}銘柄・等金額。\n")
        for title, rows in (("年率", rows_cagr), ("最大下落", rows_dd)):
            print(f"【{title}】\n")
            print(f"{'窓':<24}" + "".join(f"{c[:14]:>16}" for c in cols))
            print("-" * (24 + 16 * len(cols)))
            for r in rows:
                print(f"{r['窓']:<24}" + "".join(
                    f"{r[c]:>15.1f}%" if not pd.isna(r[c]) else f"{'—':>16}" for c in cols))
            print()

        # 本番（比べる相手の最初のルール）との差を、設定ごとに並べる
        base_lab = VARIANTS[comp[0]]["label"] if comp else None
        if base_lab:
            print(f"【「{base_lab[:30]}」との差（年率、プラスならレンジ売買が上）】\n")
            print(f"{'設定':<24}{'最初の窓':>12}{'全期間':>12}{'上回った窓':>12}")
            print("-" * 60)
            win_rows = [r for r in rows_cagr if r["窓"] != "全期間"]
            full_row = next((r for r in rows_cagr if r["窓"] == "全期間"), None)
            for c in cfgs:
                lab = c["label"]
                first = (win_rows[0][lab] - win_rows[0][base_lab]) if win_rows else float("nan")
                full = (full_row[lab] - full_row[base_lab]) if full_row else float("nan")
                wins_ = sum(1 for r in win_rows if r[lab] > r[base_lab])
                print(f"{lab[:22]:<24}{first:>+11.1f}pt{full:>+11.1f}pt"
                      f"{wins_:>8}/{len(win_rows)}")
            print()

        # レンジ売買の中身（全期間）
        for c in cfgs:
            if c["label"] not in full_runs:
                continue
            m_, r_ = full_runs[c["label"]]
            d = r_["diag"]
            hd = d.get("hold_days", [])
            print(f"■ {c['label']} の中身（全期間）")
            print(f"  買った回数 {d['buys']}回 ／ 売った回数 {d['sells']}回"
                  + (f"（うち撤退 {d['stops']}回）" if d.get("stops") else ""))
            if hd:
                print(f"  平均の保有日数 {sum(hd) / len(hd):.0f}日 ／ 平均の保有銘柄数 "
                      f"{m_.get('平均保有銘柄', 0):.1f}")
            print(f"  確定した売却益 {d.get('realized', 0):,.0f}円 ／ 配当（税引後） "
                  f"{d.get('dividend', 0):,.0f}円 ／ 含み損益 {d.get('unrealized', 0):,.0f}円")
            print(f"  税金 {d.get('tax', 0):,.0f}円 ／ 手数料とずれ {d.get('fee', 0):,.0f}円\n")

        print("  ※ レンジの上下は、その日までの過去N営業日の終値から機械的に決めています。")
        print("    目視の線とは違いますが、「上下の線の間を取る」考え方は再現しています。")
        print("  ※ 9月のレンジ売買の検証とは、対象（8条件）・配当・判定の時点が違うので、")
        print("    数字は直接比べられません。")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_cagr).to_csv(OUTDIR / "range_test.csv", index=False,
                                       encoding="utf-8-sig")
        first_rng = next((full_runs[c["label"]][1] for c in cfgs
                          if c["label"] in full_runs), None)
        if first_rng and first_rng.get("trades"):
            pd.DataFrame(first_rng["trades"]).to_csv(OUTDIR / "range_trades.csv",
                                                     index=False, encoding="utf-8-sig")
        print("\n書き出しました: data/range_test.csv, data/range_trades.csv")
        return 0

    # ══════════════════════════════════════════
    # 総当たり検証
    #   条件を1つずつ変えて試すと、そのたびに読み違いが起きる。
    #   （期間と分位を同時に変えた、窓の欄が空欄だった、など）
    #   ここでは条件をすべて機械的に組み合わせて回し、
    #   「どの条件でも成り立った結論」だけを取り出す。
    # ══════════════════════════════════════════
    if args.grid:
        ramps = [int(x) for x in str(args.grid_ramp).split(",") if x.strip() != ""]
        mys = [float(x) for x in str(args.grid_min_yield).split(",") if x.strip() != ""]
        lbs_ = [int(x) for x in str(args.grid_lookback).split(",") if x.strip() != ""]
        mxs_ = [int(x) for x in str(args.grid_max_names).split(",") if x.strip() != ""]
        nm = [x.strip() for x in args.only.split(",") if x.strip()] or \
             ["current_live", "live_budget15", "live15_holdS", "live15_holdall"]
        nm = [n for n in nm if n in VARIANTS]

        # 窓：4年の窓を1年ずつずらしたもの＋全期間
        wl = args.grid_window
        d0_, d1_ = panel["date"].min(), panel["date"].max()
        wins = []
        st_ = d1_ - pd.DateOffset(years=wl)
        while st_ >= d0_:
            wins.append((st_, d1_ if not wins else st_ + pd.DateOffset(years=wl)))
            st_ = st_ - pd.DateOffset(years=1)
        wins = [(a, a + pd.DateOffset(years=wl)) for a, _ in wins]
        wins.reverse()
        wins.append((d0_, d1_))          # 全期間も1通りとして加える

        total = len(ramps) * len(mys) * len(wins) * len(lbs_) * len(mxs_)
        log.info("総当たり：期間%d × 資金投入%d × 足切り%d × 分位%d × 保有上限%d"
                 " = %d条件 × ルール%d = %d回",
                 len(wins), len(ramps), len(mys), len(lbs_), len(mxs_),
                 total, len(nm), total * len(nm))
        if total * len(nm) > 400:
            log.warning("%d回は時間がかかります（目安 %d分）。"
                        "条件を減らすことも検討してください。",
                        total * len(nm), int(total * len(nm) * 0.12))

        rows_, done = [], 0
        stop_logs = {}       # 損切りの記録（全期間・資金投入がいちばん短い条件だけ取っておく）
        act_logs = {}        # 利確・利回り割れの記録（同じ条件だけ取っておく）
        harvest_sum = {}     # 損出しの回数と確定させた損（同じ条件だけ取っておく）
        swing_trades = {}    # うねりの売り買いの明細（同じ条件だけ取っておく。売った後・買った後の値動きを見る）
        swing_sales = {}     # うねりで売った1回ごとの（銘柄, 日, 益）（同じ条件だけ。業種別の中身を見る）
        _rb = bool(getattr(args, "robust_test", False))
        placebo_rows = []    # 12回目：ランダムな日の売り買い（資金投入がいちばん短い条件の、各期間）
        # 10回目：年率を「最後に全部売って税を払った後」（清算後）で比べる。評価額の年率も列に残す
        _liq = bool(getattr(args, "liq_metric", False))
        panels = {}
        for lb_ in lbs_:
            # 分位の期間ごとにパネルを作り直す（重いので一度だけ）
            panels[lb_] = _panel_for(lb_) if lb_ != args.lookback else panel
            log.info("  分位%dか月のパネル … %d件", lb_, len(panels[lb_]))

        for lb_ in lbs_:
          pnl = panels[lb_]
          for a, b in wins:
            sub = pnl[(pnl["date"] >= a) & (pnl["date"] <= b)]
            if sub["date"].nunique() < 24:
                continue
            span = f"{a.date()}〜{b.date()}"
            for mx_ in mxs_:
              for rp in ramps:
                for my in mys:
                    bh_ = buy_and_hold(sub, args.tax / 100.0)
                    base_mtm = bh_["年率"] if bh_ else float("nan")
                    base_r = (bh_.get("清算後年率", base_mtm) if _liq else base_mtm) if bh_ else float("nan")
                    _real_d = None
                    for n_ in nm:
                        r_ = simulate(
                            sub, VARIANTS[n_], args.capital, mx_,
                            tier_budget=args.realistic, dividends=args.realistic,
                            slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                            tax_rate=args.tax / 100.0, ramp=rp,
                            min_yield_override=my)
                        m_ = metrics(r_)
                        if m_:
                            d_ = r_.get("diag", {})
                            # 売買の多さ（2026年10月11日に追加）。外から見たときの「動いている感」の目安。
                            #   売り/年       … 1年あたりに売った回数（一部だけ売った回も1回と数える）
                            #   売買のあった月 … 最初の12か月（買い付けの時期）を除いた月のうち、
                            #                    売りか買いが1回でもあった月の割合
                            _tr = r_["trades"]
                            _dts = [pd.Timestamp(x) for x in sorted(sub["date"].unique())]
                            _yrs = max((_dts[-1] - _dts[0]).days / 365.25, 0.5)
                            _n_sell = int((_tr["side"] == "sell").sum()) if not _tr.empty else 0
                            _later = _dts[12:]
                            if _later and not _tr.empty:
                                _act_m = {pd.Timestamp(x) for x in _tr["date"]}
                                _m_act = sum(1 for d in _later if d in _act_m) / len(_later) * 100
                                # 買い/年 … 最初の12か月（買い付けの時期）を除いた買いの回数を、残りの年数で割る
                                _ld = set(_later)
                                _n_buy = int(sum(1 for x, s in zip(_tr["date"], _tr["side"])
                                                 if s == "buy" and pd.Timestamp(x) in _ld))
                                _yrs_l = max((_later[-1] - _dts[11]).days / 365.25, 0.25) \
                                    if len(_dts) > 12 else _yrs
                                _b_yr = _n_buy / _yrs_l
                            else:
                                _m_act = float("nan")
                                _b_yr = float("nan")
                            _yr = m_.get("清算後年率", m_["年率"]) if _liq else m_["年率"]
                            rows_.append({
                                "期間": span, "資金投入": rp, "利回り足切り": my,
                                "分位": lb_, "保有上限": mx_,
                                "ルール": VARIANTS[n_]["label"],
                                "年率": _yr, "最大下落": m_["最大下落"],
                                "評価額年率": m_["年率"], "清算後年率": m_.get("清算後年率", m_["年率"]),
                                "現金比率": m_.get("現金比率", float("nan")),
                                "シャープ": m_["シャープ"], "決済率": m_["決済率"],
                                "基準": base_r, "基準差": _yr - base_r, "基準_評価額": base_mtm,
                                "緊急売り": d_.get("cut_exits", 0),
                                "損切り": d_.get("stop_exits", 0),
                                "利確・利回り割れ": d_.get("act_exits", 0) + d_.get("act_partial", 0),
                                "損出し": d_.get("harvests", 0),
                                "買い増し": d_.get("topups", 0),
                                "下落で見送り": d_.get("drop_skip", 0),
                                "うねり売り": d_.get("range_sells", 0),
                                "底で買い": d_.get("swing_buys", 0),
                                "うねり1回の益": (d_.get("swing_gain_sum", 0.0) / d_["range_sells"]
                                              if d_.get("range_sells") else float("nan")),
                                "権利前の買い": d_.get("rec_buys", 0),
                                # 11回目：緊急売りの内訳（どう買った銘柄か）と、売ったときの損益の平均
                                "緊急_線の下": d_.get("cut_by_origin", {}).get("swing", 0),
                                "緊急_月末": d_.get("cut_by_origin", {}).get("month", 0),
                                "緊急_最初の月": d_.get("cut_by_origin", {}).get("first", 0),
                                "緊急の損益": (float(np.mean(d_["cut_ret"])) if d_.get("cut_ret") else float("nan")),
                                "ゆるめ買い": d_.get("rec_relaxed", 0),
                                "売り/年": _n_sell / _yrs,
                                "買い/年": _b_yr,
                                "売買のあった月": _m_act})
                            if ((a, b) == (d0_, d1_) and rp == min(ramps) and lb_ == lbs_[0]
                                    and mx_ == mxs_[0] and my == mys[0]):
                                if VARIANTS[n_].get("stop_loss"):
                                    stop_logs[n_] = (d_.get("stop_log", []), pnl)
                                if d_.get("act_log"):
                                    act_logs[n_] = (d_.get("act_log", []), pnl)
                                if VARIANTS[n_].get("harvest"):
                                    harvest_sum[n_] = (d_.get("harvests", 0),
                                                       d_.get("harvest_loss", 0.0), _yrs)
                                if VARIANTS[n_].get("swing_daily") and not _tr.empty and "day" in _tr.columns:
                                    swing_trades[n_] = _tr[_tr["day"].notna()].copy()
                                if d_.get("swing_sales"):
                                    swing_sales[n_] = list(d_["swing_sales"])
                            if _rb and n_ == "safe_line_2y_pit":
                                _real_d = d_
                    if _rb and _real_d is not None and rp == min(ramps) and lb_ == lbs_[0] \
                            and mx_ == mxs_[0] and my == mys[0]:
                        # 回数の目安は、本物の 2年線が「売ってよい日」「買ってよい日」のうち実際に売り買いした割合
                        _ps = _real_d.get("range_sells", 0) / max(1, _real_d.get("sw_elig_sell", 0))
                        _pb = _real_d.get("swing_buys", 0) / max(1, _real_d.get("sw_elig_buy", 0))
                        _wi = [w_ for w_ in wins].index((a, b)) if (a, b) in wins else 0
                        for _sd in range(PLACEBO_N):
                            _cfgp = dict(VARIANTS["safe_line_2y_pit"])
                            _cfgp["swing_daily"] = dict(_cfgp["swing_daily"],
                                                        placebo={"seed": 1000 * (_wi + 1) + _sd, "p_sell": _ps, "p_buy": _pb})
                            _rp_ = simulate(sub, _cfgp, args.capital, mx_,
                                            tier_budget=args.realistic, dividends=args.realistic,
                                            slip_bps=args.slip_bps, fee_bps=args.fee_bps,
                                            tax_rate=args.tax / 100.0, ramp=rp, min_yield_override=my)
                            _mp_ = metrics(_rp_)
                            if _mp_:
                                placebo_rows.append({"期間": span, "seed": _sd,
                                                     "清算後年率": _mp_.get("清算後年率", _mp_["年率"]),
                                                     "評価額年率": _mp_["年率"],
                                                     "うねり売り": _rp_["diag"].get("range_sells", 0),
                                                     "底で買い": _rp_["diag"].get("swing_buys", 0)})
                        log.info("  ランダムな日の売り買い（%s） … %d通り完了（売りの割合 %.4f・買いの割合 %.4f）",
                                 span, PLACEBO_N, _ps, _pb)
                    done += 1
                    if done % 5 == 0 or done == total:
                        log.info("  %d/%d 条件が完了", done, total)

        if not rows_:
            print("結果がありません。")
            return 1
        gf = pd.DataFrame(rows_)
        cond_cols = ["期間", "資金投入", "利回り足切り", "分位", "保有上限"]
        n_cond = gf.groupby(cond_cols).ngroups

        print(f"\n■ 総当たり検証（{n_cond}条件 × {len(nm)}ルール）\n")
        print(f"　 期間 {len(wins)}通り ／ 資金投入 {ramps} か月 ／ "
              f"利回り足切り {mys} ％")
        print(f"　 元本 {args.capital:,.0f}円 ／ 税率{args.tax:.3f}％ ／ "
              f"約定のずれ 片道{args.slip_bps:.0f}bps\n")
        _lt = bool(getattr(args, "line_test", False) or getattr(args, "safe_test", False)
                   or getattr(args, "robust_test", False))
        _l10 = bool(getattr(args, "line_test", False))
        if _liq and getattr(args, "robust_test", False):
            print("　 年率は「最後の月末に全部売って、約定のずれと税を払った後」（清算後）で出しています。\n")
        elif _liq:
            _rn = "11" if getattr(args, "safe_test", False) else "10"
            print("　 年率は「最後の月末に全部売って、約定のずれと税を払った後」（清算後）で出しています。"
                  f"評価額の年率は【{_rn}回目の判定】に並べます。\n")

        # 12回目のモードでは、ルールが11あって表が長くなるので、共通の表はすべて省き、【12回目】の欄にまとめる
        if not getattr(args, "robust_test", False):
            # ── 1. 基準に勝った割合 ──
            print("【全部買って放置に勝った割合】\n")
            print(f"{'ルール':<34}{'勝ち':>8}{'割合':>8}{'平均年率':>10}{'平均の差':>10}")
            print("-" * 72)
            agg = []
            for lab, g in gf.groupby("ルール"):
                w = int((g["基準差"] > 0).sum())
                agg.append((w / len(g), lab, w, len(g), g["年率"].mean(),
                            g["基準差"].mean()))
            for rate, lab, w, n, ar, ad in sorted(agg, reverse=True):
                print(f"{lab[:32]:<34}{w:>4}/{n:<3}{rate*100:>7.0f}%"
                      f"{ar:>9.1f}%{ad:>+9.1f}pt")

            # ── 1-B. 安定性で見る ──
            # 年率の高さではなく「条件が変わってもブレないか」「最悪でどこまで沈むか」。
            # 同じ数字を別の軸で読み直しているだけで、追加の検証はしていない。
            print("\n【安定性で見る】\n")
            print(f"{'ルール':<32}{'最悪の年率':>11}{'年率のばらつき':>14}"
                  f"{'最大下落の平均':>14}{'最悪の下落':>11}")
            print("-" * 84)
            st_rows = []
            for lab, g in gf.groupby("ルール"):
                st_rows.append((g["年率"].min(), lab, g["年率"].std(),
                                g["最大下落"].mean(), g["最大下落"].max()))
            for worst, lab, sd, ddm, ddw in sorted(st_rows, reverse=True):
                print(f"{lab[:30]:<32}{worst:>10.1f}%{sd:>13.1f}pt"
                      f"{ddm:>13.1f}%{ddw:>10.1f}%")
            print(f"\n  最悪の年率 … {n_cond}通りの条件のうち、いちばん悪かったときの成績。")
            print("  ばらつき … 条件によって成績がどれだけ振れるか。小さいほど読みやすい。")
            print("  安定を求めるなら、平均の高さより「最悪」と「ばらつき」を見る。")

            # ── 2. ルール同士の勝敗 ──
            piv = gf.pivot_table(index=cond_cols, columns="ルール", values="年率")
            cols = list(piv.columns)
            print(f"\n【ルール同士の勝敗（縦が横を上回った割合）】\n")
            print(" " * 26 + "".join(f"{c[:10]:>12}" for c in cols))
            print("-" * (26 + 12 * len(cols)))
            for a_ in cols:
                line = f"{a_[:24]:<26}"
                for b_ in cols:
                    if a_ == b_:
                        line += f"{'—':>12}"
                    else:
                        line += f"{(piv[a_] > piv[b_]).mean()*100:>11.0f}%"
                print(line)

            # 10回目のモードでは、まとめの表（条件による違い・堅い結論・安定・決められなかったこと）を省く。
            # Summary に出せる長さ（末尾14,000バイト）に判定の欄を収めるため。判定は【10回目の判定】にまとめる。
            if not _lt:
                # ── 3. 条件そのものの影響 ──
                print("\n【条件による違い（全ルール平均）】\n")
                for key, unit in [("資金投入", "か月"), ("利回り足切り", "％"),
                                  ("分位", "か月"), ("保有上限", "銘柄")]:
                    print(f"  {key}")
                    for v, g in gf.groupby(key):
                        print(f"    {v}{unit:<4} 平均年率 {g['年率'].mean():>5.1f}%   "
                              f"基準との差 {g['基準差'].mean():>+5.1f}pt")
                    print()

                # ── 4. 堅い結論だけを取り出す ──
                print("【条件を変えても崩れなかった結論】\n")
                found = False
                for a_ in cols:
                    for b_ in cols:
                        if a_ >= b_:
                            continue
                        r = (piv[a_] > piv[b_]).mean()
                        if r >= 0.85:
                            print(f"  ○ 「{a_[:26]}」は「{b_[:26]}」を "
                                  f"{r*100:.0f}％ の条件で上回った")
                            found = True
                        elif r <= 0.15:
                            print(f"  ○ 「{b_[:26]}」は「{a_[:26]}」を "
                                  f"{(1-r)*100:.0f}％ の条件で上回った")
                            found = True
                for rate, lab, w, n, ar, ad in agg:
                    if rate >= 0.85:
                        print(f"  ○ 「{lab[:26]}」は基準を {rate*100:.0f}％ の条件で上回った")
                        found = True
                    elif rate <= 0.15:
                        print(f"  ○ 「{lab[:26]}」は基準に {(1-rate)*100:.0f}％ の条件で負けた")
                        found = True
                if not found:
                    print("  ありません。どの比較も条件次第で入れ替わります。")

                # 安定性の観点での最良
                best_worst = max(st_rows)          # 最悪の年率がいちばんマシなもの
                best_sd = min(st_rows, key=lambda x: x[2])
                best_dd = min(st_rows, key=lambda x: x[3])
                print("\n【安定を優先するなら】\n")
                print(f"  最悪のときがいちばんマシ … {best_worst[1][:30]}"
                      f"（最悪 {best_worst[0]:.1f}％）")
                print(f"  条件によるブレが最小   … {best_sd[1][:30]}"
                      f"（ばらつき {best_sd[2]:.1f}pt）")
                print(f"  下落がいちばん浅い     … {best_dd[1][:30]}"
                      f"（平均 {best_dd[3]:.1f}％）")
                names_top = {best_worst[1], best_sd[1], best_dd[1]}
                if len(names_top) == 1:
                    print("\n  3つとも同じルールでした。安定性では明確に優れています。")
                else:
                    print("\n  3つの観点で最良が分かれています。何を重視するかで選ぶことになります。")

                print("\n【決められなかったこと】\n")
                undecided = False
                for a_ in cols:
                    for b_ in cols:
                        if a_ >= b_:
                            continue
                        r = (piv[a_] > piv[b_]).mean()
                        if 0.35 <= r <= 0.65:
                            print(f"  × 「{a_[:24]}」と「{b_[:24]}」 … "
                                  f"{r*100:.0f}％ 対 {(1-r)*100:.0f}％")
                            undecided = True
                if not undecided:
                    print("  ありません。")

                print("\n  ※ 85％以上で一貫していれば「堅い」、"
                      "35〜65％なら「決められない」としています。")
                print("  ※ 条件は互いに重なる期間を含むため、完全に独立ではありません。")

            # ── 期間ごとの年率・売買の回数・売った銘柄のその後（2026年10月11日に追加）──
            # 売る形を変えても一度も売っていなければ、土台と同じ成績になり「効果なし」と区別がつかない。
            # 回数を出しておけば、成績の差が売り方から来ているのかを確かめられる。
            _r0 = min(ramps)
            _g0 = gf[(gf["資金投入"] == _r0) & (gf["利回り足切り"] == mys[0])
                     & (gf["分位"] == lbs_[0]) & (gf["保有上限"] == mxs_[0])]
            if not _g0.empty:
                _pv = _g0.pivot_table(index="期間", columns="ルール", values="年率")
                _bh = _g0.groupby("期間")["基準"].first()
                _cl = list(_pv.columns)
                print(f"\n【期間ごとの年率（資金投入{_r0}か月）】\n")
                print(f"{'期間':<24}" + "".join(f"{c[:10]:>12}" for c in _cl) + f"{'全部買って放置':>12}")
                print("-" * (24 + 12 * (len(_cl) + 1)))
                for _p in _pv.index:
                    print(f"{_p:<24}" + "".join(f"{_pv.at[_p, c]:>11.1f}%" for c in _cl)
                          + f"{_bh.get(_p, float('nan')):>11.1f}%")
            print("\n【売買の回数（1条件あたりの平均）】\n")
            print(f"{'ルール':<34}{'減配・業績急変':>10}{'損切り':>7}{'利確・利回り割れ':>11}"
                  f"{'損出し':>7}{'買い増し':>8}{'売り/年':>8}{'買い/年':>8}{'売買のあった月':>12}")
            print("-" * 112)
            for lab, g in gf.groupby("ルール"):
                print(f"{lab[:32]:<34}{g['緊急売り'].mean():>9.1f}回{g['損切り'].mean():>6.1f}回"
                      f"{g['利確・利回り割れ'].mean():>10.1f}回{g['損出し'].mean():>6.1f}回"
                      f"{g['買い増し'].mean():>7.1f}回{g['売り/年'].mean():>7.1f}回"
                      f"{g['買い/年'].mean():>7.1f}回{g['売買のあった月'].mean():>11.0f}%")
            print("\n  売り/年 … 1年あたりに売った回数（一部だけ売った回、損出しで売った回も1回）。")
            print("  買い/年 … 最初の12か月（買い付けの時期）を除いた、1年あたりの買いの回数（買い増し・買い戻しを含む）。")
            print("  売買のあった月 … 最初の12か月を除き、売りか買いが1回でもあった月の割合。")
            print("  　　　　　　　 外から見て「動いている」と感じられるかの目安。")

        # 売買を増やす形の判定（結果を見る前に決めた基準）。比べる相手は「比較するルール」の最初のもの。
        #   次の3つをすべて満たせば「使える」（成績をほとんど削らずに売買を増やせる）
        #     ・平均年率の差が −0.5pt 以内
        #     ・最悪の年率の差が −1.0pt 以内
        #     ・相手が24条件の85％以上で上回っていない（＝相手が「堅く」勝っていない）
        #   さらに、相手を24条件の85％以上で上回れば「本番を上回る」と付ける（本番に入れる価値がある）。
        _act_keys = ("yield_exit", "tp_gain", "harvest", "topup", "rec_relax", "screen_drop",
                     "max_yield", "ex_first", "max_per_sector", "tier_weight",
                     "tiers_allowed", "cash_floor", "max_names", "tier_col", "skip_price_drop",
                     "swing_entry", "swing_daily")
        if any(any(k in VARIANTS[n_] for k in _act_keys) for n_ in nm) and len(nm) > 1 and not _lt:
            _base_lab = VARIANTS[nm[0]]["label"]
            if _base_lab in piv.columns:
                _gb = gf[gf["ルール"] == _base_lab]
                _ab, _wb = _gb["年率"].mean(), _gb["年率"].min()
                _sb, _bb = _gb["売り/年"].mean(), _gb["買い/年"].mean()
                print(f"\n【本番との差と判定（相手：{_base_lab[:28]}）】\n")
                print(f"　 相手の成績 … 平均年率 {_ab:.1f}％ ／ 最悪の年率 {_wb:.1f}％ ／ "
                      f"最悪の下落 {_gb['最大下落'].max():.1f}％ ／ 売り {_sb:.1f}回/年 ／ 買い {_bb:.1f}回/年\n")
                print(f"{'ルール':<34}{'売り/年の差':>9}{'買い/年の差':>9}{'平均年率の差':>11}{'最悪の年率の差':>12}"
                      f"{'相手が勝った':>10}{'上回った':>8}  判定")
                print("-" * 112)
                for n_ in nm[1:]:
                    lab = VARIANTS[n_]["label"]
                    if lab not in piv.columns:
                        continue
                    g = gf[gf["ルール"] == lab]
                    da, dw = g["年率"].mean() - _ab, g["年率"].min() - _wb
                    ds, db = g["売り/年"].mean() - _sb, g["買い/年"].mean() - _bb
                    lose = float((piv[_base_lab] > piv[lab]).mean())
                    win = float((piv[lab] > piv[_base_lab]).mean())
                    ok = (da >= -0.5) and (dw >= -1.0) and (lose < 0.85)
                    verdict = ("使える・本番を上回る" if (ok and win >= 0.85) else
                               "使える" if ok else "使えない")
                    print(f"{lab[:32]:<34}{ds:>+8.1f}回{db:>+8.1f}回{da:>+10.2f}pt{dw:>+11.2f}pt"
                          f"{lose * 100:>9.0f}%{win * 100:>7.0f}%  {verdict}")
                print("\n  使える … 平均年率の差 −0.5pt以内・最悪の年率の差 −1.0pt以内・相手の勝ちが85％未満、のすべてを満たす。")
                print("  本番を上回る … そのうえで、相手を24条件の85％以上で上回った（本番に入れる価値がある）。")
                print("  使えない形でも、差の大きさは「その形を入れる代金」として読めます。")
        # ── 8条件を1つずつ外した結果の判定（2026年10月11日に追加。結果を見る前に決めた基準）──
        #   効いている        … 本番（相手）が24条件の85％以上で上回った → 残す
        #   外した方がいい    … 外した形が85％以上で上回った → 外す
        #   飾り              … 勝敗が35〜65％で、平均年率の差が ±0.3pt 以内、最悪の年率の差が −0.5pt 以内 → 外しても変わらない
        #   決められない      … それ以外
        if any(VARIANTS[n_].get("screen_drop") for n_ in nm) and len(nm) > 1:
            _base_lab = VARIANTS[nm[0]]["label"]
            if _base_lab in piv.columns:
                _gb = gf[gf["ルール"] == _base_lab]
                _ab, _wb, _db = _gb["年率"].mean(), _gb["年率"].min(), _gb["最大下落"].mean()
                print(f"\n【条件を外した結果の判定（相手：{_base_lab[:28]}）】\n")
                print(f"{'外した条件':<28}{'平均年率の差':>11}{'最悪の年率の差':>12}{'下落の平均の差':>12}"
                      f"{'本番が勝った':>10}{'外した形が勝った':>12}  判定")
                print("-" * 112)
                for n_ in nm[1:]:
                    lab = VARIANTS[n_]["label"]
                    if lab not in piv.columns or not VARIANTS[n_].get("screen_drop"):
                        continue
                    g = gf[gf["ルール"] == lab]
                    da, dw = g["年率"].mean() - _ab, g["年率"].min() - _wb
                    dd = g["最大下落"].mean() - _db
                    lose = float((piv[_base_lab] > piv[lab]).mean())
                    win = float((piv[lab] > piv[_base_lab]).mean())
                    if lose >= 0.85:
                        v = "効いている（残す）"
                    elif win >= 0.85:
                        v = "外した方がいい"
                    elif 0.35 <= lose <= 0.65 and abs(da) <= 0.3 and dw >= -0.5:
                        v = "飾り（外しても変わらない）"
                    elif lose < 0.35 and win < 0.85 and abs(da) <= 0.3 and dw >= -0.5:
                        v = "飾り（外しても変わらない）"
                    else:
                        v = "決められない"
                    print(f"{lab[:26]:<28}{da:>+10.2f}pt{dw:>+11.2f}pt{dd:>+11.2f}pt"
                          f"{lose * 100:>9.0f}%{win * 100:>11.0f}%  {v}")
                print("\n  効いている … 本番が85％以上で上回った。外した方がいい … 外した形が85％以上で上回った。")
                print("  飾り … 勝敗が偏らず（本番の勝ちが65％以下）、平均年率の差が±0.3pt以内・最悪の年率の差が−0.5pt以内。")
                print("  下落の平均の差 … プラスなら、外した形の方が下落が深い。")
        if any(VARIANTS[n_].get("rec_relax") for n_ in nm):
            print("\n【配当の権利確定月の前に新しく買った回数（1条件あたりの平均）】\n")
            print(f"{'ルール':<34}{'権利月の前に買った':>14}{'うち条件をゆるめて買えた':>18}")
            print("-" * 70)
            for n_ in nm:
                lab = VARIANTS[n_]["label"]
                g = gf[gf["ルール"] == lab]
                if g.empty or not VARIANTS[n_].get("rec_relax"):
                    continue
                print(f"{lab[:32]:<34}{g['権利前の買い'].mean():>13.1f}回{g['ゆるめ買い'].mean():>17.1f}回")
            print("\n  権利月の前に買った … 次の月末に配当の権利が確定する銘柄を、新しく買った回数。")
            print("  うち条件をゆるめて買えた … 本番の条件（利回り3％・8条件）では買えなかった回数。")
            print("  これが少ないと、ゆるめても成績はほとんど変わらない（資金が埋まっていて新しく買う機会が少ないため）。")
        if harvest_sum:
            print(f"\n【損出しの中身（全期間・資金投入{min(ramps)}か月の条件）】\n")
            print(f"{'ルール':<34}{'回数':>6}{'1年あたり':>9}{'確定させた損':>14}{'戻った税金の目安':>16}")
            print("-" * 86)
            for n_, (cnt, loss, yrs_h) in harvest_sum.items():
                print(f"{VARIANTS[n_]['label'][:32]:<34}{cnt:>6}{cnt / yrs_h:>8.1f}回"
                      f"{loss:>13,.0f}円{loss * args.tax / 100.0:>15,.0f}円")
            print(f"\n  戻った税金の目安 … 確定させた損 × 税率{args.tax:.3f}％。配当や売却益と相殺しきれた場合の額。")
            print("  損出しで取得単価が下がるので、のちに売ればその分の税金を払う（売るまでの先送り）。")
            print("  売らずに持ち続ける限りは得のまま。")
        if stop_logs:
            print(f"\n【損切りした銘柄のその後（全期間・資金投入{min(ramps)}か月の条件）】\n")
            print(f"{'ルール':<34}{'回数':>6}{'売った時の損':>12}"
                  f"{'12か月後の値動き':>16}{'売値より上':>12}{'測れた数':>10}")
            print("-" * 92)
            _pxm = {}
            for n_, (lg, pn_) in stop_logs.items():
                if id(pn_) not in _pxm:
                    _pxm[id(pn_)] = pn_.pivot_table(index="date", columns="code",
                                                    values="price")
                pxm = _pxm[id(pn_)]
                losses, rets = [], []
                for e_ in lg:
                    losses.append(e_["loss"])
                    t12 = pd.Timestamp(e_["date"]) + pd.offsets.MonthEnd(12)
                    if t12 in pxm.index and e_["code"] in pxm.columns:
                        p12 = pxm.at[t12, e_["code"]]
                        if p12 is not None and not pd.isna(p12) and e_["px"] > 0:
                            rets.append(float(p12) / e_["px"] - 1)
                lab = VARIANTS[n_]["label"]
                if not lg:
                    print(f"{lab[:32]:<34}{0:>6}{'—':>12}{'—':>16}{'—':>12}{'—':>10}")
                    continue
                ml = np.mean(losses) * 100
                if rets:
                    print(f"{lab[:32]:<34}{len(lg):>6}{ml:>11.1f}%"
                          f"{np.median(rets) * 100:>+15.1f}%"
                          f"{np.mean([r > 0 for r in rets]) * 100:>11.0f}%{len(rets):>10}")
                else:
                    print(f"{lab[:32]:<34}{len(lg):>6}{ml:>11.1f}%{'—':>16}{'—':>12}{0:>10}")
            print("\n  売った時の損 … 平均取得単価に対して、いくらで売れたか（約定のずれ込み・平均）。")
            print("  12か月後の値動き … 売った値段から12か月後の月末の株価まで（中央値・配当は含まない）。")
            print("  売値より上 … 12か月後に、売った値段より株価が上だった割合。")
            print("  　　　　　　 高いほど「売らずに持っていれば戻っていた」ことが多い。")
            print("  測れた数 … 12か月後がまだ来ていない損切りは除いている。")
        if act_logs:
            print(f"\n【利確・利回り割れで売った銘柄のその後（全期間・資金投入{min(ramps)}か月の条件）】\n")
            print(f"{'ルール':<34}{'回数':>6}{'売った時の益':>12}"
                  f"{'12か月後の値動き':>16}{'売値より上':>12}{'測れた数':>10}")
            print("-" * 92)
            _pxm2 = {}
            for n_, (lg, pn_) in act_logs.items():
                if id(pn_) not in _pxm2:
                    _pxm2[id(pn_)] = pn_.pivot_table(index="date", columns="code", values="price")
                pxm = _pxm2[id(pn_)]
                gains, rets = [], []
                for e_ in lg:
                    gains.append(e_["gain"])
                    t12 = pd.Timestamp(e_["date"]) + pd.offsets.MonthEnd(12)
                    if t12 in pxm.index and e_["code"] in pxm.columns:
                        p12 = pxm.at[t12, e_["code"]]
                        if p12 is not None and not pd.isna(p12) and e_["px"] > 0:
                            rets.append(float(p12) / e_["px"] - 1)
                lab = VARIANTS[n_]["label"]
                if rets:
                    print(f"{lab[:32]:<34}{len(lg):>6}{np.mean(gains) * 100:>+11.1f}%"
                          f"{np.median(rets) * 100:>+15.1f}%"
                          f"{np.mean([r > 0 for r in rets]) * 100:>11.0f}%{len(rets):>10}")
                else:
                    print(f"{lab[:32]:<34}{len(lg):>6}{np.mean(gains) * 100:>+11.1f}%"
                          f"{'—':>16}{'—':>12}{0:>10}")
            print("\n  売った時の益 … 平均取得単価に対して、いくらで売れたか（約定のずれ込み・平均）。")
            print("  12か月後の値動き … 売った値段から12か月後の月末の株価まで（中央値・配当は含まない）。")
            print("  売値より上 … 12か月後に、売った値段より株価が上だった割合。")
            print("  　　　　　　 高いほど「売らずに持っていれば、まだ伸びていた」ことが多い。")

        # ── 5回目：Tier の後出しを外した結果の判定（2026年10月12日に追加。結果を見る前に決めた基準）──
        #   後出しの上乗せ … 本番 − 後出しなし（その時点の首位級だけで Tier）の平均年率の差
        #   首位級を先に買う価値（後出しなし同士で比べる：後出しなし vs Tierなし）
        #     価値がある           … 後出しなしが24条件の85％以上で Tierなし を上回った
        #     確かめられない       … 勝敗が35〜65％（本番の Tier の上乗せは後出しだったと読む）
        #     むしろ損             … Tierなし が85％以上で上回った
        #     どちらとも言えない   … それ以外
        if "base_div_scr" in nm and "tier_pit_lead_scr" in nm:
            _p5 = gf.pivot_table(index=cond_cols, columns="ルール", values="年率")

            def _lab5(key):
                return VARIANTS[key]["label"] if key in VARIANTS and key in nm else None

            _Lb, _Lm = _lab5("base_div_scr"), _lab5("tier_pit_mix_scr")
            _Ll, _Ln = _lab5("tier_pit_lead_scr"), _lab5("tier_none_scr")
            _rows5 = [(x, t) for x, t in ((_Lb, "本番（いまの Tier・後出しあり）"),
                                          (_Lm, "首位級だけその時点で判定"),
                                          (_Ll, "後出しなし（その時点の首位級）"),
                                          (_Ln, "Tierなし（利回り順）"))
                      if x and x in _p5.columns]

            def _st5(lab):
                g = gf[gf["ルール"] == lab]
                return {"avg": g["年率"].mean(), "worst": g["年率"].min(),
                        "dd": g["最大下落"].max(), "dd_avg": g["最大下落"].mean(),
                        "emg": g["緊急売り"].mean()}

            _S5 = {x: _st5(x) for x, _ in _rows5}
            print("\n【Tierの後出しを外した結果】（基準は結果を見る前に決めたもの）\n")
            print(f"{'':<34}{'平均年率':>8}{'最悪の年率':>10}{'最悪の下落':>10}{'下落の平均':>10}{'減配などの売り':>14}")
            print("-" * 92)
            for x, t in _rows5:
                v = _S5[x]
                print(f"{t:<34}{v['avg']:>7.1f}%{v['worst']:>9.1f}%{v['dd']:>9.1f}%"
                      f"{v['dd_avg']:>9.1f}%{v['emg']:>12.1f}回")
            print()
            if _Lb in _S5 and _Ll in _S5:
                print(f"  後出しの上乗せ（本番 − 後出しなし） … 平均年率 {_S5[_Lb]['avg'] - _S5[_Ll]['avg']:+.2f}pt"
                      f" ／ 最悪の年率 {_S5[_Lb]['worst'] - _S5[_Ll]['worst']:+.2f}pt")
                if _Lm in _S5:
                    print(f"    うち首位級の後出し（本番 − 首位級だけ直した形） … "
                          f"{_S5[_Lb]['avg'] - _S5[_Lm]['avg']:+.2f}pt")
                    print(f"    うち累進配当・DOEのリストの分（首位級だけ直した形 − 後出しなし） … "
                          f"{_S5[_Lm]['avg'] - _S5[_Ll]['avg']:+.2f}pt")
                    print("      （リストは宣言した時期が分からず、後出しかどうかを切り分けられない）")
            if _Ll in _S5 and _Ln in _S5:
                _w = float((_p5[_Ll] > _p5[_Ln]).mean())
                _l = float((_p5[_Ln] > _p5[_Ll]).mean())
                if _w >= 0.85:
                    _v = "価値がある（本番の Tier を続ける根拠になる）"
                elif _l >= 0.85:
                    _v = "むしろ損（利回り順への切り替えを検討する）"
                elif 0.35 <= _w <= 0.65:
                    _v = "確かめられない（本番の Tier の上乗せは後出しだったと読む）"
                else:
                    _v = "どちらとも言えない"
                print(f"\n  首位級を先に買う価値（後出しなし vs Tierなし） … 後出しなしが上回った {_w * 100:.0f}％"
                      f" ／ Tierなしが上回った {_l * 100:.0f}％ → {_v}")
                print(f"    平均年率の差 {_S5[_Ll]['avg'] - _S5[_Ln]['avg']:+.2f}pt ／ 最悪の年率の差 "
                      f"{_S5[_Ll]['worst'] - _S5[_Ln]['worst']:+.2f}pt ／ 減配などの売り "
                      f"{_S5[_Ll]['emg']:.1f}回 対 {_S5[_Ln]['emg']:.1f}回")
                print("    （基準：85％以上で「価値がある」、35〜65％で「確かめられない」、"
                      "Tierなしが85％以上で「むしろ損」）")
            if _Ll in _S5:
                v = _S5[_Ll]
                print(f"\n  外に出す過去の成績の目安（後出しなし） … 平均年率 {v['avg']:.1f}％ ／ 最悪の年率 {v['worst']:.1f}％"
                      f" ／ 最悪の下落 {v['dd']:.1f}％")
                print("    ※ データに潰れた会社・上場廃止の会社が入っていないので、これでもまだ甘めの数字。")

        # ── 7回目：利回りが上がった理由・増配の実績で Tier（2026年10月12日に追加。結果を見る前に決めた基準）──
        #   相手は「後出しなし（tier_pit_lead_scr）」。24条件の85％以上で上回り、かつ「使える」なら本番に入れる候補。
        #   増配の実績で Tier … 85％以上なら、手作業の累進配当リストを増配の実績で決める形に置き換える候補。
        #                       35〜65％なら確かめられない（リストの部分の価値は確かめられないまま）。
        _r7 = [n_ for n_ in nm[1:] if n_ in ("drop_skip_pit", "tier_pit_grow_scr")]
        if _r7 and nm[0] == "tier_pit_lead_scr":
            _p7 = gf.pivot_table(index=cond_cols, columns="ルール", values="年率")
            _L0 = VARIANTS[nm[0]]["label"]
            _g0 = gf[gf["ルール"] == _L0]
            print(f"\n【7回目の判定】（相手：{_L0}。基準は結果を見る前に決めたもの）\n")
            for n_ in _r7:
                _L = VARIANTS[n_]["label"]
                if _L not in _p7.columns or _L0 not in _p7.columns:
                    continue
                _g = gf[gf["ルール"] == _L]
                _w = float((_p7[_L] > _p7[_L0]).mean())
                _l = float((_p7[_L0] > _p7[_L]).mean())
                _da = _g["年率"].mean() - _g0["年率"].mean()
                _dw = _g["年率"].min() - _g0["年率"].min()
                _dd = _g["最大下落"].max() - _g0["最大下落"].max()
                _de = _g["緊急売り"].mean() - _g0["緊急売り"].mean()
                _ok = (_da >= -0.5) and (_dw >= -1.0) and (_l < 0.85)
                if _ok and _w >= 0.85:
                    _v = "本番に入れる候補"
                elif _ok:
                    _v = "入れても損はしないが、入れる根拠もない"
                else:
                    _v = "入れない"
                print(f"  {_L}")
                print(f"    上回った {_w * 100:.0f}％（相手が上回った {_l * 100:.0f}％） ／ 平均年率 {_da:+.2f}pt ／ "
                      f"最悪の年率 {_dw:+.2f}pt ／ 最悪の下落 {_dd:+.1f}pt ／ 減配などの売り {_de:+.1f}回 → {_v}")
                if n_ == "drop_skip_pit":
                    print(f"    見送った回数（1条件あたり） … {_g['下落で見送り'].mean():.0f}回（銘柄×月）")
                if n_ == "tier_pit_grow_scr":
                    if _ok and _w >= 0.85:
                        _m = "増配した会社を先に買う考え方は本物。手作業の累進配当リストを、増配の実績で決める形に置き換える候補"
                    elif 0.35 <= _w <= 0.65:
                        _m = "確かめられない。累進配当リストの部分の価値は、後出しなしでは確かめられないまま"
                    else:
                        _m = "どちらとも言えない"
                    print(f"    読み方 … {_m}")
                print()
            print("  （基準：24条件の85％以上で上回り、かつ「使える」（平均年率−0.5pt以内・最悪の年率−1.0pt以内・"
                  "相手の勝ち85％未満）なら本番に入れる候補。使えるだけでは入れない）")
            print("  最悪の下落の差はプラスなら深い。減配などの売りの差はマイナスなら少ない。")

        # ── 8回目：S は持ち続け、A・B はうねりの中で売り買い（2026年10月12日に追加。結果を見る前に決めた基準）──
        #   相手は「首位級だけその時点で判定（tier_pit_mix_scr。全部持ち続ける）」。
        #   本番に入れる候補 … 24条件の85％以上で上回り、かつ「使える」
        #   見せるための売買として使える … 「使える」（平均年率−0.5pt以内・最悪の年率−1.0pt以内・相手の勝ち85％未満）
        #   それ以外 … 年率を払う形（差が「うねり取りの代金」）
        #   3つの取り方のうち1つだけが通ったときは、たまたまの可能性が高いとして扱う。
        _r8 = [n_ for n_ in nm[1:] if VARIANTS[n_].get("swing_entry") or VARIANTS[n_].get("swing_daily")]
        if _r8 and nm[0] == "tier_pit_mix_scr" and not _lt:
            _p8 = gf.pivot_table(index=cond_cols, columns="ルール", values="年率")
            _L0 = VARIANTS[nm[0]]["label"]
            _g0 = gf[gf["ルール"] == _L0]
            print(f"\n【うねり取りの判定】（相手：{_L0}。全部持ち続ける形。基準は結果を見る前に決めたもの）\n")
            _pass = []
            for n_ in _r8:
                _L = VARIANTS[n_]["label"]
                if _L not in _p8.columns or _L0 not in _p8.columns:
                    continue
                _g = gf[gf["ルール"] == _L]
                _w = float((_p8[_L] > _p8[_L0]).mean())
                _l = float((_p8[_L0] > _p8[_L]).mean())
                _da = _g["年率"].mean() - _g0["年率"].mean()
                _dw = _g["年率"].min() - _g0["年率"].min()
                _dd = _g["最大下落"].max() - _g0["最大下落"].max()
                _ok = (_da >= -0.5) and (_dw >= -1.0) and (_l < 0.85)
                if _ok and _w >= 0.85:
                    _v = "本番に入れる候補"
                    _pass.append(_L)
                elif _ok:
                    _v = "見せるための売買として、ほぼ代金なしで使える"
                else:
                    _v = f"年率を払う形（代金 年{-_da:.1f}pt）" if _da < 0 else "入れない"
                _yrs = []
                for (_sp, _rp), _gg in _g.groupby(["期間", "資金投入"]):
                    a_, b_ = [pd.Timestamp(x) for x in str(_sp).split("〜")]
                    _yrs.append(max((b_ - a_).days / 365.25, 0.5))
                _sw = float(_g["うねり売り"].mean() / np.mean(_yrs)) if _yrs else float("nan")
                print(f"  {_L}")
                print(f"    上回った {_w * 100:.0f}％（相手が上回った {_l * 100:.0f}％） ／ 平均年率 {_da:+.2f}pt ／ "
                      f"最悪の年率 {_dw:+.2f}pt ／ 最悪の下落 {_dd:+.1f}pt → {_v}")
                print(f"    見える売買 … 売り {(_g['売り/年'].mean() - _g0['売り/年'].mean()):+.1f}回/年 ／ "
                      f"買い {(_g['買い/年'].mean() - _g0['買い/年'].mean()):+.1f}回/年 ／ 売買のあった月 "
                      f"{_g['売買のあった月'].mean():.0f}％（相手 {_g0['売買のあった月'].mean():.0f}％） ／ "
                      f"うねりで売った回数 年{_sw:.1f}回")
                if "うねり1回の益" in _g.columns and _g["うねり1回の益"].notna().any():
                    print(f"    うねりで売った1回あたりの益 … 平均 {_g['うねり1回の益'].mean() * 100:+.1f}％"
                          "（取得単価に対して。約定のずれ込み・税引前）")
                print()
            if len(_pass) == 1 and len(_r8) >= 3:
                print("  ※ 3つの取り方のうち1つだけが通った。たまたまの可能性が高いとして扱う。")
            print("  （基準：24条件の85％以上で上回り、かつ「使える」なら本番に入れる候補。"
                  "「使える」だけなら、見せるための売買としてほぼ代金なしで使える。それ以外は年率を払う形）")

        # ── 10回目：線でうねり取り（2026年10月12日に追加。結果を見る前に決めた基準）──
        #   相手は「首位級だけその時点で判定（tier_pit_mix_scr。全部持ち続ける）」。
        #   判定は清算後（最後の月末に全部売り、約定のずれと税を払った後）の年率で行う。基準はこれまでと同じ：
        #     本番に入れる候補 … 24条件の85％以上で上回り、かつ「使える」
        #     使える（見せるための売買として、ほぼ代金なし） … 平均年率の差 −0.5pt以内・最悪の年率の差 −1.0pt以内・
        #                                                     相手の勝ちが85％未満
        #     使えない … それ以外（差がその形の代金）
        #   評価額の年率でも同じ基準で判定し、2つで分類が分かれたら「決められない」とする。
        #   1年・2年・3年の線のうち1つだけが通ったときは、たまたまの可能性が高いとして扱う。
        if _l10 and nm and nm[0] == "tier_pit_mix_scr":
            _L0 = VARIANTS[nm[0]]["label"]
            _g0 = gf[gf["ルール"] == _L0]
            _pl = gf.pivot_table(index=cond_cols, columns="ルール", values="清算後年率")
            _pm = gf.pivot_table(index=cond_cols, columns="ルール", values="評価額年率")

            def _judge10(pv, col, lab_):
                g_ = gf[gf["ルール"] == lab_]
                w_ = float((pv[lab_] > pv[_L0]).mean())
                l_ = float((pv[_L0] > pv[lab_]).mean())
                da_ = float(g_[col].mean() - _g0[col].mean())
                dw_ = float(g_[col].min() - _g0[col].min())
                ok_ = (da_ >= -0.5) and (dw_ >= -1.0) and (l_ < 0.85)
                cat_ = "本番に入れる候補" if (ok_ and w_ >= 0.85) else ("使える" if ok_ else "使えない")
                return w_, l_, da_, dw_, cat_

            def _after60(tr_, arrs_, side_):
                """うねりで売った（買った）値段から60営業日後の終値までの値動き。"""
                out_ = []
                if tr_ is None or tr_.empty:
                    return out_
                for _, x_ in tr_[tr_["side"] == side_].iterrows():
                    a_ = arrs_.get(x_["code"])
                    if a_ is None:
                        continue
                    t_ns_ = pd.Timestamp(x_["day"]).value
                    j_ = int(np.searchsorted(a_[0], t_ns_))
                    if j_ < len(a_[0]) and int(a_[0][j_]) == t_ns_ and j_ + 60 < len(a_[0]) and x_["price"] > 0:
                        out_.append(float(a_[1][j_ + 60]) / float(x_["price"]) - 1.0)
                return out_

            _p0 = pd.Timestamp(panel["date"].min()).value
            print(f"\n【10回目の判定：線でうねり取り】（相手：{_L0}＝全部持ち続ける形。基準は結果を見る前に決めたもの）\n")
            print("  判定は清算後（最後に全部売って、税と約定のずれを払った後）の年率。評価額の年率でも同じ基準で判定し、"
                  "分類が分かれたら「決められない」。\n")
            _pass_lines, _n_lines = [], 0
            for n_ in nm[1:]:
                _L = VARIANTS[n_]["label"]
                if _L not in _pl.columns or _L0 not in _pl.columns:
                    continue
                _g = gf[gf["ルール"] == _L]
                wl_, ll_, dal_, dwl_, cl_ = _judge10(_pl, "清算後年率", _L)
                wm_, lm_, dam_, dwm_, cm_ = _judge10(_pm, "評価額年率", _L)
                _fin = cl_ if cl_ == cm_ else f"決められない（清算後は「{cl_}」、評価額は「{cm_}」）"
                _tg = (VARIANTS[n_].get("swing_daily") or {}).get("tag", "")
                _is_line = DAILY_BOX_SETS.get(_tg, {}).get("method") == "hline"
                _n_lines += 1 if _is_line else 0
                if _is_line and _fin == "本番に入れる候補":
                    _pass_lines.append(_L)
                _cost = f"（代金 年{-dal_:.1f}pt）" if (_fin == "使えない" and dal_ < 0) else ""
                print(f"  {_L}")
                print(f"    清算後 … 上回った {wl_ * 100:.0f}％（相手 {ll_ * 100:.0f}％） ／ 平均年率 {dal_:+.2f}pt ／ "
                      f"最悪の年率 {dwl_:+.2f}pt → {cl_}")
                print(f"    評価額 … 上回った {wm_ * 100:.0f}％（相手 {lm_ * 100:.0f}％） ／ 平均年率 {dam_:+.2f}pt ／ "
                      f"最悪の年率 {dwm_:+.2f}pt → {cm_}")
                print(f"    判定 … {_fin}{_cost}")
                _yrs = []
                for (_sp, _rp), _gg in _g.groupby(["期間", "資金投入"]):
                    a_, b_ = [pd.Timestamp(x) for x in str(_sp).split("〜")]
                    _yrs.append(max((b_ - a_).days / 365.25, 0.5))
                _sw = float(_g["うねり売り"].mean() / np.mean(_yrs)) if _yrs else float("nan")
                _gain = _g["うねり1回の益"].mean() if _g["うねり1回の益"].notna().any() else float("nan")
                _gain_s = f"{_gain * 100:+.1f}％" if _gain == _gain else "—"
                print(f"    売買 … うねりで売った回数 年{_sw:.1f}回 ／ 1回の益 {_gain_s} ／ 売買のあった月 "
                      f"{_g['売買のあった月'].mean():.0f}％（相手 {_g0['売買のあった月'].mean():.0f}％） ／ "
                      f"現金の割合 {_g['現金比率'].mean():.0f}％（相手 {_g0['現金比率'].mean():.0f}％）")
                _arrs = (SWING_DAILY or {}).get(_tg, {})
                _s60 = _after60(swing_trades.get(n_), _arrs, "sell")
                _b60 = _after60(swing_trades.get(n_), _arrs, "buy")
                if _s60 or _b60:
                    _ms = f"平均 {np.mean(_s60) * 100:+.1f}％・10％以上上がった {np.mean(np.array(_s60) >= 0.10) * 100:.0f}％" \
                        if _s60 else "—"
                    _mb = f"平均 {np.mean(_b60) * 100:+.1f}％・10％以上下がった {np.mean(np.array(_b60) <= -0.10) * 100:.0f}％" \
                        if _b60 else "—"
                    print(f"    売った後60営業日 … {_ms}（{len(_s60)}回） ／ 買った後60営業日 … {_mb}（{len(_b60)}回）")
                if _is_line and _arrs:
                    _ds_all = np.concatenate([a_[0] for a_ in _arrs.values() if len(a_[0])])
                    _ok_all = np.concatenate([a_[4] for a_ in _arrs.values() if len(a_[0])])
                    _m = _ds_all >= _p0
                    if _m.any():
                        _yy = pd.to_datetime(_ds_all[_m]).year
                        _cov = pd.Series(_ok_all[_m].astype(float)).groupby(np.asarray(_yy)).mean() * 100
                        print("    線の中にいた割合（全銘柄・日） … "
                              + "／".join(f"{int(y_)}年 {v_:.0f}" for y_, v_ in _cov.items()) + "（％）")
                print()
            if len(_pass_lines) == 1 and _n_lines >= 3:
                print("  ※ 3つの線のうち1つだけが通った。たまたまの可能性が高いとして扱う。\n")
            _snaps = (SWING_DAILY or {}).get("_snap", {})
            if _snaps:
                _names = {u.get("code"): u.get("name", "") for u in store.get("universe", [])}
                _tn = {"h1": "1年線", "h2": "2年線", "h3": "3年線"}
                print("  いただいたチャートの3銘柄の、いまの線（最後の日の終値まで）")
                for code in LINE_EXAMPLES:
                    _rs = [(t_, _snaps.get(t_, {}).get(code)) for t_ in ("h1", "h2", "h3")]
                    _rs = [(t_, r_) for t_, r_ in _rs if r_ is not None]
                    if not _rs:
                        print(f"    {code} … データがありません")
                        continue
                    _last = _rs[0][1]
                    _dts_ = pd.Timestamp(_last["date"]).date() if _last.get("date") is not None else ""
                    print(f"    {_names.get(code, '') or code}（{code}） {_dts_} の終値 {_last['last']:,.0f}円")
                    for t_, r_ in _rs:
                        if r_["why"] == "期間が足りない":
                            print(f"      {_tn[t_]} … データの期間が足りない")
                            continue
                        _s = f"      {_tn[t_]} … "
                        if np.isfinite(r_["U"]) and np.isfinite(r_["D"]):
                            _s += f"上 {r_['U']:,.0f}・下 {r_['D']:,.0f}（触れた回数 上{r_['nU']}・下{r_['nD']}"
                            if np.isfinite(r_["width"]):
                                _s += f"・幅{r_['width'] * 100:.0f}％"
                            if r_["alts"]:
                                _s += f"・行き来{r_['alts']}回"
                            _s += "）"
                        if r_["ok"]:
                            _s += f" → 線あり・{r_.get('pos', '')}"
                            if r_.get("pos") == "いまは線の中":
                                _s += f"（売り {r_['sell_px']:,.0f}・買い {r_['buy_px']:,.0f}）"
                        else:
                            _s += f" → 線なし（{r_['why']}）"
                        print(_s)
                print()
            _src = (SWING_DAILY or {}).get("_hlsrc", {})
            if _src:
                print("  線を引いたデータ … " + "／".join(f"{k_} {v_}社" for k_, v_ in _src.items())
                      + "（高値・安値が無い銘柄は終値で代用）")
            print("  （基準：清算後で24条件の85％以上で上回り、かつ「使える」なら本番に入れる候補。「使える」だけなら、"
                  "見せるための売買としてほぼ代金なし。それ以外は年率を払う形）")
            print("  売った後60営業日 … うねりで売った値段から60営業日後の終値まで。大きなプラスは、売った後も上がり続けた（上抜けを逃した）。")
            print("  買った後60営業日 … 線の下で買った値段から60営業日後まで。10％以上下がった割合は、下の線を割った（下抜け）目安。")
            print("  （売った後・買った後は、全期間・資金投入がいちばん短い条件の明細から数えたもの）")

        # ── 11回目：増配の実績がある銘柄だけで回す（2026年10月12日に追加。結果を見る前に決めた基準）──
        #   ① 1年・2年・3年線のうねり取りを「増配の実績だけ・全部持ち続ける」と比べる（安全な銘柄の中で、うねり取りが得か）
        #   ② 4つとも、今の本番（首位級だけその時点で判定＝全部持ち続ける）と比べる（今より良くなるか）
        #   判定は清算後の年率。基準はこれまでと同じ：24条件の85％以上で上回り、かつ「使える」なら採用候補。
        #   「使える」＝平均年率の差 −0.5pt以内・最悪の年率の差 −1.0pt以内・相手の勝ちが85％未満。
        #   評価額の年率でも同じ基準で判定し、分類が分かれたら「決められない」。
        #   ①で上回らなければ、うねり取りの検証は終わりにする（11回目の前に決めたこと）。
        if getattr(args, "safe_test", False) and nm and nm[0] == "tier_pit_mix_scr" \
                and VARIANTS["safe_hold_pit"]["label"] in set(gf["ルール"]):
            _pl11 = gf.pivot_table(index=cond_cols, columns="ルール", values="清算後年率")
            _pm11 = gf.pivot_table(index=cond_cols, columns="ルール", values="評価額年率")

            def _judge11(pv, col, lab_, ref_):
                g_ = gf[gf["ルール"] == lab_]
                r_ = gf[gf["ルール"] == ref_]
                w_ = float((pv[lab_] > pv[ref_]).mean())
                l_ = float((pv[ref_] > pv[lab_]).mean())
                da_ = float(g_[col].mean() - r_[col].mean())
                dw_ = float(g_[col].min() - r_[col].min())
                ok_ = (da_ >= -0.5) and (dw_ >= -1.0) and (l_ < 0.85)
                cat_ = "採用候補" if (ok_ and w_ >= 0.85) else ("使える" if ok_ else "使えない")
                return w_, l_, da_, dw_, cat_

            def _show11(lab_, ref_, short_):
                if lab_ not in _pl11.columns or ref_ not in _pl11.columns:
                    return None
                wl_, ll_, dal_, dwl_, cl_ = _judge11(_pl11, "清算後年率", lab_, ref_)
                wm_, lm_, dam_, dwm_, cm_ = _judge11(_pm11, "評価額年率", lab_, ref_)
                fin_ = cl_ if cl_ == cm_ else f"決められない（清算後「{cl_}」・評価額「{cm_}」）"
                cost_ = f"（代金 年{-dal_:.1f}pt）" if (fin_ == "使えない" and dal_ < 0) else ""
                print(f"    {short_} … 清算後 上回った {wl_ * 100:.0f}％（相手 {ll_ * 100:.0f}％）・平均 {dal_:+.2f}pt・"
                      f"最悪 {dwl_:+.2f}pt → {cl_} ／ 評価額 平均 {dam_:+.2f}pt・最悪 {dwm_:+.2f}pt → {cm_}")
                print(f"      判定 … {fin_}{cost_}")
                return fin_

            _Lp = VARIANTS["tier_pit_mix_scr"]["label"]
            _Ls = VARIANTS["safe_hold_pit"]["label"]
            _sw11 = [("safe_line_1y_pit", "1年線"), ("safe_line_2y_pit", "2年線"), ("safe_line_3y_pit", "3年線")]
            print("\n【11回目の判定：増配の実績がある銘柄だけで回す】（基準は結果を見る前に決めたもの）\n")
            print("  判定は清算後（最後に全部売って、税と約定のずれを払った後）の年率。評価額でも同じ基準で判定し、"
                  "分類が分かれたら「決められない」。\n")
            print("  ① 増配の実績がある銘柄の中で、うねり取りは持ち続けるより得か（相手：増配の実績だけ・全部持ち続ける）")
            _res1 = {}
            for n_, sh_ in _sw11:
                if n_ in nm:
                    _res1[n_] = _show11(VARIANTS[n_]["label"], _Ls, sh_)
            print("\n  ② 今の本番より良くなるか（相手：首位級だけその時点で判定＝今の本番）")
            for n_, sh_ in [("safe_hold_pit", "全部持ち続ける")] + _sw11:
                if n_ in nm:
                    _show11(VARIANTS[n_]["label"], _Lp, sh_)
            _pass1 = [n_ for n_, v_ in _res1.items() if v_ == "採用候補"]
            _use1 = [n_ for n_, v_ in _res1.items() if v_ in ("採用候補", "使える")]
            print()
            if not _pass1:
                print("  ① で持ち続ける形を上回ったうねり取りはなし → 決めておいたとおり、うねり取りの検証はここで終わり。")
            elif len(_pass1) == 1:
                print("  ※ ① で通ったのは3つの線のうち1つだけ。たまたまの可能性が高いとして扱う。")
            if _use1 and not _pass1:
                print("  　（「使える」＝見せるための売買として、ほぼ代金なしで使える形はある）")

            # 緊急売りの内訳：どう買った銘柄で、減配・業績急変の売りになったか
            print("\n  緊急売り（減配・営業利益の2割減・赤字転落）の内訳（1条件あたりの平均）")
            for n_ in nm:
                _L = VARIANTS[n_]["label"]
                _g = gf[gf["ルール"] == _L]
                if _g.empty:
                    continue
                _r = _g["緊急の損益"].mean() if _g["緊急の損益"].notna().any() else float("nan")
                _rs = f"・売ったときの損益 平均 {_r * 100:+.0f}％" if _r == _r else ""
                _sh = "本番" if n_ == "tier_pit_mix_scr" else _L.replace("増配の実績だけ・", "増配だけ・")
                print(f"    {_sh[:22]:<22} … 計 {_g['緊急売り'].mean():.1f}回（線の下で買った {_g['緊急_線の下'].mean():.1f}／"
                      f"月末に買った {_g['緊急_月末'].mean():.1f}／最初の月に買った {_g['緊急_最初の月'].mean():.1f}）{_rs}")

            # うねり取りの中身（増配の実績だけの3つ）
            print("\n  うねり取りの中身（増配の実績だけ）")
            for n_, sh_ in _sw11:
                if n_ not in nm:
                    continue
                _g = gf[gf["ルール"] == VARIANTS[n_]["label"]]
                if _g.empty:
                    continue
                _yrs = []
                for (_sp, _rp), _gg in _g.groupby(["期間", "資金投入"]):
                    a_, b_ = [pd.Timestamp(x) for x in str(_sp).split("〜")]
                    _yrs.append(max((b_ - a_).days / 365.25, 0.5))
                _swy = float(_g["うねり売り"].mean() / np.mean(_yrs)) if _yrs else float("nan")
                _gain = _g["うねり1回の益"].mean() if _g["うねり1回の益"].notna().any() else float("nan")
                _txt = f"    {sh_} … うねりで売った 年{_swy:.1f}回・1回の益 " + (f"{_gain * 100:+.1f}％" if _gain == _gain else "—")
                _tg = VARIANTS[n_]["swing_daily"]["tag"]
                _arrs = (SWING_DAILY or {}).get(_tg, {})
                _trs = swing_trades.get(n_)
                _s60, _b60 = [], []
                if _trs is not None and not _trs.empty:
                    for _, x_ in _trs.iterrows():
                        a_ = _arrs.get(x_["code"])
                        if a_ is None or not (x_["price"] > 0):
                            continue
                        t_ns_ = pd.Timestamp(x_["day"]).value
                        j_ = int(np.searchsorted(a_[0], t_ns_))
                        if j_ < len(a_[0]) and int(a_[0][j_]) == t_ns_ and j_ + 60 < len(a_[0]):
                            (_s60 if x_["side"] == "sell" else _b60).append(float(a_[1][j_ + 60]) / float(x_["price"]) - 1.0)
                if _s60:
                    _txt += f"・売った後60営業日 平均 {np.mean(_s60) * 100:+.1f}％（{len(_s60)}回）"
                if _b60:
                    _txt += f"・買った後60営業日で10％以上下がった {np.mean(np.array(_b60) <= -0.10) * 100:.0f}％（{len(_b60)}回）"
                print(_txt)

            # 増配の実績がある銘柄は、候補（8条件と利回り3％以上）のうちどれだけか
            if "grow_pit" in panel.columns and "screen_pass" in panel.columns:
                _cand = panel[panel["screen_pass"].fillna(False).astype(bool) & (panel["yield"] >= 3.0)]
                if len(_cand):
                    _sh_g = float(_cand["grow_pit"].astype(bool).mean() * 100)
                    _n_m = _cand[_cand["grow_pit"].astype(bool)].groupby("date").size().mean()
                    print(f"\n  増配の実績がある銘柄 … 候補（8条件・利回り3％以上）の {_sh_g:.0f}％"
                          f"（1か月あたり平均 {_n_m:.0f}銘柄）")
            print("  （基準：24条件の85％以上で上回り、かつ「使える」なら採用候補。「使える」＝平均年率の差 −0.5pt以内・"
                  "最悪の年率の差 −1.0pt以内・相手の勝ち85％未満。それ以外は年率を払う形）")
            print("  緊急売りの「最初の月に買った」は、検証を始めた月に買った銘柄。資金投入を分けた条件では、最初の月に買うのは一部だけ。")

        # ── 12回目：「増配の実績だけ・2年線」を崩しにいく検証（2026年10月12日に追加。基準は結果を見る前に決めたもの）──
        #   1. ランダムな日の売り買い … 同じ銘柄・同じ回数の目安・同じ「線の中」の日のうち、ランダムに選んだ日に、
        #      その日の安値〜高値のランダムな値段で売り買いした形を20通り回す（資金投入がいちばん短い条件の6期間）。
        #      6期間の平均年率で、線の形が20通りのうち19通り以上を上回れば「通過」（線の位置に意味がある）。
        #   2. 銘柄を半分に分ける … A群・B群のそれぞれで、2年線が持ち続ける形を平均で上回れば「通過」（両方で）。
        #   3. 重ならない前半・後半の4年 … 両方で、2年線が持ち続ける形を（資金投入4通りの平均で）上回れば「通過」。
        #   4. 線の設定を少し変える … 4つのうち3つ以上で、持ち続ける形を平均で上回れば「通過」。
        #   5. 業種別の中身 … 判定には使わない。
        #   総合：1が通過し、2〜4のうち2つ以上が通過すれば「線でうねりを取る意味がある（偶然ではないと言える）」。
        if getattr(args, "robust_test", False) and nm and "safe_line_2y_pit" in nm:
            _lab = {n_: VARIANTS[n_]["label"] for n_ in nm}

            def _rows12(n_):
                return gf[gf["ルール"] == _lab[n_]] if n_ in _lab else gf.iloc[0:0]

            def _diff12(n_a, n_b):
                """n_a − n_b（清算後年率）を条件ごとに。(平均の差, n_a が上回った割合, 最悪の年の差)"""
                pv_ = gf.pivot_table(index=cond_cols, columns="ルール", values="清算後年率")
                if _lab.get(n_a) not in pv_.columns or _lab.get(n_b) not in pv_.columns:
                    return float("nan"), float("nan"), float("nan")
                d_ = pv_[_lab[n_a]] - pv_[_lab[n_b]]
                return float(d_.mean()), float((d_ > 0).mean()), float(pv_[_lab[n_a]].min() - pv_[_lab[n_b]].min())

            print("\n【12回目：「増配の実績だけ・2年線」を崩しにいく検証】（基準は結果を見る前に決めたもの。年率は清算後）\n")
            print("  土台の3つ（24条件）")
            for n_, sh_ in (("tier_pit_mix_scr", "本番"), ("safe_hold_pit", "増配だけ・持ち続ける"),
                            ("safe_line_2y_pit", "増配だけ・2年線")):
                g_ = _rows12(n_)
                if g_.empty:
                    continue
                print(f"    {sh_:<14} … 平均 {g_['清算後年率'].mean():.1f}％・最悪の年 {g_['清算後年率'].min():.1f}％・"
                      f"最悪の下落 {g_['最大下落'].max():.1f}％・売り 年{g_['売り/年'].mean():.1f}回・"
                      f"売買のあった月 {g_['売買のあった月'].mean():.0f}％")
            _d1, _w1, _x1 = _diff12("safe_line_2y_pit", "safe_hold_pit")
            _d2, _w2, _x2 = _diff12("safe_line_2y_pit", "tier_pit_mix_scr")
            print(f"    2年線 − 持ち続ける … 平均 {_d1:+.2f}pt（上回った条件 {_w1 * 100:.0f}％）／ 2年線 − 本番 … "
                  f"平均 {_d2:+.2f}pt（上回った条件 {_w2 * 100:.0f}％）")

            # 1. ランダムな日の売り買い
            _t1 = None
            _r0_ = min(ramps)
            _gr = gf[(gf["ルール"] == _lab["safe_line_2y_pit"]) & (gf["資金投入"] == _r0_)].set_index("期間")
            _gh = gf[(gf["ルール"] == _lab["safe_hold_pit"]) & (gf["資金投入"] == _r0_)].set_index("期間")
            print(f"\n  1. ランダムな日に売り買いした場合（資金投入{_r0_}か月の{len(_gr)}期間の平均年率）")
            if placebo_rows and not _gr.empty:
                _pf = pd.DataFrame(placebo_rows)
                _pf = _pf[_pf["期間"].isin(_gr.index)]
                _seed_avg = _pf.groupby("seed")["清算後年率"].mean()
                _real_avg = float(_gr.loc[_pf["期間"].unique(), "清算後年率"].mean())
                _hold_avg = float(_gh.loc[_pf["期間"].unique(), "清算後年率"].mean()) if not _gh.empty else float("nan")
                _n_above = int((_real_avg > _seed_avg).sum())
                _n_seed = int(len(_seed_avg))
                _t1 = _n_above >= _n_seed - 1
                _sell_real = float(_gr.loc[_pf["期間"].unique(), "うねり売り"].mean())
                _sell_plc = float(_pf.groupby("期間")["うねり売り"].mean().mean())
                print(f"    線で売り買い {_real_avg:.1f}％ ／ ランダムな日（{_n_seed}通り）中央値 {_seed_avg.median():.1f}％"
                      f"（{_seed_avg.min():.1f}〜{_seed_avg.max():.1f}％） ／ 持ち続ける {_hold_avg:.1f}％")
                print(f"    うねりで売った回数（1期間あたり） … 線 {_sell_real:.1f}回 ／ ランダム {_sell_plc:.1f}回")
                print(f"    線の方が上だった … {_n_seed}通り中 {_n_above}通り → {'通過' if _t1 else '通過せず'}"
                      f"（{_n_seed - 1}通り以上で通過）")
            else:
                print("    ランダムな日の結果がありません（資金投入がいちばん短い条件が無いため）")

            # 2. 銘柄を半分に分ける
            _da, _wa, _ = _diff12("safe_2y_a", "safe_hold_a")
            _db, _wb, _ = _diff12("safe_2y_b", "safe_hold_b")
            _t2 = (_da == _da and _db == _db and _da > 0 and _db > 0)
            print("\n  2. 銘柄を決まった方法で半分に分けた場合（2年線 − 持ち続ける、24条件）")
            print(f"    A群 … 平均 {_da:+.2f}pt（上回った条件 {_wa * 100:.0f}％） ／ B群 … 平均 {_db:+.2f}pt"
                  f"（上回った条件 {_wb * 100:.0f}％） → {'通過' if _t2 else '通過せず'}（両方プラスで通過）")

            # 3. 重ならない前半・後半の4年
            _w4 = [w_ for w_ in wins[:-1]]
            _t3 = None
            if len(_w4) >= 2 and _w4[0][1] <= _w4[-1][0]:
                print("\n  3. 重ならない前半・後半の4年（資金投入4通りの平均）")
                _ok3 = []
                for (a_, b_) in (_w4[0], _w4[-1]):
                    _sp = f"{a_.date()}〜{b_.date()}"
                    _g = gf[gf["期間"] == _sp]
                    _p = _g.pivot_table(index="資金投入", columns="ルール", values="清算後年率")
                    if not {_lab["safe_line_2y_pit"], _lab["safe_hold_pit"], _lab["tier_pit_mix_scr"]} <= set(_p.columns):
                        continue
                    _dd1 = _p[_lab["safe_line_2y_pit"]] - _p[_lab["safe_hold_pit"]]
                    _dd2 = _p[_lab["safe_line_2y_pit"]] - _p[_lab["tier_pit_mix_scr"]]
                    _ok3.append(float(_dd1.mean()) > 0)
                    print(f"    {_sp} … 2年線 − 持ち続ける {_dd1.mean():+.2f}pt（{int((_dd1 > 0).sum())}/{len(_dd1)}で上） ／ "
                          f"2年線 − 本番 {_dd2.mean():+.2f}pt（{int((_dd2 > 0).sum())}/{len(_dd2)}で上）")
                _t3 = len(_ok3) == 2 and all(_ok3)
                print(f"    → {'通過' if _t3 else '通過せず'}（両方で「持ち続ける」を上回れば通過）")

            # 4. 線の設定を少し変える
            print("\n  4. 線の設定を少し変えた場合（2年線 − 持ち続ける、24条件）")
            _pos4, _n4 = 0, 0
            for n_ in ("safe_2y_t15", "safe_2y_t25", "safe_2y_i05", "safe_2y_i15"):
                if n_ not in _lab:
                    continue
                _dn, _wn, _ = _diff12(n_, "safe_hold_pit")
                if _dn != _dn:
                    continue
                _n4 += 1
                _pos4 += int(_dn > 0)
                print(f"    {_lab[n_].replace('増配だけ・2年線', '').strip('（）'):<14} … 平均 {_dn:+.2f}pt"
                      f"（上回った条件 {_wn * 100:.0f}％）")
            _t4 = _n4 == 4 and _pos4 >= 3
            print(f"    → プラスは{_n4}つのうち{_pos4}つ → {'通過' if _t4 else '通過せず'}（3つ以上で通過）")

            # 5. 業種別の中身
            _ss = swing_sales.get("safe_line_2y_pit") or []
            if _ss:
                _sec = {u.get("code"): (u.get("sector") or "不明") for u in store.get("universe", [])}
                _arr2 = (SWING_DAILY or {}).get("h2", {})
                _rec = []
                for c_, d_ns_, g_, sp_ in _ss:
                    a_ = _arr2.get(c_)
                    drift = float("nan")
                    if a_ is not None and sp_ > 0:
                        j_ = int(np.searchsorted(a_[0], int(d_ns_)))
                        if j_ < len(a_[0]) and int(a_[0][j_]) == int(d_ns_) and j_ + 60 < len(a_[0]):
                            drift = float(a_[1][j_ + 60]) / float(sp_) - 1.0
                    _rec.append({"業種": _sec.get(c_, "不明"), "益": g_, "後": drift})
                _sf = pd.DataFrame(_rec)
                _agg = _sf.groupby("業種").agg(回数=("益", "size"), 益=("益", "mean"), 後=("後", "mean")).sort_values(
                    "回数", ascending=False)
                print(f"\n  5. どの業種でうねりを取ったか（全期間・資金投入{_r0_}か月、売り {len(_sf)}回）")
                for sec_, r_ in _agg.head(6).iterrows():
                    _af = f"{r_['後'] * 100:+.1f}％" if r_["後"] == r_["後"] else "—"
                    print(f"    {str(sec_)[:10]:<10} … 売り {int(r_['回数'])}回・1回の益 {r_['益'] * 100:+.1f}％・"
                          f"売った後60営業日 {_af}")
                if len(_agg) > 6:
                    _rest = _agg.iloc[6:]
                    print(f"    そのほか{len(_rest)}業種 … 売り {int(_rest['回数'].sum())}回")

            # 総合
            _others = sum(1 for t_ in (_t2, _t3, _t4) if t_)
            print("\n  総合（結果を見る前に決めた基準：1が通過し、2〜4のうち2つ以上が通過すれば「意味がある」）")
            if _t1 and _others >= 2:
                _v = "線でうねりを取る意味がある（偶然ではないと言える）"
            elif _t1:
                _v = "線の位置には意味がありそうだが、銘柄や期間・設定で崩れる（言い切れない）"
            elif _others >= 2:
                _v = "ランダムな日と区別できない。効いているのは線の位置ではなく、売り買いの仕組み（入れ替えなど）の可能性"
            else:
                _v = "偶然と区別できない"
            print(f"    1 {'○' if _t1 else '×'} ／ 2 {'○' if _t2 else '×'} ／ 3 {'○' if _t3 else '×'} ／ 4 {'○' if _t4 else '×'}"
                  f" → {_v}")
            print("  （本番に入れるかどうかの基準＝24条件の85％以上で上回る、は変えていない）")

        OUTDIR.mkdir(parents=True, exist_ok=True)
        gf.to_csv(OUTDIR / "grid.csv", index=False, encoding="utf-8-sig")
        print(f"\n書き出しました: data/grid.csv（{len(gf)}行）")
        return 0

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
                                      tax_rate=args.tax / 100.0, ramp=args.ramp))
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

    # TOPIX が無いのに指数を使うルールを選んでいたら、はっきり知らせる
    _no_tpx = ("topix" not in panel.columns) or panel["topix"].isna().all()
    if _no_tpx:
        _sel = [x.strip() for x in args.only.split(",") if x.strip()] or list(VARIANTS)
        _uses = [n for n in _sel if n in VARIANTS
                 and (VARIANTS[n].get("regime_scale") or {}).get("use") == "topix"]
        if _uses:
            log.warning("TOPIX が無いため、次のルールは判定材料がなく"
                        "通常と同じ動きになります: %s", ", ".join(_uses))

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
                       tax_rate=args.tax / 100.0, ramp=args.ramp)
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
                  tax_rate=args.tax / 100.0, ramp=args.ramp)["diag"]
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

    # TOPIX との比較
    if "topix" in panel.columns and panel["topix"].notna().any():
        tp = panel.groupby("date")["topix"].first().dropna()
        if len(tp) > 12:
            yrs_ = max((tp.index[-1] - tp.index[0]).days / 365.25, 0.5)
            tot_ = tp.iloc[-1] / tp.iloc[0] - 1
            dd_ = float((1 - tp / tp.cummax()).max())
            print(f"\n■ TOPIX（指数のみ・配当を含まず）\n")
            print(f"  総リターン {tot_*100:>7.1f}%   "
                  f"年率 {((1+tot_)**(1/yrs_)-1)*100:>5.1f}%   "
                  f"最大下落 {dd_*100:>5.1f}%")
            print("  ※ 指数は配当を含みません。ルールの数字は配当込みなので、")
            print("    公平に比べるには指数側に年2％前後を足して見てください。")

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

    ts = d0.get("tier_swaps", 0)
    _sk = []
    if d0.get("swap_skip_gain"): _sk.append(f"含み益が足りず {d0['swap_skip_gain']}回")
    if d0.get("swap_skip_pct"): _sk.append(f"まだ割安で {d0['swap_skip_pct']}回")
    if d0.get("swap_skipped_div"): _sk.append(f"権利月が近く {d0['swap_skipped_div']}回")
    if d0.get("swap_skip_rich"): _sk.append(f"新規で買えたため {d0['swap_skip_rich']}回")
    if _sk:
        print(f"\n  入れ替えの見送り： " + " ／ ".join(_sk))
    if ts or d0.get("swap_skipped_div"):
        print(f"\n  質で入れ替えた回数： {ts}回"
              + (f"（権利月が近く見送り {d0['swap_skipped_div']}回）"
                 if d0.get("swap_skipped_div") else ""))
    rs = d0.get("range_sells", 0)
    if rs:
        print(f"\n  レンジ上限で降りた回数： {rs}回")
    sb = d0.get("sector_blocked", 0)
    if sb:
        print(f"\n  業種の上限で見送った回数： {sb}回")
    rm = d0.get("regime_months", {})
    if rm:
        tot_ = sum(rm.values())
        print("\n  市場の状態の内訳： " +
              " / ".join(f"{k} {v}か月（{v/tot_*100:.0f}％）" for k, v in rm.items()))
    cut = d0.get("cut_exits", 0) if "d0" in dir() else 0
    print(f"\n■ 損益の内訳（元本 {args.capital:,.0f}円）\n")
    print(f"{'ルール':<28}{'売却益':>12}{'配当':>11}{'確定した分':>13}"
          f"{'含み益':>12}{'確定の年率':>11}")
    print("-" * 90)
    yrs_ = max((panel["date"].max() - panel["date"].min()).days / 365.25, 0.5)
    for _, x in df.iterrows():
        conf = x["確定損益"]
        cy = ((args.capital + conf) / args.capital) ** (1 / yrs_) - 1
        print(f"{x['ルール'][:26]:<28}{x['売却益']:>11,.0f}円{x['配当']:>10,.0f}円"
              f"{conf:>12,.0f}円{x['含み損益']:>11,.0f}円{cy*100:>10.1f}%")
    print("\n  売却益と配当は税引後。確定した分＝売却益＋配当。")

    # 値上がり益と配当に分ける。成績の差がどちらから来ているかを見るため。
    print(f"\n■ 値上がり益と配当に分ける（{yrs_:.1f}年間）\n")
    print(f"{'ルール':<28}{'値上がり益':>13}{'配当':>12}{'配当の割合':>11}"
          f"{'配当/年（元本比）':>16}{'評価額':>14}")
    print("-" * 96)
    for _, x in df.iterrows():
        cap_g = x["売却益"] + x["含み損益"]
        div = x["配当"]
        tot = cap_g + div
        share = div / tot * 100 if tot > 0 else float("nan")
        per_y = div / yrs_ / args.capital * 100
        print(f"{x['ルール'][:26]:<28}{cap_g:>12,.0f}円{div:>11,.0f}円"
              f"{share:>10.0f}%{per_y:>14.2f}%{args.capital + tot:>13,.0f}円")
    print("\n  値上がり益＝売却益＋含み益。配当は税引後。")
    print("  配当/年（元本比）＝1年あたりに受け取った配当が、元本の何％にあたるか。")
    print("  含み益は、まだ売っていない建玉の評価上の損益。")
    print("  ※ 売らない戦略ほど「確定した分」は小さくなる。")
    print("    これは成績が悪いのではなく、利益を確定させていないだけ。")

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

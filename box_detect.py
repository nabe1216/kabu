#!/usr/bin/env python3
"""
ボックス／チャネル検出

「多少はみ出てもボックスとして扱う」という運用に合わせた設計にしてある。
上下限は最大値・最小値ではなく分位点で引き、帯から外れた本数は割合で許容する。
1本のヒゲでボックス判定が壊れないようにするためのつくり。

判定は5つの指標を重みづけして 0〜100 のスコアにする。
  1. 効率性比      行って戻ってを繰り返しているか（トレンドでないか）
  2. 幅            狭すぎず広すぎないか
  3. タッチ回数    上下限で実際に反発しているか（ボックスと停滞を分ける核心）
  4. はみ出し率    帯からの逸脱が許容範囲に収まっているか
  5. 平均回帰性    上がったら戻る性質が統計的にあるか

傾きは減点材料にせず、種別の判定に使う。
アコムのような上昇チャネルも取引対象になるため、水平だけを正解にしない。

使い方:
    from box_detect import detect_box
    r = detect_box(close_series, high=high_series, low=low_series)
    print(r["score"], r["type"], r["position"])

単体で実行すると、合成データによる自己テストが走る:
    python box_detect.py --selftest
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# ────────────────────────────────── 設定
@dataclass
class BoxConfig:
    window: int = 120           # 判定期間（営業日）。120 ≒ 6か月
    lower_q: float = 0.08       # 下限の分位点。0にすると安値1本で決まってしまう
    upper_q: float = 0.92       # 上限の分位点
    touch_band: float = 0.18    # 上下限から幅の何％以内を「タッチ」とみなすか
    min_touches: int = 2        # 上下それぞれ最低何回の反発を求めるか（足切り用）
    ideal_crosses: int = 18     # 中心線をこの回数またいでいれば満点（実データは往復が多い）
    outside_tol: float = 0.30   # 帯幅の何割まで外側を許容するか（はみ出し許容の本体）
    max_outside: float = 0.06   # 許容線をも超えた本数の上限割合
    width_min: float = 0.06     # 幅の下限（中心価格比）。狭すぎるのは単なる停滞
    width_ideal_lo: float = 0.10
    width_ideal_hi: float = 0.30
    width_max: float = 0.45     # 幅の上限。広すぎるとボックスとは呼べない
    slope_flat: float = 0.15    # 年率換算の傾きがこの範囲なら「水平ボックス」
    slope_max: float = 0.60     # これを超える傾きはチャネルとしても対象外
    break_margin: float = 0.02  # 直近終値が帯をこの割合超えたらブレイク扱い


# ────────────────────────────────── 補助指標
def efficiency_ratio(p: np.ndarray) -> float:
    """始点から終点への正味移動 ÷ 値動きの総和。
       トレンドなら1に近く、行って戻ってを繰り返すと0に近づく。"""
    total = np.abs(np.diff(p)).sum()
    if total == 0:
        return 1.0
    return abs(p[-1] - p[0]) / total


def adf_tstat(p: np.ndarray) -> float:
    """トレンド項つきの単位根検定（ADF）の t 値。

        Δy(t) = a + c・t + b・y(t-1) + ε        の b の t 値

    b が有意に負なら「トレンドの周りで往復している」＝ボックスまたはチャネル。
    トレンド項 c を入れているのが要点で、これがないと上昇チャネルが弾かれる。
    アコムのような右上がりのボックスも対象にしたいので、この形にしている。

    目安（トレンド項つきの臨界値）:
        t ≒ -2 前後 … ランダムウォークでも普通に出る水準
        t < -3.4    … 統計的に平均回帰と言える水準
    """
    y = np.log(p)
    dy = np.diff(y)
    ylag = y[:-1]
    n = len(dy)
    if n < 30:
        return 0.0
    t_idx = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), t_idx, ylag])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    dof = n - X.shape[1]
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0
    se = math.sqrt(max(cov[2, 2], 1e-18))
    return float(np.clip(beta[2] / se, -12.0, 6.0))


def count_touches(series: np.ndarray, level: float, band: float, upper: bool) -> int:
    """上限（下限）付近に到達した「回数」を数える。連続した滞在は1回として扱う。
       スコアには使わず、最低限の反発があるかの足切りにのみ使う。"""
    if upper:
        near = series >= level - band
    else:
        near = series <= level + band
    cnt, prev = 0, False
    for x in near:
        if x and not prev:
            cnt += 1
        prev = bool(x)
    return cnt


def cross_count(resid: np.ndarray) -> int:
    """中心線をまたいだ回数。
       上下限へのタッチ回数は、分位点で線を引く以上ほぼ全銘柄で発生してしまう。
       「何度も往復しているか」を見るには、中心をまたいだ回数の方が素直に効く。"""
    sign = np.sign(resid)
    sign[sign == 0] = 1
    return int(np.sum(np.diff(sign) != 0))


def _score_width(w: float, c: BoxConfig) -> float:
    """幅は「適正域で満点、外れるほど減点」の台形で評価する"""
    if w < c.width_min or w > c.width_max:
        return 0.0
    if c.width_ideal_lo <= w <= c.width_ideal_hi:
        return 1.0
    if w < c.width_ideal_lo:
        return (w - c.width_min) / (c.width_ideal_lo - c.width_min)
    return (c.width_max - w) / (c.width_max - c.width_ideal_hi)


# ────────────────────────────────── 本体
def detect_box(close, high=None, low=None, cfg: BoxConfig | None = None) -> dict:
    cfg = cfg or BoxConfig()
    s = pd.Series(close).dropna().astype(float)
    if len(s) < max(40, cfg.window // 3):
        return {"score": 0, "type": "データ不足", "reason": f"{len(s)}本しかありません"}

    s = s.iloc[-cfg.window:]
    hi = pd.Series(high).dropna().astype(float).iloc[-len(s):] if high is not None else s
    lo = pd.Series(low).dropna().astype(float).iloc[-len(s):] if low is not None else s
    p = s.to_numpy()
    n = len(p)

    # ── 傾き（対数で見て年率換算）────────────────
    x = np.arange(n)
    slope_log, intercept = np.polyfit(x, np.log(p), 1)
    slope_annual = float(np.expm1(slope_log * 245))   # 245営業日 ≒ 1年

    # ── 上下限を分位点で引く（はみ出しを織り込む）──
    trend = np.exp(intercept + slope_log * x)         # 回帰直線
    resid = p / trend - 1.0                           # 直線からの乖離率
    lo_r = float(np.quantile(resid, cfg.lower_q))
    hi_r = float(np.quantile(resid, cfg.upper_q))

    lower_line = trend * (1 + lo_r)
    upper_line = trend * (1 + hi_r)
    center = float(np.mean(trend))
    width = float(np.mean(upper_line - lower_line) / center)

    # ── 各指標 ───────────────────────────
    er = efficiency_ratio(p)
    tstat = adf_tstat(p)

    band = (hi_r - lo_r) * cfg.touch_band
    up_touch = count_touches(resid, hi_r, band, upper=True)
    dn_touch = count_touches(resid, lo_r, band, upper=False)

    # 上下限は分位点で引いているので、単純に「帯の外」を数えると定義上ほぼ一定になる。
    # そこで帯幅の一定割合ぶん外側に許容線を引き、そこを超えた本数だけを数える。
    tol = (hi_r - lo_r) * cfg.outside_tol
    outside = float(np.mean((resid > hi_r + tol) | (resid < lo_r - tol)))

    # ── スコア化 ──────────────────────────
    sc_er = float(np.clip(1 - er / 0.28, 0, 1))            # ER 0.28以上でトレンド扱い
    sc_width = _score_width(width, cfg)
    crosses = cross_count(resid)
    sc_osc = float(np.clip(crosses / cfg.ideal_crosses, 0, 1))
    sc_out = float(np.clip(1 - outside / max(cfg.max_outside * 2, 1e-9), 0, 1))
    sc_mr = float(np.clip((-tstat - 1.8) / 1.8, 0, 1))     # t = -3.6 以下で満点

    # 配点は合成データで分離が最大になるよう調整したもの。
    # 実データでは「方向感」「往復」「収まり」が大半の銘柄で満点近くになり
    # 選別に効かなかったため、統計的な裏づけである平均回帰の比重を上げている。
    score = 100 * (0.25 * sc_er + 0.10 * sc_width +
                   0.15 * sc_osc + 0.05 * sc_out + 0.45 * sc_mr)

    # ── 種別と足切り ─────────────────────
    a = abs(slope_annual)
    if a <= cfg.slope_flat:
        btype = "水平ボックス"
    elif a <= cfg.slope_max:
        btype = "上昇チャネル" if slope_annual > 0 else "下降チャネル"
    else:
        btype = "トレンド"

    reasons = []
    if btype == "トレンド":
        reasons.append(f"傾きが大きすぎます（年率 {slope_annual*100:+.0f}％）")
    if min(up_touch, dn_touch) < cfg.min_touches:
        reasons.append(f"反発回数が不足（上{up_touch}回・下{dn_touch}回）")
    if tstat > -0.6:
        reasons.append(f"平均回帰性がまったく見られません（t={tstat:.1f}）")
    if crosses < 3:
        reasons.append(f"往復が少なく、レンジと呼べません（{crosses}回）")
    if outside > cfg.max_outside:
        reasons.append(f"はみ出しが多い（{outside*100:.0f}％）")
    if width < cfg.width_min:
        reasons.append(f"幅が狭すぎます（{width*100:.1f}％）")
    if width > cfg.width_max:
        reasons.append(f"幅が広すぎます（{width*100:.1f}％）")
    if reasons:
        score = min(score, 45)   # 足切り条件に触れたら合格圏に入れない

    # ── 現在位置とブレイク判定 ─────────────
    last = float(p[-1])
    lo_now, hi_now = float(lower_line[-1]), float(upper_line[-1])
    position = (last - lo_now) / (hi_now - lo_now) * 100 if hi_now > lo_now else 50.0

    if last > hi_now * (1 + cfg.break_margin):
        status = "上抜け"
    elif last < lo_now * (1 - cfg.break_margin):
        status = "下抜け"
    elif position <= 25:
        status = "下限圏"
    elif position >= 75:
        status = "上限圏"
    else:
        status = "レンジ中央"

    return {
        "score": round(score, 1),
        "type": btype,
        "status": status,
        "position": round(float(np.clip(position, -20, 120)), 1),
        "upper": round(hi_now, 1),
        "lower": round(lo_now, 1),
        "width_pct": round(width * 100, 1),
        "slope_annual_pct": round(slope_annual * 100, 1),
        "efficiency_ratio": round(er, 3),
        "adf_t": round(tstat, 2),
        "crosses": crosses,
        "touch_up": up_touch,
        "touch_dn": dn_touch,
        "outside_pct": round(outside * 100, 1),
        "bars": n,
        "reasons": reasons,
        "parts": {"方向感": round(sc_er, 2), "幅": round(sc_width, 2),
                  "往復": round(sc_osc, 2), "収まり": round(sc_out, 2),
                  "平均回帰": round(sc_mr, 2)},
    }


# ────────────────────────────────── 高精度判定（検証つき）
def detect_box_validated(close, high=None, low=None,
                         cfg: BoxConfig | None = None,
                         holdout: int = 40) -> dict:
    """検証期間を分けて判定する。

    通常の detect_box は、上下限を引いたのと同じデータで当てはまりを測っている。
    これでは「よく当てはまって当然」で、実際に守られるかは分からない。

    そこで期間を2つに割る。

        [────── 学習：ここで上下限を引く ──────][── 検証：守られたかを見る ──]

    学習部分で引いた線を検証期間まで延ばし、その間の終値が枠の中に
    収まっていたか、両側で反発していたか、抜けなかったかを測る。
    実際に効いたボックスだけが高い点になる。
    """
    cfg = cfg or BoxConfig()
    s = pd.Series(close).dropna().astype(float)
    if len(s) < cfg.window * 0.8:
        return {"score": 0, "type": "データ不足",
                "reason": f"{len(s)}本しかありません", "validated": False}

    s = s.iloc[-cfg.window:]
    holdout = max(20, min(holdout, len(s) // 3))
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]

    hi_s = pd.Series(high).dropna().astype(float).iloc[-len(s):] if high is not None else s
    lo_s = pd.Series(low).dropna().astype(float).iloc[-len(s):] if low is not None else s

    base = detect_box(train, hi_s.iloc[:-holdout], lo_s.iloc[:-holdout],
                      BoxConfig(**{**cfg.__dict__, "window": len(train)}))
    if base.get("score", 0) <= 0:
        return {**base, "validated": False}

    # 学習部分の回帰直線を検証期間まで延長する
    p_tr = train.to_numpy()
    x_tr = np.arange(len(p_tr))
    slope, intercept = np.polyfit(x_tr, np.log(p_tr), 1)
    resid_tr = p_tr / np.exp(intercept + slope * x_tr) - 1.0
    lo_r = float(np.quantile(resid_tr, cfg.lower_q))
    hi_r = float(np.quantile(resid_tr, cfg.upper_q))

    x_te = np.arange(len(p_tr), len(p_tr) + len(test))
    trend_te = np.exp(intercept + slope * x_te)
    p_te = test.to_numpy()
    resid_te = p_te / trend_te - 1.0

    tol = (hi_r - lo_r) * cfg.outside_tol
    inside = float(np.mean((resid_te <= hi_r + tol) & (resid_te >= lo_r - tol)))

    band = (hi_r - lo_r) * cfg.touch_band
    up_te = count_touches(resid_te, hi_r, band, upper=True)
    dn_te = count_touches(resid_te, lo_r, band, upper=False)
    both_sides = up_te >= 1 and dn_te >= 1

    last = float(p_te[-1])
    lo_now, hi_now = float(trend_te[-1] * (1 + lo_r)), float(trend_te[-1] * (1 + hi_r))
    broke = last > hi_now * (1 + cfg.break_margin) or last < lo_now * (1 - cfg.break_margin)
    position = (last - lo_now) / (hi_now - lo_now) * 100 if hi_now > lo_now else 50.0

    # 検証期間の成績で、学習時の点数を割り引く
    keep = 0.55 * min(inside / 0.85, 1.0) + 0.25 * (1.0 if both_sides else 0.0) \
         + 0.20 * (0.0 if broke else 1.0)
    score = base["score"] * keep

    reasons = list(base.get("reasons", []))
    if inside < 0.75:
        reasons.append(f"検証期間で枠から外れがち（枠内 {inside*100:.0f}％）")
    if broke:
        reasons.append("検証期間の最後にレンジを抜けています")
    if not both_sides:
        reasons.append("検証期間に両側での反発がありません")

    status = ("上抜け" if last > hi_now * (1 + cfg.break_margin) else
              "下抜け" if last < lo_now * (1 - cfg.break_margin) else
              "下限圏" if position <= 25 else
              "上限圏" if position >= 75 else "レンジ中央")

    return {**base,
            "score": round(score, 1),
            "train_score": base["score"],
            "validated": True,
            "holdout_bars": len(test),
            "inside_pct": round(inside * 100, 1),
            "touch_both": both_sides,
            "status": status,
            "position": round(float(np.clip(position, -20, 120)), 1),
            "upper": round(hi_now, 1),
            "lower": round(lo_now, 1),
            "reasons": reasons}


def detect_box_multi(close, high=None, low=None,
                     windows=(120, 240), holdout: int = 40) -> dict:
    """複数の期間で判定し、どれでも成立したものだけを高く評価する。

    6か月では横ばいでも1年で見れば下降トレンド、ということは普通に起きる。
    期間を変えても同じ結論になる銘柄だけを残すと、信頼度が上がる。

    最終スコアは各期間の最小値。ひとつでも崩れたら評価を下げる考え方。
    """
    results = {}
    for w in windows:
        r = detect_box_validated(close, high, low, BoxConfig(window=w), holdout)
        if r.get("score", 0) > 0 and r.get("validated"):
            results[w] = r

    if not results:
        return {"score": 0, "type": "判定不能", "windows_ok": 0,
                "reasons": ["どの期間でも成立しませんでした"]}

    # 種別が期間で食い違う場合は減点（例: 6か月は水平だが1年では下降）
    types = {r["type"] for r in results.values()}
    agree = len(types) == 1
    base = min(r["score"] for r in results.values())
    score = base * (1.0 if agree else 0.75) * (len(results) / len(windows))

    primary = results[min(results)]      # 短い期間を表示の基準にする
    return {**primary,
            "score": round(score, 1),
            "windows_ok": len(results),
            "windows_total": len(windows),
            "type_agree": agree,
            "per_window": {w: r["score"] for w, r in results.items()},
            "types_by_window": {w: r["type"] for w, r in results.items()}}


# ────────────────────────────────── 自己テスト
def _ou(n, mu, theta, sigma, rng, drift=0.0):
    """平均回帰過程（OU過程）。実際のレンジ相場に近い、ノイズを含む往復を作る。
       theta が大きいほど強く中心に引き戻される。"""
    x = np.zeros(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = x[i-1] + theta * (mu + drift * i - x[i-1]) + rng.normal(0, sigma)
    return x


def _synth(kind: str, n: int = 140, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    if kind == "box":                     # NTT型：水平で往復するレンジ
        p = _ou(n, 155.0, 0.055, 1.9, rng)
    elif kind == "channel":               # アコム型：右上がりのチャネル
        p = _ou(n, 450.0, 0.050, 6.0, rng, drift=0.34)
    elif kind == "cycle":                 # きれいな循環（ADFが苦手な形）
        t = np.arange(n)
        p = 155 + 9 * np.sin(t / 9.0) + rng.normal(0, 1.2, n)
    elif kind == "trend":                 # 一方向のトレンド
        p = 100 * np.cumprod(1 + rng.normal(0.004, 0.010, n))
    elif kind == "flat":                  # ほとんど動かない
        p = 200 + rng.normal(0, 0.4, n)
    else:                                 # ランダムウォーク
        p = 300 * np.cumprod(1 + rng.normal(0, 0.013, n))
    return pd.Series(p)


def selftest():
    cases = [("box", "NTT型・水平ボックス"), ("channel", "アコム型・上昇チャネル"),
             ("cycle", "きれいな循環"), ("trend", "一方向トレンド"),
             ("flat", "停滞（動かない）"), ("random", "ランダムウォーク")]
    print(f"{'ケース':<24}{'スコア':>7}{'種別':>14}{'状態':>10}{'位置':>7}  内訳")
    print("-" * 92)
    for kind, label in cases:
        r = detect_box(_synth(kind))
        parts = " ".join(f"{k}{v:.1f}" for k, v in r["parts"].items())
        print(f"{label:<24}{r['score']:>7.1f}{r['type']:>14}{r['status']:>10}"
              f"{r['position']:>6.0f}%  {parts}")
        if r["reasons"]:
            print(f"{'':<24}  除外理由: " + " / ".join(r["reasons"]))


def scan(path: str, cfg: BoxConfig, top: int = 40, min_score: float = 70.0):
    """複数銘柄をまとめて判定して並べ替える。

    入力CSVの想定列: code, date, close （あれば name, high, low）
    DIVIDEND HEIST の日次データをそのまま流し込めます。
    """
    df = pd.read_csv(path)
    need = {"code", "date", "close"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSVに {need} の列が必要です。実際の列: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])

    rows = []
    for code, g in df.groupby("code"):
        r = detect_box(g["close"], g.get("high"), g.get("low"), cfg)
        if r.get("score", 0) < min_score:
            continue
        rows.append({
            "code": code,
            "name": g["name"].iloc[-1] if "name" in g.columns else "",
            "score": r["score"], "type": r["type"], "status": r["status"],
            "position": r["position"], "lower": r["lower"], "upper": r["upper"],
            "width_pct": r["width_pct"], "slope": r["slope_annual_pct"],
            "adf_t": r["adf_t"], "crosses": r["crosses"],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        print(f"{min_score}点以上の銘柄はありませんでした。")
        return out
    # 買い場に近い順に並べる：スコアが高く、レンジ下限に近いものを上に
    out["rank"] = out["score"] - out["position"] * 0.25
    out = out.sort_values("rank", ascending=False).head(top).drop(columns="rank")

    print(f"{'コード':<8}{'銘柄':<16}{'点':>6}{'種別':>12}{'状態':>9}{'位置':>7}"
          f"{'下限':>9}{'上限':>9}{'幅':>7}")
    print("-" * 90)
    for _, x in out.iterrows():
        print(f"{str(x['code']):<8}{str(x['name'])[:14]:<16}{x['score']:>6.0f}"
              f"{x['type']:>12}{x['status']:>9}{x['position']:>6.0f}%"
              f"{x['lower']:>9.1f}{x['upper']:>9.1f}{x['width_pct']:>6.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="合成データで挙動を確認する")
    ap.add_argument("--csv", help="1銘柄の終値CSV。列名は close / high / low")
    ap.add_argument("--scan", help="複数銘柄のCSV。列は code / date / close（＋name/high/low）")
    ap.add_argument("--out", help="スキャン結果の書き出し先CSV")
    ap.add_argument("--window", type=int, default=120, help="判定期間（営業日）")
    ap.add_argument("--min-score", type=float, default=70.0, help="この点数以上のみ表示")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    cfg = BoxConfig(window=args.window)

    if args.selftest:
        selftest()
        return
    if args.scan:
        res = scan(args.scan, cfg, args.top, args.min_score)
        if args.out and not res.empty:
            res.to_csv(args.out, index=False, encoding="utf-8-sig")
            print(f"\n書き出しました: {args.out}")
        return
    if not args.csv:
        ap.error("--csv / --scan / --selftest のいずれかを指定してください")

    df = pd.read_csv(args.csv)
    r = detect_box(df["close"], df.get("high"), df.get("low"), cfg)
    for k, v in r.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

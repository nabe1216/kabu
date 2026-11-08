"""DIVUP の並走記録（2026年11月9日に作成）。

計画（決まり・判定・予想）はプロジェクトの divup_forward_plan_2026-11-08（記録を始める前に決めたもの）。
増配の予想の発表（決算短信の要約で、1回の配当の予想が前の予想・前の年度の実績より10％以上高い）のあとに買う形が、
まだ誰も見ていない2026年12月1日からの期間でも「でたらめより良い」かを記録する。

使い方（GitHub の Actions → divup-track → Run workflow）
  mode=update … 足りない日の株価（全銘柄・日付ごと）・決算短信の要約（日付ごと）・月末の銘柄一覧を取り、
                合図と成績を作り直して data/divup/ に書く（生のデータは Actions のキャッシュにだけ持ち、リポジトリには入れない）
  mode=check  … 記録を始める前の道具の確かめ（計画の8）：直近12か月（2025年12月〜2026年11月）の合図の数と例だけを出す。成績は計算しない

生のデータ（data/divup_raw.pkl.gz）は J-Quants から取ったものなので、公開のリポジトリには入れない（計画の9）。
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import yield_backtest as yb

OUT = Path("data")
RAW = OUT / "divup_raw.pkl.gz"
PUB = OUT / "divup"
JST = timezone(timedelta(hours=9))

# ── 計画（divup_forward_plan_2026-11-08）で決めたこと ──
REC_FROM = "2026-12-01"            # 記録の始まり（買う日）
REC_END1 = "2027-11-30"            # 12か月の判定に入れる最後の買う日
REC_END2 = "2028-11-30"            # 延ばしたときの最後の買う日（1回だけ）
CHECK_FROM, CHECK_TO = "2025-12-01", "2026-11-30"    # 記録の前の道具の確かめ（数と例だけ）
SUMM_FROM = "2024-10-01"           # 比べる相手（前の予想・前の年度の実績）を作るために、要約をここから取る
BARS_FROM = "2024-10-01"           # 範囲（60営業日の売買代金）と、比べる相手との間の分割・併合を読むために、株価をここから取る
TH = 0.10                          # 増配の幅の下限
H_MAIN, H_SUB = 60, 120            # 判定に使う日数・記録して見せる日数
SLIP = 0.002                       # 株の片道の費用（往復で約0.4％）
ETF_SLIP = 0.001                   # 1306 の片道の費用
TAX = 0.20315
CAP = 1e7
MIN_VA, MIN_MCAP, STALE = 5e7, 1e10, 5
MKT_OK = {"0111", "0112", "0113"}  # プライム・スタンダード・グロース
NW_LAG = 12                        # 週ごとにまとめたときの重なり（60営業日＝約12週）
T_TH, MEAN_TH = 2.2, 0.012         # 判定の基準（計画の5）
QS = ("1Q", "2Q", "3Q", "FY")
NMAX_B = 20
BUDGET_MIN = float(os.environ.get("DIVUP_BUDGET_MIN", "100"))
T_START = time.monotonic()
SUMM_KEEP = ("Code", "DiscDate", "DiscTime", "DocType", "CurPerType", "CurFYSt", "CurFYEn", "NxtFYSt", "NxtFYEn",
             "ShOutFY", "TrShFY") + tuple(f"Div{q}" for q in QS) + tuple(f"FDiv{q}" for q in QS) + tuple(f"NxFDiv{q}" for q in QS)
# 制限値幅（東証。基準値段 → 上下の値幅）。wide_lab.py と同じ表
LIMIT_TABLE = ((100, 30), (200, 50), (500, 80), (700, 100), (1000, 150), (1500, 300), (2000, 400), (3000, 500), (5000, 700),
               (7000, 1000), (10000, 1500), (15000, 3000), (20000, 4000), (30000, 5000), (50000, 7000), (70000, 10000),
               (100000, 15000), (150000, 30000), (200000, 40000), (300000, 50000), (500000, 70000), (700000, 100000),
               (1000000, 150000), (1500000, 300000), (2000000, 400000), (3000000, 500000), (5000000, 700000),
               (7000000, 1000000), (10000000, 1500000), (15000000, 3000000), (20000000, 4000000), (30000000, 5000000),
               (50000000, 7000000))


def log(msg: str) -> None:
    print(f"[{datetime.now(JST).strftime('%H:%M:%S')}] {msg}", flush=True)


def elapsed() -> float:
    return (time.monotonic() - T_START) / 60


def num(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def first(r: dict, *ks):
    for k in ks:
        if k in r and r[k] not in (None, ""):
            return r[k]
    return None


# ───────────────────────── 生のデータ ─────────────────────────
def load_raw() -> dict:
    if RAW.exists():
        with gzip.open(RAW, "rb") as f:
            return pickle.load(f)
    return {"bars": {}, "holidays": set(), "summ": {}, "summ_done": set(), "master": {}}


def save_raw(raw: dict, note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = RAW.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, RAW)
    if note:
        log(f"  {note}（{RAW.stat().st_size / 1e6:.1f}MB・開始から{elapsed():.0f}分）")


def pack_day_bars(rows: list) -> dict | None:
    """1日分の全銘柄の株価を、列ごとの配列にして持つ（その日の終値と高値と安値・売買代金・分割の係数・ストップ高安の印）。"""
    if not rows:
        return None
    codes, cols = [], {k: [] for k in ("craw", "h", "l", "va", "af", "ul", "ll")}
    for r in rows:
        c = yb.norm_code(r.get("Code") or "")
        if not c:
            continue
        codes.append(c)
        cols["craw"].append(num(first(r, "C", "Close")))
        cols["h"].append(num(first(r, "H", "High")))
        cols["l"].append(num(first(r, "L", "Low")))
        va = num(first(r, "Va", "TurnoverValue"))
        if va is None:
            vo, cr = num(first(r, "Vo", "Volume")), num(first(r, "C", "Close"))
            va = vo * cr if (vo is not None and cr is not None) else None
        cols["va"].append(va)
        cols["af"].append(num(first(r, "AdjFactor", "AdjustmentFactor")) or 1.0)
        cols["ul"].append(1.0 if str(first(r, "UL", "UpperLimit") or "0") == "1" else 0.0)
        cols["ll"].append(1.0 if str(first(r, "LL", "LowerLimit") or "0") == "1" else 0.0)
    out = {"code": np.array(codes, dtype=object)}
    for k, v in cols.items():
        out[k] = np.array([np.nan if x is None else x for x in v], dtype=np.float32)
    return out


def fetch(jq, raw: dict, today: str) -> bool:
    """足りない日の株価・要約と、月末の銘柄一覧を取る。時間切れなら取った分を保存して False。"""
    last = time.monotonic()
    days_b = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(BARS_FROM, today)]
    todo_b = [d for d in days_b if d not in raw["bars"] and d not in raw["holidays"]]
    if todo_b:
        log(f"株価（全銘柄・日付ごと）を取っています（残り {len(todo_b)}日）…")
    for d in todo_b:
        if elapsed() >= BUDGET_MIN:
            save_raw(raw, "株価を途中まで保存しました")
            return False
        time.sleep(yb.API_SLEEP)
        b = pack_day_bars(jq.get("/v2/equities/bars/daily", {"date": d}))
        if b is None:
            if d < today:
                raw["holidays"].add(d)          # 過去の平日で何もなければ休みの日
            continue                            # 今日の分がまだなければ、次の実行で取る
        raw["bars"][d] = b
        if (time.monotonic() - last) / 60 >= 15:
            save_raw(raw, "途中まで保存しました")
            last = time.monotonic()
    days_s = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(SUMM_FROM, today)]
    todo_s = [d for d in days_s if d not in raw["summ_done"] and d not in raw["holidays"]]
    if todo_s:
        log(f"決算短信の要約（日付ごと）を取っています（残り {len(todo_s)}日）…")
    for d in todo_s:
        if elapsed() >= BUDGET_MIN:
            save_raw(raw, "要約を途中まで保存しました")
            return False
        time.sleep(yb.API_SLEEP)
        rows = jq.get("/v2/fins/summary", {"date": d})
        raw["summ"][d] = [{k: r.get(k) for k in SUMM_KEEP if r.get(k) not in (None, "")} for r in rows or []]
        if d < today:
            raw["summ_done"].add(d)             # 今日の分は、あとで出る開示があるので次の実行でも取り直す
        if (time.monotonic() - last) / 60 >= 15:
            save_raw(raw, "途中まで保存しました")
            last = time.monotonic()
    cal = sorted(raw["bars"])
    mes = month_end_days(cal)
    for me in mes:
        ym = me[:7]
        if ym in raw["master"]:
            continue
        time.sleep(yb.API_SLEEP)
        rows = jq.get("/v2/equities/master", {"date": me})
        raw["master"][ym] = {yb.norm_code(r.get("Code") or ""): (str(r.get("Mkt") or ""), str(r.get("S33") or ""), str(r.get("CoName") or ""))
                             for r in rows or [] if r.get("Code")}
    save_raw(raw, "取り終えました")
    return True


def month_end_days(cal: list) -> list:
    """株価のある日の中で、月の最後の日（いまの月は、月が終わっていなければ入れない）。"""
    out = []
    for k in range(len(cal) - 1):
        if cal[k][:7] != cal[k + 1][:7]:
            out.append(cal[k])
    return out


# ───────────────────────── 行列にする ─────────────────────────
def panel(raw: dict) -> dict:
    cal = sorted(raw["bars"])
    T = len(cal)
    codes = sorted({c for d in cal for c in raw["bars"][d]["code"]})
    ci = {c: i for i, c in enumerate(codes)}
    S = len(codes)
    P = {k: np.full((T, S), np.nan) for k in ("craw", "h", "l", "va")}
    AF = np.ones((T, S))
    UL = np.zeros((T, S), dtype=bool)
    LL = np.zeros((T, S), dtype=bool)
    for t, d in enumerate(cal):
        b = raw["bars"][d]
        idx = np.array([ci[c] for c in b["code"]], dtype=int)
        for k in P:
            P[k][t, idx] = b[k]
        AF[t, idx] = np.where(np.isfinite(b["af"]) & (b["af"] > 0), b["af"], 1.0)
        UL[t, idx] = b["ul"] > 0
        LL[t, idx] = b["ll"] > 0
    traded = np.isfinite(P["craw"]) & (P["craw"] > 0)
    # 分割を調整した終値は、その日の終値を、その日までの分割・併合の係数の積で割って自分で作る。
    # 日付ごとに取った J-Quants の調整後終値は、取ったあとに起きた分割では直らないので使わない（最初の日の株数の基準にそろう）
    cum = np.cumprod(AF, axis=0)
    C = np.where(traded, P["craw"] / cum, np.nan)
    CF = pd.DataFrame(C).ffill().to_numpy()
    # ストップ高・ストップ安で引けた日：印（UL・LL）があればそれと「終値＝高値（安値）」、なければ値幅から計算する
    prev_raw = pd.DataFrame(np.where(traded, P["craw"], np.nan)).ffill().shift(1).to_numpy()
    base = prev_raw * AF                      # その日に分割があれば、前の日の終値をその日の基準に直す
    edges = np.array([e for e, _ in LIMIT_TABLE], dtype=float)
    widths = np.array([w for _, w in LIMIT_TABLE] + [10000000], dtype=float)
    with np.errstate(invalid="ignore"):
        w = widths[np.searchsorted(edges, np.nan_to_num(base, nan=0.0), side="right")]
        at_h = np.abs(P["craw"] - P["h"]) <= 1e-6 * np.maximum(P["craw"], 1.0)
        at_l = np.abs(P["craw"] - P["l"]) <= 1e-6 * np.maximum(P["craw"], 1.0)
        lim_u = (P["craw"] >= base + w - 1e-4 * np.maximum(base, 1.0)) & at_h
        lim_d = (P["craw"] <= base - w + 1e-4 * np.maximum(base, 1.0)) & at_l
    LIMU = traded & ((UL & at_h) | lim_u)
    LIMD = traded & ((LL & at_l) | lim_d)
    return {"cal": cal, "codes": codes, "ci": ci, "C": C, "CRAW": P["craw"], "VA": P["va"], "AF": AF, "CF": CF,
            "traded": traded, "LIMU": LIMU, "LIMD": LIMD}


def split_factor(Pn: dict, i: int, d1: str, d2: str) -> float:
    """d1 より後、d2 まで（d2 を含む）の分割・併合の係数の積（1:2 の分割なら 0.5）。d1・d2 は日付の文字列。"""
    cal = Pn["cal"]
    a = int(np.searchsorted(cal, d1, side="right"))
    b = int(np.searchsorted(cal, d2, side="right"))
    if b <= a:
        return 1.0
    return float(np.prod(Pn["AF"][a:b, i]))


# ───────────────────────── 合図（計画の2） ─────────────────────────
def fy_back(fy_end: str) -> str:
    """前の年度の年度末（同じ月の1年前。年と月だけで合わせる）。"""
    y, m = int(fy_end[:4]), fy_end[5:7]
    return f"{y - 1}-{m}"


def signals(raw: dict, Pn: dict) -> list:
    """開示ごとに、1回の配当ごとの予想を、前の予想（なければ前の年度の実績）と比べる。+10％以上なら合図（1つの開示で1つ）。
    分割・併合をはさむ比べは、いまの株数の基準に直した読み方と、直さない読み方のうち、変化の小さいほう（倍率が1に近いほう）を使う
    （分割を決めた開示で、すでに分割のあとの基準で予想を書く会社がある。どちらの基準か分からないとき、分割を増配と読む誤りを避ける）。"""
    by_code: dict = {}
    for d in sorted(raw["summ"]):
        for r in raw["summ"][d]:
            c = yb.norm_code(r.get("Code") or "")
            dd = str(r.get("DiscDate") or d)[:10]
            if c:
                by_code.setdefault(c, []).append((dd, str(r.get("DiscTime") or ""), r))
    out = []
    ci = Pn["ci"]
    for c, lst in by_code.items():
        lst.sort(key=lambda x: (x[0], x[1]))
        known: dict = {}      # (年度末の年月, 回) → (値, 開示日)
        actual: dict = {}
        i = ci.get(c)
        for dd, _tm, r in lst:
            fy_c = str(r.get("CurFYEn") or "")[:7]
            fy_n = str(r.get("NxtFYEn") or "")[:7]
            for q in QS:
                v = num(r.get(f"Div{q}"))
                if v is not None and fy_c:
                    actual[(fy_c, q)] = (v, dd)
            cands = []
            for q in QS:
                for fy, key in ((fy_c, f"FDiv{q}"), (fy_n, f"NxFDiv{q}")):
                    v = num(r.get(key))
                    if v is None or not fy or (fy, q) in actual:
                        continue
                    cands.append((fy, q, v))
            best = None
            for fy, q, v in cands:
                prev = known.get((fy, q))
                src = "前の予想"
                if prev is None:
                    prev = actual.get((fy_back(fy + "-01"), q))
                    src = "前の年度の実績"
                known[(fy, q)] = (v, dd)
                if prev is None or prev[0] <= 0 or v < 0:
                    continue
                f = split_factor(Pn, i, prev[1], dd) if i is not None else 1.0
                adj = prev[0] * f
                chg_adj = v / adj - 1.0 if adj > 0 else -1.0
                chg_raw = v / prev[0] - 1.0
                if abs(f - 1.0) < 1e-9 or chg_adj <= -1.0:
                    chg = chg_adj if abs(f - 1.0) < 1e-9 else chg_raw
                else:
                    chg = chg_adj if abs(np.log1p(chg_adj)) <= abs(np.log1p(chg_raw)) else chg_raw
                if chg >= TH and (best is None or chg > best["chg"]):
                    best = {"code": c, "disc": dd, "q": q, "fy": fy, "prev": prev[0], "prev_src": src, "new": v, "chg": chg,
                            "split": abs(f - 1.0) >= 1e-9}
            if best:
                out.append(best)
    return out


# ───────────────────────── 範囲（計画の3） ─────────────────────────
def universe_at(raw: dict, Pn: dict, me: str, shares: dict) -> np.ndarray:
    """月末 me の範囲（プライム・スタンダード・グロースの普通株・60営業日の売買代金の中央値5,000万円以上・時価総額100億円以上・5営業日以内に売買）。"""
    cal, codes = Pn["cal"], Pn["codes"]
    t = cal.index(me)
    S = len(codes)
    ok = np.zeros(S, dtype=bool)
    ms = raw["master"].get(me[:7]) or {}
    va = Pn["VA"][max(0, t - 59):t + 1]
    med = np.nanmedian(np.where(np.isfinite(va), va, 0.0), axis=0)
    tr = Pn["traded"][max(0, t - STALE):t + 1].any(axis=0)
    for i, c in enumerate(codes):
        m = ms.get(c)
        if not m or m[0] not in MKT_OK or m[1] in ("9999", "", "-"):
            continue
        if not tr[i] or med[i] < MIN_VA:
            continue
        sh = shares.get(c)
        if not sh:
            continue
        k = int(np.searchsorted([x[0] for x in sh], me, side="right")) - 1     # 月末の日までに出た最後の開示
        if k < 0:
            continue
        d_sh, n_sh = sh[k]
        px = Pn["CRAW"][t, i]
        if not (px == px and px > 0):
            px_c = Pn["CF"][t, i]
            if not (px_c == px_c and px_c > 0):
                continue
            px = px_c          # その日に売買がなければ、最後の値段（分割の調整のずれは小さいので、範囲の判定にだけ使う）
        mcap = px * n_sh / split_factor(Pn, i, d_sh, me)
        ok[i] = mcap >= MIN_MCAP
    return ok


def shares_table(raw: dict) -> dict:
    """銘柄ごとの（開示日, 自己株を除く株数）の並び。"""
    sh: dict = {}
    for d in sorted(raw["summ"]):
        for r in raw["summ"][d]:
            n, tr = num(r.get("ShOutFY")), num(r.get("TrShFY")) or 0.0
            c = yb.norm_code(r.get("Code") or "")
            if c and n and n - tr > 0:
                sh.setdefault(c, []).append((str(r.get("DiscDate") or d)[:10], n - tr))
    for c in sh:
        sh[c].sort()
    return sh


# ───────────────────────── 測り方（計画の4） ─────────────────────────
def exit_index(Pn: dict, i: int, t_exit: int) -> int | None:
    """売る日の番号。ストップ安で引けたら、そうでない最初の日。売買がなければ、その日までの最後に売買があった日（上場をやめたときも）。
    データがまだ足りなければ None。"""
    T = len(Pn["cal"])
    if t_exit >= T:
        return None
    t = t_exit
    while t < T and Pn["traded"][t, i] and Pn["LIMD"][t, i]:
        t += 1
    if t >= T:
        return None
    if Pn["traded"][t, i]:
        return t
    tr = np.nonzero(Pn["traded"][:t + 1, i])[0]
    return int(tr[-1]) if len(tr) else None


def ret_hold(Pn: dict, i: int, t_buy: int, H: int) -> float | None:
    te = exit_index(Pn, i, t_buy + H)
    if te is None:
        return None
    return float(Pn["C"][te, i] * (1.0 - SLIP) / (Pn["C"][t_buy, i] * (1.0 + SLIP)) - 1.0)


def nw_t(weeks: list, ys: dict, ns: dict, lag: int = NW_LAG) -> tuple:
    """週ごとの合計から、超過の平均と t（Newey-West・ずらし lag 週）。合図のない週も0として並べる。"""
    N = sum(ns.values())
    if N == 0:
        return float("nan"), float("nan")
    m = sum(ys.values()) / N
    u = np.array([ys.get(w, 0.0) - m * ns.get(w, 0) for w in weeks], dtype=float)
    s = float(np.sum(u * u))
    for L in range(1, min(lag, len(u) - 1) + 1):
        s += 2.0 * (1.0 - L / (lag + 1.0)) * float(np.sum(u[L:] * u[:-L]))
    se = np.sqrt(max(s, 0.0)) / N
    return m, (m / se if se > 0 else float("nan"))


def measure(sig: list, raw: dict, Pn: dict, frm: str, to: str, outcomes: bool = True) -> list:
    """記録の期間の合図に、範囲・買えたか・数えるか・成績（60と120営業日・範囲の全銘柄の平均・TOPIX〈1306〉）をつける。
    outcomes=False なら成績はつけない（記録の前の確かめで、成績を見ないため）。"""
    cal, ci = Pn["cal"], Pn["ci"]
    T = len(cal)
    mes = month_end_days(cal)
    sh = shares_table(raw)
    uni_cache: dict = {}
    ctl_cache: dict = {}
    i1306 = ci.get("1306")
    rows = []
    for s in sorted(sig, key=lambda x: (x["disc"], -x["chg"], x["code"])):
        t_b = int(np.searchsorted(cal, s["disc"], side="right"))      # 開示日の次の営業日
        if t_b >= T:
            continue                                                 # 買う日のデータがまだない（次の実行で入る）
        buy = cal[t_b]
        if not (frm <= buy <= to):
            continue
        me = next((m for m in reversed(mes) if m < buy), None)
        r = dict(s, buy=buy)
        i = ci.get(s["code"])
        if me is None or i is None:
            r["status"] = "範囲を作れない"
            rows.append(r)
            continue
        if me not in uni_cache:
            uni_cache[me] = universe_at(raw, Pn, me, sh)
        if not uni_cache[me][i]:
            r["status"] = "範囲の外"
        elif not Pn["traded"][t_b, i]:
            r["status"] = "買う日に売買なし"
        elif Pn["LIMU"][t_b, i]:
            r["status"] = "ストップ高で買えない"
        else:
            r["status"] = "対象"
        r["t_b"] = t_b
        rows.append(r)
    # 同じ銘柄で、数えた合図の買う日から60営業日のうちの次の合図は数えない（同じ日に2つの開示があっても1つ）
    lastc: dict = {}
    for r in rows:
        if r.get("status") != "対象":
            continue
        t0 = lastc.get(r["code"])
        if t0 is not None and r["t_b"] - t0 < H_MAIN:
            r["status"] = "60営業日のうちの2回目"
            continue
        lastc[r["code"]] = r["t_b"]
    if not outcomes:
        return rows
    sig_days: dict = {}
    for r in rows:
        if r.get("status") == "対象":
            sig_days.setdefault(r["t_b"], set()).add(ci[r["code"]])
    for r in rows:
        if r.get("status") != "対象":
            continue
        t_b, i = r["t_b"], ci[r["code"]]
        me = next(m for m in reversed(mes) if m < r["buy"])
        for H in (H_MAIN, H_SUB):
            rr = ret_hold(Pn, i, t_b, H)
            key = (t_b, H)
            if key not in ctl_cache:
                pool = np.nonzero(uni_cache[me] & Pn["traded"][t_b] & ~Pn["LIMU"][t_b])[0]
                pool = [j for j in pool if j not in sig_days.get(t_b, set())]
                vals = [ret_hold(Pn, j, t_b, H) for j in pool] if t_b + H < T else []
                vals = [v for v in vals if v is not None]
                ctl_cache[key] = (float(np.mean(vals)) if vals else None, len(vals))
            ctl, nctl = ctl_cache[key]
            tx = ret_hold(Pn, i1306, t_b, H) if i1306 is not None else None
            r[f"ret{H}"] = rr
            r[f"ctl{H}"] = ctl
            r[f"n_ctl{H}"] = nctl
            r[f"ex{H}"] = (rr - ctl) if (rr is not None and ctl is not None) else None
            r[f"tx{H}"] = (rr - tx) if (rr is not None and tx is not None) else None
    return rows


def aggregate(rows: list, frm: str, to: str, H: int) -> dict:
    """計画の4のまとめ方：全部の合図の超過の単純平均と、週ごとの合計からの t。"""
    done = [r for r in rows if r.get("status") == "対象" and frm <= r["buy"] <= to and r.get(f"ex{H}") is not None]
    pend = [r for r in rows if r.get("status") == "対象" and frm <= r["buy"] <= to and r.get(f"ex{H}") is None]
    wk = lambda d: pd.Timestamp(d).to_period("W-SUN").start_time.strftime("%Y-%m-%d")
    if not done:
        return {"n": 0, "pending": len(pend)}
    ys, ns = {}, {}
    for r in done:
        w = wk(r["buy"])
        ys[w] = ys.get(w, 0.0) + r[f"ex{H}"]
        ns[w] = ns.get(w, 0) + 1
    weeks = [x.strftime("%Y-%m-%d") for x in pd.period_range(min(ys), max(ys), freq="W-SUN").start_time]
    m, t = nw_t(weeks, ys, ns)
    tx = [r[f"tx{H}"] for r in done if r.get(f"tx{H}") is not None]
    return {"n": len(done), "pending": len(pend), "mean_ex": m, "t": t, "mean_tx": float(np.mean(tx)) if tx else None,
            "win_ctl": float(np.mean([r[f"ex{H}"] > 0 for r in done]))}


def verdict(a: dict) -> str:
    if not a.get("n"):
        return "まだ数がない"
    m, t = a["mean_ex"], a["t"]
    if m >= MEAN_TH and t >= T_TH:
        return "残っている（過去と同じくらい）"
    if m > 0 and t >= T_TH:
        return "残っている（過去より小さい）"
    if m > 0:
        return "決まらない"
    return "消えた"


# ───────────────────────── 紙の上の運用（計画の6の B・B2） ─────────────────────────
def paper(rows: list, raw: dict, Pn: dict, frm: str, H: int, top: int | None = None) -> dict:
    """全銘柄（top なら時価総額の上位 top 社）・20銘柄・H 営業日持つ・空いている資金は 1306。配当と分配金は入れない（両方とも）。"""
    cal, ci = Pn["cal"], Pn["ci"]
    T = len(cal)
    i1306 = ci.get("1306")
    if i1306 is None or frm > cal[-1]:
        return {}
    t0 = int(np.searchsorted(cal, frm, side="left"))
    cands: dict = {}
    mes = month_end_days(cal)
    sh = shares_table(raw) if top else None
    top_cache: dict = {}
    for r in rows:
        if r.get("status") not in ("対象", "60営業日のうちの2回目") or r["buy"] < frm:
            continue
        i = ci[r["code"]]
        if top:
            me = next(m for m in reversed(mes) if m < r["buy"])
            if me not in top_cache:
                top_cache[me] = top_mask(raw, Pn, me, sh, top)
            if not top_cache[me][i]:
                continue
        cands.setdefault(r["t_b"], []).append((r["chg"], i))
    px = lambda i, t: float(Pn["CF"][t, i]) if Pn["CF"][t, i] == Pn["CF"][t, i] else 0.0
    cash, loss, eu, ecost = CAP, 0.0, 0, 0.0
    pos: dict = {}
    vals = []
    nb = 0

    def realize(g):
        nonlocal cash, loss
        if g > 0:
            cash -= max(0.0, g - loss) * TAX
            loss = max(0.0, loss - g)
        else:
            loss += -g
    for t in range(t0, T):
        ep = px(i1306, t)
        for i in [k for k, p in pos.items() if p[2] <= t]:
            if Pn["traded"][t, i] and Pn["LIMD"][t, i] and t < T - 1:
                continue
            sh_, cost, _o = pos.pop(i)
            eff = px(i, t) * (1.0 - SLIP)
            cash += sh_ * eff
            realize(sh_ * eff - cost)
        cl = cands.get(t)
        if cl:
            value = cash + eu * ep + sum(p[0] * px(k, t) for k, p in pos.items())
            tgt = value / NMAX_B
            for _c, i in sorted(cl, key=lambda x: (-x[0], x[1])):
                if len(pos) >= NMAX_B or i in pos:
                    continue
                if not Pn["traded"][t, i] or Pn["LIMU"][t, i]:
                    continue
                p = float(Pn["C"][t, i]) * (1.0 + SLIP)
                if cash < tgt and eu > 0:
                    u = min(eu, int(np.ceil((tgt - cash) / (ep * (1.0 - ETF_SLIP)))))
                    c_part = ecost * u / eu
                    cash += u * ep * (1.0 - ETF_SLIP)
                    realize(u * ep * (1.0 - ETF_SLIP) - c_part)
                    eu -= u
                    ecost -= c_part
                n = int(min(tgt, cash) / p)
                if n < 1:
                    continue
                cash -= n * p
                pos[i] = [n, n * p, t + H]
                nb += 1
        value = cash + eu * ep + sum(p[0] * px(k, t) for k, p in pos.items())
        if cash > 0.01 * value and ep > 0:
            u = int(cash / (ep * (1.0 + ETF_SLIP)))
            if u > 0:
                cash -= u * ep * (1.0 + ETF_SLIP)
                eu += u
                ecost += u * ep * (1.0 + ETF_SLIP)
        vals.append((cal[t], cash + eu * ep + sum(p[0] * px(k, t) for k, p in pos.items())))
    # 1306 を最初の日に買って持ち続けた場合（最後に売ったとして税を引く）
    e0 = px(i1306, t0)
    u0 = CAP / (e0 * (1.0 + ETF_SLIP))
    e1 = px(i1306, T - 1)
    bench = u0 * e1 * (1.0 - ETF_SLIP)
    bench -= TAX * max(0.0, bench - CAP)
    liq, gain = cash, 0.0
    for k, p in pos.items():
        eff = px(k, T - 1) * (1.0 - SLIP)
        liq += p[0] * eff
        gain += p[0] * eff - p[1]
    if eu > 0:
        liq += eu * e1 * (1.0 - ETF_SLIP)
        gain += eu * e1 * (1.0 - ETF_SLIP) - ecost
    liq -= TAX * max(0.0, gain - loss)
    return {"from": cal[t0], "to": cal[-1], "value": vals[-1][1] if vals else CAP, "liq": liq, "bench_liq": bench, "buys": nb,
            "held": len(pos), "monthly": [(d, v) for d, v in vals if d in set(mes) or d == cal[-1]]}


def top_mask(raw: dict, Pn: dict, me: str, sh: dict, top: int) -> np.ndarray:
    uni = universe_at(raw, Pn, me, sh)
    t = Pn["cal"].index(me)
    mc = np.full(len(Pn["codes"]), np.nan)
    for i in np.nonzero(uni)[0]:
        c = Pn["codes"][i]
        k = int(np.searchsorted([x[0] for x in sh[c]], me, side="right")) - 1
        d_sh, n_sh = sh[c][k]
        p = Pn["CF"][t, i]
        mc[i] = p * n_sh / split_factor(Pn, i, d_sh, me)
    idx = np.nonzero(np.isfinite(mc))[0]
    keep = idx[np.argsort(-mc[idx])][:top]
    out = np.zeros(len(Pn["codes"]), dtype=bool)
    out[keep] = True
    return out


# ───────────────────────── 出力 ─────────────────────────
def names_at(raw: dict) -> dict:
    out = {}
    for ym in sorted(raw["master"]):
        for c, v in raw["master"][ym].items():
            out[c] = v[2]
    return out


def run_check(raw: dict, Pn: dict) -> int:
    """計画の8：直近12か月の合図の数と、決算短信と見比べるための例（成績は計算しない）。"""
    sig = signals(raw, Pn)
    rows = measure(sig, raw, Pn, CHECK_FROM, CHECK_TO, outcomes=False)
    nm = names_at(raw)
    print("\n" + "=" * 90)
    print("【DIVUP の並走記録：記録を始める前の道具の確かめ（計画の8）】成績は計算しない")
    print(f"  株価 {Pn['cal'][0]}〜{Pn['cal'][-1]}（{len(Pn['cal'])}営業日）・要約 {len(raw['summ'])}日・月末の銘柄一覧 {len(raw['master'])}か月")
    st = pd.Series([r["status"] for r in rows]).value_counts()
    print("  合図の内訳 … " + "・".join(f"{k} {v}" for k, v in st.items()))
    cnt = pd.Series([r["buy"][:7] for r in rows if r["status"] == "対象"]).value_counts().sort_index()
    print("  月ごとの合図（範囲の中・買えた・数えるもの）… " + "・".join(f"{k} {v}" for k, v in cnt.items())
          + f"（平均 {cnt.mean():.0f}回／月。計画の予想は約70回）")
    spl = [r for r in rows if r.get("split")]
    print(f"  分割・併合をはさんだ比べ … {len(spl)}件（直す読み方と直さない読み方のうち、変化の小さいほうで判定している）")
    rg = np.random.default_rng(1)
    ok = [r for r in rows if r["status"] == "対象"]
    pick = [ok[k] for k in sorted(rg.choice(len(ok), size=min(20, len(ok)), replace=False))] if ok else []
    print("\n  決算短信と見比べる例（20件・決まった種でえらんだもの）")
    for r in pick:
        print(f"    {r['code']} {nm.get(r['code'], '')[:14]}  開示 {r['disc']} → 買う日 {r['buy']}  {r['fy']} 年度の {r['q']}：{r['prev']:g}円（{r['prev_src']}）→ {r['new']:g}円（{r['chg'] * 100:+.0f}％）"
              + ("・分割をはさむ" if r.get("split") else ""))
    big = sorted(ok, key=lambda r: -r["chg"])[:10]
    print("\n  増配の幅が大きい10件（読み違いがないかの確かめ）")
    for r in big:
        print(f"    {r['code']} {nm.get(r['code'], '')[:14]}  開示 {r['disc']}  {r['fy']} 年度の {r['q']}：{r['prev']:g}円（{r['prev_src']}）→ {r['new']:g}円（{r['chg'] * 100:+.0f}％）")
    return 0


def run_update(raw: dict, Pn: dict) -> int:
    sig = signals(raw, Pn)
    rows = measure(sig, raw, Pn, REC_FROM, REC_END2)
    nm = names_at(raw)
    a12 = {H: aggregate(rows, REC_FROM, REC_END1, H) for H in (H_MAIN, H_SUB)}
    a24 = {H: aggregate(rows, REC_FROM, REC_END2, H) for H in (H_MAIN, H_SUB)}
    B = paper(rows, raw, Pn, REC_FROM, H_SUB)
    B2 = paper(rows, raw, Pn, REC_FROM, H_MAIN, top=1000)
    PUB.mkdir(parents=True, exist_ok=True)
    keep = ("code", "disc", "buy", "q", "fy", "prev", "prev_src", "new", "chg", "split", "status",
            "ret60", "ctl60", "ex60", "tx60", "ret120", "ctl120", "ex120", "tx120")
    pub_rows = [dict({k: r.get(k) for k in keep}, name=nm.get(r["code"], "")) for r in rows]
    with open(PUB / "signals.json", "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now(JST).isoformat(timespec="seconds"), "rows": pub_rows}, f, ensure_ascii=False, indent=0, default=str)
    status = {"generated_at": datetime.now(JST).isoformat(timespec="seconds"), "last_date": Pn["cal"][-1], "record_from": REC_FROM,
              "note": "途中の数字（判定には使わない）。判定は2027年11月30日に買った合図の60営業日後がそろってから（計画の5）",
              "counts": pd.Series([r["status"] for r in rows]).value_counts().to_dict(),
              "a12": {str(H): a12[H] for H in a12}, "a24": {str(H): a24[H] for H in a24},
              "verdict12_if_now": verdict(a12[H_MAIN]),
              "paper_B": {k: v for k, v in B.items() if k != "monthly"}, "paper_B2": {k: v for k, v in B2.items() if k != "monthly"},
              "paper_B_monthly": B.get("monthly", []), "paper_B2_monthly": B2.get("monthly", [])}
    with open(PUB / "status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=1, default=str)
    print("\n" + "=" * 90)
    print("【DIVUP の並走記録】途中の数字（判定には使わない。計画は divup_forward_plan_2026-11-08）")
    print(f"  データ {Pn['cal'][0]}〜{Pn['cal'][-1]}・記録の始まり {REC_FROM}")
    print("  合図の内訳 … " + "・".join(f"{k} {v}" for k, v in status["counts"].items()))
    for H in (H_MAIN, H_SUB):
        a = a12[H]
        if a.get("n"):
            print(f"  {H}営業日 … そろった {a['n']}件（待ち {a['pending']}件）・超過の平均 {a['mean_ex'] * 100:+.2f}％・t {a['t']:.2f}"
                  + (f"・TOPIX との差の平均 {a['mean_tx'] * 100:+.2f}％" if a.get("mean_tx") is not None else ""))
        else:
            print(f"  {H}営業日 … まだそろった合図なし（待ち {a.get('pending', 0)}件）")
    for nmB, b in (("B 全銘柄・20銘柄・120日", B), ("B2 上位1,000社・20銘柄・60日", B2)):
        if b:
            print(f"  {nmB} … 清算後 {b['liq']:,.0f}円・1306 を持ち続けた場合 {b['bench_liq']:,.0f}円・買い {b['buys']}回")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["update", "check"], default="update")
    args = ap.parse_args()
    key = os.environ.get("J_QUANTS_API_KEY")
    if not key:
        print("J_QUANTS_API_KEY が設定されていません")
        return 2
    today = datetime.now(JST).strftime("%Y-%m-%d")
    raw = load_raw()
    log(f"生のデータ … 株価 {len(raw['bars'])}日・要約 {len(raw['summ_done'])}日・銘柄一覧 {len(raw['master'])}か月")
    if not fetch(yb.JQ(key), raw, today):
        print("\n■ まだ終わっていません（データを取っている途中です）。同じ設定のまま、もう一度押してください。")
        return 3
    Pn = panel(raw)
    log(f"行列 … {len(Pn['cal'])}営業日 × {len(Pn['codes'])}銘柄")
    return run_check(raw, Pn) if args.mode == "check" else run_update(raw, Pn)


if __name__ == "__main__":
    sys.exit(main())

"""東証の全銘柄で、日本株の手法の型をまとめて確かめる道具（wide_lab）。2026年11月4日に作成。

計画と記録はプロジェクトの method_map_2026-11-04（地図と共通の決まり）と wide_lab_plan_2026-11-04（1回目の型と順位の付け方）。
今までのツール（yield_backtest.py）・キャッシュ（data/yield_backtest_cache.pkl）には触らない。データは data/wide_cache.pkl に別に持つ。

使い方（GitHub の Actions → wide-lab → Run workflow）
  mode=fetch … 東証全銘柄のデータ（各月の銘柄一覧・株価・財務の要約）を集める。時間切れなら取った分を保存して止まる（もう一度押すと続きから）
  mode=lab1  … 足りないデータがあれば先に集め、そろっていれば1回目の型をまとめて回す
"""
import argparse
import os
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import yield_backtest as yb

OUT = Path("data")
CACHE = OUT / "wide_cache.pkl"
YEARS = int(os.environ.get("WIDE_YEARS", "9"))
BUDGET_MIN = float(os.environ.get("FETCH_BUDGET_MIN", "130"))
COMPUTE_LIMIT_MIN = float(os.environ.get("COMPUTE_START_LIMIT_MIN", "140"))
SAVE_EVERY_MIN = 15
EXIT_UNFINISHED = 3
T0 = time.monotonic()

# 東証の株の市場（一部・二部・マザーズ・JASDAQ スタンダード・JASDAQ グロース・プライム・スタンダード・グロース）。
# TOKYO PRO MARKET（0105）とその他（0109）は入れない
TSE_MKT = {"0101", "0102", "0104", "0106", "0107", "0111", "0112", "0113"}
FIN_KEEP = ("Code", "DiscDate", "DiscTime", "DocType", "CurPerType", "CurPerSt", "CurPerEn", "CurFYSt", "CurFYEn",
            "Sales", "OP", "OdP", "NP", "EPS", "BPS", "TA", "Eq", "EqAR", "CFO", "CFI", "CashEq",
            "DivAnn", "Div2Q", "FDivAnn", "ShOutFY", "TrShFY", "FSales", "FOP", "FNP", "NxFSales", "NxFOP", "NxFNp", "NxFNP")

# ── 共通の決まり（method_map_2026-11-04 で、結果を見る前に決めたもの）──
MIN_LIQ = 5e7            # その月末までの60営業日の売買代金の中央値の下限（円）
MIN_MCAP = 1e10          # 時価総額の下限（円）
STALE_DAYS = 5           # 最後に売買があった日が、月末からこの営業日より前なら、その月は買えない
NMAX = 20
KEEPS = (("A", 20), ("B", 60))     # A＝20位より下で売る・B＝60位より下で売る
SLIP = 0.002             # 片道0.2％
TAX = 0.20315
CAP = 1e7
N_RAND = 20
PT_TX = 2.0              # TOPIX（税あり）を上回る幅（年率pt）
PT_RND = 2.0             # でたらめな20銘柄の平均を上回る幅（年率pt）
BEAT_MIN = 18            # でたらめな20回のうち、上回る回数の下限
SPLIT = pd.Timestamp("2022-11-01")    # 前半＝これより前・後半＝これ以後（wide_lab_plan で決めた）
START_T = {"REV24": 504}   # 2年の戻りは2年前の株価が要るので、比べる期間を2年目の終わりから始める（wide_lab_plan に書いた）
SURP_DAYS = 400          # 決算の驚き：通期の決算短信が出てから、次の通期の決算短信が出るまで（約1年）使う
LARGE_N, MID_N = 500, 1500


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def elapsed() -> float:
    return (time.monotonic() - T0) / 60.0


def load() -> dict:
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def save(store: dict, note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CACHE)
    if note:
        log(f"  {note}（{CACHE.stat().st_size / 1e6:,.0f}MB・開始から{elapsed():.0f}分）")


def num(v):
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


# ───────────────────────── データを集める ─────────────────────────
def month_ends() -> list:
    today = pd.Timestamp(date.today())
    start = today - pd.DateOffset(years=YEARS) + pd.DateOffset(days=40)
    me = pd.bdate_range(start, today, freq="BME")
    return [d for d in me if d <= today]


def fetch_months(jq, store) -> bool:
    mm = store.setdefault("months", {})
    need = [d for d in month_ends() if d.strftime("%Y-%m") not in mm]
    if need:
        log(f"各月の銘柄一覧を取っています（{len(need)}か月分）…")
    seen = {}
    for d in need:
        if elapsed() >= BUDGET_MIN:
            save(store, "銘柄一覧を途中まで保存しました")
            return False
        time.sleep(yb.API_SLEEP)
        rows = jq.get("/v2/equities/master", {"date": d.strftime("%Y-%m-%d")})
        keep = []
        for r in rows:
            mk = str(r.get("Mkt") or "")
            seen[mk] = seen.get(mk, 0) + 1
            s33 = str(r.get("S33") or "")
            # ETF・REIT など（業種が「その他」・コード9999・「-」）は入れない
            if mk not in TSE_MKT or s33 == "9999" or str(r.get("S33Nm") or "") in ("-", "その他"):
                continue
            c = yb.norm_code(r.get("Code", ""))
            if c:
                keep.append((c, mk, str(r.get("ScaleCat") or ""), str(r.get("S33Nm") or ""), str(r.get("CoName") or "")))
        mm[d.strftime("%Y-%m")] = keep
    if seen:
        log("  受け取った市場コード … " + "／".join(f"{k}:{v}" for k, v in sorted(seen.items(), key=lambda x: -x[1])[:10]))
    return True


def pack_bars(rows) -> dict | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "Date" not in df.columns:
        return None

    def col(*ks):
        k = next((x for x in ks if x in df.columns), None)
        return pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float) if k else np.full(len(df), np.nan)
    d = pd.to_datetime(df["Date"]).to_numpy(dtype="datetime64[D]")
    o = np.argsort(d, kind="stable")
    c = col("AdjC", "AdjustmentClose", "C", "Close")
    va = col("Va", "TurnoverValue")
    v = col("AdjVo", "AdjustmentVolume", "Vo", "Volume")
    if np.isnan(va).all():
        va = c * v          # 分割の調整は株価と出来高で打ち消し合うので、終値×出来高で売買代金になる
    f = col("AdjFactor", "AdjustmentFactor")
    f = np.where(np.isfinite(f) & (f > 0), f, 1.0)[o]
    fm = np.abs(f - 1.0) > 1e-12          # 分割・併合の係数は、1でない日だけ持つ（キャッシュを小さくする）
    return {"d": d[o], "c": c[o].astype(np.float32), "va": va[o].astype(np.float32), "fd": d[o][fm], "ff": f[fm]}


def prune_fins(st) -> list:
    out = []
    for r in st or []:
        out.append({k: r.get(k) for k in FIN_KEEP if r.get(k) not in (None, "")})
    return out


def fetch_all(jq, store) -> bool:
    """銘柄一覧に一度でも出た銘柄の、株価と財務の要約を取る。取り終えた銘柄は覚えておき、続きから取る。"""
    if not fetch_months(jq, store):
        return False
    codes = sorted({r[0] for v in store["months"].values() for r in v})
    bars = store.setdefault("bars", {})
    fins = store.setdefault("fins", {})
    done = set(store.get("done") or [])
    nodata = set(store.get("nodata") or [])
    todo = [c for c in codes if c not in done]
    if "etf1306" not in store:
        time.sleep(yb.API_SLEEP)
        store["etf1306"] = pack_bars(yb.fetch_bars(jq, "1306", YEARS))
    if todo:
        log(f"株価と財務を取っています（全部で {len(codes)}銘柄・残り {len(todo)}銘柄・開始から{elapsed():.0f}分）…")
    last = time.monotonic()
    for i, c in enumerate(todo, 1):
        if elapsed() >= BUDGET_MIN:
            store["done"], store["nodata"] = sorted(done), sorted(nodata)
            save(store, "株価と財務を途中まで保存しました")
            return False
        try:
            time.sleep(yb.API_SLEEP)
            b = pack_bars(yb.fetch_bars(jq, c, YEARS))
            time.sleep(yb.API_SLEEP)
            st = jq.get("/v2/fins/summary", {"code": c})
        except Exception as e:
            log(f"  取得に失敗 {c}: {e}")
            continue
        if b is None:
            nodata.add(c)
        else:
            bars[c] = b
            fins[c] = prune_fins(st)
        done.add(c)
        if i % 200 == 0:
            log(f"  {i}/{len(todo)}（開始から{elapsed():.0f}分）")
        if (time.monotonic() - last) / 60 >= SAVE_EVERY_MIN:
            store["done"], store["nodata"] = sorted(done), sorted(nodata)
            save(store, "途中まで保存しました")
            last = time.monotonic()
    store["done"], store["nodata"] = sorted(done), sorted(nodata)
    store["complete"] = True
    save(store, "全銘柄のデータがそろいました")
    return True


# ───────────────────────── 月ごとの指標 ─────────────────────────
def fac_after(fd: np.ndarray, ff: np.ndarray, t) -> float:
    """t より後（t を含まない）に起きた分割・併合の係数の積（株価と同じ基準にそろえるため）。"""
    if len(fd) == 0:
        return 1.0
    k = int(np.searchsorted(fd, np.datetime64(pd.Timestamp(t).date()), side="right"))
    return float(np.prod(ff[k:])) if k < len(ff) else 1.0


def fin_timeline(st: list) -> list:
    rows = []
    for r in st or []:
        d = str(r.get("DiscDate") or "")[:10]
        if len(d) != 10:
            continue
        rows.append((d + " " + str(r.get("DiscTime") or ""), r))
    rows.sort(key=lambda x: x[0])
    return [r for _, r in rows]


def is_fy(r) -> bool:
    return str(r.get("CurPerType") or "") == "FY" and "FinancialStatements" in str(r.get("DocType") or "")


def build(store: dict) -> dict:
    """月末ごとの値段・時価総額・売買代金・配当・各型の点数を作る。"""
    tp = store.get("etf1306")
    if not tp:
        raise SystemExit("TOPIX（1306）の株価がありません")
    cal = tp["d"]
    T = len(cal)
    codes = sorted(store["bars"])
    S = len(codes)
    ci = {c: i for i, c in enumerate(codes)}
    C = np.full((S, T), np.nan, dtype=np.float32)
    VA = np.zeros((S, T), dtype=np.float32)
    LT = np.full((S, T), -1, dtype=np.int32)     # その日までの最後に売買があった日の番号
    for c, i in ci.items():
        b = store["bars"][c]
        k = np.searchsorted(cal, b["d"])
        ok = (k < T)
        ok[ok] = cal[k[ok]] == b["d"][ok]
        C[i, k[ok]] = b["c"][ok]
        VA[i, k[ok]] = np.nan_to_num(b["va"][ok])
        idx = np.full(T, -1, dtype=np.int32)
        idx[k[ok]] = k[ok]
        LT[i] = np.maximum.accumulate(idx)
    CF = pd.DataFrame(C).ffill(axis=1).to_numpy(dtype=np.float32)
    mret = np.r_[np.nan, tp["c"][1:] / tp["c"][:-1] - 1.0]
    me_keys, me_dates, me_idx = [], [], []
    for k in sorted(store["months"]):
        y, m = map(int, k.split("-"))
        end = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)
        t = int(np.searchsorted(cal, np.datetime64(end.date()), side="right")) - 1
        if t >= 0:
            me_keys.append(k)
            me_dates.append(end)
            me_idx.append(t)
    M = len(me_dates)
    members = {}
    for j, k in enumerate(me_keys):
        for c, mk, sc, s33, nm in store["months"][k]:
            members.setdefault(c, {})[j] = (mk, s33, nm)
    sector = {c: next(iter(v.values()))[1] for c, v in members.items()}
    names = {c: next(iter(v.values()))[2] for c, v in members.items()}
    F = {}
    keys = ("EP", "BP", "SP", "CASH", "FCF", "DY", "ROE", "ROA", "EQR", "SGROW", "AGROW", "SHR", "SURP", "NXG")
    for k in keys:
        F[k] = np.full((S, M), np.nan)
    MCAP = np.full((S, M), np.nan)
    DIV = np.zeros((S, M))
    me_np = np.array([np.datetime64(d.date()) for d in me_dates])
    me_ym = [d.strftime("%Y-%m") for d in me_dates]
    ym_j = {k: j for j, k in enumerate(me_ym)}
    for c, i in ci.items():
        b = store["bars"][c]
        fd, ff = b["fd"], b["ff"]
        tl = fin_timeline(store["fins"].get(c))
        if not tl:
            continue
        dd = np.array([np.datetime64(str(r["DiscDate"])[:10]) for r in tl])
        # 配当（受け取る側）：通期の決算短信の実績。中間は期末の6か月前の月、期末は期末の月に、前の月末から持っていれば受け取る
        seen_fy = set()
        for r in tl:
            if not is_fy(r):
                continue
            fy_end = str(r.get("CurFYEn") or "")[:10]
            if len(fy_end) != 10 or fy_end in seen_fy:
                continue
            seen_fy.add(fy_end)
            ann, q2 = num(r.get("DivAnn")), num(r.get("Div2Q"))
            if not ann or ann <= 0:
                continue
            q2 = q2 if (q2 is not None and 0 <= q2 <= ann) else 0.0
            e = pd.Timestamp(fy_end)
            for amt, rec in ((ann - q2, e), (q2, e - pd.DateOffset(months=6))):
                if amt > 0:
                    j = ym_j.get((rec + pd.offsets.MonthEnd(0)).strftime("%Y-%m"))
                    if j is not None:
                        DIV[i, j] += amt * fac_after(fd, ff, rec + pd.offsets.MonthEnd(0))
        for j in range(M):
            k = int(np.searchsorted(dd, me_np[j], side="left")) - 1      # 月末の日より前に出た開示だけ
            if k < 0:
                continue
            m_date = me_dates[j]
            fy = [r for r in tl[:k + 1] if is_fy(r)]
            fy_by_end = {}
            for r in fy:
                fy_by_end[str(r.get("CurFYEn") or "")[:10]] = r
            fy_list = [fy_by_end[e] for e in sorted(fy_by_end)]
            # 株数（自己株を除く）：直近の開示。いまの株数の基準へ
            sh_r = next((r for r in reversed(tl[:k + 1]) if num(r.get("ShOutFY"))), None)
            if sh_r is None:
                continue
            sh = num(sh_r.get("ShOutFY")) - (num(sh_r.get("TrShFY")) or 0.0)
            sh_cur = sh / fac_after(fd, ff, str(sh_r["DiscDate"])[:10])
            px = CF[i, me_idx[j]]
            if not (px == px and px > 0 and sh_cur > 0):
                continue
            mcap = float(px) * sh_cur
            MCAP[i, j] = mcap
            eq_r = next((r for r in reversed(tl[:k + 1]) if num(r.get("Eq")) is not None), None)
            ta_r = next((r for r in reversed(tl[:k + 1]) if num(r.get("TA")) is not None), None)
            cash_r = next((r for r in reversed(tl[:k + 1]) if num(r.get("CashEq")) is not None), None)
            eq = num(eq_r.get("Eq")) if eq_r else None
            ta = num(ta_r.get("TA")) if ta_r else None
            if eq is not None:
                F["BP"][i, j] = eq / mcap
            if cash_r is not None:
                F["CASH"][i, j] = num(cash_r.get("CashEq")) / mcap
            if eq is not None and ta and ta > 0:
                F["EQR"][i, j] = eq / ta
            if fy_list:
                r = fy_list[-1]
                np_, sales, op = num(r.get("NP")), num(r.get("Sales")), num(r.get("OP"))
                if np_ is not None:
                    F["EP"][i, j] = np_ / mcap
                    if eq and eq > 0:
                        F["ROE"][i, j] = np_ / eq
                if sales is not None:
                    F["SP"][i, j] = sales / mcap
                if op is not None and ta and ta > 0:
                    F["ROA"][i, j] = op / ta
                cfo, cfi = num(r.get("CFO")), num(r.get("CFI"))
                if cfo is not None and cfi is not None:
                    F["FCF"][i, j] = (cfo + cfi) / mcap
                ann = num(r.get("DivAnn"))
                if ann is not None:
                    F["DY"][i, j] = ann * fac_after(fd, ff, str(r["DiscDate"])[:10]) / float(px)
                nx = num(r.get("NxFNp")) if num(r.get("NxFNp")) is not None else num(r.get("NxFNP"))
                if nx is not None and np_ and np_ > 0:
                    F["NXG"][i, j] = nx / np_ - 1.0
                # 決算の驚き：通期の実績 ÷ その期の最後の会社予想（決算短信が出てから SURP_DAYS 日まで）
                fy_end = str(r.get("CurFYEn") or "")[:10]
                r_d = str(r["DiscDate"])[:10]
                if np_ is not None and (m_date - pd.Timestamp(r_d)).days <= SURP_DAYS:
                    fc = None
                    for q in tl[:k + 1]:
                        if str(q["DiscDate"])[:10] >= r_d:
                            break
                        if str(q.get("CurFYEn") or "")[:10] == fy_end and num(q.get("FNP")):
                            fc = num(q.get("FNP"))
                    if fc and fc > 0:
                        F["SURP"][i, j] = np_ / fc - 1.0
                if len(fy_list) >= 2:
                    p = fy_list[-2]
                    s0, s1 = num(p.get("Sales")), sales
                    if s0 and s0 > 0 and s1 is not None:
                        F["SGROW"][i, j] = s1 / s0 - 1.0
                    a0, a1 = num(p.get("TA")), num(r.get("TA"))
                    if a0 and a0 > 0 and a1 is not None:
                        F["AGROW"][i, j] = -(a1 / a0 - 1.0)
                    h0, h1 = num(p.get("ShOutFY")), num(r.get("ShOutFY"))
                    if h0 and h1:
                        h0 = (h0 - (num(p.get("TrShFY")) or 0.0)) / fac_after(fd, ff, str(p["DiscDate"])[:10])
                        h1 = (h1 - (num(r.get("TrShFY")) or 0.0)) / fac_after(fd, ff, str(r["DiscDate"])[:10])
                        if h0 > 0:
                            F["SHR"][i, j] = -(h1 / h0 - 1.0)
    # 値動きから作る型
    for k in ("MOM12", "MOM6", "HI52", "REV1", "REV24", "LOWVOL", "LOWBETA", "LOTTERY", "ILLIQ", "SMALL"):
        F[k] = np.full((S, M), np.nan)
    LIQ = np.full((S, M), np.nan)
    TRADE_OK = np.zeros((S, M), dtype=bool)
    for j, t in enumerate(me_idx):
        LIQ[:, j] = np.median(VA[:, max(0, t - 59):t + 1], axis=1)
        TRADE_OK[:, j] = (LT[:, t] >= 0) & (t - LT[:, t] <= STALE_DAYS)
        px = CF[:, t]
        if t >= 252:
            F["MOM12"][:, j] = CF[:, t - 21] / CF[:, t - 252] - 1.0
            w = CF[:, t - 251:t + 1]
            F["HI52"][:, j] = px / np.nanmax(w, axis=1)
            r = w[:, 1:] / w[:, :-1] - 1.0
            mr = mret[t - 250:t + 1]
            ok = np.isfinite(r).sum(axis=1) >= 200
            sd = np.nanstd(r, axis=1)
            F["LOWVOL"][:, j] = np.where(ok, -sd, np.nan)
            rm = np.where(np.isfinite(r), r, np.nan)
            mm_ = np.nanmean(mr)
            cov = np.nanmean((rm - np.nanmean(rm, axis=1, keepdims=True)) * (mr - mm_), axis=1)
            var = np.nanvar(mr)
            F["LOWBETA"][:, j] = np.where(ok & (var > 0), -cov / var, np.nan)
            F["LOTTERY"][:, j] = -np.nanmax(r[:, -21:], axis=1)
        if t >= 126:
            F["MOM6"][:, j] = CF[:, t - 21] / CF[:, t - 126] - 1.0
        if t >= 21:
            F["REV1"][:, j] = -(px / CF[:, t - 21] - 1.0)
        if t >= 504:
            F["REV24"][:, j] = -(px / CF[:, t - 504] - 1.0)
        F["ILLIQ"][:, j] = -(VA[:, max(0, t - 20):t + 1].sum(axis=1) / MCAP[:, j])
        F["SMALL"][:, j] = -MCAP[:, j]
    UNI = np.zeros((S, M), dtype=bool)
    for c, i in ci.items():
        for j in members.get(c, {}):
            if j < M:
                UNI[i, j] = True
    UNI &= TRADE_OK & (LIQ >= MIN_LIQ) & (MCAP >= MIN_MCAP)
    fin = np.array([sector.get(c, "") in yb.FIN_SECTORS for c in codes])
    for k in ("CASH", "FCF", "ROA", "EQR", "AGROW"):
        F[k][fin, :] = np.nan
    PXM = np.stack([CF[:, t] for t in me_idx], axis=1).astype(float)
    return {"codes": codes, "names": names, "sector": sector, "dates": me_dates, "me_idx": me_idx, "F": F, "UNI": UNI,
            "MCAP": MCAP, "LIQ": LIQ, "PXM": PXM, "DIV": DIV, "cal": cal, "tp": tp}


# ───────────────────────── 実際の運用の形のエンジン ─────────────────────────
def sim(ranks: list, PXM: np.ndarray, DIV: np.ndarray, j0: int, keep: int, nmax: int = NMAX, cap: float = CAP,
        slip: float = SLIP, tax: float = TAX):
    """月末ごとに、順位の上から買い、keep 位より下になったら売る。1株単位・税（繰り越しの損あり）・配当は受け取るときに税。
    ranks[j] … その月末の順位（銘柄の番号の並び。範囲の中だけ）。返す値：j0 から最後までの月末の評価額、最後に全部売って税を払った額、診断。"""
    cash = float(cap)
    pos: dict = {}
    loss = 0.0
    vals = []
    st = {"n_buy": 0, "n_sell": 0, "tax": 0.0}
    M = PXM.shape[1]

    def p_(i, j):
        v = PXM[i, j]
        return float(v) if (v == v and v > 0) else 0.0

    for j in range(j0, M):
        for i, (sh, _c) in pos.items():
            d = DIV[i, j]
            if d > 0:
                g = sh * d
                cash += g * (1.0 - tax)
                st["tax"] += g * tax
        rk = ranks[j]
        rp = {i: k for k, i in enumerate(rk)}
        for i in list(pos):
            if i not in rp or rp[i] >= keep:
                sh, cost = pos.pop(i)
                eff = p_(i, j) * (1.0 - slip)
                cash += sh * eff
                g = sh * eff - cost
                if g > 0:
                    tx = max(0.0, g - loss) * tax
                    loss = max(0.0, loss - g)
                    cash -= tx
                    st["tax"] += tx
                else:
                    loss += -g
                st["n_sell"] += 1
        value = cash + sum(sh * p_(i, j) for i, (sh, _c) in pos.items())
        tgt = value / nmax
        for i in rk[:nmax]:
            if len(pos) >= nmax:
                break
            if i in pos:
                continue
            p = p_(i, j)
            if not p:
                continue
            px = p * (1.0 + slip)
            sh = int(min(tgt, cash) / px)
            if sh < 1:
                continue
            cash -= sh * px
            pos[i] = [sh, sh * px]
            st["n_buy"] += 1
        vals.append(cash + sum(sh * p_(i, j) for i, (sh, _c) in pos.items()))
    liq, gain = cash, 0.0
    for i, (sh, cost) in pos.items():
        eff = p_(i, M - 1) * (1.0 - slip)
        liq += sh * eff
        gain += sh * eff - cost
    liq -= tax * max(0.0, gain - loss)
    return np.array(vals), liq, st


def ranks_from(score: np.ndarray, uni: np.ndarray) -> list:
    """月末ごとに、範囲の中で点数の大きい順に並べた銘柄の番号。点数のない銘柄は入れない。"""
    out = []
    for j in range(score.shape[1]):
        ok = uni[:, j] & np.isfinite(score[:, j])
        idx = np.nonzero(ok)[0]
        out.append(list(idx[np.argsort(-score[idx, j], kind="stable")]))
    return out


def topix_taxed(D: dict, j0: int) -> tuple:
    tp, cal = D["tp"], D["cal"]
    dist = [(np.datetime64(pd.Timestamp(d).date()), a / 100.0) for d, a in yb.ETF_DIST.get("1306", ())]
    px = [float(tp["c"][t]) for t in D["me_idx"]]
    units = CAP / (px[j0] * (1.0 + SLIP))
    cost = CAP
    vals = [units * px[j0]]
    for j in range(j0 + 1, len(px)):
        d0, d1 = cal[D["me_idx"][j - 1]], cal[D["me_idx"][j]]
        dd = sum(a for d, a in dist if d0 < d <= d1)
        if dd > 0:
            add = units * dd * (1.0 - TAX)
            units += add / (px[j] * (1.0 + SLIP))
            cost += add
        vals.append(units * px[j])
    fin = units * px[-1] * (1.0 - SLIP)
    return np.array(vals), fin - TAX * max(0.0, fin - cost)


def half_stats(vals: np.ndarray, dates: list) -> dict:
    s = pd.Series(vals, index=pd.DatetimeIndex(dates))
    r = s.pct_change().iloc[1:]
    out = {}
    for h, m in (("前半", r.index < SPLIT), ("後半", r.index >= SPLIT), ("通し", np.ones(len(r), dtype=bool))):
        out[h] = yb.rev2_stats(r[m])
    return out


# 1回目の型（wide_lab_plan_2026-11-04 で、結果を見る前に決めたもの）。点数は大きいほど上
LAB1 = (("EP", "利益に対する安さ（純利益÷時価総額）"), ("BP", "純資産に対する安さ"), ("SP", "売上に対する安さ"),
        ("CASH", "現金に対する安さ（金融を除く）"), ("FCF", "稼ぐ現金に対する安さ（金融を除く）"), ("DY", "配当利回り（実績）"),
        ("MOM12", "12か月の勢い（直近1か月を除く）"), ("MOM6", "6か月の勢い（直近1か月を除く）"), ("HI52", "52週高値への近さ"),
        ("REV1", "1か月の戻り（下げた順）"), ("REV24", "2年の戻り（下げた順）"),
        ("ROE", "自己資本の利益率"), ("ROA", "総資産の営業利益率（金融を除く）"), ("EQR", "自己資本比率（金融を除く）"),
        ("SGROW", "売上の伸び"), ("AGROW", "資産の増え方の小ささ（金融を除く）"), ("SHR", "株数の減り（自社株買いなど）"),
        ("SURP", "決算が予想を上回る（次の決算まで）"), ("NXG", "来期予想の伸び"),
        ("LOWVOL", "値動きの小ささ"), ("LOWBETA", "市場との連動の小ささ"), ("LOTTERY", "宝くじ株を避ける（1日の最大の上げが小さい）"),
        ("ILLIQ", "売買の少なさ"), ("SMALL", "小型株（時価総額の小さい順）"))


def lab1(store: dict) -> int:
    log("1回目の型をまとめて回します（指標を作っています）…")
    D = build(store)
    dates, UNI, F, PXM, DIV = D["dates"], D["UNI"], D["F"], D["PXM"], D["DIV"]
    M = len(dates)
    j0 = next((j for j, t in enumerate(D["me_idx"]) if t >= 252), None)
    if j0 is None or M - j0 < 36:
        print("\n■ 比べる期間が短すぎます。")
        return 1
    ev = dates[j0:]
    log(f"  比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月）・範囲の銘柄数 平均 {UNI[:, j0:].sum(axis=0).mean():.0f}")
    base = {}

    def baseline(jx: int):
        """比べる相手（TOPIX（税あり）と、でたらめな20銘柄を20回）。でたらめは、銘柄ごとに決めた点数を期間のあいだずっと使う。"""
        if jx not in base:
            evx = dates[jx:]
            tv, tl = topix_taxed(D, jx)
            rng = np.random.default_rng(20261104)
            rnd = {nm: [] for nm, _ in KEEPS}
            for _k in range(N_RAND):
                sc_r = np.repeat(rng.random(UNI.shape[0])[:, None], M, axis=1)
                rk_r = ranks_from(sc_r, UNI)
                for nm, kp in KEEPS:
                    v, _l, _s = sim(rk_r, PXM, DIV, jx, kp)
                    rnd[nm].append(half_stats(v, evx))
            base[jx] = (evx, half_stats(tv, evx), tl, rnd)
        return base[jx]
    _ev, ST_TX, TXliq, RND = baseline(j0)
    # 大きさの区分（その月末の範囲の中の時価総額の順位）
    rank_mc = np.full(UNI.shape, np.inf)
    for j in range(M):
        idx = np.nonzero(UNI[:, j])[0]
        o = idx[np.argsort(-D["MCAP"][idx, j])]
        rank_mc[o, j] = np.arange(len(o))
    buckets = {"大型": rank_mc < LARGE_N, "中型": (rank_mc >= LARGE_N) & (rank_mc < MID_N), "小型": rank_mc >= MID_N}
    res = {}
    for key, lab in LAB1:
        sc = F[key]
        jf = j0 if key not in START_T else next((j for j, t in enumerate(D["me_idx"]) if t >= START_T[key]), M)
        if M - jf < 36:
            log(f"  {key} … 比べる期間が短すぎるので飛ばします")
            continue
        evf, tx_f, txliq_f, rnd_f = baseline(jf)
        cov = [float((UNI[:, j] & np.isfinite(sc[:, j])).sum() / max(1, UNI[:, j].sum())) for j in range(jf, M)]
        r = {"lab": lab, "cov": cov, "low": sum(1 for x in cov if x < 0.5), "start": evf[0], "n": len(evf),
             "yrs": max((evf[-1] - evf[0]).days / 365.25, 0.5), "tx": tx_f, "txliq": txliq_f}
        rk = ranks_from(sc, UNI)
        for nm, kp in KEEPS:
            v, liq, s_ = sim(rk, PXM, DIV, jf, kp)
            r[nm] = half_stats(v, evf)
            r[nm + "_liq"] = liq
            r[nm + "_st"] = s_
        rk_low = ranks_from(-sc, UNI)
        v, _l, _s = sim(rk_low, PXM, DIV, jf, 20)
        r["low20"] = half_stats(v, evf)
        r["bk"] = {}
        for bn, bm in buckets.items():
            v, _l, _s = sim(ranks_from(sc, UNI & bm), PXM, DIV, jf, 20)
            r["bk"][bn] = half_stats(v, evf)
        # 判定
        ok_tx, ok_rnd = True, True
        detail = {}
        for nm, _kp in KEEPS:
            for h in ("前半", "後半"):
                a = r[nm][h]["ann"]
                rnd = [x[h]["ann"] for x in rnd_f[nm]]
                d_tx = a - tx_f[h]["ann"]
                d_rn = a - float(np.mean(rnd))
                beat = int(sum(1 for x in rnd if a > x))
                detail[(nm, h)] = (d_tx, d_rn, beat)
                ok_tx &= d_tx >= PT_TX
                ok_rnd &= (d_rn >= PT_RND) and (beat >= BEAT_MIN)
        r["detail"] = detail
        r["verdict"] = "保留（データ不足）" if r["low"] >= 3 else ("候補" if (ok_tx and ok_rnd) else ("範囲の中では効く" if ok_rnd else "効かない"))
        res[key] = r
        log(f"  {key} … {r['verdict']}")
    report(D, res, RND, ST_TX, TXliq, ev, j0)
    return 0


def report(D, res, RND, ST_TX, TXliq, ev, j0):
    HH = ("前半", "後半")
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    print("\n【日本株の手法の型・1回目（wide_lab）：東証全銘柄・実際の運用の形】（基準は結果を見る前に決めたもの。計画はプロジェクトの method_map_2026-11-04 と wide_lab_plan_2026-11-04）")
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    nu = D["UNI"][:, j0:].sum(axis=0)
    print(f"  範囲 … 東証の全市場の普通株（配当のない株・上場をやめた銘柄も入れる）。売買代金の中央値5,000万円以上・時価総額100億円以上。"
          f"1か月あたり平均 {nu.mean():.0f}銘柄（最少 {nu.min()}・最多 {nu.max()}）")
    print(f"  決まり … 元本1,000万円・20銘柄・1株単位・月末の終値・片道{SLIP * 100:.1f}％・税20.315％（繰り越しの損あり・配当は受け取るときに）。"
          "売る決まり A＝20位より下で売る・B＝60位より下で売る")
    print(f"  判定 … 両方の期間で、A・B の両方とも ①TOPIX（税あり）を +{PT_TX:.1f}pt 以上 ②でたらめな20銘柄の平均を +{PT_RND:.1f}pt 以上かつ20回のうち{BEAT_MIN}回以上上回る"
          "→ ①②とも「候補」・②だけ「範囲の中では効く」・それ以外「効かない」。点数がある銘柄が範囲の半分未満の月が3か月以上なら保留")
    rA = {h: np.mean([x[h]["ann"] for x in RND["A"]]) for h in HH + ("通し",)}
    rB = {h: np.mean([x[h]["ann"] for x in RND["B"]]) for h in HH + ("通し",)}
    print(f"\n  比べる相手（年率％。「前半 ／ 後半」） … TOPIX（税あり） {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}"
          f"（通し {ST_TX['通し']['ann']:.1f}・最大下落 {ST_TX['通し']['mdd']:.0f}％） ・ でたらめな20銘柄の平均 A {rA['前半']:.1f} ／ {rA['後半']:.1f}"
          f" ・ B {rB['前半']:.1f} ／ {rB['後半']:.1f}")
    print("\n  1. 型ごとの成績（年率％。「前半 ／ 後半」。A・B は売る決まり。差は TOPIX との差 ／ でたらめの平均との差・でたらめを上回った回数）")
    for key, r in res.items():
        a, b = r["A"], r["B"]
        dA = " ".join(f"{r['detail'][('A', h)][0]:+.1f}/{r['detail'][('A', h)][1]:+.1f}・{r['detail'][('A', h)][2]}回" for h in HH)
        dB = " ".join(f"{r['detail'][('B', h)][0]:+.1f}/{r['detail'][('B', h)][1]:+.1f}・{r['detail'][('B', h)][2]}回" for h in HH)
        extra = "" if r["start"] == ev[0] else f"（比べる期間は {r['start'].date()} から {r['n']}か月。比べる相手もこの期間で作り直した。TOPIX {r['tx']['前半']['ann']:.1f} ／ {r['tx']['後半']['ann']:.1f}）"
        print(f"     {key:<8}{r['lab']}{extra}")
        print(f"         A {a['前半']['ann']:6.1f} ／ {a['後半']['ann']:6.1f}（最大下落 {a['通し']['mdd']:.0f}％）［{dA}］"
              f"  B {b['前半']['ann']:6.1f} ／ {b['後半']['ann']:6.1f}（最大下落 {b['通し']['mdd']:.0f}％）［{dB}］ … {r['verdict']}")
        bk = " ・ ".join(f"{bn} {v['前半']['ann']:.1f} ／ {v['後半']['ann']:.1f}" for bn, v in r["bk"].items())
        print(f"         参考：下から20（A） {r['low20']['前半']['ann']:.1f} ／ {r['low20']['後半']['ann']:.1f} ・ 大きさ別（A） {bk}"
              f" ・ 点数のある割合 平均 {np.mean(r['cov']) * 100:.0f}％・清算後の通し A {((r['A_liq'] / CAP) ** (1 / r['yrs']) - 1) * 100:.1f}％"
              f"・買い A {r['A_st']['n_buy'] / r['yrs']:.0f}回／年・B {r['B_st']['n_buy'] / r['yrs']:.0f}回／年")
    print("\n  2. 判定のまとめ")
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in res.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fetch", "lab1"], default="lab1")
    args = ap.parse_args()
    store = load()
    if not store.get("complete"):
        key = os.environ.get("J_QUANTS_API_KEY")
        if not key:
            print("J_QUANTS_API_KEY が設定されていません")
            return 2
        jq = yb.JQ(key)
        if not fetch_all(jq, store):
            print("\n■ まだ終わっていません（東証全銘柄のデータを取っている途中です）")
            print(f"　 取った分は保存しました（開始から{elapsed():.0f}分）。同じ設定のまま、もう一度「Run workflow」を押してください。")
            return EXIT_UNFINISHED
    n = len(store.get("bars", {}))
    log(f"データ … 銘柄一覧 {len(store.get('months', {}))}か月・株価 {n}銘柄・株価が取れなかった {len(store.get('nodata', []))}銘柄")
    if args.mode == "fetch":
        print("\n■ 東証全銘柄のデータがそろっています。")
        return 0
    if elapsed() > COMPUTE_LIMIT_MIN:
        print("\n■ まだ終わっていません（データは取り終えましたが、計算の時間が残っていません）。もう一度押してください。")
        return EXIT_UNFINISHED
    return lab1(store)


if __name__ == "__main__":
    sys.exit(main())

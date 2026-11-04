"""東証の全銘柄で、日本株の手法の型をまとめて確かめる道具（wide_lab）。2026年11月4日に作成。

計画と記録はプロジェクトの method_map_2026-11-04（地図と共通の決まり）と wide_lab_plan_2026-11-04（1回目の型と順位の付け方）。
今までのツール（yield_backtest.py）・キャッシュ（data/yield_backtest_cache.pkl）には触らない。データは data/wide_cache.pkl に別に持つ。

使い方（GitHub の Actions → wide-lab → Run workflow）
  mode=fetch … 東証全銘柄のデータ（各月の銘柄一覧・株価・財務の要約）を集める。時間切れなら取った分を保存して止まる（もう一度押すと続きから）
  mode=lab1  … 足りないデータがあれば先に集め、そろっていれば1回目の型をまとめて回す
  mode=lab2  … 同じく、2回目の型（出来事と暦）をまとめて回す
"""
import argparse
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import yield_backtest as yb

OUT = Path("data")
YEARS = int(os.environ.get("WIDE_YEARS", "9"))
LONG = YEARS > 10        # 20年分（プレミアム）で確かめる段階。データは別のファイルに持ち、プレミアムをやめる前に long-cleanup で消す
CACHE = OUT / ("wide_long.pkl" if LONG else "wide_cache.pkl")
LONG_FIRST_ME = pd.Timestamp("2008-05-30")
KEEP_OHLC = LONG or os.environ.get("WIDE_OHLC") == "1"     # 20年分は始値・高値・安値も持つ（11月4日夜に追加）
# 20年分のデータで「探す段階」を回すとき（WIDE_STAGE=explore）は、9年分のキャッシュと同じ期間だけを使う（見ていない期間を見ないため）
STAGE = os.environ.get("WIDE_STAGE", "confirm" if LONG else "explore")
EXPLORE_FROM = np.datetime64("2017-10-05")      # 9年分のキャッシュの株価の最初の日
EXPLORE_FIRST_MONTH = "2017-12"                 # 9年分のキャッシュの銘柄一覧の最初の月    # J-Quants の株価は2008年5月7日から。銘柄一覧もこの月末から取る
BUDGET_MIN = float(os.environ.get("FETCH_BUDGET_MIN", "130"))
COMPUTE_LIMIT_MIN = float(os.environ.get("COMPUTE_START_LIMIT_MIN", "140"))
SAVE_EVERY_MIN = 15
# 株価と財務を同時に取る数。プレミアムは呼び出しの上限が大きい（1分に500回ほど）ので増やせる。スタンダード（1分に120回ほど）は1にする
THREADS = max(1, int(os.environ.get("WIDE_THREADS", "") or "1"))
EXIT_UNFINISHED = 3
T0 = time.monotonic()

# 東証の株の市場（一部・二部・マザーズ・JASDAQ スタンダード・JASDAQ グロース・プライム・スタンダード・グロース）。
# TOKYO PRO MARKET（0105）とその他（0109）は入れない
TSE_MKT = {"0101", "0102", "0104", "0106", "0107", "0111", "0112", "0113"}
FIN_KEEP = ("Code", "DiscDate", "DiscTime", "DocType", "CurPerType", "CurPerSt", "CurPerEn", "CurFYSt", "CurFYEn",
            "Sales", "OP", "OdP", "NP", "EPS", "BPS", "TA", "Eq", "EqAR", "CFO", "CFI", "CashEq",
            "DivAnn", "Div2Q", "FDivAnn", "ShOutFY", "TrShFY", "FSales", "FOP", "FNP", "NxFSales", "NxFOP", "NxFNp", "NxFNP",
            "FNCNP", "NxFNCNP", "NxFNCNp")      # 単体だけの会社の予想（連結の予想の項目が空のときに使う）

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
SPLIT = pd.Timestamp("2018-08-01" if (LONG and STAGE == "confirm") else "2022-11-01")
# 探す段階（9年分）：前半＝2022年10月まで・後半＝11月から。確かめる段階（20年分）：前半＝見ていない期間（2018年7月まで）・後半＝最近（wide_lab_plan で決めた）
START_T = {"REV24": 504}   # 2年の戻りは2年前の株価が要るので、比べる期間を2年目の終わりから始める（wide_lab_plan に書いた）
SURP_DAYS = 400          # 決算の驚き：通期の決算短信が出てから、次の通期の決算短信が出るまで（約1年）使う
LARGE_N, MID_N = 500, 1500
GAP_DAYS = 60            # 売買のない日がこれより長く続いたら、その前とあとを別の銘柄とみなす（コードの使い回し・長い売買停止）


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
    if LONG:
        start = max(start, LONG_FIRST_ME)
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
    out = {"d": d[o], "c": c[o].astype(np.float32), "va": va[o].astype(np.float32), "fd": d[o][fm], "ff": f[fm]}
    if KEEP_OHLC:              # 短期売買の型のために、始値・高値・安値・出来高（どれも分割を調整したもの）も持つ
        out["o"] = col("AdjO", "AdjustmentOpen", "O", "Open")[o].astype(np.float32)
        out["h"] = col("AdjH", "AdjustmentHigh", "H", "High")[o].astype(np.float32)
        out["l"] = col("AdjL", "AdjustmentLow", "L", "Low")[o].astype(np.float32)
        out["v"] = v[o].astype(np.float32)
    return out


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
    # 始値などを持っていない銘柄（11月4日夜より前に取った分）は、取り直す
    redo = {c for c in done if KEEP_OHLC and c in bars and "o" not in bars[c]}
    if redo:
        log(f"  始値・高値・安値がない {len(redo)}銘柄を取り直します")
    todo = [c for c in codes if c not in done or c in redo]
    if "etf1306" not in store:
        time.sleep(yb.API_SLEEP)
        store["etf1306"] = pack_bars(yb.fetch_bars(jq, "1306", YEARS))
    if todo:
        log(f"株価と財務を取っています（全部で {len(codes)}銘柄・残り {len(todo)}銘柄・開始から{elapsed():.0f}分）…")
    last = time.monotonic()
    local = threading.local()

    def one(c):
        """1銘柄の株価と財務。同時に取るときは、取る役ごとに別の接続を使う。"""
        cli = jq
        if THREADS > 1:
            if not hasattr(local, "jq"):
                local.jq = type(jq)(os.environ.get("J_QUANTS_API_KEY", ""))
            cli = local.jq
        try:
            time.sleep(yb.API_SLEEP)
            b = pack_bars(yb.fetch_bars(cli, c, YEARS))
            time.sleep(yb.API_SLEEP)
            st = cli.get("/v2/fins/summary", {"code": c})
            return c, b, st, None
        except Exception as e:
            return c, None, None, e
    if THREADS > 1:
        log(f"  同時に {THREADS}銘柄ずつ取ります")
    i = 0
    chunk = max(1, THREADS * 10)
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        for k in range(0, len(todo), chunk):
            if elapsed() >= BUDGET_MIN:
                store["done"], store["nodata"] = sorted(done), sorted(nodata)
                save(store, "株価と財務を途中まで保存しました")
                return False
            for c, b, st, err in ex.map(one, todo[k:k + chunk]):
                i += 1
                if err is not None:
                    log(f"  取得に失敗 {c}: {err}")
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


def first_num(r, keys):
    for k in keys:
        v = num(r.get(k))
        if v is not None:
            return v
    return None


def build(store: dict) -> dict:
    """月末ごとの値段・時価総額・売買代金・配当・各型の点数を作る。"""
    tp = store.get("etf1306")
    if not tp:
        raise SystemExit("TOPIX（1306）の株価がありません")
    # 1306 の値がつかなかった日（取った時点でまだ値が入っていない最後の日など）は、前の日の値で埋める
    n_nan = int(np.isnan(tp["c"]).sum())
    if n_nan:
        log(f"  1306 の値がない日 {n_nan}日（前の日の値で埋めました。いちばん新しい日 {tp['d'][np.isnan(tp['c'])][-1]}）")
    tp = dict(tp, c=pd.Series(tp["c"]).ffill().bfill().to_numpy(dtype=np.float32))
    months_keys = sorted(store["months"])
    if LONG and STAGE == "explore":
        keep = tp["d"] >= EXPLORE_FROM
        tp = {k: (v[keep] if (isinstance(v, np.ndarray) and len(v) == len(keep)) else v) for k, v in tp.items()}
        months_keys = [k for k in months_keys if k >= EXPLORE_FIRST_MONTH]
        log(f"  探す段階：20年分のうち {str(tp['d'][0])} からだけを使います（見ていない期間は使いません）")
    cal = tp["d"]
    T = len(cal)
    codes = sorted(store["bars"])
    S = len(codes)
    ci = {c: i for i, c in enumerate(codes)}
    C = np.full((S, T), np.nan, dtype=np.float32)
    has_ohlc = bool(codes) and all("o" in store["bars"][c] for c in codes)
    OHLV = {k: np.full((S, T), np.nan, dtype=np.float32) for k in ("o", "h", "l", "v")} if has_ohlc else None
    VA = np.zeros((S, T), dtype=np.float32)
    LT = np.full((S, T), -1, dtype=np.int32)     # その日までの最後に売買があった日の番号
    SEG = np.full((S, T), T, dtype=np.int32)     # その日が入っている「ひと続きの売買」の最初の日の番号（まだ売買がなければ T）
    seg_first = np.full(S, T, dtype=np.int32)    # 最初のひと続きの最初の日
    for c, i in ci.items():
        b = store["bars"][c]
        k = np.searchsorted(cal, b["d"])
        ok = (k < T)
        ok[ok] = cal[k[ok]] == b["d"][ok]
        ok &= np.isfinite(b["c"]) & (b["c"] > 0)          # 値のつかなかった日は、売買のない日として扱う
        kk = k[ok]
        C[i, kk] = b["c"][ok]
        if OHLV is not None:
            for kx in ("o", "h", "l", "v"):
                OHLV[kx][i, kk] = b[kx][ok]
        VA[i, kk] = np.nan_to_num(b["va"][ok])
        idx = np.full(T, -1, dtype=np.int32)
        idx[kk] = kk
        LT[i] = np.maximum.accumulate(idx)
        if len(kk):
            st_ = kk[np.r_[True, np.diff(kk) > GAP_DAYS]]
            sidx = np.full(T, -1, dtype=np.int32)
            sidx[st_] = st_
            sg = np.maximum.accumulate(sidx)
            SEG[i] = np.where(sg < 0, T, sg)
            seg_first[i] = st_[0]
    CF = pd.DataFrame(C).ffill(axis=1).to_numpy(dtype=np.float32)
    mret = np.r_[np.nan, tp["c"][1:] / tp["c"][:-1] - 1.0]
    me_keys, me_dates, me_idx = [], [], []
    for k in months_keys:
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
        cal_i = cal.astype("datetime64[D]")
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
            # 売買が長く途切れたあと（コードの使い回しなど）は、途切れたあとに出た開示だけを使う
            sg = int(SEG[i, me_idx[j]])
            k_lo = int(np.searchsorted(dd, cal_i[sg], side="left")) if (sg < T and sg > seg_first[i]) else 0
            if k_lo > k:
                continue
            sub = tl[k_lo:k + 1]
            m_date = me_dates[j]
            fy = [r for r in sub if is_fy(r)]
            fy_by_end = {}
            for r in fy:
                fy_by_end[str(r.get("CurFYEn") or "")[:10]] = r
            fy_list = [fy_by_end[e] for e in sorted(fy_by_end)]
            # 株数（自己株を除く）：直近の開示。いまの株数の基準へ
            sh_r = next((r for r in reversed(sub) if num(r.get("ShOutFY"))), None)
            if sh_r is None:
                continue
            sh = num(sh_r.get("ShOutFY")) - (num(sh_r.get("TrShFY")) or 0.0)
            sh_cur = sh / fac_after(fd, ff, str(sh_r["DiscDate"])[:10])
            px = CF[i, me_idx[j]]
            if not (px == px and px > 0 and sh_cur > 0):
                continue
            mcap = float(px) * sh_cur
            MCAP[i, j] = mcap
            eq_r = next((r for r in reversed(sub) if num(r.get("Eq")) is not None), None)
            ta_r = next((r for r in reversed(sub) if num(r.get("TA")) is not None), None)
            cash_r = next((r for r in reversed(sub) if num(r.get("CashEq")) is not None), None)
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
                nc = "NonConsolidated" in str(r.get("DocType") or "")      # 単体だけの会社は、単体の予想も見る
                k_fc = ("FNP",) + (("FNCNP",) if nc else ())
                nx = first_num(r, ("NxFNp", "NxFNP") + (("NxFNCNP", "NxFNCNp") if nc else ()))
                if nx is not None and np_ and np_ > 0:
                    F["NXG"][i, j] = nx / np_ - 1.0
                # 決算の驚き：通期の実績 ÷ その期の最後の会社予想（決算短信が出てから SURP_DAYS 日まで）
                fy_end = str(r.get("CurFYEn") or "")[:10]
                r_d = str(r["DiscDate"])[:10]
                if np_ is not None and (m_date - pd.Timestamp(r_d)).days <= SURP_DAYS:
                    fc = None
                    for q in sub:
                        if str(q["DiscDate"])[:10] >= r_d:
                            break
                        if str(q.get("CurFYEn") or "")[:10] == fy_end and first_num(q, k_fc):
                            fc = first_num(q, k_fc)
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
        seg_t = SEG[:, t]

        def inseg(L):
            return seg_t <= t - L          # L 営業日前から、ひと続きの売買の中にいる
        if t >= 252:
            F["MOM12"][:, j] = np.where(inseg(252), CF[:, t - 21] / CF[:, t - 252] - 1.0, np.nan)
            w = CF[:, t - 251:t + 1]
            F["HI52"][:, j] = np.where(inseg(252), px / np.nanmax(w, axis=1), np.nan)
            r = w[:, 1:] / w[:, :-1] - 1.0
            mr = mret[t - 250:t + 1]
            ok = (np.isfinite(r).sum(axis=1) >= 200) & inseg(251)
            sd = np.nanstd(r, axis=1)
            F["LOWVOL"][:, j] = np.where(ok, -sd, np.nan)
            rm = np.where(np.isfinite(r), r, np.nan)
            mm_ = np.nanmean(mr)
            cov = np.nanmean((rm - np.nanmean(rm, axis=1, keepdims=True)) * (mr - mm_), axis=1)
            var = np.nanvar(mr)
            F["LOWBETA"][:, j] = np.where(ok & (var > 0), -cov / var, np.nan)
            F["LOTTERY"][:, j] = np.where(inseg(21), -np.nanmax(r[:, -21:], axis=1), np.nan)
        if t >= 126:
            F["MOM6"][:, j] = np.where(inseg(126), CF[:, t - 21] / CF[:, t - 126] - 1.0, np.nan)
        if t >= 21:
            F["REV1"][:, j] = np.where(inseg(21), -(px / CF[:, t - 21] - 1.0), np.nan)
        if t >= 504:
            F["REV24"][:, j] = np.where(inseg(504), -(px / CF[:, t - 504] - 1.0), np.nan)
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
            "MCAP": MCAP, "LIQ": LIQ, "PXM": PXM, "DIV": DIV, "cal": cal, "tp": tp, "C": C, "CF": CF, "SEG": SEG, "seg_first": seg_first,
            "OHLV": OHLV, "VA": VA}


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
        if rk is None:            # 入れ替えない月：配当を受け取って評価するだけ
            vals.append(cash + sum(sh * p_(i, j) for i, (sh, _c) in pos.items()))
            continue
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
ONLY: set = set()        # 回す型（空なら全部）
LAB_TITLE, LAB_PLAN = "1回目", "wide_lab_plan_2026-11-04"     # 結果の見出し（3回目・4回目はここを変える）

LAB1 = (("EP", "利益に対する安さ（純利益÷時価総額）"), ("BP", "純資産に対する安さ"), ("SP", "売上に対する安さ"),
        ("CASH", "現金に対する安さ（金融を除く）"), ("FCF", "稼ぐ現金に対する安さ（金融を除く）"), ("DY", "配当利回り（実績）"),
        ("MOM12", "12か月の勢い（直近1か月を除く）"), ("MOM6", "6か月の勢い（直近1か月を除く）"), ("HI52", "52週高値への近さ"),
        ("REV1", "1か月の戻り（下げた順）"), ("REV24", "2年の戻り（下げた順）"),
        ("ROE", "自己資本の利益率"), ("ROA", "総資産の営業利益率（金融を除く）"), ("EQR", "自己資本比率（金融を除く）"),
        ("SGROW", "売上の伸び"), ("AGROW", "資産の増え方の小ささ（金融を除く）"), ("SHR", "株数の減り（自社株買いなど）"),
        ("SURP", "決算が予想を上回る（次の決算まで）"), ("NXG", "来期予想の伸び"),
        ("LOWVOL", "値動きの小ささ"), ("LOWBETA", "市場との連動の小ささ"), ("LOTTERY", "宝くじ株を避ける（1日の最大の上げが小さい）"),
        ("ILLIQ", "売買の少なさ"), ("SMALL", "小型株（時価総額の小さい順）"))


def pct_rank(sc: np.ndarray, uni: np.ndarray) -> np.ndarray:
    """月末ごとに、範囲の中で点数を0（いちばん下）〜1（いちばん上）の順位にする。点数のない銘柄は空。"""
    out = np.full(sc.shape, np.nan)
    for j in range(sc.shape[1]):
        idx = np.nonzero(uni[:, j] & np.isfinite(sc[:, j]))[0]
        n = len(idx)
        if n > 1:
            r = np.argsort(np.argsort(sc[idx, j], kind="stable"), kind="stable")
            out[idx, j] = r / (n - 1)
    return out


def combo_score(F: dict, keys: list, uni: np.ndarray) -> np.ndarray:
    """組み合わせ：型ごとの順位（0〜1）の平均。半分以上の型に点数がある銘柄だけ。"""
    R = np.stack([pct_rank(F[k], uni) for k in keys])
    cnt = np.isfinite(R).sum(axis=0)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(R, axis=0)
    return np.where(cnt >= int(np.ceil(len(keys) / 2)), m, np.nan)


# 3回目の組み合わせ（wide_lab_plan3_2026-11-04 で、1回目の結果を見る前に決めたもの）
VQM = ("BP", "ROE", "MOM12")      # 割安・質・勢いの古典的な組み合わせ（結果を見ずに決めた固定の組み合わせ）


def lab1(store: dict, combos: dict | None = None, extra: dict | None = None) -> int:
    log("1回目の型をまとめて回します（指標を作っています）…" if not (combos or extra) else
        ("3回目（組み合わせ）を回します（指標を作っています）…" if combos else "4回目（売買の内訳）を回します（指標を作っています）…"))
    D = build(store)
    dates, UNI, F, PXM, DIV = D["dates"], D["UNI"], D["F"], D["PXM"], D["DIV"]
    items = LAB1
    if extra:
        items = []
        for nm, (fn, lab) in extra.items():
            F[nm] = fn(D)
            items.append((nm, lab))
    if combos:
        items = []
        for nm, (keys, lab) in combos.items():
            F[nm] = combo_score(F, list(keys), UNI)
            items.append((nm, f"{lab}（{'・'.join(keys)} の順位の平均）"))
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
    for key, lab in items:
        if ONLY and key not in ONLY:
            continue
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


def stage_note() -> None:
    if LONG and STAGE == "explore":
        print("  ※ 探す段階（20年分のデータのうち、9年分のキャッシュと同じ期間だけを使う。見ていない期間（2018年7月まで）は使わない）"
              + (f"。回した型：{'・'.join(sorted(ONLY))}" if ONLY else ""))
    elif LONG:
        print("  ※ 確かめる段階（プレミアム・20年分）。「前半」＝見ていない期間（2018年7月まで）・「後半」＝最近（2018年8月から）。"
              "両方で「候補」の基準を満たせば「確立」（method_map_2026-11-04）" + (f"。回した型：{'・'.join(sorted(ONLY))}" if ONLY else ""))


def report(D, res, RND, ST_TX, TXliq, ev, j0):
    HH = ("前半", "後半")
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    print(f"\n【日本株の手法の型・{LAB_TITLE}（wide_lab）：東証全銘柄・実際の運用の形】（基準は結果を見る前に決めたもの。計画はプロジェクトの method_map_2026-11-04 と {LAB_PLAN}）")
    stage_note()
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


# ───────────────────────── 2回目：出来事と暦の型 ─────────────────────────
# 計画は wide_lab_plan2_2026-11-04（結果を見る前に決めたもの）。比べる期間・前半と後半の分け方・判定の基準は1回目と同じ。
# 1回目の「売る決まり A・B」の代わりに、型ごとに決めた2通り（持つ日数など）の両方で基準を満たしたときだけ通す。
REACT_TH = 0.05                         # 決算発表の反応（TOPIX との差）の下限
DY_TH = 0.03                            # 権利付き最終日の前に買う型：予想の配当利回りの下限（直近の通期の年間配当 ÷ 株価）
ETF_SLIP = 0.001                        # 暦の型（1306 を売り買い）の値段のずれ（片道）
EV_MIN_BUYS = 50                        # 前半・後半のどちらかで、実際に買った回数がこれ未満なら保留
RND_FILL_MIN = 0.8                      # でたらめの比べの買いの回数が、型の買いの回数の8割に満たなければ保留（比べる相手が細すぎる）
IPO_MIN_N, IPO_MIN_SHARE = 20, 0.9      # 上場直後の株：点数のある銘柄が20以上の月が9割に満たなければ保留
T2_START = np.datetime64("2019-07-16")  # 受け渡しが2営業日になった日（これより前は、権利付き最終日が権利確定日の3営業日前）


def t_cum_of(cal: np.ndarray, t_rec: int) -> int:
    """権利確定日（その日までの最後の営業日の番号）から、権利付き最終日の番号。"""
    return t_rec - (2 if cal[t_rec] >= T2_START else 3)


def last_td(cal: np.ndarray, d) -> int:
    """その日までの最後の営業日の番号（なければ −1。データの最後の日より後の日付は、まだ来ていないので −1）。"""
    dd = np.datetime64(pd.Timestamp(d).date())
    if len(cal) == 0 or dd > cal[-1]:
        return -1
    return int(np.searchsorted(cal, dd, side="right")) - 1


def ev_extract(store: dict, D: dict) -> dict:
    """出来事を取り出す：決算短信の発表・配当を受け取る日（権利付き最終日）・権利付き最終日の前に買う型の候補・株式分割の効力発生日。"""
    cal, codes = D["cal"], D["codes"]
    T = len(cal)
    annc, recs, splits = [], [], []
    divpay: dict = {}
    for i, c in enumerate(codes):
        b = store["bars"][c]
        fd, ff = b["fd"], b["ff"]
        for d0, f0 in zip(fd, ff):
            if f0 < 1.0:
                t_s = int(np.searchsorted(cal, d0))
                if t_s < T and cal[t_s] == d0:
                    splits.append((i, t_s))
        tl = fin_timeline(store["fins"].get(c))
        seen_d, seen_fy = set(), set()
        for r in tl:
            d = str(r["DiscDate"])[:10]
            if "FinancialStatements" in str(r.get("DocType") or "") and d not in seen_d:
                seen_d.add(d)
                dd = np.datetime64(d)
                t_prev = int(np.searchsorted(cal, dd, side="left")) - 1
                t_a = int(np.searchsorted(cal, dd, side="right"))
                if t_prev >= 0 and t_a < T:
                    annc.append((i, t_prev, t_a))
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
            # 受け取る配当（実績）：期末の分は期末の月、中間の分は6か月前の月。権利付き最終日の終わりに持っていれば受け取る
            for amt, rec in ((ann - q2, e), (q2, e - pd.DateOffset(months=6))):
                if amt > 0:
                    me = rec + pd.offsets.MonthEnd(0)
                    t_rec = last_td(cal, me)
                    if t_rec >= 3:
                        divpay.setdefault(t_cum_of(cal, t_rec), []).append((i, amt * fac_after(fd, ff, me)))
            # 権利付き最終日の前に買う型の候補：この決算短信のあとに来る、次の期の中間と期末の権利確定日
            t_disc = int(np.searchsorted(cal, np.datetime64(d), side="left"))
            dps = ann * fac_after(fd, ff, d)
            nxt = e + pd.DateOffset(years=1)
            for rec, ok in ((nxt - pd.DateOffset(months=6), q2 > 0), (nxt, True)):
                if ok:
                    t_rec = last_td(cal, rec + pd.offsets.MonthEnd(0))
                    if t_rec >= 3:
                        recs.append((i, t_cum_of(cal, t_rec), dps, t_disc))
    return {"annc": annc, "recs": recs, "splits": splits, "divpay": divpay}


def ev_sim(cands: dict, D: dict, divpay: dict, t0: int, nmax: int = NMAX, cap: float = CAP, slip: float = SLIP, tax: float = TAX):
    """出来事で買って、決めた日に売る運用。cands … {買う日の番号: [(優先度, 銘柄の番号, 売る日の番号), …]}。
    毎日、①売る日が来た銘柄を終値で売る ②その日の候補を優先度の高い順に買う（20銘柄まで・そのときの資産の20分の1まで・1株単位）
    ③権利付き最終日の終わりに持っている銘柄の配当を受け取る（税）。上場をやめた銘柄は最後の値段で売る。月末ごとの評価額を返す。"""
    CF, C, me_idx = D["CF"], D["C"], D["me_idx"]
    T = CF.shape[1]
    me_set = {t: j for j, t in enumerate(me_idx)}
    cash, loss = float(cap), 0.0
    pos: dict = {}
    vals = []
    st = {"n_buy": 0, "buy_t": [], "tax": 0.0, "held": []}

    def px(i, t):
        v = CF[i, t]
        return float(v) if (v == v and v > 0) else 0.0
    for t in range(t0, T):
        for i in [k for k, p in pos.items() if p[2] <= t]:
            sh, cost, _o = pos.pop(i)
            eff = px(i, t) * (1.0 - slip)
            cash += sh * eff
            g = sh * eff - cost
            if g > 0:
                tx = max(0.0, g - loss) * tax
                loss = max(0.0, loss - g)
                cash -= tx
                st["tax"] += tx
            else:
                loss += -g
        cl = cands.get(t)
        if cl:
            value = cash + sum(p[0] * px(k, t) for k, p in pos.items())
            tgt = value / nmax
            for _pr, i, t_out in sorted(cl, key=lambda x: (-x[0], x[1])):
                if len(pos) >= nmax:
                    break
                if i in pos or t_out <= t:
                    continue
                v = C[i, t]
                if not (v == v and v > 0):          # その日に売買がなければ買わない
                    continue
                p = float(v) * (1.0 + slip)
                sh = int(min(tgt, cash) / p)
                if sh < 1:
                    continue
                cash -= sh * p
                pos[i] = [sh, sh * p, t_out]
                st["n_buy"] += 1
                st["buy_t"].append(t)
        for i, amt in divpay.get(t, ()):
            if i in pos:
                g = pos[i][0] * amt
                cash += g * (1.0 - tax)
                st["tax"] += g * tax
        if t in me_set:
            vals.append(cash + sum(p[0] * px(k, t) for k, p in pos.items()))
            st["held"].append(len(pos))
    liq, gain = cash, 0.0
    for k, p in pos.items():
        eff = px(k, T - 1) * (1.0 - slip)
        liq += p[0] * eff
        gain += p[0] * eff - p[1]
    liq -= tax * max(0.0, gain - loss)
    return np.array(vals), liq, st


def cal_sim(inmkt: np.ndarray, D: dict, t0: int, slip: float = ETF_SLIP, tax: float = TAX, cap: float = CAP):
    """1306 を、inmkt[t] が真の日の終わりに持つように終値で売り買いする（暦の型）。分配金は権利付き最終日の終わりに持っていれば受け取る。"""
    tp, cal, me_idx = D["tp"], D["cal"], D["me_idx"]
    T = len(cal)
    me_set = set(me_idx)
    dist = {}
    for d, a in yb.ETF_DIST.get("1306", ()):
        t_rec = last_td(cal, d)
        if t_rec >= 3:
            dist[t_cum_of(cal, t_rec)] = a / 100.0
    cash, units, cost, loss = float(cap), 0, 0.0, 0.0
    vals = []
    st = {"n_buy": 0, "buy_t": [], "tax": 0.0}
    for t in range(t0, T):
        p = float(tp["c"][t])
        if inmkt[t] and units == 0:
            units = int(cash / (p * (1.0 + slip)))
            cost = units * p * (1.0 + slip)
            cash -= cost
            st["n_buy"] += 1
            st["buy_t"].append(t)
        elif (not inmkt[t]) and units > 0:
            pr = units * p * (1.0 - slip)
            g = pr - cost
            cash += pr
            if g > 0:
                tx = max(0.0, g - loss) * tax
                loss = max(0.0, loss - g)
                cash -= tx
                st["tax"] += tx
            else:
                loss += -g
            units = 0
        if units > 0 and t in dist:
            g = units * dist[t]
            cash += g * (1.0 - tax)
            st["tax"] += g * tax
        if t in me_set:
            vals.append(cash + units * p)
    pr = units * float(tp["c"][T - 1]) * (1.0 - slip)
    liq = cash + pr - tax * max(0.0, (pr - cost if units else 0.0) - loss)
    return np.array(vals), liq, st


def month_bounds(cal: np.ndarray) -> list:
    """月ごとの（最初の営業日の番号, 最後の営業日の番号）。"""
    ym = cal.astype("datetime64[M]")
    out, s = [], 0
    for t in range(1, len(cal) + 1):
        if t == len(cal) or ym[t] != ym[s]:
            out.append((s, t - 1))
            s = t
    return out


def tom_mask(cal: np.ndarray, pre: int) -> np.ndarray:
    """月末月初：月の最後の営業日の pre 日前の終値で買い、次の月の3日目の終値で売る（その日の終わりに持っている日が真）。"""
    mb = month_bounds(cal)
    x = np.zeros(len(cal), dtype=bool)
    for k in range(len(mb) - 1):
        s_, e_ = mb[k]
        s2, e2_ = mb[k + 1]
        if e_ - s_ < pre or e2_ - s2 < 3:
            continue
        x[e_ - pre:s2 + 2] = True
    return x


def may_mask(cal: np.ndarray, last_mon: int) -> np.ndarray:
    """冬だけ持つ：10月の最後の営業日の終値で買い、次の last_mon 月の最後の営業日の終値で売る。"""
    mb = month_bounds(cal)
    mon = (cal.astype("datetime64[M]").astype(int) % 12) + 1
    x = np.zeros(len(cal), dtype=bool)
    for _s, e_ in mb:
        if mon[e_] != 10:
            continue
        nxt = [e2_ for (_s2, e2_) in mb if e2_ > e_ and mon[e2_] == last_mon]
        if nxt:
            x[e_:nxt[0]] = True
    return x


def lab2(store: dict) -> int:
    log("2回目の型（出来事と暦）をまとめて回します（指標を作っています）…")
    D = build(store)
    dates, UNI, me_idx, cal = D["dates"], D["UNI"], D["me_idx"], D["cal"]
    M, T = len(dates), len(cal)
    C, CF = D["C"], D["CF"]
    TPc = D["tp"]["c"].astype(float)
    j0 = next((j for j, t in enumerate(me_idx) if t >= 252), None)
    if j0 is None or M - j0 < 36:
        print("\n■ 比べる期間が短すぎます。")
        return 1
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    E = ev_extract(store, D)
    log(f"  出来事 … 決算短信の発表 {len(E['annc']):,}回・配当を受け取る日 {sum(len(v) for v in E['divpay'].values()):,}回・"
        f"権利確定日の候補 {len(E['recs']):,}回・株式分割 {len(E['splits']):,}回")
    jm = np.searchsorted(np.array(me_idx), np.arange(T), side="left") - 1      # その日より前の最後の月末の番号
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    elig_cache: dict = {}

    def elig_list(t):
        if t not in elig_cache:
            j = jm[t]
            if j < 0:
                elig_cache[t] = np.zeros(0, dtype=int)
            else:
                v = C[:, t]
                elig_cache[t] = np.nonzero(UNI[:, j] & np.isfinite(v) & (v > 0))[0]
        return elig_cache[t]

    def is_elig(i, t):
        j = jm[t]
        return j >= 0 and bool(UNI[i, j]) and C[i, t] == C[i, t] and C[i, t] > 0

    def placebo(cands, seed, pool_fn):
        """同じ日に同じ数だけ、でたらめな銘柄を同じ日数だけ持つ候補（優先度もでたらめ）。"""
        rng = np.random.default_rng(seed)
        out = {}
        for t, cl in cands.items():
            pool = pool_fn(t)
            if len(pool) == 0:
                continue
            pick = rng.choice(pool, size=min(len(cl), len(pool)), replace=False)
            out[t] = [(float(rng.random()), int(i), t_out) for i, (_p, _i, t_out) in zip(pick, cl)]
        return out

    def run(cands, rnd_pool_fn, tag):
        v, liq, s_ = ev_sim(cands, D, E["divpay"], t0)
        rnd, rnd_nb = [], []
        for k in range(N_RAND):
            pv, _l, ps_ = ev_sim(placebo(cands, 1000 + k, rnd_pool_fn), D, E["divpay"], t0)
            rnd.append(half_stats(pv, ev))
            rnd_nb.append(ps_["n_buy"])
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
              "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        return {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "nb": nb, "held": float(np.mean(s_["held"])) if s_["held"] else 0.0,
                "fill": float(np.mean(rnd_nb)) / max(1, s_["n_buy"])}

    # でたらめの比べで「同じ出来事が持つ期間に入っている銘柄」を外すための印
    S = len(D["codes"])
    ANNP = np.zeros((S, T + 1), dtype=bool)          # 決算短信の発表の前の営業日
    for i, t_prev, _t_a in E["annc"]:
        ANNP[i, t_prev] = True
    RECM = np.zeros((S, T + 1), dtype=bool)          # 権利付き最終日
    DPSD = np.full((S, T), np.nan, dtype=np.float32)  # その日に分かっている年間配当（分割のあとの基準）
    for i, t_c, dps, t_disc in sorted(E["recs"], key=lambda x: (x[0], x[3])):
        if 0 <= t_c < T:
            RECM[i, t_c] = True
        if t_disc + 1 < T:
            DPSD[i, t_disc + 1:] = dps
    for t_c, lst in E["divpay"].items():
        for i, _a in lst:
            if 0 <= t_c < T:
                RECM[i, t_c] = True

    def pool_e2(L):
        def f(t):
            el = elig_list(t)
            return el[~ANNP[el, t:t + L + 2].any(axis=1)] if len(el) else el
        return f

    def pool_e3(L):
        def f(t):
            el = elig_list(t)
            if not len(el):
                return el
            with np.errstate(invalid="ignore", divide="ignore"):
                y = DPSD[el, t] / CF[el, t]
            return el[(y >= DY_TH) & ~RECM[el, t:t + L + 1].any(axis=1)]
        return f

    ann_pool: dict = {}
    for i, t_prev, t_a in E["annc"]:
        if t_a >= t0 and is_elig(i, t_a) and C[i, t_prev] == C[i, t_prev] and C[i, t_prev] > 0:
            ann_pool.setdefault(t_a, []).append(i)

    def e1(H, good=True):
        out = {}
        for i, t_prev, t_a in E["annc"]:
            if t_a < t0 or not is_elig(i, t_a):
                continue
            p0 = C[i, t_prev]
            if not (p0 == p0 and p0 > 0):
                continue
            re = float(C[i, t_a] / p0) - float(TPc[t_a] / TPc[t_prev])
            if (re >= REACT_TH) if good else (re <= -REACT_TH):
                out.setdefault(t_a, []).append((re if good else -re, i, t_a + H))
        return out

    def e2(L):
        rng = np.random.default_rng(1)
        out = {}
        for i, t_prev, _t_a in sorted(E["annc"]):
            t_in = t_prev - L
            if t_in < t0 or not is_elig(i, t_in):
                continue
            out.setdefault(t_in, []).append((float(rng.random()), i, t_prev))
        return out

    def e3(L):
        out = {}
        for i, t_c, dps, t_disc in E["recs"]:
            t_in = t_c - L
            if t_in < t0 or t_in <= t_disc or t_c >= T or not is_elig(i, t_in):
                continue
            y = dps / float(CF[i, t_in])
            if y >= DY_TH:
                out.setdefault(t_in, []).append((y, i, t_c))
        return out

    def e7(H):
        out = {}
        for i, t_s in E["splits"]:
            if t_s >= t0 and is_elig(i, t_s):
                out.setdefault(t_s, []).append((0.0, i, t_s + H))
        return out

    def ann_pool_fn(t):
        return np.array(ann_pool.get(t, []), dtype=int)

    # でたらめの比べ：E1＝同じ日に決算を発表した銘柄から（反応は問わない）・E2＝持つ期間に決算の発表がない銘柄から・
    # E3＝配当利回り3％以上で、持つ期間に権利付き最終日がない銘柄から・E7＝範囲の銘柄から
    TYPES = (("E1", "決算発表のあとの値動きの続き（反応が TOPIX より +5％以上）",
              (("A", "20日持つ", lambda: e1(20), ann_pool_fn), ("B", "60日持つ", lambda: e1(60), ann_pool_fn))),
             ("E2", "決算発表の前に買う", (("A", "10日前に買う", lambda: e2(10), pool_e2(10)), ("B", "5日前に買う", lambda: e2(5), pool_e2(5)))),
             ("E3", f"権利付き最終日の前に買う（配当利回り{DY_TH * 100:.0f}％以上・権利付き最終日に売る）",
              (("A", "20日前に買う", lambda: e3(20), pool_e3(20)), ("B", "40日前に買う", lambda: e3(40), pool_e3(40)))),
             ("E7", "株式分割の効力発生日に買う", (("A", "20日持つ", lambda: e7(20), elig_list), ("B", "60日持つ", lambda: e7(60), elig_list))))
    res = {}
    for key, lab, variants in TYPES:
        if ONLY and key not in ONLY:
            continue
        r = {"lab": lab, "var": {}}
        for nm, vlab, mk, pool_fn in variants:
            cands = mk()
            r["var"][nm] = dict(run(cands, pool_fn, key + nm), vlab=vlab, n_cand=sum(len(v) for v in cands.values()))
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda x: min(x["nb"].values()) < EV_MIN_BUYS or x["fill"] < RND_FILL_MIN)
        res[key] = r
        log(f"  {key} … {r['verdict']}")
    # 参考：決算発表の反応が悪い株を買う（判定には使わない）
    ref_bad = run(e1(60, good=False), ann_pool_fn, "E1bad") if (not ONLY or "E1" in ONLY) else None
    # E4：上場直後の株（毎月の順位で回す。売る決まりは1回目と同じ A・B。比べるでたらめも1回目と同じ）
    run_e4 = not ONLY or "E4" in ONLY
    SEG = D["SEG"]
    sc4 = np.full(UNI.shape, np.nan)
    for j, t in enumerate(me_idx):
        s = SEG[:, t]
        age = t - s
        ok = (s < T) & (s >= 21) & (age >= 21) & (age <= 252)
        sc4[ok, j] = -age[ok]
    nsc = [int((UNI[:, j] & np.isfinite(sc4[:, j])).sum()) for j in range(j0, M)]
    rk4 = ranks_from(sc4, UNI)
    r4 = {"lab": "上場直後の株（上場から1か月〜1年。新しい順）", "var": {}, "nsc": nsc}
    rng = np.random.default_rng(20261104)
    rnd4 = {nm: [] for nm, _ in KEEPS}
    for _k in range(N_RAND if run_e4 else 0):
        sc_r = np.repeat(rng.random(UNI.shape[0])[:, None], M, axis=1)
        rk_r = ranks_from(sc_r, UNI)
        for nm, kp in KEEPS:
            v, _l, _s = sim(rk_r, D["PXM"], D["DIV"], j0, kp)
            rnd4[nm].append(half_stats(v, ev))
    if run_e4:
        for nm, kp in KEEPS:
            v, liq, s_ = sim(rk4, D["PXM"], D["DIV"], j0, kp)
            r4["var"][nm] = {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd4[nm], "vlab": f"{kp}位より下で売る",
                             "nb": {}, "held": float("nan"), "n_cand": int(np.isfinite(sc4).sum())}
        share_ok = float(np.mean([n >= IPO_MIN_N for n in nsc])) if nsc else 0.0
        r4["verdict"], r4["detail"] = judge2(r4["var"], ST_TX, lambda _x: share_ok < IPO_MIN_SHARE)
        res["E4"] = r4
        log(f"  E4 … {r4['verdict']}")
    # 暦の型（1306 を売り買い）
    mb = month_bounds(cal)
    mon = (cal.astype("datetime64[M]").astype(int) % 12) + 1

    def tom(pre):
        return tom_mask(cal, pre)

    def sim_may(last_mon):
        return may_mask(cal, last_mon)

    def cal_placebo_tom(n_days, seed):
        rng_ = np.random.default_rng(seed)
        x = np.zeros(T, dtype=bool)
        for s_, e_ in mb:
            if e_ - s_ + 1 > n_days:
                st_ = int(rng_.integers(s_, e_ - n_days + 1))
                x[st_:st_ + n_days] = True
        return x

    def cal_placebo_may(n_mon, seed):
        rng_ = np.random.default_rng(seed)
        x = np.zeros(T, dtype=bool)
        mends = [e_ for (_s, e_) in mb]
        oct_k = [k for k, e_ in enumerate(mends) if mon[e_] == 10]
        for k in oct_k:
            off = int(rng_.integers(0, 12 - n_mon + 1))
            a, b_ = k + off, k + off + n_mon
            if b_ < len(mends):
                x[mends[a]:mends[b_]] = True
        return x

    CAL_TYPES = (("E5", "月末月初だけ TOPIX を持つ", (("A", "月末の2日前に買い、3日目に売る（4日）", lambda: tom(1), lambda s: cal_placebo_tom(4, s)),
                                                ("B", "月末の4日前に買い、3日目に売る（6日）", lambda: tom(3), lambda s: cal_placebo_tom(6, s)))),
                 ("E6", "冬だけ TOPIX を持つ（10月末に買う）", (("A", "4月末に売る（6か月）", lambda: sim_may(4), lambda s: cal_placebo_may(6, s)),
                                                      ("B", "5月末に売る（7か月）", lambda: sim_may(5), lambda s: cal_placebo_may(7, s)))))
    for key, lab, variants in CAL_TYPES:
        if ONLY and key not in ONLY:
            continue
        r = {"lab": lab, "var": {}}
        for nm, vlab, mk, pmk in variants:
            x = mk()
            v, liq, s_ = cal_sim(x, D, t0)
            rnd = [half_stats(cal_sim(pmk(2000 + k), D, t0)[0], ev) for k in range(N_RAND)]
            nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
                  "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
            r["var"][nm] = {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "vlab": vlab, "nb": nb,
                            "held": float(x[t0:].mean()), "n_cand": int(x[t0:].sum())}
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda _x: False)
        res[key] = r
        log(f"  {key} … {r['verdict']}")
    report2(res, ref_bad, ST_TX, TXliq, ev, yrs, D, j0)
    return 0


def judge2(var: dict, ST_TX: dict, short) -> tuple:
    """2通りの両方・前半と後半の両方で、①TOPIX（税あり）を +2pt 以上 ②でたらめの平均を +2pt 以上かつ20回のうち18回以上。"""
    ok_tx, ok_rnd = True, True
    detail = {}
    for nm, x in var.items():
        for h in ("前半", "後半"):
            a = x["st"][h]["ann"]
            rnd = [y[h]["ann"] for y in x["rnd"]]
            d_tx = a - ST_TX[h]["ann"]
            d_rn = a - float(np.mean(rnd))
            beat = int(sum(1 for y in rnd if a > y))
            detail[(nm, h)] = (d_tx, d_rn, beat)
            ok_tx &= d_tx >= PT_TX
            ok_rnd &= (d_rn >= PT_RND) and (beat >= BEAT_MIN)
    if any(short(x) for x in var.values()):
        return "保留（データ不足）", detail
    return ("候補" if (ok_tx and ok_rnd) else ("範囲の中では効く" if ok_rnd else "効かない")), detail


def report2(res, ref_bad, ST_TX, TXliq, ev, yrs, D, j0):
    HH = ("前半", "後半")
    print("\n【日本株の手法の型・2回目（wide_lab）：出来事と暦・東証全銘柄・実際の運用の形】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan2_2026-11-04）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  決まり … 元本1,000万円・20銘柄まで・1株単位・終値で売り買い・片道{SLIP * 100:.1f}％（暦の型の 1306 は{ETF_SLIP * 100:.1f}％）・税20.315％（繰り越しの損あり）・"
          "使っていないお金に利息はつけない。買う候補は、その日より前の最後の月末に範囲（1回目と同じ）に入っていた銘柄")
    print(f"  判定 … 型ごとの2通り（A・B）の両方・前半と後半の両方で、①TOPIX（税あり）を +{PT_TX:.1f}pt 以上 ②でたらめ（同じ日に同じ数だけ、でたらめな銘柄を同じ日数持つ。"
          f"暦の型は同じ長さのでたらめな期間）の平均を +{PT_RND:.1f}pt 以上かつ20回のうち{BEAT_MIN}回以上上回る → ①②とも「候補」・②だけ「範囲の中では効く」・それ以外「効かない」。"
          f"出来事の型で、前半・後半のどちらかの買いが{EV_MIN_BUYS}回未満、またはでたらめの買いの回数が型の{RND_FILL_MIN * 100:.0f}％未満なら保留")
    print(f"\n  比べる相手 … TOPIX（税あり） {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}（年率％。「前半 ／ 後半」。通し {ST_TX['通し']['ann']:.1f}・最大下落 {ST_TX['通し']['mdd']:.0f}％）")
    print("\n  1. 型ごとの成績（年率％。「前半 ／ 後半」。［TOPIX との差 ／ でたらめの平均との差・でたらめを上回った回数］）")
    for key, r in res.items():
        print(f"     {key:<4}{r['lab']} … {r['verdict']}")
        for nm, x in r["var"].items():
            rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
            dd = " ".join(f"{r['detail'][(nm, h)][0]:+.1f}/{r['detail'][(nm, h)][1]:+.1f}・{r['detail'][(nm, h)][2]}回" for h in HH)
            nb = f"買い 前半 {x['nb'].get('前半', '-')}回・後半 {x['nb'].get('後半', '-')}回・" if x["nb"] else ""
            hold = f"平均の持ち数 {x['held']:.1f}銘柄・でたらめの買いの回数 {x['fill'] * 100:.0f}％・" if key not in ("E5", "E6", "E4") else (f"持っていた日の割合 {x['held'] * 100:.0f}％・" if key in ("E5", "E6") else "")
            print(f"         {nm}（{x['vlab']}） {x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}（最大下落 {x['st']['通し']['mdd']:.0f}％）［{dd}］"
                  f" でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}・{nb}{hold}清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％")
        if key == "E4":
            ns = r["nsc"]
            print(f"         点数のある銘柄の数（範囲の中） 平均 {np.mean(ns):.0f}・最少 {min(ns)}・20以上の月 {np.mean([n >= IPO_MIN_N for n in ns]) * 100:.0f}％")
    if ref_bad is not None:
        print(f"\n  参考（判定には使わない）：決算発表の反応が TOPIX より −5％以下の株を買って60日持つ … {ref_bad['st']['前半']['ann']:.1f} ／ {ref_bad['st']['後半']['ann']:.1f}"
              f"（でたらめ {np.mean([y['前半']['ann'] for y in ref_bad['rnd']]):.1f} ／ {np.mean([y['後半']['ann'] for y in ref_bad['rnd']]):.1f}・"
              f"買い 前半 {ref_bad['nb']['前半']}回・後半 {ref_bad['nb']['後半']}回）")
    print("\n  2. 判定のまとめ")
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in res.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")


# ───────────────────────── 6段目：短期売買・テクニカル ─────────────────────────
# 計画は wide_lab_plan6_2026-11-04（結果を見る前に決めたもの）。日足の始値・高値・安値・終値・出来高で決まる合図で買い、決めた日（または合図）で売る。
# 運用のエンジンは2回目と同じ（ev_sim・20銘柄まで・その日の終値で売り買い）。でたらめは、同じ日に同じ数だけ、範囲のでたらめな銘柄を同じ日数持つ


def rolling_mean(X: np.ndarray, w: int) -> np.ndarray:
    """横（日）方向の移動平均（値のない日は前の値で埋めた上で）。最初の w−1 日は空。"""
    cs = np.nancumsum(np.nan_to_num(X, nan=0.0), axis=1, dtype=np.float64)
    out = np.full(X.shape, np.nan)
    out[:, w - 1:] = (cs[:, w - 1:] - np.concatenate([np.zeros((X.shape[0], 1)), cs[:, :-w]], axis=1)) / w
    return out


def rolling_max(X: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(X.T).rolling(w, min_periods=w).max().to_numpy().T


def rsi(CF: np.ndarray, w: int = 14) -> np.ndarray:
    d = np.diff(CF, axis=1, prepend=np.nan)
    up = pd.DataFrame(np.where(d > 0, d, 0.0).T).ewm(alpha=1 / w, min_periods=w).mean().to_numpy().T
    dn = pd.DataFrame(np.where(d < 0, -d, 0.0).T).ewm(alpha=1 / w, min_periods=w).mean().to_numpy().T
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100.0 - 100.0 / (1.0 + up / dn)


def spells_placebo(x: np.ndarray, t0: int, seed: int) -> np.ndarray:
    """持っている期間と持っていない期間の長さをそのままに、順番をでたらめに入れ替える（暦の型・TOPIX の切り替えのでたらめ）。"""
    rng = np.random.default_rng(seed)
    seg = x[t0:]
    runs, cur, n = [], seg[0], 0
    for v in seg:
        if v == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = v, 1
    runs.append((cur, n))
    ins = [n for v, n in runs if v]
    outs = [n for v, n in runs if not v]
    rng.shuffle(ins)
    rng.shuffle(outs)
    seq = []
    first_in = bool(runs[0][0])
    a, b = list(ins), list(outs)
    turn = first_in
    while a or b:
        if turn and a:
            seq += [True] * a.pop()
        elif (not turn) and b:
            seq += [False] * b.pop()
        elif a:
            seq += [True] * a.pop()
        else:
            seq += [False] * b.pop()
        turn = not turn
    out = x.copy()
    out[t0:] = np.array(seq[:len(seg)], dtype=bool)
    return out


def lab6(store: dict) -> int:
    log("6段目（短期売買・テクニカル）を回します（指標を作っています）…")
    D = build(store)
    if D.get("OHLV") is None:
        print("\n■ 始値・高値・安値がありません（20年分のデータで回してください）")
        return 2
    dates, UNI, me_idx, cal = D["dates"], D["UNI"], D["me_idx"], D["cal"]
    M, T = len(dates), len(cal)
    C, CF, O, H, L, V = D["C"], D["CF"], D["OHLV"]["o"], D["OHLV"]["h"], D["OHLV"]["l"], D["OHLV"]["v"]
    j0 = next((j for j, t in enumerate(me_idx) if t >= 252), None)
    if j0 is None or M - j0 < 36:
        print("\n■ 比べる期間が短すぎます。")
        return 1
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    E = ev_extract(store, D)
    jm = np.searchsorted(np.array(me_idx), np.arange(T), side="left") - 1
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    traded = np.isfinite(C) & (C > 0)
    ELIG = np.zeros((UNI.shape[0], T), dtype=bool)          # その日に買える（前の月末に範囲に入っていて、その日に売買がある）
    for t in range(t0, T):
        if jm[t] >= 0:
            ELIG[:, t] = UNI[:, jm[t]] & traded[:, t]
    log("  合図を作っています…")
    PC = np.concatenate([np.full((C.shape[0], 1), np.nan), CF[:, :-1]], axis=1)    # 前の日の終値
    PH = np.concatenate([np.full((C.shape[0], 1), np.nan), pd.DataFrame(H).ffill(axis=1).to_numpy()[:, :-1]], axis=1)
    PL = np.concatenate([np.full((C.shape[0], 1), np.nan), pd.DataFrame(L).ffill(axis=1).to_numpy()[:, :-1]], axis=1)
    rng_pr = np.random.default_rng(6)
    RP = rng_pr.random(C.shape)                               # 優先度を決めない型の、でたらめな優先度（種を固定）

    def events_from(sig: np.ndarray, prio: np.ndarray, hold, exit_fn=None) -> dict:
        out = {}
        ii, tt = np.nonzero(sig & ELIG)
        for i, t in zip(ii, tt):
            if t < t0:
                continue
            t_out = exit_fn(i, t) if exit_fn else t + hold
            out.setdefault(int(t), []).append((float(prio[i, t]), int(i), int(min(t_out, T - 1 if t_out >= T else t_out))))
        return out

    def ma_cross(sw, lw):
        ms, ml = rolling_mean(CF, sw), rolling_mean(CF, lw)
        above = ms > ml
        prev = np.concatenate([np.zeros((C.shape[0], 1), dtype=bool), above[:, :-1]], axis=1)
        sig = above & ~prev & np.isfinite(ml)
        below_t = {}

        def ex(i, t):
            if i not in below_t:
                below_t[i] = np.nonzero(~above[i] & np.isfinite(ml[i]))[0]
            arr = below_t[i]
            k = np.searchsorted(arr, t + 1)
            return int(arr[k]) if k < len(arr) else min(t + 250, T - 1)
        return events_from(sig, RP, None, ex)

    def breakout(w, hold=20):
        hi = rolling_max(CF, w)
        prevhi = np.concatenate([np.full((C.shape[0], 1), np.nan), hi[:, :-1]], axis=1)
        sig = traded & (C >= prevhi) & np.isfinite(prevhi)
        return events_from(sig, RP, hold)

    R14 = rsi(CF)

    def rsi_low(th, hold=10):
        return events_from(traded & (R14 < th), 100.0 - np.nan_to_num(R14, nan=100.0), hold)

    def boll(k, hold=10):
        m = rolling_mean(CF, 20)
        sd = pd.DataFrame(CF.T).rolling(20, min_periods=20).std().to_numpy().T
        z = (CF - m) / sd
        return events_from(traded & (z < -k), -np.nan_to_num(z, nan=0.0), hold)

    def gap_up(hold):
        g = O / PH - 1.0
        return events_from(traded & (g >= 0.03), np.nan_to_num(g), hold)

    def gap_down(hold):
        g = 1.0 - O / PL
        return events_from(traded & (g >= 0.03), np.nan_to_num(g), hold)

    VM = rolling_mean(np.nan_to_num(V), 20)
    PVM = np.concatenate([np.full((C.shape[0], 1), np.nan), VM[:, :-1]], axis=1)

    def vol_spike(hold):
        r = V / PVM
        return events_from(traded & (r >= 5.0) & (CF > PC), np.nan_to_num(r), hold)

    C5 = np.concatenate([np.full((C.shape[0], 5), np.nan), CF[:, :-5]], axis=1)

    def drop5(hold):
        r = CF / C5 - 1.0
        return events_from(traded & (r <= -0.15), -np.nan_to_num(r), hold)

    def run(cands, tag):
        v, liq, s_ = ev_sim(cands, D, E["divpay"], t0)
        rnd, rnd_nb = [], []
        for k in range(N_RAND):
            rg = np.random.default_rng(3000 + k)
            pl = {}
            for t, cl in cands.items():
                pool = np.nonzero(ELIG[:, t])[0]
                if len(pool) == 0:
                    continue
                pick = rg.choice(pool, size=min(len(cl), len(pool)), replace=False)
                pl[t] = [(float(rg.random()), int(i), t_out) for i, (_p, _i, t_out) in zip(pick, cl)]
            pv, _l, ps_ = ev_sim(pl, D, E["divpay"], t0)
            rnd.append(half_stats(pv, ev))
            rnd_nb.append(ps_["n_buy"])
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
              "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        return {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "nb": nb,
                "held": float(np.mean(s_["held"])) if s_["held"] else 0.0, "fill": float(np.mean(rnd_nb)) / max(1, s_["n_buy"]),
                "n_cand": sum(len(x) for x in cands.values())}

    TYPES = (("T1", "移動平均の上抜け（上抜けで買い、下抜けで売る）", (("A", "5日と25日", lambda: ma_cross(5, 25)), ("B", "25日と75日", lambda: ma_cross(25, 75)))),
             ("T2", "高値の更新（終値が過去の最高値以上で買い、20日持つ）", (("A", "60日の高値", lambda: breakout(60)), ("B", "250日の高値", lambda: breakout(250)))),
             ("T3", "RSI の売られすぎ（14日 RSI が下回ったら買い、10日持つ）", (("A", "30未満", lambda: rsi_low(30)), ("B", "20未満", lambda: rsi_low(20)))),
             ("T4", "ボリンジャーバンドの下抜け（20日の平均から下に k 標準偏差、10日持つ）", (("A", "2標準偏差", lambda: boll(2.0)), ("B", "3標準偏差", lambda: boll(3.0)))),
             ("T5", "上への窓開け（始値が前の日の高値より3％以上上）", (("A", "5日持つ", lambda: gap_up(5)), ("B", "20日持つ", lambda: gap_up(20)))),
             ("T6", "下への窓開け（始値が前の日の安値より3％以上下）", (("A", "5日持つ", lambda: gap_down(5)), ("B", "20日持つ", lambda: gap_down(20)))),
             ("T7", "出来高の急増（20日平均の5倍以上で、上げて引けた日）", (("A", "5日持つ", lambda: vol_spike(5)), ("B", "20日持つ", lambda: vol_spike(20)))),
             ("T8", "5日で15％以上下げた株（数日の戻り）", (("A", "5日持つ", lambda: drop5(5)), ("B", "20日持つ", lambda: drop5(20)))))
    res = {}
    for key, lab, variants in TYPES:
        if ONLY and key not in ONLY:
            continue
        r = {"lab": lab, "var": {}}
        for nm, vlab, mk in variants:
            cands = mk()
            r["var"][nm] = dict(run(cands, key + nm), vlab=vlab)
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda x: min(x["nb"].values()) < EV_MIN_BUYS or x["fill"] < RND_FILL_MIN)
        res[key] = r
        log(f"  {key} … {r['verdict']}")
    # T9：TOPIX を移動平均より上のときだけ持つ（1306・暦の型と同じエンジン）
    if not ONLY or "T9" in ONLY:
        tpc = D["tp"]["c"].astype(float)
        r = {"lab": "TOPIX を移動平均より上のときだけ持つ（1306）", "var": {}}
        for nm, w in (("A", 200), ("B", 100)):
            ma = pd.Series(tpc).rolling(w, min_periods=w).mean().to_numpy()
            x = np.nan_to_num(tpc > ma, nan=False).astype(bool)
            v, liq, s_ = cal_sim(x, D, t0)
            rnd = [half_stats(cal_sim(spells_placebo(x, t0, 4000 + k), D, t0)[0], ev) for k in range(N_RAND)]
            nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
                  "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
            r["var"][nm] = {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "vlab": f"{w}日の移動平均", "nb": nb,
                            "held": float(x[t0:].mean()), "n_cand": int(x[t0:].sum()), "fill": 1.0}
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda _x: False)
        res["T9"] = r
        log(f"  T9 … {r['verdict']}")
    report6(res, ST_TX, TXliq, ev, yrs)
    return 0


def report6(res, ST_TX, TXliq, ev, yrs):
    HH = ("前半", "後半")
    print("\n【日本株の手法・6段目（短期売買・テクニカル）（wide_lab）：東証全銘柄・実際の運用の形】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan6_2026-11-04）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  決まり … 2回目と同じ（元本1,000万円・20銘柄まで・その日の終値で売り買い・片道{SLIP * 100:.1f}％・税20.315％）。T9 は 1306 を売り買い（片道{ETF_SLIP * 100:.1f}％）")
    print(f"  判定 … 2通り（A・B）の両方・前半と後半の両方で、TOPIX +{PT_TX:.1f}pt・でたらめ（同じ日に同じ数だけ範囲のでたらめな銘柄を同じ日数持つ。T9 は持つ期間の長さをそのままに順番を入れ替える）"
          f" +{PT_RND:.1f}pt かつ20回のうち{BEAT_MIN}回以上")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    print("\n  1. 型ごとの成績（年率％。「前半 ／ 後半」。［TOPIX との差 ／ でたらめの平均との差・でたらめを上回った回数］）")
    for key, r in res.items():
        print(f"     {key:<4}{r['lab']} … {r['verdict']}")
        for nm, x in r["var"].items():
            rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
            dd = " ".join(f"{r['detail'][(nm, h)][0]:+.1f}/{r['detail'][(nm, h)][1]:+.1f}・{r['detail'][(nm, h)][2]}回" for h in HH)
            extra = (f"持っていた日の割合 {x['held'] * 100:.0f}％" if key == "T9" else
                     f"合図 {x['n_cand']:,}回・買い 前半 {x['nb']['前半']}回・後半 {x['nb']['後半']}回・平均の持ち数 {x['held']:.1f}・でたらめの買いの回数 {x['fill'] * 100:.0f}％")
            print(f"         {nm}（{x['vlab']}） {x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}（最大下落 {x['st']['通し']['mdd']:.0f}％）［{dd}］"
                  f" でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}・{extra}・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％")
    print("\n  2. 判定のまとめ")
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in res.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")


# ───────────────────────── 5段目：残りの型 ─────────────────────────
# 計画は wide_lab_plan5_2026-11-04（結果を見る前に決めたもの）。
# 毎月の点数の型：ACCR（利益の質＝アクルーアルの小ささ）・SECT（業種の勢い）・SSR（大口の空売りの報告がない株）・
#                 MLONG（信用買い残の重さ）・MBAL（信用の売り残と買い残の釣り合い）
# 出来事の型：DIVUP（増配の予想の発表）・E2C（決算発表予定日の前に買う＝2回目の E2 を、前もって公表された予定日で作り直したもの）
X_CACHE = OUT / ("wide_x_long.pkl" if LONG else "wide_x.pkl")
# 名前 → （道、データの始まり、日付を渡す名前、D＝営業日ごと・W＝週の最後の営業日ごと）
X_SETS = {"div": ("/v2/fins/dividend", "2013-02-20", "date", "D"),               # 公表日ごと（プレミアムだけ）
          "ssr": ("/v2/markets/short-sale-report", "2013-11-07", "disc_date", "D"),  # 公表日ごと
          "ed": ("/v2/fins/earnings-date", "2013-01-01", "date", "D"),              # 予定日を公表・変更した日ごと（始まりは仕様にない）
          "mi": ("/v2/markets/margin-interest", "2008-05-01", "date", "W")}         # 信用取引の週末残高（週の最後の営業日）
X_FIELDS = {"div": {"pub": ("PubDate",), "time": ("PubTime",), "code": ("Code",), "stat": ("StatCode",), "if": ("IFCode",),
                    "fr": ("FRCode",), "term": ("IFTerm",), "rate": ("DivRate",), "cs": ("CommSpecCode",), "rec": ("RecDate",)},
            "ssr": {"disc": ("DiscDate",), "calc": ("CalcDate",), "code": ("Code",), "seller": ("SSName",),
                    "dic": ("DICName",), "fund": ("FundName",), "ratio": ("ShrtPosToSO",)},
            "ed": {"pub": ("PubDate",), "sch": ("SchDate",), "fq": ("FQName",), "fye": ("FYE",), "code": ("Code",)},
            "mi": {"date": ("Date",), "code": ("Code",), "long": ("LongVol",), "short": ("ShrtVol",), "iss": ("IssType",)}}
X_OPTIONAL = {("ssr", "dic"), ("ssr", "fund"), ("div", "time"), ("ed", "fye"), ("mi", "iss")}


def x_load() -> dict:
    if X_CACHE.exists():
        with open(X_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def x_save(xs: dict, note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = X_CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(xs, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, X_CACHE)
    if note:
        log(f"  {note}（{X_CACHE.stat().st_size / 1e6:,.1f}MB・開始から{elapsed():.0f}分）")


def x_days(store: dict, name: str) -> list:
    """その種類で取る日。探す段階では、探す期間の始まりの500日前から（それより前は使わないので取らない）。"""
    _p, st_, _k, kind = X_SETS[name]
    start = np.datetime64(st_)
    if STAGE == "explore":
        start = max(start, EXPLORE_FROM - np.timedelta64(500, "D"))
    days = [d for d in store["etf1306"]["d"] if d >= start]
    if kind == "W":
        wk = {}
        for d in days:
            ic = pd.Timestamp(d).isocalendar()
            wk[(ic[0], ic[1])] = d
        days = sorted(wk.values())[:-1] if wk else []      # 最後の週は、まだ公表されていないことがあるので外す
    return [str(d) for d in days]


def x_complete(store: dict, xs: dict) -> bool:
    for n in X_SETS:
        st = xs.get(n, {})
        if st.get("bad"):
            continue
        if not set(x_days(store, n)) <= (st.get("done", set()) | st.get("fail", set())):
            return False
    return True


def fetch_x(jq, store: dict, xs: dict) -> bool:
    """配当金情報・空売り残高の報告・決算発表予定日・信用取引の週末残高を、日ごとに取る。時間切れなら取った分を保存して False。
    取れない種類（プランにない・項目が違う・最近の日でも取れない）は、その種類だけ「保留」にして先へ進む。"""
    last = time.monotonic()
    for name, (path, _start, prm, _kind) in X_SETS.items():
        st = xs.setdefault(name, {"done": set(), "fail": set(), "rows": [], "keys": None})
        st.setdefault("fail", set())
        if st.get("bad"):
            continue
        days = x_days(store, name)
        todo = [d for d in days if d not in st["done"] and d not in st["fail"]]
        if not todo:
            continue
        log(f"{name} を取っています（{len(days)}日・残り {len(todo)}）…")
        fmap = st.get("fmap")

        def bad(why):
            st["bad"] = why
            log(f"  ■ {name} は取るのをやめます（{why}）。この種類の型は「保留」になります")
        # 最初に、最近の日で取れるか確かめる（取れなければ、この種類は保留）
        if not st["done"]:
            probe = days[-3] if len(days) >= 3 else days[-1]
            try:
                jq.get(path, {prm: probe})
            except yb.NotAllowed as e:
                bad(f"プランに含まれていない可能性があります：{e}")
                continue
            except Exception as e:
                bad(f"最近の日（{probe}）でも取れません：{e}")
                continue
        fails, ok_once, shown = 0, bool(st["done"]), 0
        for n, d in enumerate(todo, 1):
            if elapsed() >= BUDGET_MIN:
                x_save(xs, f"{name} を途中まで保存しました")
                return False
            time.sleep(yb.API_SLEEP)
            try:
                rows = jq.get(path, {prm: d})
            except yb.NotAllowed as e:
                bad(f"プランに含まれていない可能性があります：{e}")
                break
            except Exception as e:
                st["fail"].add(d)
                fails += 1
                if shown < 3 or fails % 100 == 0:
                    log(f"  取れない日 {name} {d}（{fails}日続けて）: {str(e)[:100]}")
                    shown += 1
                if ok_once and fails >= 20:
                    bad("取れていたのに、続けて20日取れません")
                    break
                continue
            fails, ok_once = 0, True
            if rows and fmap is None:
                ks = list(rows[0].keys())
                st["keys"] = ks
                log("  受け取った項目 … " + "・".join(ks))
                fmap = {}
                for lk, cands in X_FIELDS[name].items():
                    k = next((c for c in cands if c in ks), None)
                    if k is None and (name, lk) not in X_OPTIONAL:
                        bad(f"項目 {lk} が見つかりません")
                        break
                    fmap[lk] = k
                if st.get("bad"):
                    break
                st["fmap"] = fmap
            for r in rows or []:
                st["rows"].append(tuple(r.get(fmap[lk]) if fmap[lk] else None for lk in X_FIELDS[name]))
            st["done"].add(d)
            if n % 500 == 0:
                log(f"  {n}/{len(todo)}（開始から{elapsed():.0f}分）")
            if (time.monotonic() - last) / 60 >= SAVE_EVERY_MIN:
                x_save(xs, "途中まで保存しました")
                last = time.monotonic()
        if st["fail"] and not st.get("bad"):
            log(f"  {name} の取れなかった日 {len(st['fail'])}日（いちばん新しい日 {max(st['fail'])}）")
        x_save(xs, f"{name} を取り終えました（{len(st['rows']):,}件）")
    return True


def x_rows(xs: dict, name: str) -> list:
    """取った行を、項目名つきの辞書のリストにする。"""
    ks = list(X_FIELDS[name])
    return [dict(zip(ks, r)) for r in xs.get(name, {}).get("rows", [])]


def accr_score(store: dict):
    """ACCR：−（直近の通期の純利益 − 営業CF）÷ 総資産（金融を除く）。利益のうち現金を伴わない部分が小さい会社ほど上。"""
    def fn(D):
        codes, dates = D["codes"], D["dates"]
        out = np.full((len(codes), len(dates)), np.nan)
        me_np = np.array([np.datetime64(d.date()) for d in dates])
        for i, c in enumerate(codes):
            if D["sector"].get(c, "") in yb.FIN_SECTORS:
                continue
            tl = [r for r in fin_timeline(store["fins"].get(c)) if is_fy(r)]
            if not tl:
                continue
            dd = np.array([np.datetime64(str(r["DiscDate"])[:10]) for r in tl])
            for j in range(len(dates)):
                k = int(np.searchsorted(dd, me_np[j], side="left")) - 1
                if k < 0:
                    continue
                r = tl[k]
                np_, cfo, ta = num(r.get("NP")), num(r.get("CFO")), num(r.get("TA"))
                if np_ is not None and cfo is not None and ta and ta > 0:
                    out[i, j] = -(np_ - cfo) / ta
        return out
    return fn


def sect_score(D):
    """SECT：その月末の範囲の中で、業種（33業種）ごとの12か月の勢い（直近1か月を除く）の平均。業種の中では時価総額の大きい順に並ぶよう、ごく小さく足す。"""
    UNI, MOM, MC = D["UNI"], D["F"]["MOM12"], D["MCAP"]
    codes = D["codes"]
    sec = np.array([D["sector"].get(c, "") for c in codes])
    out = np.full(UNI.shape, np.nan)
    for j in range(UNI.shape[1]):
        ok = UNI[:, j] & np.isfinite(MOM[:, j])
        if ok.sum() < 30:
            continue
        for sname in set(sec[ok]):
            m = ok & (sec == sname)
            if m.sum() < 3:
                continue
            v = float(np.mean(MOM[m, j]))
            mc = MC[m, j]
            out[m, j] = v + 1e-6 * (pd.Series(mc).rank(pct=True).to_numpy())
    return out


def ssr_score(xs: dict):
    """SSR：その月末に、大口の空売りの報告（0.5％以上）が出ていない株ほど上。−（売り手ごとの最新の報告の比率の合計）。
    最新の報告の計算日が月末から365日より前なら、もう出ていないとみなす。同じ点数（報告なし）どうしは、銘柄ごとに固定したでたらめの順で並べる。"""
    rows = x_rows(xs, "ssr")
    rs = [num(r.get("ratio")) for r in rows]
    rs = [x for x in rs if x is not None]
    scale = 100.0 if (rs and float(np.median(rs)) < 0.05) else 1.0       # 割合（0.005）なら％にそろえる
    log(f"  空売り残高の報告 … {len(rows):,}件（比率の中央値 {float(np.median(rs)) if rs else float('nan'):.4f} → ％にする倍率 {scale:.0f}）")

    def fn(D):
        codes, dates = D["codes"], D["dates"]
        ci = {c: i for i, c in enumerate(codes)}
        out = np.zeros((len(codes), len(dates)))
        by = {}
        for r in rows:
            c = yb.norm_code(r.get("code") or "")
            if c in ci:
                who = "｜".join(str(r.get(k) or "") for k in ("seller", "dic", "fund"))     # 売り手・投資一任の契約の相手・ファンドの組
                by.setdefault(c, []).append((str(r.get("disc") or "")[:10], str(r.get("calc") or "")[:10], who, num(r.get("ratio"))))
        for c, lst in by.items():
            lst.sort(key=lambda x: (x[0], x[1]))
            i = ci[c]
            for j, d in enumerate(dates):
                me = d.strftime("%Y-%m-%d")
                lim = (d - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
                latest = {}
                for disc, calc, seller, ratio in lst:
                    if disc >= me:
                        break
                    latest[seller] = (calc, ratio)
                tot = 0.0
                for calc, ratio in latest.values():
                    if ratio is not None and calc >= lim:
                        rr = ratio * scale
                        if rr >= 0.5:
                            tot += rr
                out[i, j] = -tot
        noise = np.random.default_rng(55).random(len(codes))[:, None] * 1e-6     # 銘柄ごとに固定（毎月入れ替わらないように）
        return np.where(D["UNI"], out + noise, np.nan)
    return fn


def mi_score(store: dict, xs: dict, kind: str):
    """信用取引の週末残高（週の最後の営業日の残高。翌週の2営業日目に公表）から、月末の点数を作る。
    月末の7日前までの日付の残高のうち最新のもの（30日より古ければ使わない）。
    MLONG：−（信用買い残の金額 ÷ 直近20営業日の1日の売買代金の平均）＝買い残が何日分の売買にあたるか。少ないほど上
    MBAL：（売り残 − 買い残）÷（売り残 + 買い残）。売り残が多いほど上（買い戻しが入りやすい）"""
    rows = x_rows(xs, "mi")

    def fn(D):
        codes, dates, me_idx = D["codes"], D["dates"], D["me_idx"]
        ci = {c: i for i, c in enumerate(codes)}
        CF, VA = D["CF"], D["VA"]
        out = np.full((len(codes), len(dates)), np.nan)
        by = {}
        for r in rows:
            c = yb.norm_code(r.get("code") or "")
            lv, sv = num(r.get("long")), num(r.get("short"))
            if c in ci and lv is not None and sv is not None:
                by.setdefault(c, []).append((str(r.get("date") or "")[:10], lv, sv))
        for c, lst in by.items():
            lst.sort()
            ds = np.array([x[0] for x in lst])
            i = ci[c]
            b = store["bars"].get(c)
            if b is None:
                continue
            for j, d in enumerate(dates):
                hi = (d - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
                lo = (d - pd.Timedelta(days=37)).strftime("%Y-%m-%d")
                k = int(np.searchsorted(ds, hi, side="right")) - 1
                if k < 0 or ds[k] < lo:
                    continue
                dd, lv, sv = lst[k]
                if kind == "MBAL":
                    if lv + sv > 0:
                        out[i, j] = (sv - lv) / (lv + sv)
                    continue
                t = me_idx[j]
                px = CF[i, t]
                va = float(np.mean(VA[i, max(0, t - 19):t + 1]))
                if px == px and px > 0 and va > 0:
                    out[i, j] = -(lv / fac_after(b["fd"], b["ff"], dd)) * float(px) / va
        return out
    return fn


def lab5(store: dict, xs: dict) -> int:
    log("5段目（残りの型）を回します…")
    # 毎月の点数の型（1回目と同じ道具・同じ判定）
    extra = {"ACCR": (accr_score(store), "利益の質（アクルーアルの小ささ。金融を除く）"),
             "SECT": (sect_score, "業種の勢い（33業種の12か月の勢いの平均。業種の中では大きい順）"),
             "SSR": (ssr_score(xs), "大口の空売りの報告がない株（報告の比率の合計の小さい順）"),
             "MLONG": (mi_score(store, xs, "MLONG"), "信用買い残の軽さ（買い残が何日分の売買にあたるかの少ない順）"),
             "MBAL": (mi_score(store, xs, "MBAL"), "信用の売り残の多さ（（売り残 − 買い残）÷（売り残 + 買い残）の大きい順）")}
    for nm, k in (("SSR", "ssr"), ("MLONG", "mi"), ("MBAL", "mi")):
        if xs.get(k, {}).get("bad"):
            log(f"  {nm} … データが取れなかったので飛ばします（保留）")
            extra.pop(nm)
    lab1(store, extra=extra)
    # 出来事の型
    log("5段目の出来事の型を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, me_idx, cal = D["dates"], D["UNI"], D["me_idx"], D["cal"]
    T = len(cal)
    C = D["C"]
    j0 = next((j for j, t in enumerate(me_idx) if t >= 252), None)
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    E = ev_extract(store, D)
    jm = np.searchsorted(np.array(me_idx), np.arange(T), side="left") - 1
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    codes = D["codes"]
    ci = {c: i for i, c in enumerate(codes)}
    traded = np.isfinite(C) & (C > 0)

    def elig(i, t):
        return 0 <= t < T and jm[t] >= 0 and bool(UNI[i, jm[t]]) and bool(traded[i, t])

    def pool_all(t):
        j = jm[t]
        return np.nonzero(UNI[:, j] & traded[:, t])[0] if j >= 0 else np.zeros(0, dtype=int)
    # DIVUP：増配の予想の発表
    divs = {}
    for r in x_rows(xs, "div"):
        c = yb.norm_code(r.get("code") or "")
        if c in ci:
            divs.setdefault(c, []).append(r)
    evs_div = []
    for c, lst in divs.items():
        i = ci[c]
        b = store["bars"][c]
        lst.sort(key=lambda r: (str(r.get("pub") or ""), str(r.get("time") or "")))
        known = {}                 # (IFTerm, IFCode) → 最新の値（いまの株数の基準）
        result = {}                # (IFTerm, IFCode) → 実績
        for r in lst:
            if str(r.get("stat")) == "3":
                continue
            rate = num(r.get("rate"))
            rec = str(r.get("rec") or "")[:10]
            term, ifc, frc = str(r.get("term") or ""), str(r.get("if") or ""), str(r.get("fr") or "")
            if rate is None or len(rec) != 10 or not term:
                continue
            v = rate * fac_after(b["fd"], b["ff"], rec)
            key = (term, ifc)
            if frc == "1":
                result[key] = v
                known[key] = v
                continue
            if str(r.get("cs") or "0") not in ("0", "", "None"):     # 記念・特別配当を含む予想は比べない
                continue
            prev = known.get(key)
            if prev is None:
                try:
                    py = (pd.Period(term, "M") - 12).strftime("%Y-%m")
                except Exception:
                    py = None
                prev = result.get((py, ifc)) if py else None
            known[key] = v
            if prev and prev > 0 and v / prev - 1.0 >= 0.10:
                pubd = np.datetime64(str(r.get("pub"))[:10])
                t_a = int(np.searchsorted(cal, pubd, side="right"))
                evs_div.append((i, t_a, v / prev - 1.0))
    # E2C：決算発表予定日の前に買う。予定日は、買う日より前に公表され、その日にまだ変わっていないものだけを使う
    hist = {}
    for r in x_rows(xs, "ed"):
        c = yb.norm_code(r.get("code") or "")
        pub, sch = str(r.get("pub") or "")[:10], str(r.get("sch") or "")[:10]
        if c in ci and len(pub) == 10:
            hist.setdefault((ci[c], str(r.get("fq") or ""), str(r.get("fye") or "")), []).append((pub, sch))
    sched_recs = []          # （銘柄, 公表日, 予定日の番号, 次の公表日）
    SCHP = np.zeros((len(codes), T + 1), dtype=bool)
    for (i, _fq, _fye), lst in hist.items():
        lst.sort()
        for k, (pub, sch) in enumerate(lst):
            if len(sch) != 10 or sch <= pub:
                continue
            t_s = int(np.searchsorted(cal, np.datetime64(sch), side="left"))
            if not (0 < t_s < T):
                continue
            nxt = lst[k + 1][0] if k + 1 < len(lst) else "9999-12-31"
            sched_recs.append((i, pub, t_s, nxt))
            SCHP[i, t_s - 1] = True
    evs_cal = sorted({(i, t_s) for i, _p, t_s, _n in sched_recs})
    log(f"  出来事 … 増配の予想 {len(evs_div):,}回・決算発表予定日 {len(evs_cal):,}回")

    def placebo_run(cands, pool_fn):
        v, liq, s_ = ev_sim(cands, D, E["divpay"], t0)
        rnd, rnb = [], []
        for k in range(N_RAND):
            rg = np.random.default_rng(5000 + k)
            pl = {}
            for t, cl in cands.items():
                pool = pool_fn(t)
                if len(pool) == 0:
                    continue
                pick = rg.choice(pool, size=min(len(cl), len(pool)), replace=False)
                pl[t] = [(float(rg.random()), int(i), t_out) for i, (_p, _i, t_out) in zip(pick, cl)]
            pv, _l, ps_ = ev_sim(pl, D, E["divpay"], t0)
            rnd.append(half_stats(pv, ev))
            rnb.append(ps_["n_buy"])
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
              "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        return {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "nb": nb, "held": float(np.mean(s_["held"])) if s_["held"] else 0.0,
                "fill": float(np.mean(rnb)) / max(1, s_["n_buy"]), "n_cand": sum(len(x) for x in cands.values())}

    def divup(H):
        out = {}
        for i, t_a, chg in evs_div:
            if t_a >= t0 and elig(i, t_a):
                out.setdefault(t_a, []).append((chg, i, t_a + H))
        return out

    def e2c(L):
        rg = np.random.default_rng(1)
        out, seen = {}, set()
        for i, pub, t_s, nxt in sorted(sched_recs, key=lambda x: (x[2], x[0], x[1])):
            t_in = t_s - 1 - L
            if t_in < t0 or (i, t_s) in seen or not elig(i, t_in):
                continue
            day = str(cal[t_in])[:10]
            if not (pub < day <= nxt):          # 買う日の前に公表され、買う日の時点でまだ変わっていない
                continue
            seen.add((i, t_s))
            out.setdefault(t_in, []).append((float(rg.random()), i, t_s - 1))
        return out

    def pool_nosched(L):
        def f(t):
            el = pool_all(t)
            return el[~SCHP[el, t:t + L + 2].any(axis=1)] if len(el) else el
        return f
    TYPES = (("DIVUP", "増配の予想の発表（前の予想・前の年の実績より10％以上。記念・特別配当は除く）",
              (("A", "60日持つ", lambda: divup(60), pool_all), ("B", "20日持つ", lambda: divup(20), pool_all))),
             ("E2C", "決算発表予定日の前に買う（予定日の前の営業日に売る）",
              (("A", "10日前に買う", lambda: e2c(10), pool_nosched(10)), ("B", "5日前に買う", lambda: e2c(5), pool_nosched(5)))))
    res = {}
    for key, lab, variants in TYPES:
        if ONLY and key not in ONLY:
            continue
        r = {"lab": lab, "var": {}}
        for nm, vlab, mk, pf in variants:
            r["var"][nm] = dict(placebo_run(mk(), pf), vlab=vlab)
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda x: min(x["nb"].values()) < EV_MIN_BUYS or x["fill"] < RND_FILL_MIN)
        res[key] = r
        log(f"  {key} … {r['verdict']}")
    HH = ("前半", "後半")
    print("\n【日本株の手法・5段目（残りの型：出来事）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan5_2026-11-04）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）。決まりと判定は2回目と同じ")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    for key, r in res.items():
        print(f"     {key:<6}{r['lab']} … {r['verdict']}")
        for nm, x in r["var"].items():
            rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
            dd = " ".join(f"{r['detail'][(nm, h)][0]:+.1f}/{r['detail'][(nm, h)][1]:+.1f}・{r['detail'][(nm, h)][2]}回" for h in HH)
            print(f"         {nm}（{x['vlab']}） {x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}（最大下落 {x['st']['通し']['mdd']:.0f}％）［{dd}］"
                  f" でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}・合図 {x['n_cand']:,}回・買い 前半 {x['nb']['前半']}回・後半 {x['nb']['後半']}回・"
                  f"でたらめの買いの回数 {x['fill'] * 100:.0f}％・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％")
    print("\n  判定のまとめ（出来事）")
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in res.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    # SSR の空売りの形：大口の空売りの報告が多い株を売る（LS・TXS。8段目と同じ道具・同じ判定）
    if (not ONLY or "SSR" in ONLY) and not xs.get("ssr", {}).get("bad"):
        lab8(store, extra={"SSR": (ssr_score(xs), "大口の空売りの報告")},
             head="【日本株の手法・5段目（残りの型：SSR の空売りの形）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan5_2026-11-04。"
                  "売る側は、大口の空売りの報告の比率の合計が大きい順）")
    return 0


# ───────────────────────── 9段目：多くの型をまとめて学ばせる ─────────────────────────
# 計画は wide_lab_plan9_2026-11-04（結果を見る前に決めたもの）。月末ごとに、それまでの36か月だけを使って型の重みを決め（先の情報は使わない）、
# その重みで点数を作って運用する。ML1＝型ごとの「次の月の成績との順位の相関（IC）」の平均を重みにする（正のものだけ）。ML2＝リッジ回帰
ML_WIN = 36                  # 重みを決めるのに使う月数
ML_MIN = 24                  # これより少ない月しかないときは点数を出さない
ML_RIDGE = 10.0              # リッジ回帰の強さ（型の順位を平均0にしたあとの、1銘柄・1か月あたり）
ML_CONS = (("ALL", 20, "M"), ("TOP500", 20, "Q"), ("TOP1000", 40, "Q"))


def ml_scores(F: dict, keys: list, UNI: np.ndarray, PXM: np.ndarray, DIV: np.ndarray) -> tuple:
    """月末ごとに、その月までに分かっていることだけで2つの点数（ML1・ML2）を作る。"""
    S, M = UNI.shape
    R = np.stack([pct_rank(F[k], UNI) for k in keys])           # 型 × 銘柄 × 月（0〜1）
    X = np.nan_to_num(R - 0.5, nan=0.0)                           # 点数のない型は真ん中
    fr = np.full((S, M), np.nan)                                  # 次の月の成績（配当込み）
    with np.errstate(invalid="ignore", divide="ignore"):
        fr[:, :-1] = (PXM[:, 1:] + DIV[:, 1:]) / PXM[:, :-1] - 1.0
    Y = pct_rank(fr, UNI) - 0.5                                   # 次の月の成績の順位（範囲の中）
    K = len(keys)
    ic = np.full((K, M), np.nan)                                  # 月ごとの順位の相関
    for j in range(M - 1):
        ok = UNI[:, j] & np.isfinite(Y[:, j])
        if ok.sum() < 30:
            continue
        y = Y[ok, j]
        for k in range(K):
            x = X[k, ok, j]
            if np.std(x) > 0:
                ic[k, j] = float(np.corrcoef(x, y)[0, 1])
    ML1 = np.full((S, M), np.nan)
    ML2 = np.full((S, M), np.nan)
    W1, W2 = [], []
    for j in range(M):
        # 月 m の IC は m+1 月末に分かる → j より前の月まで。半分以上の型に IC がある月だけ（2年の戻りなど、遅れて始まる型があるため）
        past = [m for m in range(max(0, j - ML_WIN), j) if m + 1 <= j and np.isfinite(ic[:, m]).sum() >= K / 2]
        if len(past) < ML_MIN:
            W1.append(None)
            W2.append(None)
            continue
        with np.errstate(invalid="ignore"):
            w1 = np.clip(np.nan_to_num(np.nanmean(ic[:, past], axis=1), nan=0.0), 0.0, None)
        w1 = w1 / w1.sum() if w1.sum() > 0 else np.full(K, 1.0 / K)
        ok_now = UNI[:, j]
        ML1[ok_now, j] = np.tensordot(w1, X[:, ok_now, j], axes=1)
        # リッジ回帰（過去の月の、銘柄 × 型 → 次の月の順位）
        xs, ys = [], []
        for m in past:
            ok = UNI[:, m] & np.isfinite(Y[:, m])
            xs.append(X[:, ok, m].T)
            ys.append(Y[ok, m])
        A = np.vstack(xs)
        b = np.concatenate(ys)
        w2 = np.linalg.solve(A.T @ A + ML_RIDGE * len(b) / 100.0 * np.eye(K), A.T @ b)
        ML2[ok_now, j] = X[:, ok_now, j].T @ w2
        W1.append(w1)
        W2.append(w2)
    return ML1, ML2, W1, W2, ic


def lab9(store: dict) -> int:
    log("9段目（多くの型をまとめて学ばせる）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, F, PXM, DIV = D["dates"], D["UNI"], D["F"], D["PXM"], D["DIV"]
    M = len(dates)
    keys = [k for k, _ in LAB1]
    ML1, ML2, W1, W2, ic = ml_scores(F, keys, UNI, PXM, DIV)
    F["ML1"], F["ML2"] = ML1, ML2
    jstart = next((j for j in range(M) if W1[j] is not None), None)
    if jstart is None or M - jstart < 36:
        print("\n■ 学ばせる期間を取ると、比べる期間が短すぎます。")
        return 1
    j0 = max(jstart, next(j for j, t in enumerate(D["me_idx"]) if t >= 252))
    ev = dates[j0:]
    log(f"  点数が出るのは {dates[jstart].date()} から。比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月）")
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    rank_mc = np.full(UNI.shape, np.inf)
    for j in range(M):
        idx = np.nonzero(UNI[:, j])[0]
        o = idx[np.argsort(-D["MCAP"][idx, j])]
        rank_mc[o, j] = np.arange(len(o))
    unis = {nm: (UNI if lim is None else (UNI & (rank_mc < lim))) for nm, lim in G7_UNIS}
    months = [d.month for d in dates]
    fqs = dict(G7_FREQ)

    def thin(rk, fq):
        return rk if fq is None else [r if months[j] in fq else None for j, r in enumerate(rk)]
    rng = np.random.default_rng(20261104)
    seeds = [rng.random(UNI.shape[0]) for _ in range(N_RAND)]
    res = []
    for un, n, fn in ML_CONS:
        U, fq = unis[un], fqs[fn]
        rks_r = [thin(ranks_from(np.repeat(sd[:, None], M, axis=1), U), fq) for sd in seeds]
        rnd = {rule: [half_stats(sim(rk, PXM, DIV, j0, kp, nmax=n)[0], ev) for rk in rks_r] for rule, kp in (("A", n), ("B", 3 * n))}
        for key in ("ML1", "ML2"):
            if ONLY and key not in ONLY:
                continue
            rk = thin(ranks_from(F[key], U), fq)
            r = {"key": key, "uni": un, "n": n, "freq": fn, "rnd": rnd}
            ok_tx, ok_rn, worst = True, True, np.inf
            for rule, kp in (("A", n), ("B", 3 * n)):
                v, liq, s_ = sim(rk, PXM, DIV, j0, kp, nmax=n)
                stt = half_stats(v, ev)
                r[rule], r[rule + "_liq"] = stt, liq
                for h in ("前半", "後半"):
                    a = stt[h]["ann"]
                    rr = [x[h]["ann"] for x in rnd[rule]]
                    d_tx, d_rn = a - ST_TX[h]["ann"], a - float(np.mean(rr))
                    beat = int(sum(1 for x in rr if a > x))
                    r[(rule, h)] = (d_tx, d_rn, beat)
                    ok_tx &= d_tx >= PT_TX
                    ok_rn &= (d_rn >= PT_RND) and (beat >= BEAT_MIN)
                    worst = min(worst, d_tx)
            r["verdict"] = "候補" if (ok_tx and ok_rn) else ("範囲の中では効く" if ok_rn else "効かない")
            r["worst"] = worst
            res.append(r)
            log(f"  {key}／{un}／{n}銘柄／{fn} … {r['verdict']}")
    # 報告
    HH = ("前半", "後半")
    print("\n【日本株の手法・9段目（多くの型をまとめて学ばせる）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan9_2026-11-04）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）。"
          f"重みは、その月末までの{ML_WIN}か月（{ML_MIN}か月未満なら点数なし）だけで決める")
    print("  ML1＝型ごとの順位の相関（次の月の成績と）の平均を重みにする（正のものだけ）・ML2＝リッジ回帰（型24通りの順位から、次の月の成績の順位を当てる）")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    print("\n  1. 成績（年率％。［TOPIX との差 ／ でたらめの平均との差・回数］）")
    for r in res:
        d = " ".join(f"{r[(ru, h)][0]:+.1f}/{r[(ru, h)][1]:+.1f}・{r[(ru, h)][2]}" for ru in ("A", "B") for h in HH)
        rn = " ／ ".join(f"{np.mean([x[h]['ann'] for x in r['rnd']['A']]):.1f}" for h in HH)
        print(f"     {r['key']}／{r['uni']}／{r['n']}銘柄／{r['freq']} … A {r['A']['前半']['ann']:.1f}／{r['A']['後半']['ann']:.1f}（最大下落 {r['A']['通し']['mdd']:.0f}％）"
              f" B {r['B']['前半']['ann']:.1f}／{r['B']['後半']['ann']:.1f}［{d}］ でたらめ（A） {rn} … {r['verdict']}")
    print("\n  2. 重みの平均（ML1。比べる期間の月の平均。大きい順に上から8）")
    ws = np.array([w for w in W1[j0:] if w is not None])
    if len(ws):
        mw = ws.mean(axis=0)
        print("     " + "・".join(f"{keys[k]} {mw[k] * 100:.0f}％" for k in np.argsort(-mw)[:8]))
    icm = np.nanmean(ic[:, j0:], axis=1)
    print("     （型ごとの順位の相関の平均：" + "・".join(f"{keys[k]} {icm[k]:+.3f}" for k in np.argsort(-icm)) + "）")
    print("\n  3. 判定のまとめ")
    for v in ("候補", "範囲の中では効く", "効かない"):
        ks = [f"{r['key']}／{r['uni']}／{r['n']}／{r['freq']}" for r in res if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    return 0


# ───────────────────────── 8段目：空売り・ロングショート ─────────────────────────
# 計画は wide_lab_plan8_2026-11-04（結果を見る前に決めたもの）。点数の下位の銘柄（制度信用で売れる貸借銘柄だけ）を信用で売る。
# 形は2つ：LS＝上位を買い（現金）、下位を売る（同じ額）。TXS＝TOPIX（1306）を全額持ち、下位を同じ額だけ売る
LEND_FEE = 0.0115            # 貸株料（年率。制度信用の目安）
SHORT_MAX_MONTHS = 6         # 制度信用の返済期限（6か月）。期限が来たら一度買い戻し、まだ下位なら売り直す


def fetch_margin(jq, store: dict) -> bool:
    """各月の銘柄一覧から、貸借銘柄（制度信用で売れる銘柄）を取り出して覚える。"""
    mg = store.setdefault("mrgn", {})
    need = [k for k in sorted(store["months"]) if k not in mg]
    if not need:
        return True
    log(f"貸借銘柄を取っています（{len(need)}か月分）…")
    shown = False
    for k in need:
        if elapsed() >= BUDGET_MIN:
            save(store, "貸借銘柄を途中まで保存しました")
            return False
        y, m = map(int, k.split("-"))
        d = pd.bdate_range(pd.Timestamp(year=y, month=m, day=1), pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0), freq="BME")[-1]
        time.sleep(yb.API_SLEEP)
        rows = jq.get("/v2/equities/master", {"date": d.strftime("%Y-%m-%d")})
        if rows and not shown:
            ks = [x for x in rows[0].keys() if "rgn" in x or "argin" in x]
            log("  信用の区分の項目 … " + ("・".join(f"{x}={rows[0].get(x)}" for x in ks) if ks else "見つかりません"))
            shown = True
        ok = set()
        for r in rows:
            code_v = str(r.get("Mrgn") or r.get("MarginCode") or "")
            name_v = str(r.get("MrgnNm") or r.get("MarginCodeName") or "")
            if code_v == "2" or "貸借" in name_v:
                c = yb.norm_code(r.get("Code", ""))
                if c:
                    ok.add(c)
        mg[k] = ok
    save(store, "貸借銘柄をそろえました")
    log(f"  貸借銘柄の数 … 最初の月 {len(mg[sorted(mg)[0]])}・最後の月 {len(mg[sorted(mg)[-1]])}")
    return True


def sim_ls(long_rk, short_rk, PXM, DIV, j0, keep, nmax=NMAX, cap=CAP, slip=SLIP, tax=TAX, fee=LEND_FEE, txpx=None, txdist=None):
    """long_rk[j]・short_rk[j]：その月末の買う順位・売る順位（None は売り買いしない月）。txpx があれば TOPIX（1306）を全額持つ形（TXS）。
    資産＝自由に使える現金＋空売りの担保（売った代金）＋買っている株の評価額＋1306 の評価額 − 空売りの買い戻しの額。"""
    M = PXM.shape[1]
    free = float(cap)          # 自由に使える現金
    coll = 0.0                 # 空売りの担保（売った代金。使えない）
    longs, shorts = {}, {}
    loss = 0.0
    tx_units, tx_cost = 0.0, 0.0
    vals = []
    st = {"n_long": 0, "n_short": 0, "fee": 0.0, "divpay": 0.0, "tax": 0.0}

    def p_(i, j):
        v = PXM[i, j]
        return float(v) if (v == v and v > 0) else 0.0

    def realize(g):
        nonlocal loss
        if g > 0:
            t = max(0.0, g - loss) * tax
            loss = max(0.0, loss - g)
            st["tax"] += t
            return t
        loss += -g
        return 0.0

    def equity(j):
        e = free + coll + sum(sh * p_(i, j) for i, (sh, _c) in longs.items()) - sum(sh * p_(i, j) for i, (sh, _p, _e) in shorts.items())
        if txpx is not None:
            e += tx_units * txpx[j]
        return e

    for j in range(j0, M):
        if txpx is not None and j == j0:
            px = txpx[j] * (1.0 + slip / 2)
            tx_units = free / px
            tx_cost = free
            free = 0.0
        # 配当：買っている株は受け取る（税）、売っている株は払う（配当落調整金）
        for i, (sh, _c) in longs.items():
            d = DIV[i, j]
            if d > 0:
                free += sh * d * (1.0 - tax)
                st["tax"] += sh * d * tax
        for i, (sh, _p, _e) in shorts.items():
            d = DIV[i, j]
            if d > 0:
                free -= sh * d
                st["divpay"] += sh * d
        if txpx is not None and txdist is not None and txdist[j] > 0:
            free += tx_units * txdist[j] * (1.0 - tax)
        # 貸株料（月ごと）
        sv = sum(sh * p_(i, j) for i, (sh, _p, _e) in shorts.items())
        free -= sv * fee / 12.0
        st["fee"] += sv * fee / 12.0
        lr, sr = (long_rk[j] if long_rk is not None else None), short_rk[j]
        if sr is not None:
            if lr is not None:
                lp = {i: k for k, i in enumerate(lr)}
                for i in list(longs):
                    if i not in lp or lp[i] >= keep:
                        sh, cost = longs.pop(i)
                        pr = sh * p_(i, j) * (1.0 - slip)
                        free += pr - realize(pr - cost)
            sp = {i: k for k, i in enumerate(sr)}
            for i in list(shorts):
                sh, proceeds, e0 = shorts[i]
                if i not in sp or sp[i] >= keep or (j - e0) >= SHORT_MAX_MONTHS:
                    shorts.pop(i)
                    cover = sh * p_(i, j) * (1.0 + slip)
                    coll -= proceeds
                    free += proceeds - cover - realize(proceeds - cover)
            eq = equity(j)
            tgt = eq / nmax
            if lr is not None:
                for i in lr[:nmax]:
                    if len(longs) >= nmax:
                        break
                    if i in longs or not p_(i, j):
                        continue
                    px = p_(i, j) * (1.0 + slip)
                    sh = int(min(tgt, free) / px)
                    if sh < 1:
                        continue
                    free -= sh * px
                    longs[i] = [sh, sh * px]
                    st["n_long"] += 1
            for i in sr[:nmax]:
                if len(shorts) >= nmax:
                    break
                if i in shorts or i in longs or not p_(i, j):
                    continue
                px = p_(i, j) * (1.0 - slip)
                sh = int(tgt / px)
                if sh < 1:
                    continue
                coll += sh * px
                shorts[i] = [sh, sh * px, j]
                st["n_short"] += 1
        vals.append(equity(j))
    # 最後に全部閉じる
    liq = free + coll
    g = 0.0
    for i, (sh, cost) in longs.items():
        pr = sh * p_(i, M - 1) * (1.0 - slip)
        liq += pr
        g += pr - cost
    for i, (sh, proceeds, _e) in shorts.items():
        cover = sh * p_(i, M - 1) * (1.0 + slip)
        liq -= cover
        g += proceeds - cover
    if txpx is not None:
        pr = tx_units * txpx[M - 1] * (1.0 - slip / 2)
        liq += pr
        g += pr - tx_cost
    liq -= tax * max(0.0, g - loss)
    return np.array(vals), liq, st


def lab8(store: dict, extra: dict | None = None, head: str | None = None) -> int:
    log("8段目（空売り・ロングショート）を回します（指標を作っています）…" if not extra else "空売りの形（LS・TXS）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, F, PXM, DIV = D["dates"], D["UNI"], D["F"], D["PXM"], D["DIV"]
    F["VQM"] = combo_score(F, list(VQM), UNI)
    for nm, (fn, _lab) in (extra or {}).items():
        F[nm] = fn(D)
    M = len(dates)
    j0 = next((j for j, t in enumerate(D["me_idx"]) if t >= 252), None)
    if j0 is None or M - j0 < 36:
        print("\n■ 比べる期間が短すぎます。")
        return 1
    ev = dates[j0:]
    mg = store.get("mrgn", {})
    codes = D["codes"]
    ci = {c: i for i, c in enumerate(codes)}
    SH = np.zeros(UNI.shape, dtype=bool)
    for j, d in enumerate(dates):
        for c in mg.get(d.strftime("%Y-%m"), ()):
            i = ci.get(c)
            if i is not None:
                SH[i, j] = True
    SHU = UNI & SH
    share = [float(SHU[:, j].sum()) / max(1, UNI[:, j].sum()) for j in range(j0, M)]
    log(f"  売れる銘柄（範囲の中の貸借銘柄） … 平均 {np.mean([SHU[:, j].sum() for j in range(j0, M)]):.0f}銘柄（範囲の {np.mean(share) * 100:.0f}％）")
    if np.mean(share) < 0.2:
        print("\n■ 貸借銘柄が取れていません（信用の区分の項目を確かめてください）")
        return 2
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    txpx = np.array([float(D["tp"]["c"][t]) for t in D["me_idx"]])
    txdist = np.zeros(M)
    for dd, a in yb.ETF_DIST.get("1306", ()):
        k = int(np.searchsorted(np.array([np.datetime64(x.date()) for x in dates]), np.datetime64(pd.Timestamp(dd).date())))
        if k < M:
            txdist[k] += a / 100.0
    months = [d.month for d in dates]

    def thin(rk, fq):
        return rk if fq is None else [r if months[j] in fq else None for j, r in enumerate(rk)]
    rng = np.random.default_rng(20261104)
    seeds = [(rng.random(UNI.shape[0]), rng.random(UNI.shape[0])) for _ in range(N_RAND)]
    rnd = {}
    for fn, fq in G7_FREQ:
        rl = [thin(ranks_from(np.repeat(a[:, None], M, axis=1), UNI), fq) for a, _b in seeds]
        rs = [thin(ranks_from(np.repeat(b[:, None], M, axis=1), SHU), fq) for _a, b in seeds]
        for rule, kp in (("A", NMAX), ("B", 3 * NMAX)):
            rnd[("LS", fn, rule)] = [half_stats(sim_ls(rl[k], rs[k], PXM, DIV, j0, kp)[0], ev) for k in range(N_RAND)]
            rnd[("TXS", fn, rule)] = [half_stats(sim_ls(None, rs[k], PXM, DIV, j0, kp, txpx=txpx, txdist=txdist)[0], ev) for k in range(N_RAND)]
    log(f"  でたらめの比べ … {len(rnd)}通り × {N_RAND}回")
    keys = list(extra) if extra else [k for k, _ in LAB1] + ["VQM"]
    res = []
    for key in keys:
        if ONLY and key not in ONLY:
            continue
        sc = F[key]
        rl_full = ranks_from(sc, UNI)
        rs_full = ranks_from(-sc, SHU)            # 点数の低い順（売る側）
        for form in ("LS", "TXS"):
            for fn, fq in G7_FREQ:
                r = {"key": key, "form": form, "freq": fn}
                ok_tx, ok_rn, worst = True, True, np.inf
                for rule, kp in (("A", NMAX), ("B", 3 * NMAX)):
                    v, liq, s_ = sim_ls(thin(rl_full, fq) if form == "LS" else None, thin(rs_full, fq), PXM, DIV, j0, kp,
                                        txpx=txpx if form == "TXS" else None, txdist=txdist if form == "TXS" else None)
                    stt = half_stats(v, ev)
                    r[rule], r[rule + "_st"], r[rule + "_liq"], r[rule + "_avg"] = stt, s_, liq, float(np.mean(v))
                    for h in ("前半", "後半"):
                        a = stt[h]["ann"]
                        rr = [x[h]["ann"] for x in rnd[(form, fn, rule)]]
                        d_tx, d_rn = a - ST_TX[h]["ann"], a - float(np.mean(rr))
                        beat = int(sum(1 for x in rr if a > x))
                        r[(rule, h)] = (d_tx, d_rn, beat)
                        ok_tx &= d_tx >= PT_TX
                        ok_rn &= (d_rn >= PT_RND) and (beat >= BEAT_MIN)
                        worst = min(worst, d_tx)
                r["verdict"] = "候補" if (ok_tx and ok_rn) else ("範囲の中では効く" if ok_rn else "効かない")
                r["worst"] = worst
                res.append(r)
        log(f"  {key} … " + "・".join(f"{r['form']}{r['freq']} {r['verdict']}" for r in res if r["key"] == key))
    report8(res, rnd, ST_TX, TXliq, ev, head)
    return 0


def report8(res, rnd, ST_TX, TXliq, ev, head=None):
    HH = ("前半", "後半")
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    print("\n" + (head or "【日本株の手法・8段目（空売り・ロングショート）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan8_2026-11-04）"))
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  形 … LS＝点数の上位20を買い（現金）、下位20を信用で売る（同じ額）・TXS＝TOPIX（1306）を全額持ち、下位20を同じ額だけ信用で売る。"
          f"売れるのはその月末の貸借銘柄だけ。貸株料 年{LEND_FEE * 100:.2f}％・配当落調整金を払う・6か月で一度買い戻す・片道{SLIP * 100:.1f}％・税（買いと売りの損益を合わせる）")
    print("  判定 … 1回目と同じ（前半・後半の両方で、A（20位）・B（60位）の両方とも TOPIX +2pt・でたらめ（同じ形ででたらめに選ぶ）+2pt かつ20回のうち18回）")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    print("  でたらめの平均（A） … " + " ・ ".join(f"{fm}{fn} {np.mean([x['前半']['ann'] for x in rnd[(fm, fn, 'A')]]):.1f}／{np.mean([x['後半']['ann'] for x in rnd[(fm, fn, 'A')]]):.1f}"
                                          for fm in ("LS", "TXS") for fn, _f in G7_FREQ))
    cand = [r for r in res if r["verdict"] == "候補"]
    eff = [r for r in res if r["verdict"] == "範囲の中では効く"]
    print(f"\n  1. 判定のまとめ … {len(res)}通りのうち 候補 {len(cand)}・範囲の中では効く {len(eff)}・効かない {len(res) - len(cand) - len(eff)}")

    def line(r):
        d = " ".join(f"{r[(ru, h)][0]:+.1f}/{r[(ru, h)][1]:+.1f}・{r[(ru, h)][2]}" for ru in ("A", "B") for h in HH)
        return (f"     {r['key']:<7}{r['form']:<4}{r['freq']} … A {r['A']['前半']['ann']:.1f}／{r['A']['後半']['ann']:.1f}（最大下落 {r['A']['通し']['mdd']:.0f}％）"
                f" B {r['B']['前半']['ann']:.1f}／{r['B']['後半']['ann']:.1f}［{d}］ 清算後の通し A {((r['A_liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％・"
                f"貸株料 {r['A_st']['fee'] / r['A_avg'] * 100 / yrs:.1f}％／年・配当落 {r['A_st']['divpay'] / r['A_avg'] * 100 / yrs:.1f}％／年（資産の平均に対して） … {r['verdict']}")
    print("\n  2. 候補")
    for r in sorted(cand, key=lambda x: -x["worst"]):
        print(line(r))
    if not cand:
        print("     なし")
    print("\n  3. 全部（TOPIX との差のいちばん悪い値の大きい順）")
    for r in sorted(res, key=lambda x: -x["worst"]):
        print(line(r))
    carry = []
    for r in sorted(cand, key=lambda x: -x["worst"]):
        if r["key"] not in {c["key"] for c in carry}:
            carry.append(r)
        if len(carry) >= G7_CARRY:
            break
    print("\n  4. 確かめる段階に持っていくもの（候補の中から点数の違うものを上から3つまで）")
    print("     " + ("・".join(f"{r['key']}/{r['form']}/{r['freq']}" for r in carry) if carry else "なし"))
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")


# ───────────────────────── 7段目：組み立て方の探索 ─────────────────────────
# 計画は wide_lab_plan7_2026-11-04（結果を見る前に決めたもの）。1回目の24通り＋VQM の点数ごとに、
# 範囲（全銘柄／時価総額の上から1,000位／500位）× 持つ数（10・20・40・100）× 入れ替え（毎月／四半期）を回す。売る決まりは A（持つ数）・B（持つ数の3倍）
G7_UNIS = (("ALL", None), ("TOP1000", 1000), ("TOP500", 500))
G7_NS = (10, 20, 40, 100)
G7_FREQ = (("M", None), ("Q", (3, 6, 9, 12)))
G7_CARRY = 3                 # 確かめる段階に持っていく数の上限
CFG: list = []               # 確かめる段階で回す組み立て（"点数/範囲/持つ数/入れ替え" のリスト。空なら全部）


def lab7(store: dict) -> int:
    log("7段目（組み立て方の探索）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, F, PXM, DIV = D["dates"], D["UNI"], D["F"], D["PXM"], D["DIV"]
    F["VQM"] = combo_score(F, list(VQM), UNI)
    M = len(dates)
    j0 = next((j for j, t in enumerate(D["me_idx"]) if t >= 252), None)
    if j0 is None or M - j0 < 36:
        print("\n■ 比べる期間が短すぎます。")
        return 1
    ev = dates[j0:]
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    rank_mc = np.full(UNI.shape, np.inf)
    for j in range(M):
        idx = np.nonzero(UNI[:, j])[0]
        o = idx[np.argsort(-D["MCAP"][idx, j])]
        rank_mc[o, j] = np.arange(len(o))
    unis = {nm: (UNI if lim is None else (UNI & (rank_mc < lim))) for nm, lim in G7_UNIS}
    months = [d.month for d in dates]

    def thin(rk, fq):
        return rk if fq is None else [r if months[j] in fq else None for j, r in enumerate(rk)]
    # でたらめ（範囲・持つ数・入れ替えごとに20回）
    rnd = {}
    rng = np.random.default_rng(20261104)
    seeds = [rng.random(UNI.shape[0]) for _ in range(N_RAND)]
    want = {tuple(c.split("/")) for c in CFG}
    need = {(u, int(n), f) for (_k, u, n, f) in want} if want else None
    for un, U in unis.items():
        if need is not None and not any(x[0] == un for x in need):
            continue
        rks = [ranks_from(np.repeat(sd[:, None], M, axis=1), U) for sd in seeds]
        for n in G7_NS:
            for fn, fq in G7_FREQ:
                if need is not None and (un, n, fn) not in need:
                    continue
                for rule, kp in (("A", n), ("B", 3 * n)):
                    rnd[(un, n, fn, rule)] = [half_stats(sim(thin(rk, fq), PXM, DIV, j0, kp, nmax=n)[0], ev) for rk in rks]
    log(f"  でたらめの比べ … {len(rnd)}通り × {N_RAND}回")
    keys = [k for k, _ in LAB1] + ["VQM"]
    labs = dict(LAB1)
    labs["VQM"] = "割安・質・勢い（固定）"
    res = []
    for key in keys:
        if ONLY and key not in ONLY:
            continue
        for un, U in unis.items():
            if want and not any(w[0] == key and w[1] == un for w in want):
                continue
            rk_full = ranks_from(F[key], U)
            for n in G7_NS:
                for fn, fq in G7_FREQ:
                    if want and (key, un, str(n), fn) not in want:
                        continue
                    rk = thin(rk_full, fq)
                    r = {"key": key, "uni": un, "n": n, "freq": fn}
                    ok_tx, ok_rn, worst = True, True, np.inf
                    for rule, kp in (("A", n), ("B", 3 * n)):
                        v, liq, s_ = sim(rk, PXM, DIV, j0, kp, nmax=n)
                        stt = half_stats(v, ev)
                        r[rule] = stt
                        r[rule + "_buy"] = s_["n_buy"]
                        for h in ("前半", "後半"):
                            a = stt[h]["ann"]
                            rr = [x[h]["ann"] for x in rnd[(un, n, fn, rule)]]
                            d_tx, d_rn = a - ST_TX[h]["ann"], a - float(np.mean(rr))
                            beat = int(sum(1 for x in rr if a > x))
                            r[(rule, h)] = (d_tx, d_rn, beat)
                            ok_tx &= d_tx >= PT_TX
                            ok_rn &= (d_rn >= PT_RND) and (beat >= BEAT_MIN)
                            worst = min(worst, d_tx)
                    r["verdict"] = "候補" if (ok_tx and ok_rn) else ("範囲の中では効く" if ok_rn else "効かない")
                    r["worst"] = worst
                    r["allpos"] = all(r[(ru, h)][0] > 0 and r[(ru, h)][1] > 0 for ru in ("A", "B") for h in ("前半", "後半"))
                    res.append(r)
        log(f"  {key} … 候補 {sum(1 for x in res if x['key'] == key and x['verdict'] == '候補')}・"
            f"範囲の中では効く {sum(1 for x in res if x['key'] == key and x['verdict'] == '範囲の中では効く')}")
    report7(res, rnd, ST_TX, TXliq, ev, labs)
    return 0


def report7(res, rnd, ST_TX, TXliq, ev, labs):
    HH = ("前半", "後半")
    print("\n【日本株の手法・7段目（組み立て方の探索）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan7_2026-11-04）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  試した組み立て … {len(res)}通り（それぞれ売る決まり A・B）" + (f"。指定した組み立てだけ：{'・'.join(CFG)}" if CFG else
          f"（点数 {len({r['key'] for r in res})}通り × 範囲 {len(G7_UNIS)} × 持つ数 {len(G7_NS)} × 入れ替え {len(G7_FREQ)}）"))
    print(f"  判定 … 1回目と同じ（前半・後半の両方で、A・B の両方とも TOPIX +{PT_TX:.1f}pt・でたらめ +{PT_RND:.1f}pt かつ20回のうち{BEAT_MIN}回以上）。"
          f"でたらめは、同じ範囲・同じ持つ数・同じ入れ替えで作る")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    print("\n  でたらめの平均（年率％。前半 ／ 後半。A）")
    for un, _l in G7_UNIS:
        line = " ・ ".join(f"{n}銘柄{fn} {np.mean([x['前半']['ann'] for x in rnd[(un, n, fn, 'A')]]):.1f}／{np.mean([x['後半']['ann'] for x in rnd[(un, n, fn, 'A')]]):.1f}"
                           for n in G7_NS for fn, _f in G7_FREQ if (un, n, fn, "A") in rnd)
        if line:
            print(f"     {un} … {line}")
    cand = [r for r in res if r["verdict"] == "候補"]
    eff = [r for r in res if r["verdict"] == "範囲の中では効く"]
    print(f"\n  1. 判定のまとめ … 候補 {len(cand)}通り・範囲の中では効く {len(eff)}通り・効かない {len(res) - len(cand) - len(eff)}通り")

    def line(r):
        d = " ".join(f"{r[(rule, h)][0]:+.1f}/{r[(rule, h)][1]:+.1f}・{r[(rule, h)][2]}" for rule in ("A", "B") for h in HH)
        return (f"     {r['key']:<7}{r['uni']:<8}{r['n']:>3}銘柄 {r['freq']} … A {r['A']['前半']['ann']:.1f}／{r['A']['後半']['ann']:.1f}"
                f" B {r['B']['前半']['ann']:.1f}／{r['B']['後半']['ann']:.1f}［{d}］ 最大下落 A {r['A']['通し']['mdd']:.0f}％ … {r['verdict']}")
    print("\n  2. 候補（TOPIX との差のいちばん悪い値の大きい順）")
    for r in sorted(cand, key=lambda x: -x["worst"]):
        print(line(r))
    if not cand:
        print("     なし")
    print("\n  3. 範囲の中では効く（同じ順に上から20）")
    for r in sorted(eff, key=lambda x: -x["worst"])[:20]:
        print(line(r))
    if not eff:
        print("     なし")
    print("\n  4. 点数ごとのいちばん良い組み立て（TOPIX との差のいちばん悪い値で）")
    for k in sorted({r["key"] for r in res}, key=lambda x: list(labs).index(x) if x in labs else 99):
        rs = [r for r in res if r["key"] == k]
        b = max(rs, key=lambda x: x["worst"])
        print(line(b))
    carry = []
    for r in sorted(cand, key=lambda x: -x["worst"]):
        if r["key"] not in {c["key"] for c in carry}:
            carry.append(r)
        if len(carry) >= G7_CARRY:
            break
    print("\n  5. 確かめる段階に持っていくもの（計画どおり、候補の中から点数の違うものを上から3つまで）")
    print("     " + ("・".join(f"{r['key']}／{r['uni']}／{r['n']}銘柄／{r['freq']}" for r in carry) if carry else "なし"))
    # 準候補（11月4日夜、探す段階の結果を見たあと・見ていない期間を見る前に足した決まり。wide_lab_plan7 の6）
    semi = []
    for r in sorted([r for r in res if r["allpos"] and r["verdict"] != "候補"], key=lambda x: -x["worst"]):
        if r["key"] not in {c["key"] for c in semi} | {c["key"] for c in carry}:
            semi.append(r)
        if len(semi) >= G7_CARRY:
            break
    print("\n  6. 準候補（4つの欄すべてで TOPIX とでたらめの平均を上回ったが、+2pt に届かない欄があるもの。点数の違うものを上から3つまで）")
    for r in semi:
        print(line(r))
    print("     " + ("・".join(f"{r['key']}/{r['uni']}/{r['n']}/{r['freq']}" for r in semi) if semi else "なし"))
    if CFG:
        print("\n  7. 確かめる段階の判定（前半＝見ていない期間で、候補の基準を満たしたか）")
        for r in res:
            ok = all(r[(ru, "前半")][0] >= PT_TX and r[(ru, "前半")][1] >= PT_RND and r[(ru, "前半")][2] >= BEAT_MIN for ru in ("A", "B"))
            print(f"     {r['key']}/{r['uni']}/{r['n']}/{r['freq']} … 見ていない期間で {'基準を満たした' if ok else '基準を満たさない'}（両方の期間では {r['verdict']}）")
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / max((ev[-1] - ev[0]).days / 365.25, 0.5)) - 1) * 100:.1f}％）")


# ───────────────────────── 4回目：売買の内訳（プレミアムだけのデータ）─────────────────────────
# 計画は wide_lab_plan4_2026-11-04（結果を見る前に決めたもの）。毎日の銘柄ごとの売買の内訳（実売り・空売り・信用の新規と返済）を、
# 「その日のデータが使えるようになる次の営業日」の月に足し上げ、月末ごとの比率を点数にする。比べ方と判定は1回目と同じ。
# プレミアムのデータなので、スタンダードに戻す前に long-cleanup で消す（キャッシュの名前を yield-backtest-long- で始める）。
BD_CACHE = OUT / ("wide_bd_long.pkl" if LONG else "wide_bd.pkl")
BD_KEYS = ("LongSellVa", "ShrtNoMrgnVa", "MrgnSellNewVa", "MrgnSellCloseVa", "LongBuyVa", "MrgnBuyNewVa", "MrgnBuyCloseVa")


def bd_load() -> dict:
    if BD_CACHE.exists():
        with open(BD_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def bd_save(bd: dict, note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = BD_CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(bd, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, BD_CACHE)
    if note:
        log(f"  {note}（{BD_CACHE.stat().st_size / 1e6:,.0f}MB・開始から{elapsed():.0f}分）")


def fetch_bd(jq, store: dict, bd: dict) -> bool:
    """営業日ごとに、全銘柄の売買の内訳を取って、月ごとに足し上げる。時間切れなら取った分を保存して False。"""
    cal = store["etf1306"]["d"]
    first = pd.Timestamp(min(store["months"]) + "-01") - pd.DateOffset(days=31)
    ks = [k for k in range(len(cal) - 1) if cal[k] >= np.datetime64(first.date())]
    done = bd.setdefault("done", set())
    mm_all = bd.setdefault("m", {})
    todo = [k for k in ks if str(cal[k]) not in done]
    if todo:
        log(f"売買の内訳を取っています（全部で {len(ks)}営業日・残り {len(todo)}営業日）…")
    last = time.monotonic()
    shown = bool(bd.get("keys_seen"))
    for n, k in enumerate(todo, 1):
        if elapsed() >= BUDGET_MIN:
            bd_save(bd, "売買の内訳を途中まで保存しました")
            return False
        d = str(cal[k])
        time.sleep(yb.API_SLEEP)
        try:
            rows = jq.get("/v2/markets/breakdown", {"date": d})
        except yb.NotAllowed as e:
            raise SystemExit(f"■ 売買の内訳が取れません（プレミアムではない可能性があります）：{e}")
        except Exception as e:
            log(f"  取得に失敗 {d}: {e}")
            continue
        if rows and not shown:
            bd["keys_seen"] = sorted(rows[0].keys())
            log("  受け取った項目 … " + "・".join(bd["keys_seen"]))
            shown = True
        ym = str(cal[k + 1])[:7]          # この日のデータは次の営業日から使える
        mm = mm_all.setdefault(ym, {})
        for r in rows:
            c = yb.norm_code(r.get("Code", ""))
            if not c:
                continue
            v = np.array([num(r.get(x)) or 0.0 for x in BD_KEYS])
            mm[c] = mm[c] + v if c in mm else v
        done.add(d)
        if n % 200 == 0:
            log(f"  {n}/{len(todo)}（開始から{elapsed():.0f}分）")
        if (time.monotonic() - last) / 60 >= SAVE_EVERY_MIN:
            bd_save(bd, "途中まで保存しました")
            last = time.monotonic()
    bd["complete"] = True
    bd_save(bd, "売買の内訳がそろいました")
    return True


def bd_score(bd: dict, kind: str):
    """月末ごとの点数。SHORT＝空売りの比率の小ささ・MBUY＝信用の新規買いの比率の小ささ（どちらも、その月に使えるようになった分の合計で）。"""
    def fn(D):
        codes, dates = D["codes"], D["dates"]
        out = np.full((len(codes), len(dates)), np.nan)
        ci = {c: i for i, c in enumerate(codes)}
        for j, d in enumerate(dates):
            mm = bd.get("m", {}).get(d.strftime("%Y-%m"))
            if not mm:
                continue
            for c, v in mm.items():
                i = ci.get(c)
                if i is None:
                    continue
                if kind == "SHORT":
                    den = v[0] + v[1] + v[2] + v[3]
                    if den > 0:
                        out[i, j] = -(v[1] + v[2]) / den
                else:
                    den = v[4] + v[5] + v[6]
                    if den > 0:
                        out[i, j] = -v[5] / den
        return out
    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fetch", "lab1", "lab2", "lab3", "lab4", "lab5", "lab6", "lab7", "lab8", "lab9"], default="lab1")
    ap.add_argument("--combo", default="", help="3回目：1回目で「範囲の中では効く」以上だった型の記号（カンマ区切り）。2つ以上なら SEL として回す")
    ap.add_argument("--only", default="", help="回す型の記号（カンマ区切り。空欄なら全部）。確かめる段階で候補だけを回すとき")
    ap.add_argument("--cfg", default="", help="7段目の確かめる段階で回す組み立て（例: VQM/TOP500/20/Q,NXG/TOP500/20/Q）")
    args = ap.parse_args()
    global ONLY, LAB_TITLE, LAB_PLAN, CFG
    ONLY = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    CFG = [x.strip() for x in args.cfg.split(",") if x.strip()]
    if args.mode == "lab3":
        LAB_TITLE, LAB_PLAN = "3回目（組み合わせ）", "wide_lab_plan3_2026-11-04"
    elif args.mode == "lab4":
        LAB_TITLE, LAB_PLAN = "4回目（売買の内訳）", "wide_lab_plan4_2026-11-04"
    elif args.mode == "lab5":
        LAB_TITLE, LAB_PLAN = "5段目（残りの型：毎月の点数）", "wide_lab_plan5_2026-11-04"
    if LONG:
        yb.LONG_MODE = True          # 株価を「20年前まで」で取る。取れなければプレミアムではないので止まる
        yb.API_SLEEP = float(os.environ.get("LONG_API_SLEEP", "0.25"))    # プレミアムは呼び出しの上限が大きい
        log(f"確かめる段階（{YEARS}年分・データは {CACHE}）")
    store = load()
    need_ohlc = KEEP_OHLC and any("o" not in b for b in store.get("bars", {}).values())
    if not store.get("complete") or need_ohlc:
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
    if args.mode == "lab7":
        return lab7(store)
    if args.mode == "lab6":
        return lab6(store)
    if args.mode == "lab9":
        return lab9(store)
    if args.mode == "lab5":
        xs = x_load()
        if not x_complete(store, xs):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_x(yb.JQ(key), store, xs):
                print("\n■ まだ終わっていません（配当金情報などを取っている途中です）。同じ設定のまま、もう一度押してください。")
                return EXIT_UNFINISHED
        if not all(k in store.get("mrgn", {}) for k in store["months"]):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_margin(yb.JQ(key), store):
                print("\n■ まだ終わっていません（貸借銘柄を取っている途中です）。もう一度押してください。")
                return EXIT_UNFINISHED
        return lab5(store, xs)
    if args.mode == "lab8":
        if not all(k in store.get("mrgn", {}) for k in store["months"]):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_margin(yb.JQ(key), store):
                print("\n■ まだ終わっていません（貸借銘柄を取っている途中です）。もう一度押してください。")
                return EXIT_UNFINISHED
        return lab8(store)
    if args.mode == "lab4":
        bd = bd_load()
        if not bd.get("complete"):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_bd(yb.JQ(key), store, bd):
                print("\n■ まだ終わっていません（売買の内訳を取っている途中です）。同じ設定のまま、もう一度「Run workflow」を押してください。")
                return EXIT_UNFINISHED
        log(f"売買の内訳 … {len(bd.get('done', []))}営業日・{len(bd.get('m', {}))}か月")
        extra = {"SHORT": (bd_score(bd, "SHORT"), "空売りの少ない株（その月の空売り（信用の新規売りを含む）÷ 売り全体の小さい順）"),
                 "MBUY": (bd_score(bd, "MBUY"), "信用の新規買いの少ない株（その月の信用の新規買い ÷ 買い全体の小さい順）")}
        return lab1(store, extra=extra)
    if args.mode == "lab3":
        sel = [x.strip().upper() for x in args.combo.split(",") if x.strip()]
        bad = [k for k in sel if k not in dict(LAB1)]
        if bad:
            print(f"\n■ 組み合わせに使えない記号があります：{'・'.join(bad)}")
            return 2
        combos = {"VQM": (VQM, "割安・質・勢い（固定）")}
        if len(sel) >= 2:
            combos["SEL"] = (tuple(sel), "1回目で効いた型の組み合わせ")
        return lab1(store, combos)
    return lab1(store) if args.mode == "lab1" else lab2(store)


if __name__ == "__main__":
    sys.exit(main())

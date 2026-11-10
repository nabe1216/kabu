"""東証の全銘柄で、日本株の手法の型をまとめて確かめる道具（wide_lab）。2026年11月4日に作成。

計画と記録はプロジェクトの method_map_2026-11-04（地図と共通の決まり）と wide_lab_plan_2026-11-04（1回目の型と順位の付け方）。
今までのツール（yield_backtest.py）・キャッシュ（data/yield_backtest_cache.pkl）には触らない。データは data/wide_cache.pkl に別に持つ。

使い方（GitHub の Actions → wide-lab → Run workflow）
  mode=fetch … 東証全銘柄のデータ（各月の銘柄一覧・株価・財務の要約）を集める。時間切れなら取った分を保存して止まる（もう一度押すと続きから）
  mode=lab1  … 足りないデータがあれば先に集め、そろっていれば1回目の型をまとめて回す
  mode=lab2  … 同じく、2回目の型（出来事と暦）をまとめて回す
"""
import argparse
import bisect
import json
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
CFG36: set = set()       # 36段目 B：回す組み立て（cfg で指定。空なら全部）
FAIR = True              # 30段目：判定のでたらめを「同じ数・同じ入れ替え」にそろえ、ぶれの幅を出す（wide_lab_plan30_2026-11-10）


def uni_txt() -> str:
    return f"売買代金の中央値{MIN_LIQ / 1e4:,.0f}万円以上・時価総額{MIN_MCAP / 1e8:,.0f}億円以上" + ("（30段目の小型株を含む範囲）" if MIN_MCAP < 1e10 else "")
STALE_DAYS = 5           # 最後に売買があった日が、月末からこの営業日より前なら、その月は買えない
NMAX = 20
KEEPS = (("A", 20), ("B", 60))     # A＝20位より下で売る・B＝60位より下で売る
# 費用と税の前提（14段目で足した。S0＝第1部と同じ・S1＝低い費用・S2＝低い費用で税なし。wide_lab_plan14_2026-11-05）
COSTS = {"S0": (0.002, 0.001, 0.20315), "S1": (0.001, 0.0005, 0.20315), "S2": (0.001, 0.0005, 0.0)}
COST = (os.environ.get("WIDE_COST", "") or "S0").strip().upper()[:2]
if COST not in COSTS:
    COST = "S0"
SLIP = COSTS[COST][0]    # 株の値段のずれ（片道）。S0 は0.2％
TAX = COSTS[COST][2]
TAX_TXT = f"税{TAX * 100:.3f}％".replace("0％", "％") if TAX > 0 else "税なし"
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


# 29段目のあと（別のレビューで見つかった誤りを直したもの。2026年11月10日）：
# 期末を基準日にした株式分割では、権利落ちの日（＝調整係数のつく日）が期末より前に来る。決算短信の実績の配当と期末の株数は、
# 分割前の実額のこともあれば、分割後に直してあることもある。前の値（いまの基準）に近いほうを選ぶ
SPLIT_DIAG: dict = {"div": set(), "div_post": set(), "sh": set(), "sh_post": set()}


def _d7(t) -> pd.Timestamp:
    return pd.Timestamp(str(t)[:10]) - pd.Timedelta(days=7)


def div_post_map(code: str, fd: np.ndarray, ff: np.ndarray, tl: list) -> dict:
    """期末の日 → True（その期の配当の実績は分割後の基準に直してある）。期末の7日前より後〜開示の日までに分割・併合がなければ False。"""
    out, ref = {}, None
    for r in tl:
        if not is_fy(r):
            continue
        fy_end = str(r.get("CurFYEn") or "")[:10]
        if len(fy_end) != 10 or fy_end in out:
            continue
        ann = num(r.get("DivAnn"))
        a, b = fac_after(fd, ff, _d7(fy_end)), fac_after(fd, ff, str(r["DiscDate"])[:10])
        post = False
        if abs(a - b) > 1e-9:
            SPLIT_DIAG["div"].add((code, fy_end))
            if ref and ann and ann > 0:
                post = bool(abs(np.log(ann * b / ref)) < abs(np.log(ann * a / ref)))
            if post:
                SPLIT_DIAG["div_post"].add((code, fy_end))
        out[fy_end] = post
        if ann and ann > 0:
            ref = ann * (b if post else a)
    return out


def dps_now(fd: np.ndarray, ff: np.ndarray, r: dict, amt: float, rec_me, post: bool) -> float:
    """決算短信の実績の配当 amt（rec_me＝権利確定の月末）を、いまの株価の基準へ。"""
    if post:
        return amt * fac_after(fd, ff, str(r["DiscDate"])[:10])
    return amt * fac_after(fd, ff, _d7(rec_me))


def sh_now_list(code: str, fd: np.ndarray, ff: np.ndarray, tl: list) -> list:
    """開示ごとの、自己株を除く株数（いまの基準）。期間の末の7日前より後〜開示の日までに分割・併合があれば、前の開示の株数に近いほうを選ぶ。"""
    out, ref = [], None
    for r in tl:
        sh = num(r.get("ShOutFY"))
        if not sh:
            out.append(None)
            continue
        sh = sh - (num(r.get("TrShFY")) or 0.0)
        pe = str(r.get("CurPerEn") or r.get("CurFYEn") or "")[:10]
        b = fac_after(fd, ff, str(r["DiscDate"])[:10])
        a = fac_after(fd, ff, _d7(pe)) if len(pe) == 10 else b
        v = sh / b
        if abs(a - b) > 1e-9:
            key = (code, str(r["DiscDate"])[:10])
            SPLIT_DIAG["sh"].add(key)
            va = sh / a
            if not (ref and ref > 0 and sh > 0 and abs(np.log(v / ref)) < abs(np.log(va / ref))):
                v = va
            else:
                SPLIT_DIAG["sh_post"].add(key)
        out.append(v if v > 0 else None)
        if v > 0:
            ref = v
    return out


def interim_only(tl: list, seen_fy: set) -> list:
    """通期の決算短信がまだない期の中間配当（2Q の決算短信の実績）。[(行, 期末, 額, 権利確定の月末)]。期ごとに1回だけ。"""
    out, done = [], set()
    for r in tl:
        if str(r.get("CurPerType") or "") != "2Q" or "FinancialStatements" not in str(r.get("DocType") or ""):
            continue
        fy_end = str(r.get("CurFYEn") or "")[:10]
        if len(fy_end) != 10 or fy_end in seen_fy or fy_end in done:
            continue
        q2 = num(r.get("Div2Q"))
        if q2 and q2 > 0:
            done.add(fy_end)
            out.append((r, fy_end, q2, pd.Timestamp(fy_end) - pd.DateOffset(months=6) + pd.offsets.MonthEnd(0)))
    return out


BOOT_N, BOOT_BLOCK = 2000, 6      # 30段目：ぶれの幅（月ごとの値動きを6か月のかたまりで組み直す・2,000回。12か月のかたまりは96か月では幅が狭く出やすい）


def boot_ci(v: np.ndarray, tv: np.ndarray, liq: float, tliq: float, yrs: float, seed: int = 20261110) -> tuple:
    """清算後の通しの TOPIX との差（年率pt）と、その90％の幅・差が正になった割合。
    月末の評価額（v）と TOPIX（tv）の月ごとの値動きを、同じ月をそろえて12か月のかたまりで組み直す（循環ブロック・ブートストラップ）。
    中心は清算後の通しの差（税を両方とも払った後）。幅は、組み直した年率の差のぶれを足したもの。"""
    v, tv = np.asarray(v, dtype=float), np.asarray(tv, dtype=float)
    n = min(len(v), len(tv)) - 1
    L = (((liq / CAP) ** (1 / yrs)) - ((tliq / CAP) ** (1 / yrs))) * 100.0
    if n < 24 or not np.all(v[:n + 1] > 0):
        return L, float("nan"), float("nan"), float("nan")
    rs, rt = v[1:n + 1] / v[:n] - 1.0, tv[1:n + 1] / tv[:n] - 1.0
    ls, lt = np.log1p(rs), np.log1p(rt)
    base = (np.exp(ls.mean() * 12) - np.exp(lt.mean() * 12)) * 100.0
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / BOOT_BLOCK))
    st = rng.integers(0, n, size=(BOOT_N, nb))
    idx = ((st[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]) % n).reshape(BOOT_N, -1)[:, :n]
    d = (np.exp(ls[idx].mean(axis=1) * 12) - np.exp(lt[idx].mean(axis=1) * 12)) * 100.0
    out = L + (d - base)
    return L, float(np.percentile(out, 5)), float(np.percentile(out, 95)), float((out > 0).mean())


def ci_txt(c: tuple) -> str:
    L, lo, hi, p = c
    if lo != lo:
        return f"清算後の通しの TOPIX との差 {L:+.1f}pt"
    return f"清算後の通しの TOPIX との差 {L:+.1f}pt［90％の幅 {lo:+.1f}〜{hi:+.1f}・組み直して差が正だった割合 {p * 100:.0f}％］"


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
        dpost = div_post_map(c, fd, ff, tl)
        shn = sh_now_list(c, fd, ff, tl)
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
                        DIV[i, j] += dps_now(fd, ff, r, amt, rec + pd.offsets.MonthEnd(0), dpost.get(fy_end, False))
        # 30段目（レビューの指摘）：通期の決算短信がまだ出ていない期の中間配当は、2Q の決算短信の実績から入れる
        for r, fy_end, amt, rec_me in interim_only(tl, seen_fy):
            j = ym_j.get(rec_me.strftime("%Y-%m"))
            if j is not None:
                DIV[i, j] += dps_now(fd, ff, r, amt, rec_me, False)
        for j in range(M):
            k = int(np.searchsorted(dd, cal_i[me_idx[j]], side="left")) - 1      # 月末の最後の営業日より前に出た開示だけ（その日の引けのあとの開示を使わない）
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
            k_sh = next((kk for kk in range(k, k_lo - 1, -1) if shn[kk]), None)
            if k_sh is None:
                continue
            sh_cur = shn[k_sh]
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
                    fe_ = str(r.get("CurFYEn") or "")[:10]
                    F["DY"][i, j] = dps_now(fd, ff, r, ann, fe_, dpost.get(fe_, False)) / float(px)
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
                    h0, h1 = shn[tl.index(p)], shn[tl.index(r)]
                    if h0 and h1:
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
    LIMU, LIMD = limit_masks(store, codes, cal, C, CF, OHLV) if OHLV is not None else (None, None)
    return {"codes": codes, "names": names, "sector": sector, "dates": me_dates, "me_idx": me_idx, "F": F, "UNI": UNI,
            "MCAP": MCAP, "LIQ": LIQ, "PXM": PXM, "DIV": DIV, "cal": cal, "tp": tp, "C": C, "CF": CF, "SEG": SEG, "seg_first": seg_first,
            "OHLV": OHLV, "VA": VA, "LIMU": LIMU, "LIMD": LIMD}


# 制限値幅（東証。基準値段 → 上下の値幅）。https://www.jpx.co.jp/equities/trading/domestic/06.html
LIMIT_TABLE = ((100, 30), (200, 50), (500, 80), (700, 100), (1000, 150), (1500, 300), (2000, 400), (3000, 500), (5000, 700),
               (7000, 1000), (10000, 1500), (15000, 3000), (20000, 4000), (30000, 5000), (50000, 7000), (70000, 10000),
               (100000, 15000), (150000, 30000), (200000, 40000), (300000, 50000), (500000, 70000), (700000, 100000),
               (1000000, 150000), (1500000, 300000), (2000000, 400000), (3000000, 500000), (5000000, 700000),
               (7000000, 1000000), (10000000, 1500000), (15000000, 3000000), (20000000, 4000000), (30000000, 5000000),
               (50000000, 7000000))


def price_limit(base: np.ndarray) -> np.ndarray:
    edges = np.array([e for e, _ in LIMIT_TABLE], dtype=float)
    widths = np.array([w for _, w in LIMIT_TABLE] + [10000000], dtype=float)
    return widths[np.searchsorted(edges, base, side="right")]


def limit_masks(store: dict, codes: list, cal: np.ndarray, C: np.ndarray, CF: np.ndarray, OHLV: dict):
    """ストップ高・ストップ安で引けた日の印。終値が高値（安値）と同じで、前の日の終値から制限値幅いっぱいまで動いた日。
    値幅は分割の調整をはずした「その日の株価」で決まるので、調整後の株価を、その日より後の分割の係数で割って戻す。
    ストップ高で引けた日は、終値では買えない（買いが残る）。ストップ安で引けた日は、終値では売れない。"""
    S, T = C.shape
    LIMU = np.zeros((S, T), dtype=bool)
    LIMD = np.zeros((S, T), dtype=bool)
    H, L = OHLV["h"], OHLV["l"]
    calD = cal.astype("datetime64[D]")
    for i, c in enumerate(codes):
        b = store["bars"][c]
        fd, ff = b["fd"], b["ff"]
        if len(fd):
            suf = np.r_[np.cumprod(ff[::-1])[::-1], 1.0]                 # suf[k] = ff[k:] の積
            fac = suf[np.searchsorted(fd, calD, side="right")]
        else:
            fac = np.ones(T)
        cl = C[i]
        ok = np.isfinite(cl) & (cl > 0)
        ok[0] = False
        if not ok.any():
            continue
        t = np.nonzero(ok)[0]
        base = CF[i, t - 1] / fac[t]                  # その日の株価の基準（前の日の終値を、その日の基準に）
        close = cl[t] / fac[t]
        w = price_limit(base)
        tol = 1e-4 * np.maximum(base, 1.0)
        up = (close >= base + w - tol) & (np.abs(cl[t] - H[i, t]) <= 1e-6 * np.maximum(cl[t], 1.0))
        dn = (close <= base - w + tol) & (np.abs(cl[t] - L[i, t]) <= 1e-6 * np.maximum(cl[t], 1.0))
        LIMU[i, t] = up
        LIMD[i, t] = dn
    return LIMU, LIMD


# ───────────────────────── 実際の運用の形のエンジン ─────────────────────────
W_CAP = 0.15             # 16段目：加重するときの1銘柄の上限（資産の15％）


def cap_weights(w: np.ndarray, cap: float) -> np.ndarray:
    """重みを合計1にし、上限を超えた分を、上限に届いていない銘柄に重みの比で配り直す（16段目）。"""
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        return np.full(len(w), 1.0 / max(1, len(w)))
    w = w / w.sum()
    if cap * len(w) < 1.0:
        return np.full(len(w), 1.0 / len(w))
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        ex = float((w[over] - cap).sum())
        w[over] = cap
        under = w < cap - 1e-12
        if not under.any():
            break
        su = float(w[under].sum())
        if su > 0:
            w[under] += ex * w[under] / su
        else:                      # 重みのない銘柄しか残っていなければ、等しく配る
            w[under] += ex / int(under.sum())
    return w


def sim(ranks: list, PXM: np.ndarray, DIV: np.ndarray, j0: int, keep: int, nmax: int | None = None, cap: float = CAP,
        slip: float = SLIP, tax: float = TAX, W: np.ndarray | None = None):
    """月末ごとに、順位の上から買い、keep 位より下になったら売る。1株単位・税（繰り越しの損あり）・配当は受け取るときに税。
    ranks[j] … その月末の順位（銘柄の番号の並び。範囲の中だけ）。返す値：j0 から最後までの月末の評価額、最後に全部売って税を払った額、診断。
    nmax を渡さなければ、その時点の NMAX（36段目で持つ数を変えるため、呼ぶときに読む）。"""
    if nmax is None:
        nmax = NMAX
    cash = float(cap)
    pos: dict = {}
    loss = 0.0
    vals = []
    st = {"n_buy": 0, "n_sell": 0, "tax": 0.0, "hold": []}      # hold … 月末ごとの持ち銘柄の数（23段目の現金の割合の目安）
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
            st["hold"].append(len(pos))
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
        wt = None
        if W is not None:              # 16段目：上位 nmax 銘柄の重み（時価総額・値動きの小ささ）で、新しく買う銘柄の額を決める
            top = rk[:nmax]
            raw = np.array([W[i, j] if (W[i, j] == W[i, j] and W[i, j] > 0) else 0.0 for i in top])
            if (raw > 0).any():           # 重みが分からない銘柄（上場して1年たっていないなど）は、ほかの銘柄の重みの中央値にする
                raw = np.where(raw > 0, raw, float(np.median(raw[raw > 0])))
            wt = dict(zip(top, cap_weights(raw, W_CAP))) if len(top) else {}
        for i in rk[:nmax]:
            if len(pos) >= nmax:
                break
            if i in pos:
                continue
            p = p_(i, j)
            if not p:
                continue
            px = p * (1.0 + slip)
            sh = int(min(tgt if wt is None else value * wt.get(i, 0.0), cash) / px)
            if sh < 1:
                continue
            cash -= sh * px
            pos[i] = [sh, sh * px]
            st["n_buy"] += 1
        vals.append(cash + sum(sh * p_(i, j) for i, (sh, _c) in pos.items()))
        st["hold"].append(len(pos))
    st["pos"] = {i: (sh, c_) for i, (sh, c_) in pos.items()}      # 31段目：最後の持ち株（並走記録で見せる）
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


PUB_FROM = None     # 24段目：本の出版のあとの期間（この月末の次の月からの成績を「出版後」として足す。判定には使わない）


def half_stats(vals: np.ndarray, dates: list) -> dict:
    s = pd.Series(vals, index=pd.DatetimeIndex(dates))
    r = s.pct_change().iloc[1:]
    out = {}
    for h, m in (("前半", r.index < SPLIT), ("後半", r.index >= SPLIT), ("通し", np.ones(len(r), dtype=bool))):
        out[h] = yb.rev2_stats(r[m])
    if PUB_FROM is not None and (r.index > PUB_FROM).sum() >= 6:
        out["出版後"] = yb.rev2_stats(r[r.index > PUB_FROM])
    return out


# 1回目の型（wide_lab_plan_2026-11-04 で、結果を見る前に決めたもの）。点数は大きいほど上
ONLY: set = set()        # 回す型（空なら全部）
LAST_LAB1: dict = {}     # 直前の1回目の結果（14段目で3回目の SEL を作るため）
LAB_NOTE = ""             # 報告の「決まり」の下に書き足す注（16段目の範囲と加重など）
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


def lab1(store: dict, combos: dict | None = None, extra: dict | None = None, uni_fn=None, w_fn=None, filt: bool = False) -> int:
    log("1回目の型をまとめて回します（指標を作っています）…" if not (combos or extra) else
        ("3回目（組み合わせ）を回します（指標を作っています）…" if combos else "4回目（売買の内訳）を回します（指標を作っています）…"))
    D = build(store)
    dates, UNI, F, PXM, DIV = D["dates"], D["UNI"], D["F"], D["PXM"], D["DIV"]
    if uni_fn is not None:           # 16段目：範囲を絞る（上位500社など）
        UNI = UNI & uni_fn(D)
    WM = w_fn(D) if w_fn is not None else None      # 16段目：加重（なければ同じ額ずつ）
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
                    v, _l, _s = sim(rk_r, PXM, DIV, jx, kp, W=WM)
                    rnd[nm].append(half_stats(v, evx))
            base[jx] = (evx, half_stats(tv, evx), tl, rnd)
            TXV[jx] = tv
        return base[jx]
    TXV: dict = {}
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
             "yrs": max((evf[-1] - evf[0]).days / 365.25, 0.5), "tx": tx_f, "txliq": txliq_f, "rnd": rnd_f}
        if filt:      # 23段目：条件に合う銘柄だけに点数がある型。点数のある割合ではなく、合格した銘柄の数で保留を決める（平均5銘柄未満なら保留）
            npass = [int((UNI[:, j] & np.isfinite(sc[:, j])).sum()) for j in range(jf, M)]
            r["npass"] = npass
            r["low"] = 3 if np.mean(npass) < 5 else 0
            # 参考：でたらめ（同じ数）… 月末ごとに、合格した銘柄と同じ数だけ範囲からでたらめに選ぶ（現金の割合をそろえ、選び方だけの効き目を見る。売る決まり A）
            npf = [0] * jf + npass
            rng_n = np.random.default_rng(20261109)
            r["rnd_n"] = []
            for _k in range(N_RAND):
                rk_r = ranks_from(np.repeat(rng_n.random(UNI.shape[0])[:, None], M, axis=1), UNI)
                rk_r = [x[:npf[j]] for j, x in enumerate(rk_r)]
                v, _l, _s = sim(rk_r, PXM, DIV, jf, KEEPS[0][1], W=WM)
                r["rnd_n"].append(half_stats(v, evf))
        rk = ranks_from(sc, UNI)
        for nm, kp in KEEPS:
            v, liq, s_ = sim(rk, PXM, DIV, jf, kp, W=WM)
            r[nm] = half_stats(v, evf)
            r[nm + "_liq"] = liq
            r[nm + "_st"] = s_
            r[nm + "_ci"] = boot_ci(v, TXV[jf], liq, txliq_f, r["yrs"])
        if FAIR:
            # 30段目（レビューの指摘）：判定に使うでたらめを、戦略と同じ数（合格した銘柄の数）・同じ入れ替えの速さにそろえる。
            # 銘柄ごとの点数を、毎月 q の確率で引き直す（q＝戦略の1か月あたりの売りの数 ÷ 持ち銘柄の数）
            npf_ = ([0] * jf + r["npass"]) if filt else None
            r["rnd_fair"] = {}
            for nm, kp in KEEPS:
                s_ = r[nm + "_st"]
                q = min(1.0, s_["n_sell"] / max(1.0, float(np.sum(s_["hold"]))))
                r[nm + "_q"] = q
                rng_q = np.random.default_rng(20261110 + kp)
                outs = []
                for _k in range(N_RAND):
                    scr = np.empty((UNI.shape[0], M))
                    scr[:, 0] = rng_q.random(UNI.shape[0])
                    for jj in range(1, M):
                        col = scr[:, jj - 1].copy()
                        m_ = rng_q.random(UNI.shape[0]) < q
                        col[m_] = rng_q.random(int(m_.sum()))
                        scr[:, jj] = col
                    rk_r = ranks_from(scr, UNI)
                    if npf_ is not None:
                        rk_r = [x[:npf_[j]] for j, x in enumerate(rk_r)]
                    v_, _l, _s = sim(rk_r, PXM, DIV, jf, kp, W=WM)
                    outs.append(half_stats(v_, evf))
                r["rnd_fair"][nm] = outs
        rk_low = ranks_from(-sc, UNI)
        v, _l, _s = sim(rk_low, PXM, DIV, jf, 20, W=WM)
        r["low20"] = half_stats(v, evf)
        r["bk"] = {}
        for bn, bm in buckets.items():
            v, _l, _s = sim(ranks_from(sc, UNI & bm), PXM, DIV, jf, 20, W=WM)
            r["bk"][bn] = half_stats(v, evf)
        # 判定
        ok_tx, ok_rnd = True, True
        detail = {}
        for nm, _kp in KEEPS:
            for h in ("前半", "後半"):
                a = r[nm][h]["ann"]
                rnd = [x[h]["ann"] for x in (r["rnd_fair"][nm] if FAIR else rnd_f[nm])]
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
    global LAST_LAB1
    LAST_LAB1 = res
    report(D, res, RND, ST_TX, TXliq, ev, j0)
    return 0


def stage_note() -> None:
    if COST != "S0":
        print(f"  ※ 費用と税の前提 {COST}（14段目）：株 片道{SLIP * 100:.1f}％・1306 片道{ETF_SLIP * 100:.2f}％・{TAX_TXT}"
              + ("（比べる相手の TOPIX も税なし）" if TAX == 0 else "") + "。それ以外の決まりは第1部と同じ")
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
    print(f"  範囲 … 東証の全市場の普通株（配当のない株・上場をやめた銘柄も入れる）。{uni_txt()}。"
          f"1か月あたり平均 {nu.mean():.0f}銘柄（最少 {nu.min()}・最多 {nu.max()}）")
    print(f"  決まり … 元本1,000万円・20銘柄・1株単位・月末の終値・片道{SLIP * 100:.1f}％・{TAX_TXT}（繰り越しの損あり・配当は受け取るときに）。"
          "売る決まり A＝20位より下で売る・B＝60位より下で売る")
    if LAB_NOTE:
        print(f"  ※ {LAB_NOTE}")
    if FAIR:
        print("  ※ 30段目から：判定の②のでたらめは、型ごとに「同じ数・同じ入れ替えの速さ」にそろえたもの（条件で絞る型は合格した銘柄と同じ数。下の行の「判定に使ったでたらめ」）。"
              "下の「でたらめな20銘柄の平均」は前と同じ作り方の参考。ぶれの幅は、月ごとの値動きを6か月のかたまりで2,000回組み直したもの")
    print(f"  判定 … 両方の期間で、A・B の両方とも ①TOPIX（税あり）を +{PT_TX:.1f}pt 以上 ②でたらめな20銘柄の平均を +{PT_RND:.1f}pt 以上かつ20回のうち{BEAT_MIN}回以上上回る"
          "→ ①②とも「候補」・②だけ「範囲の中では効く」・それ以外「効かない」。"
          + ("合格した銘柄が1か月あたり平均5未満なら保留" if any("npass" in r for r in res.values()) else "点数がある銘柄が範囲の半分未満の月が3か月以上なら保留"))
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
        if "rnd_n" in r:
            sel = []
            for h in HH:
                rn = [x[h]["ann"] for x in r["rnd_n"]]
                sel.append((a[h]["ann"] - float(np.mean(rn)), int(sum(1 for x in rn if a[h]["ann"] > x)), float(np.mean(rn))))
            up = all(d >= PT_RND and bt >= BEAT_MIN for d, bt, _m in sel)
            dn = all(d <= -PT_RND and bt <= N_RAND - BEAT_MIN for d, bt, _m in sel)
            r["sel"] = "ある" if up else ("裏目" if dn else "見えない")
            print(f"         でたらめ（同じ数・A） {sel[0][2]:.1f} ／ {sel[1][2]:.1f}［差 {sel[0][0]:+.1f}・{sel[0][1]}回 ／ {sel[1][0]:+.1f}・{sel[1][1]}回］"
                  f" … 選び方の効き目 {r['sel']} ・ 持ち銘柄の数 平均 {np.mean(r['A_st']['hold']):.1f}（A）・{np.mean(r['B_st']['hold']):.1f}（B）")
        if FAIR:
            fm = {nm: [np.mean([x[h]["ann"] for x in r["rnd_fair"][nm]]) for h in HH] for nm, _kp in KEEPS}
            print(f"         {ci_txt(r['A_ci'])}（A）・{ci_txt(r['B_ci'])}（B）")
            print(f"         判定に使ったでたらめ（同じ数・同じ入れ替え）の平均 A {fm['A'][0]:.1f} ／ {fm['A'][1]:.1f}（毎月 {r['A_q'] * 100:.0f}％を入れ替え）"
                  f" ・ B {fm['B'][0]:.1f} ／ {fm['B'][1]:.1f}（{r['B_q'] * 100:.0f}％）")
        bk = " ・ ".join(f"{bn} {v['前半']['ann']:.1f} ／ {v['後半']['ann']:.1f}" for bn, v in r["bk"].items())
        print(f"         参考：下から20（A） {r['low20']['前半']['ann']:.1f} ／ {r['low20']['後半']['ann']:.1f} ・ 大きさ別（A） {bk}"
              + (f" ・ 合格した銘柄 平均 {np.mean(r['npass']):.0f}（{min(r['npass'])}〜{max(r['npass'])}。0の月 {sum(1 for x in r['npass'] if x == 0)}）" if "npass" in r else
                 f" ・ 点数のある割合 平均 {np.mean(r['cov']) * 100:.0f}％")
              + f"・清算後の通し A {((r['A_liq'] / CAP) ** (1 / r['yrs']) - 1) * 100:.1f}％"
              f"・買い A {r['A_st']['n_buy'] / r['yrs']:.0f}回／年・B {r['B_st']['n_buy'] / r['yrs']:.0f}回／年")
    print("\n  2. 判定のまとめ")
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in res.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    if any("sel" in r for r in res.values()):
        for v in ("ある", "見えない", "裏目"):
            ks = [k for k, r in res.items() if r.get("sel") == v]
            print(f"     選び方の効き目（でたらめ〈同じ数〉と比べて） {v} … " + ("・".join(ks) if ks else "なし"))
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")


# ───────────────────────── 2回目：出来事と暦の型 ─────────────────────────
# 計画は wide_lab_plan2_2026-11-04（結果を見る前に決めたもの）。比べる期間・前半と後半の分け方・判定の基準は1回目と同じ。
# 1回目の「売る決まり A・B」の代わりに、型ごとに決めた2通り（持つ日数など）の両方で基準を満たしたときだけ通す。
REACT_TH = 0.05                         # 決算発表の反応（TOPIX との差）の下限
DY_TH = 0.03                            # 権利付き最終日の前に買う型：予想の配当利回りの下限（直近の通期の年間配当 ÷ 株価）
ETF_SLIP = COSTS[COST][1]               # 暦の型（1306 を売り買い）の値段のずれ（片道）。S0 は0.1％
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
        dpost = div_post_map(c, fd, ff, tl)
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
                        divpay.setdefault(t_cum_of(cal, t_rec), []).append((i, dps_now(fd, ff, r, amt, me, dpost.get(fy_end, False))))
            # 権利付き最終日の前に買う型の候補：この決算短信のあとに来る、次の期の中間と期末の権利確定日
            t_disc = int(np.searchsorted(cal, np.datetime64(d), side="left"))
            dps = ann * fac_after(fd, ff, d)
            nxt = e + pd.DateOffset(years=1)
            for rec, ok in ((nxt - pd.DateOffset(months=6), q2 > 0), (nxt, True)):
                if ok:
                    t_rec = last_td(cal, rec + pd.offsets.MonthEnd(0))
                    if t_rec >= 3:
                        recs.append((i, t_cum_of(cal, t_rec), dps, t_disc))
        # 30段目（レビューの指摘）：通期の決算短信がまだ出ていない期の中間配当は、2Q の決算短信の実績から入れる
        for r, fy_end, amt, me in interim_only(tl, seen_fy):
            t_rec = last_td(cal, me)
            if t_rec >= 3:
                divpay.setdefault(t_cum_of(cal, t_rec), []).append((i, dps_now(fd, ff, r, amt, me, False)))
    return {"annc": annc, "recs": recs, "splits": splits, "divpay": divpay}


def ev_sim(cands: dict, D: dict, divpay: dict, t0: int, nmax: int = NMAX, cap: float = CAP, slip: float = SLIP, tax: float = TAX):
    """出来事で買って、決めた日に売る運用。cands … {買う日の番号: [(優先度, 銘柄の番号, 売る日の番号), …]}。
    毎日、①売る日が来た銘柄を終値で売る ②その日の候補を優先度の高い順に買う（20銘柄まで・そのときの資産の20分の1まで・1株単位）
    ③権利付き最終日の終わりに持っている銘柄の配当を受け取る（税）。上場をやめた銘柄は最後の値段で売る。月末ごとの評価額を返す。"""
    CF, C, me_idx = D["CF"], D["C"], D["me_idx"]
    LIMU, LIMD = D.get("LIMU"), D.get("LIMD")       # 始値・高値・安値があるときだけ（20年分のデータ）
    T = CF.shape[1]
    me_set = {t: j for j, t in enumerate(me_idx)}
    cash, loss = float(cap), 0.0
    pos: dict = {}
    vals = []
    st = {"n_buy": 0, "buy_t": [], "tax": 0.0, "held": [], "skip_up": 0, "late_dn": 0}

    def px(i, t):
        v = CF[i, t]
        return float(v) if (v == v and v > 0) else 0.0
    for t in range(t0, T):
        for i in [k for k, p in pos.items() if p[2] <= t]:
            if LIMD is not None and LIMD[i, t] and t < T - 1:     # ストップ安で引けた日は売れないので、次の日に売る
                st["late_dn"] += 1
                continue
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
                if LIMU is not None and LIMU[i, t]:     # ストップ高で引けた日は、終値では買えない
                    st["skip_up"] += 1
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


def ev_sim_fill(cands: dict, D: dict, divpay: dict, t0: int, nmax: int = NMAX, cap: float = CAP, slip: float = SLIP, tax: float = TAX,
                eslip: float = ETF_SLIP):
    """ev_sim と同じ運用で、空いている資金をいつも 1306 で持つ（10段目）。
    毎日、①売る日が来た銘柄を売る ②候補を買う。お金が足りなければ、その分の 1306 を売る ③残ったお金で 1306 を買う（資産の1％を超えるとき）。
    1306 の分配金（yb.ETF_DIST）は、その日以降の最初の営業日に受け取る（税）。1306 の売買の益と損は、株の益と損と合わせて税を計算する。"""
    CF, C, me_idx, cal = D["CF"], D["C"], D["me_idx"], D["cal"]
    LIMU, LIMD = D.get("LIMU"), D.get("LIMD")
    tpc = np.asarray(D["tp"]["c"], dtype=float)
    T = CF.shape[1]
    me_set = set(me_idx)
    dist: dict = {}
    for d_, a_ in yb.ETF_DIST.get("1306", ()):
        tt = int(np.searchsorted(cal, np.datetime64(pd.Timestamp(d_).date()), side="left"))
        if tt < T:
            dist[tt] = dist.get(tt, 0.0) + a_ / 100.0
    cash, loss = float(cap), 0.0
    eu, ecost = 0, 0.0
    pos: dict = {}
    vals = []
    st = {"n_buy": 0, "buy_t": [], "tax": 0.0, "held": [], "skip_up": 0, "late_dn": 0, "etf_share": []}

    def px(i, t):
        v = CF[i, t]
        return float(v) if (v == v and v > 0) else 0.0

    def realize(g):
        nonlocal cash, loss
        if g > 0:
            tx = max(0.0, g - loss) * tax
            loss = max(0.0, loss - g)
            cash -= tx
            st["tax"] += tx
        else:
            loss += -g

    def etf_sell(units, t):
        nonlocal eu, ecost, cash
        units = min(int(units), eu)
        if units <= 0:
            return
        pr = tpc[t] * (1.0 - eslip)
        c_part = ecost * units / eu
        cash += units * pr
        eu -= units
        ecost -= c_part
        realize(units * pr - c_part)
    for t in range(t0, T):
        ep = float(tpc[t])
        if t in dist and eu > 0:
            g = eu * dist[t]
            cash += g * (1.0 - tax)
            st["tax"] += g * tax
        for i in [k for k, p in pos.items() if p[2] <= t]:
            if LIMD is not None and LIMD[i, t] and t < T - 1:
                st["late_dn"] += 1
                continue
            sh, cost, _o = pos.pop(i)
            eff = px(i, t) * (1.0 - slip)
            cash += sh * eff
            realize(sh * eff - cost)
        cl = cands.get(t)
        if cl:
            value = cash + eu * ep + sum(p[0] * px(k, t) for k, p in pos.items())
            tgt = value / nmax
            for _pr, i, t_out in sorted(cl, key=lambda x: (-x[0], x[1])):
                if len(pos) >= nmax:
                    break
                if i in pos or t_out <= t:
                    continue
                v = C[i, t]
                if not (v == v and v > 0):
                    continue
                if LIMU is not None and LIMU[i, t]:
                    st["skip_up"] += 1
                    continue
                p = float(v) * (1.0 + slip)
                if cash < tgt and eu > 0:
                    etf_sell(np.ceil((tgt - cash) / (ep * (1.0 - eslip))), t)
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
        value = cash + eu * ep + sum(p[0] * px(k, t) for k, p in pos.items())
        if cash > 0.01 * value and ep > 0:
            u = int(cash / (ep * (1.0 + eslip)))
            if u > 0:
                cash -= u * ep * (1.0 + eslip)
                eu += u
                ecost += u * ep * (1.0 + eslip)
        if t in me_set:
            value = cash + eu * ep + sum(p[0] * px(k, t) for k, p in pos.items())
            vals.append(value)
            st["held"].append(len(pos))
            st["etf_share"].append(eu * ep / value if value > 0 else 0.0)
    liq, gain = cash, 0.0
    for k, p in pos.items():
        eff = px(k, T - 1) * (1.0 - slip)
        liq += p[0] * eff
        gain += p[0] * eff - p[1]
    if eu > 0:
        eff = tpc[T - 1] * (1.0 - eslip) * eu
        liq += eff
        gain += eff - ecost
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
    print(f"  決まり … 元本1,000万円・20銘柄まで・1株単位・終値で売り買い・片道{SLIP * 100:.1f}％（暦の型の 1306 は{ETF_SLIP * 100:.2f}％）・{TAX_TXT}（繰り越しの損あり）・"
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
    report6(res, ST_TX, TXliq, ev, yrs, D.get("LIMU") is not None)
    return 0


def report6(res, ST_TX, TXliq, ev, yrs, has_lim=False):
    HH = ("前半", "後半")
    print("\n【日本株の手法・6段目（短期売買・テクニカル）（wide_lab）：東証全銘柄・実際の運用の形】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan6_2026-11-04）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  決まり … 2回目と同じ（元本1,000万円・20銘柄まで・その日の終値で売り買い・片道{SLIP * 100:.1f}％・{TAX_TXT}）。T9 は 1306 を売り買い（片道{ETF_SLIP * 100:.2f}％）")
    if has_lim:
        print("  ストップ高・ストップ安 … ストップ高で引けた日は終値で買えない（その銘柄は買わない）・ストップ安で引けた日は終値で売れない（次の日に売る）。でたらめの比べも同じ（11月5日に追加）")
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
                     f"合図 {x['n_cand']:,}回・買い 前半 {x['nb']['前半']}回・後半 {x['nb']['後半']}回・平均の持ち数 {x['held']:.1f}・でたらめの買いの回数 {x['fill'] * 100:.0f}％・"
                     f"ストップ高で買えなかった {x['s'].get('skip_up', 0):,}回・ストップ安で売りを次の日へ {x['s'].get('late_dn', 0):,}回")
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
          "mi": ("/v2/markets/margin-interest", "2008-05-01", "date", "W"),         # 信用取引の週末残高（週の最後の営業日）
          "fsd2": ("/v2/fins/details", "2008-05-01", "date", "D"),                 # 財務の明細（開示日ごと。プレミアムだけ。12段目。項目は名前でしぼって値ごと持つ）
          # 13段目（J-Quants の残りのデータ）
          "inv": ("/v2/equities/investor-types", "2008-01-16", None, "R"),          # 投資部門別売買状況（週ごと。期間でまとめて取る）
          "srt": ("/v2/markets/short-ratio", "2008-11-05", "date", "D"),            # 業種別空売り比率（日ごと。全業種を足す）
          "mal": ("/v2/markets/margin-alert", "2008-05-08", "date", "D"),           # 日々公表信用取引残高
          "o225": ("/v2/derivatives/bars/daily/options/225", "2008-05-07", "date", "D"),   # 日経225オプション（日ごとに集計して持つ）
          "fut": ("/v2/derivatives/bars/daily/futures", "2008-05-07", "date", "D"),        # 先物（プレミアムだけ。商品ごとに集計）
          "opt": ("/v2/derivatives/bars/daily/options", "2008-05-07", "date", "D"),        # オプション（プレミアムだけ。商品ごとに集計）
          "msh": ("/v2/edinet/major-shareholders", "2016-06-01", "date", "D"),      # 大株主（提出日ごと）
          "xsh": ("/v2/edinet/cross-shareholdings", "2020-03-31", "date", "D"),     # 政策保有株式（提出日ごと）
          "lvh": ("/v2/edinet/large-volume-shareholders", "2021-07-01", "date", "D"),      # 大量保有報告書（提出日ごと）
          # 18段目：個別株オプション（全オプションから、原資産ごとに集計し直す。月ごとのファイルで取る）
          "optu": ("/v2/derivatives/bars/daily/options", "2008-05-07", "date", "D")}
X_EXTRA_SETS = {"optu"}          # 名前を指定したときだけ取る（それまでの段の「そろっているか」の確かめには入れない）
X_FIELDS = {"div": {"pub": ("PubDate",), "time": ("PubTime",), "code": ("Code",), "stat": ("StatCode",), "if": ("IFCode",),
                    "fr": ("FRCode",), "term": ("IFTerm",), "rate": ("DivRate",), "cs": ("CommSpecCode",), "rec": ("RecDate",)},
            "ssr": {"disc": ("DiscDate",), "calc": ("CalcDate",), "code": ("Code",), "seller": ("SSName",),
                    "dic": ("DICName",), "fund": ("FundName",), "ratio": ("ShrtPosToSO",)},
            "ed": {"pub": ("PubDate",), "sch": ("SchDate",), "fq": ("FQName",), "fye": ("FYE",), "code": ("Code",)},
            "mi": {"date": ("Date",), "code": ("Code",), "long": ("LongVol",), "short": ("ShrtVol",), "iss": ("IssType",)},
            "fsd2": {"disc": ("DiscDate",), "time": ("DiscTime",), "code": ("Code",), "doc": ("DocType",), "fs": ()},   # fs は項目名 → 値
            # 13段目：取り出し方は X_REDUCE（ここは持つ形の名前だけ）
            "inv": {"pub": (), "st": (), "en": (), "sec": (), "fbuy": (), "fsell": (), "ibuy": (), "isell": ()},
            "srt": {"date": (), "sell": (), "s1": (), "s2": ()},
            "mal": {"pub": (), "app": (), "code": ()},
            "o225": {"date": (), "pvo": (), "cvo": (), "bv": ()},
            "fut": {"date": (), "prod": (), "oi": (), "vo": ()},
            "opt": {"date": (), "prod": (), "pvo": (), "cvo": ()},
            "msh": {"sub": (), "code": (), "doc": (), "trust": (), "top": ()},
            "xsh": {"sub": (), "code": (), "per": (), "book": (), "dec": (), "inc": (), "iss": ()},
            "lvh": {"sub": (), "code": (), "doc": (), "typ": (), "ratio": (), "last": ()},
            "optu": {"date": (), "und": (), "n": (), "pv": (), "cv": (), "iv": ()}}
X_OPTIONAL = {("ssr", "dic"), ("ssr", "fund"), ("div", "time"), ("ed", "fye"), ("mi", "iss"), ("fsd2", "time"), ("fsd2", "doc")}
FS_KEEP_WORDS = ("gross", "assets", "liabilities", "loans", "bonds", "borrowings", "debt", "sales", "revenue")


def fs_keep(fs: dict) -> dict:
    """明細のうち、12段目で使うかもしれない項目（名前に gross・assets・liabilities・loans・bonds・borrowings・debt・sales・revenue を含む）だけを、値（数）にして持つ。"""
    out = {}
    for k, v in (fs or {}).items():
        kl = str(k).lower()
        if any(w in kl for w in FS_KEEP_WORDS):
            x = num(v)
            if x is not None:
                out[str(k)] = x
    return out


def fs_parse(k: str):
    """明細の項目名を（名前, 区分）に分ける。会計基準の印（(IFRS) など）と、末尾の区分の記号（-CA・- NCL・-CMD など）を外す。"""
    import re
    k = re.sub(r"\s*\((IFRS|US GAAP|USGAAP|JMIS|Japanese GAAP|J-GAAP|JGAAP)\)\s*$", "", str(k)).strip()
    sec = None
    m = re.match(r"^(.*?)\s*-\s*([A-Z][A-Za-z]{1,6})$", k)
    if m and " " not in m.group(2):
        k, sec = m.group(1).strip(), m.group(2).upper()
    k = re.sub(r",\s*DEI$", "", k).strip()
    return k.lower(), sec


FS_NAMES = {"gp": ("gross profit", "gross profit (loss)", "gross profit on sales", "gross profit (loss) on sales",
                   "operating gross profit", "operating gross profit (loss)", "gross operating profit"),
            "sales": ("net sales", "revenue", "revenues", "operating revenue", "operating revenues", "net sales of goods", "sales",
                      "net sales of completed construction contracts", "operating revenue and net sales"),
            "ta": ("assets", "total assets"),
            "ca": ("current assets", "total current assets"),
            "cl": ("current liabilities", "total current liabilities")}
FS_LTD_TOTAL = ("bonds and borrowings", "interest-bearing liabilities", "borrowings")
FS_LTD_PARTS = ("long-term loans payable", "long-term borrowings", "bonds payable", "bonds", "long-term debt",
                "convertible bond-type bonds with subscription rights to shares", "convertible bonds")


def fs_pick(fs: dict) -> dict:
    """財務の明細（FS）から、売上総利益・売上・総資産・流動資産・流動負債・長期の借入と社債を取り出す。見つからなければ None。
    長期の借入は、区分が固定負債（NCL）または区分の記号がない項目だけ（流動負債（CL）の借入は入れない）。"""
    base0, base1, ncl = {}, {}, {}
    for k, v in (fs or {}).items():
        nm, sec = fs_parse(k)
        x = num(v)
        if x is None:
            continue
        if sec is None:
            base0.setdefault(nm, x)
        base1.setdefault(nm, x)
        if sec in (None, "NCL"):
            ncl.setdefault(nm, x)
    out = {}
    for lk, names in FS_NAMES.items():
        v = next((base0[n] for n in names if base0.get(n) is not None), None)
        out[lk] = v if v is not None else next((base1[n] for n in names if base1.get(n) is not None), None)
    tot = next((ncl[n] for n in FS_LTD_TOTAL if ncl.get(n) is not None), None)
    if tot is None:
        parts = [ncl[n] for n in FS_LTD_PARTS if ncl.get(n) is not None]
        tot = float(sum(parts)) if parts else (0.0 if out["ta"] is not None else None)   # 借入の項目がなければ0（借入なし）とみなす
    out["ltd"] = tot
    return out


def _red_inv(rows, d):
    return [(r.get("PubDate"), r.get("StDate"), r.get("EnDate"), r.get("Section"), num(r.get("FrgnBuy")), num(r.get("FrgnSell")),
             num(r.get("IndBuy")), num(r.get("IndSell"))) for r in rows]


def _red_srt(rows, d):
    if not rows:
        return []
    f = lambda k: float(sum(num(r.get(k)) or 0.0 for r in rows))
    return [(d, f("SellExShortVa"), f("ShrtWithResVa"), f("ShrtNoResVa"))]


def _red_mal(rows, d):
    return [(r.get("PubDate"), r.get("AppDate"), r.get("Code")) for r in rows]


def _red_o225(rows, d):
    rows = _not_emg(rows)
    if not rows:
        return []
    pv = float(sum(num(r.get("Vo")) or 0.0 for r in rows if str(r.get("PCDiv")) == "1"))
    cv = float(sum(num(r.get("Vo")) or 0.0 for r in rows if str(r.get("PCDiv")) == "2"))
    cms = sorted({str(r.get("CM")) for r in rows if num(r.get("BaseVol")) is not None})
    bv = None
    if cms:
        vals = [num(r.get("BaseVol")) for r in rows if str(r.get("CM")) == cms[0] and num(r.get("BaseVol")) is not None]
        bv = float(np.median(vals)) if vals else None
    return [(d, pv, cv, bv)]


def _not_emg(rows):
    """緊急取引証拠金が発動された日は、発動のときの値（001）と、ふだんの清算のときの値（002）の2つが来る。002 だけを使う（二重に数えない）。"""
    return [r for r in rows if str(r.get("EmMrgnTrgDiv") or "002") != "001"]


def _red_fut(rows, d):
    g = {}
    for r in _not_emg(rows):
        k = str(r.get("ProdCat") or "")
        a = g.setdefault(k, [0.0, 0.0])
        a[0] += num(r.get("OI")) or 0.0
        a[1] += num(r.get("Vo")) or 0.0
    return [(d, k, v[0], v[1]) for k, v in g.items()]


def _red_opt(rows, d):
    g = {}
    for r in _not_emg(rows):
        k = str(r.get("ProdCat") or "")
        a = g.setdefault(k, [0.0, 0.0])
        if str(r.get("PCDiv")) == "1":
            a[0] += num(r.get("Vo")) or 0.0
        elif str(r.get("PCDiv")) == "2":
            a[1] += num(r.get("Vo")) or 0.0
    return [(d, k, v[0], v[1]) for k, v in g.items()]


def _red_msh(rows, d):
    out = []
    for r in rows:
        hs = r.get("Hldrs") or []
        trust = float(sum(num(h.get("ShsRatio")) or 0.0 for h in hs if "信託口" in str(h.get("HldrName") or "")))
        top = float(sum(num(h.get("ShsRatio")) or 0.0 for h in hs))
        out.append((r.get("SubDate"), r.get("Code"), r.get("DocTypeCode"), trust, top))
    return out


def _red_xsh(rows, d):
    out = []
    for r in rows:
        rp = r.get("Report") or {}
        out.append((r.get("SubDate"), r.get("Code"), r.get("PerEn"), num(rp.get("ListedBookVal")), num(rp.get("ListedDecSaleAmt")),
                    num(rp.get("ListedIncAcqCost")), num(rp.get("ListedIss"))))
    return out


def _red_lvh(rows, d):
    return [(r.get("SubDate"), r.get("Code"), r.get("DocTypeCode"), r.get("LargeHldgTypeCode"), num(r.get("TotalShsRatio")),
             num(r.get("TotalShsRatioLast"))) for r in rows]


def _red_optu(rows, d):
    """個別株オプション（商品区分 EQOP）を、原資産ごとに：出来高のある銘柄の数・プットとコールの出来高・いちばん近い限月の IV の中央値。"""
    g = {}
    for r in _not_emg(rows):
        if str(r.get("ProdCat") or "") != "EQOP":
            continue
        u = str(r.get("UndSSO") or "")
        if not u:
            continue
        a = g.setdefault(u, {"n": 0, "pv": 0.0, "cv": 0.0, "iv": {}})
        vo = num(r.get("Vo")) or 0.0
        if vo > 0:
            a["n"] += 1
        pc = _bulk_norm_code(r.get("PCDiv"))
        if pc == "1":
            a["pv"] += vo
        elif pc == "2":
            a["cv"] += vo
        iv = num(r.get("IV"))
        if iv is not None and iv > 0:
            a["iv"].setdefault(str(r.get("CM") or ""), []).append(iv)
    out = []
    for u, a in g.items():
        ivm = None
        if a["iv"]:
            cm = sorted(a["iv"])[0]
            ivm = float(np.median(a["iv"][cm]))
        out.append((d, u, a["n"], a["pv"], a["cv"], ivm))
    return out


X_REDUCE = {"optu": _red_optu, "inv": _red_inv, "srt": _red_srt, "mal": _red_mal, "o225": _red_o225, "fut": _red_fut, "opt": _red_opt,
            "msh": _red_msh, "xsh": _red_xsh, "lvh": _red_lvh}
X13 = ("inv", "srt", "mal", "o225", "fut", "opt", "msh", "xsh", "lvh")


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
    if kind == "R":
        return ["ALL"]
    start = np.datetime64(st_)
    if STAGE == "explore":
        start = max(start, EXPLORE_FROM - np.timedelta64(500, "D"))
    days = [d for d in store["etf1306"]["d"] if d >= start]
    if name == "fsd2":
        c0 = store["etf1306"]["d"][0]
        if start < c0:
            pre = [np.datetime64(x.date()) for x in pd.bdate_range(pd.Timestamp(str(start)), pd.Timestamp(str(c0)) - pd.Timedelta(days=1))]
            days = pre + days
    if kind == "W":
        wk = {}
        for d in days:
            ic = pd.Timestamp(d).isocalendar()
            wk[(ic[0], ic[1])] = d
        days = sorted(wk.values())[:-1] if wk else []      # 最後の週は、まだ公表されていないことがあるので外す
    return [str(d) for d in days]


def x_complete(store: dict, xs: dict, names=None) -> bool:
    for n in (names or [k for k in X_SETS if k not in X_EXTRA_SETS]):
        st = xs.get(n, {})
        if st.get("bad"):
            continue
        if not set(x_days(store, n)) <= (st.get("done", set()) | st.get("fail", set())):
            return False
    return True


# 日ごとに取ると時間がかかりすぎるデータは、月ごとのファイル（/v2/bulk。中身は API と同じ日ごとの四本値）でまとめて取る。
# ファイルがない月（いまの月など）や、ファイルの形が思ったものと違うときは、これまでどおり日ごとに取る
X_BULK = {"opt": ("/derivatives/bars/daily/options", ("Date", "ProdCat", "PCDiv", "Vo", "EmMrgnTrgDiv")),
          "optu": ("/derivatives/bars/daily/options", ("Date", "ProdCat", "UndSSO", "PCDiv", "Vo", "IV", "CM", "EmMrgnTrgDiv"))}


def _bulk_norm_date(s: str) -> str:
    s = str(s).strip()[:10]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else s


def _bulk_norm_code(s) -> str:
    """'1'・'1.0'・'001' を同じに扱う（CSV で数字の形が変わっても、区分を取り違えないため）。"""
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.lstrip("0") or "0"


def fetch_x_bulk(jq, name: str, st: dict, todo: list) -> None:
    """月ごとのファイルで取れる日を取り、st["done"] に入れる。取れなかった日は残す（呼び出し側が日ごとに取る）。"""
    import gzip
    import io
    import re
    import requests
    ep, need = X_BULK[name]
    key = os.environ.get("J_QUANTS_API_KEY", "")
    want = set(todo)
    months = sorted({d[:7] for d in todo})
    try:
        items = jq.get("/v2/bulk/list", {"endpoint": ep, "from": months[0], "to": months[-1]})
    except Exception as e:
        log(f"  {name}：月ごとのファイルの一覧が取れません（{str(e)[:120]}）。日ごとに取ります")
        return
    files = []
    for it in items or []:
        k = str(it.get("Key") or "")
        m = re.search(r"_(\d{6}|\d{8})\.csv\.gz$", k)
        if not m:
            continue
        tag = m.group(1)
        ds = {d for d in want if d.replace("-", "").startswith(tag)}
        if ds:
            files.append((tag, k, it.get("Size")))
    files.sort()
    log(f"  {name}：月ごとのファイル {len(files)}個で取ります（一覧 {len(items or [])}個。例：{files[0][1] if files else 'なし'}）")
    shown = False
    for i, (tag, k, size) in enumerate(files, 1):
        if elapsed() >= BUDGET_MIN:
            return
        try:
            time.sleep(yb.API_SLEEP)
            r = requests.get(f"{yb.JQUANTS_BASE}/v2/bulk/get", params={"key": k}, headers={"x-api-key": key}, timeout=60)
            r.raise_for_status()
            url = r.json().get("url")
            g = requests.get(url, timeout=600)
            g.raise_for_status()
            raw = gzip.decompress(g.content)
            head = raw[:4000].decode("utf-8-sig", errors="replace").splitlines()[0]
            cols = [c.strip().strip('"') for c in head.split(",")]
            if not shown:
                log("  ファイルの項目 … " + "・".join(cols[:40]))
                shown = True
            miss = [c for c in need if c not in cols]
            if miss:
                log(f"  ■ {name}：ファイルに {'・'.join(miss)} がありません。日ごとに取ります")
                return
            df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, usecols=list(need), encoding="utf-8-sig")
        except Exception as e:
            log(f"  {name}：ファイル {k} が取れません（{str(e)[:120]}）。この月は日ごとに取ります")
            continue
        df["Date"] = df["Date"].map(_bulk_norm_date)
        df["PCDiv"] = df["PCDiv"].map(_bulk_norm_code)
        df["EmMrgnTrgDiv"] = df["EmMrgnTrgDiv"].map(lambda v: "001" if _bulk_norm_code(v) == "1" else "002")
        n_day = 0
        for d, grp in df.groupby("Date", sort=True):
            if d not in want or d in st["done"]:
                continue
            st["rows"].extend(X_REDUCE[name](grp.to_dict("records"), d))
            st["done"].add(d)
            st["fail"].discard(d)
            n_day += 1
        if i % 12 == 0 or i == len(files):
            log(f"  {i}/{len(files)}ファイル（{tag}：{n_day}日・{len(df):,}行・開始から{elapsed():.0f}分）")


def _x_iter(jq, path, prm, todo, par):
    """日ごとに取った結果を、(番号, 日, 行, 例外) の形で順に返す。par のときは THREADS 個の接続で同時に取る（返す順は日の順のまま）。"""
    if not par:
        for n, d in enumerate(todo, 1):
            time.sleep(yb.API_SLEEP)
            try:
                yield n, d, jq.get(path, {prm: d}), None
            except Exception as e:
                yield n, d, None, e
        return
    local = threading.local()

    def one(d):
        if not hasattr(local, "jq"):
            local.jq = type(jq)(os.environ.get("J_QUANTS_API_KEY", ""))
        time.sleep(yb.API_SLEEP)
        try:
            return d, local.jq.get(path, {prm: d}), None
        except Exception as e:
            return d, None, e
    chunk = THREADS * 10
    n = 0
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        for k in range(0, len(todo), chunk):      # 時間切れは呼び出し側が見て、保存して止まる（ここでは止めない）
            for d, rows, err in ex.map(one, todo[k:k + chunk]):
                n += 1
                yield n, d, rows, err


def fetch_x(jq, store: dict, xs: dict, names=None) -> bool:
    """配当金情報・空売り残高の報告・決算発表予定日・信用取引の週末残高を、日ごとに取る。時間切れなら取った分を保存して False。
    取れない種類（プランにない・項目が違う・最近の日でも取れない）は、その種類だけ「保留」にして先へ進む。"""
    last = time.monotonic()
    for name, (path, _start, prm, _kind) in X_SETS.items():
        if names and name not in names:
            continue
        if not names and name in X_EXTRA_SETS:
            continue
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
        if name in X_BULK and os.environ.get("WIDE_BULK", "1") == "1":
            fetch_x_bulk(jq, name, st, todo)
            todo = [d for d in days if d not in st["done"] and d not in st["fail"]]
            x_save(xs, f"{name}：月ごとのファイルのあと、取れた日 {len(days) - len(todo)}日・残り {len(todo)}日")
            if elapsed() >= BUDGET_MIN:
                return False
            if not todo:
                x_save(xs, f"{name} を取り終えました（{len(st['rows']):,}件）")
                continue
        if name in X_REDUCE and X_SETS[name][3] == "R":          # 期間でまとめて取る（投資部門別売買状況）
            try:
                st_ = X_SETS[name][1]
                if STAGE == "explore":
                    st_ = max(np.datetime64(st_), EXPLORE_FROM - np.timedelta64(500, "D")).astype(str)
                rows = jq.get(path, {"from": str(st_), "to": str(date.today())})
            except yb.NotAllowed as e:
                bad(f"プランに含まれていない可能性があります：{e}")
                continue
            except Exception as e:
                bad(f"取れません：{e}")
                continue
            if rows:
                st["keys"] = list(rows[0].keys())
                log("  受け取った項目 … " + "・".join(st["keys"][:20]) + ("…" if len(st["keys"]) > 20 else ""))
            st["rows"] = X_REDUCE[name](rows or [], "ALL")
            st["done"].add("ALL")
            x_save(xs, f"{name} を取り終えました（{len(st['rows']):,}件）")
            continue
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
        par = THREADS > 1 and name in X_REDUCE     # 13段目のデータは、同時に何日か取れる（プレミアムの上限は1分に500回）
        if par:
            log(f"  同時に {THREADS}日ずつ取ります")
        for n, d, rows, err in _x_iter(jq, path, prm, todo, par):
            if elapsed() >= BUDGET_MIN:
                x_save(xs, f"{name} を途中まで保存しました")
                return False
            if isinstance(err, yb.NotAllowed):
                bad(f"プランに含まれていない可能性があります：{err}")
                break
            if err is not None:
                e = err
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
            if name in X_REDUCE:
                if rows and st["keys"] is None:
                    st["keys"] = list(rows[0].keys())
                    log("  受け取った項目 … " + "・".join(st["keys"][:40]) + ("…" if len(st["keys"]) > 40 else ""))
                st["rows"].extend(X_REDUCE[name](rows or [], d))
                st["done"].add(d)
                if n % 500 == 0:
                    log(f"  {n}/{len(todo)}（開始から{elapsed():.0f}分）")
                if (time.monotonic() - last) / 60 >= SAVE_EVERY_MIN:
                    x_save(xs, "途中まで保存しました")
                    last = time.monotonic()
                continue
            if name == "fsd2":
                if rows and st["keys"] is None:
                    st["keys"] = list(rows[0].keys())
                    fs0 = rows[0].get("FS") or {}
                    log(f"  書類の種類の例 … {rows[0].get('DocType')}")
                    log("  受け取った項目 … " + "・".join(st["keys"]) + f"（明細の項目 {len(fs0)}個。例：" +
                        "・".join([k for k in fs0 if any(w in k.lower() for w in ("gross", "current", "borrow", "bond", "total assets", "net sales", "revenue"))][:12]) + "）")
                for r in rows or []:
                    st["rows"].append((r.get("DiscDate"), r.get("DiscTime"), r.get("Code"), r.get("DocType"), fs_keep(r.get("FS") or {})))
                st["done"].add(d)
                if n % 500 == 0:
                    log(f"  {n}/{len(todo)}（開始から{elapsed():.0f}分）")
                if (time.monotonic() - last) / 60 >= SAVE_EVERY_MIN:
                    x_save(xs, "途中まで保存しました")
                    last = time.monotonic()
                continue
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


def fsd_timeline(store: dict, xs: dict) -> dict:
    """銘柄ごとに、通期の決算（決算短信の要約）に、財務の明細の売上総利益・流動資産・流動負債・長期の借入を合わせた並びを作る。
    明細は、要約の開示日の3日前〜100日後の、通期の明細のうち最初のもの。使えるのは、両方の開示日の遅いほうの翌日から。"""
    det = {}
    docs, kc = {}, {}
    rows_all = x_rows(xs, "fsd2")
    for r in rows_all:
        docs[str(r.get("doc"))] = docs.get(str(r.get("doc")), 0) + 1
    log("  財務の明細の書類の種類（多い順） … " + "・".join(f"{k} {v:,}" for k, v in sorted(docs.items(), key=lambda x: -x[1])[:6]))
    for r in rows_all:
        c = yb.norm_code(r.get("code") or "")
        d = str(r.get("disc") or "")[:10]
        doc = str(r.get("doc") or "")
        if len(d) == 10 and ("FY" in doc or not doc):
            fs = r.get("fs") or {}
            for k in fs:
                kc[k] = kc.get(k, 0) + 1
            det.setdefault(c, []).append((d, fs_pick(fs)))
    top = sorted(kc.items(), key=lambda x: -x[1])[:40]
    log("  通期の明細でよく出る項目（上から40） … " + "・".join(f"{k}（{v:,}）" for k, v in top))
    for c in det:
        det[c].sort(key=lambda x: x[0])
    out, stat = {}, {"fy": 0, "gp": 0, "ca": 0, "cl": 0}
    for c, st in store["fins"].items():
        tl = [r for r in fin_timeline(st) if is_fy(r)]
        if not tl:
            continue
        dl = det.get(c, [])
        dd = [x[0] for x in dl]
        rows = []
        for r in tl:
            d0 = str(r["DiscDate"])[:10]
            lo = (pd.Timestamp(d0) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            hi = (pd.Timestamp(d0) + pd.Timedelta(days=100)).strftime("%Y-%m-%d")
            k = int(np.searchsorted(dd, lo, side="left"))
            m = dl[k] if k < len(dl) and dl[k][0] <= hi else None
            x = m[1] if m else {}
            avail = max(d0, m[0]) if m else d0
            rows.append({"avail": avail, "disc": d0, "np": num(r.get("NP")), "cfo": num(r.get("CFO")), "ta": num(r.get("TA")),
                         "sales": num(r.get("Sales")), "sh": num(r.get("ShOutFY")), "tr": num(r.get("TrShFY")) or 0.0,
                         "gp": x.get("gp"), "ca": x.get("ca"), "cl": x.get("cl"), "ltd": x.get("ltd"), "ta_d": x.get("ta")})
            stat["fy"] += 1
            stat["gp"] += x.get("gp") is not None
            stat["ca"] += x.get("ca") is not None
            stat["cl"] += x.get("cl") is not None
        out[c] = rows
    return out, stat


def fscore_items(cur: dict, prv: dict, fd, ff) -> list:
    """Fスコアの9項目（1か0。計算できない項目は None）。"""
    def ratio(a, b):
        return a / b if (a is not None and b is not None and b > 0) else None
    ta_c = cur["ta"] if cur["ta"] else cur.get("ta_d")
    ta_p = prv["ta"] if prv["ta"] else prv.get("ta_d")
    roa_c, roa_p = ratio(cur["np"], ta_c), ratio(prv["np"], ta_p)
    it = [None] * 9
    it[0] = None if roa_c is None else int(roa_c > 0)
    it[1] = None if cur["cfo"] is None else int(cur["cfo"] > 0)
    it[2] = None if (roa_c is None or roa_p is None) else int(roa_c > roa_p)
    it[3] = None if (cur["cfo"] is None or cur["np"] is None) else int(cur["cfo"] > cur["np"])
    lv_c, lv_p = ratio(cur["ltd"], ta_c), ratio(prv["ltd"], ta_p)
    it[4] = None if (lv_c is None or lv_p is None) else int(lv_c <= lv_p)
    cr_c, cr_p = ratio(cur["ca"], cur["cl"]), ratio(prv["ca"], prv["cl"])
    it[5] = None if (cr_c is None or cr_p is None) else int(cr_c > cr_p)
    if cur["sh"] and prv["sh"]:
        s_c = (cur["sh"] - cur["tr"]) / fac_after(fd, ff, cur["disc"])
        s_p = (prv["sh"] - prv["tr"]) / fac_after(fd, ff, prv["disc"])
        it[6] = int(s_c <= s_p * 1.001)
    gm_c, gm_p = ratio(cur["gp"], cur["sales"]), ratio(prv["gp"], prv["sales"])
    it[7] = None if (gm_c is None or gm_p is None) else int(gm_c > gm_p)
    at_c, at_p = ratio(cur["sales"], ta_c), ratio(prv["sales"], ta_p)
    it[8] = None if (at_c is None or at_p is None) else int(at_c > at_p)
    return it


def lab12_scores(store: dict, xs: dict):
    tl, stat = fsd_timeline(store, xs)
    log(f"  財務の明細 … 通期の決算 {stat['fy']:,}件のうち、売上総利益 {stat['gp'] / max(1, stat['fy']) * 100:.0f}％・"
        f"流動資産 {stat['ca'] / max(1, stat['fy']) * 100:.0f}％・流動負債 {stat['cl'] / max(1, stat['fy']) * 100:.0f}％が取れた（金融を含む）")
    cache = {}

    def mats(D):
        if "m" in cache:
            return cache["m"]
        codes, dates = D["codes"], D["dates"]
        GPA = np.full((len(codes), len(dates)), np.nan)
        FS = np.full((len(codes), len(dates)), np.nan)
        me = [d.strftime("%Y-%m-%d") for d in dates]
        for i, c in enumerate(codes):
            if D["sector"].get(c, "") in yb.FIN_SECTORS or c not in tl:
                continue
            rows = tl[c]
            b = store["bars"].get(c, {"fd": np.array([], dtype="datetime64[D]"), "ff": np.array([])})
            av = [r["avail"] for r in rows]
            for j, m in enumerate(me):
                k = int(np.searchsorted(av, m, side="left")) - 1       # 月末より前に使えるようになった最新の通期
                if k < 0:
                    continue
                r = rows[k]
                ta = r["ta"] if r["ta"] else r.get("ta_d")
                if r["gp"] is not None and ta and ta > 0:
                    GPA[i, j] = r["gp"] / ta
                if k >= 1:
                    it = fscore_items(r, rows[k - 1], b["fd"], b["ff"])
                    ok = [x for x in it if x is not None]
                    if len(ok) >= 8:
                        FS[i, j] = sum(ok) * 9.0 / len(ok)
        cache["m"] = (GPA, FS)
        return cache["m"]
    noise_c = {}

    def noise(D):
        if "n" not in noise_c:
            noise_c["n"] = np.random.default_rng(1205).random(len(D["codes"]))[:, None] * 1e-6
        return noise_c["n"]

    def gpa(D):
        return mats(D)[0]

    def fscore(D):
        return mats(D)[1] + noise(D)

    def fsv(D):
        FS = mats(D)[1]
        BP, UNI = D["F"]["BP"], D["UNI"]
        out = np.full(FS.shape, np.nan)
        for j in range(FS.shape[1]):
            ok = UNI[:, j] & np.isfinite(BP[:, j])
            if ok.sum() < 30:
                continue
            pr = pd.Series(BP[ok, j]).rank(pct=True).to_numpy()
            idx = np.nonzero(ok)[0]
            val = idx[pr >= 0.7]
            col = np.full(FS.shape[0], -1000.0)                      # 割安の上位30％に入らない株は買わない（いちばん下）
            col[val] = np.nan_to_num(FS[val, j], nan=-500.0)
            out[:, j] = np.where(UNI[:, j], col, np.nan)
        return out + noise(D)
    return {"GPA": (gpa, "粗利益の大きさ（売上総利益 ÷ 総資産。金融を除く）"),
            "FSCORE": (fscore, "Fスコア（9項目の合計。金融を除く）"),
            "FSV": (fsv, "割安株の中のFスコア（純資産に対して安い上位30％の中で、Fスコアの大きい順）")}


def lab13(store: dict, xs: dict) -> int:
    """13段目：J-Quants の残りのデータの型（計画は wide_lab_plan13_2026-11-05）。"""
    log("13段目（J-Quants の残りのデータの型）を回します（指標を作っています）…")
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
    calD = cal.astype("datetime64[D]")
    cal_s = [str(x)[:10] for x in calD]

    def t_after(dstr):           # その日より後の最初の営業日の番号（その日に公表されたものは、次の営業日から使える）
        return int(np.searchsorted(calD, np.datetime64(str(dstr)[:10]), side="right"))

    def t_on(dstr):              # その日（営業日でなければ次の営業日）の番号
        return int(np.searchsorted(calD, np.datetime64(str(dstr)[:10]), side="left"))

    def daily(name, fn):         # 日ごとの値（その日のデータ）を暦にそろえる。ない日は NaN
        out = np.full(T, np.nan)
        for r in x_rows(xs, name):
            k = t_on(r["date"])
            if k < T and cal_s[k] == str(r["date"])[:10]:
                v = fn(r)
                if v is not None:
                    out[k] = v
        return out

    def lag1(x):                 # 前の営業日までのデータで決める（その日の終値で売り買い）
        return np.r_[False, x[:-1]]

    def roll_q(x, w, q):
        return pd.Series(x).rolling(w, min_periods=int(w * 0.8)).quantile(q).to_numpy()

    res_t = {}

    def run_timing(key, lab, variants):
        if (ONLY and key not in ONLY):
            return
        r = {"lab": lab, "var": {}}
        for nm, vlab, x in variants:
            x = np.asarray(x, dtype=bool)
            if x[t0:].sum() == 0:
                log(f"  {key}{nm} … 一度も持たない（データがない）")
            v, liq, s_ = cal_sim(x, D, t0)
            rnd = [half_stats(cal_sim(spells_placebo(x, t0, 7000 + k), D, t0)[0], ev) for k in range(N_RAND)]
            nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
                  "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
            r["var"][nm] = {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "vlab": vlab, "nb": nb,
                            "held": float(x[t0:].mean()), "n_cand": int(x[t0:].sum()), "fill": 1.0}
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda xx: xx["held"] <= 0.0)
        res_t[key] = r
        log(f"  {key} … {r['verdict']}")
    # ── 投資部門別売買状況（東証一部、2022年4月からはプライム）
    inv = x_rows(xs, "inv")
    if inv:
        secs = {}
        for r in inv:
            secs[str(r["sec"])] = secs.get(str(r["sec"]), 0) + 1
        log("  投資部門別の市場区分（件数） … " + "・".join(f"{k} {v}" for k, v in sorted(secs.items(), key=lambda x: -x[1])[:10]))
        first = {}
        for r in inv:
            if str(r["sec"]) not in ("TSE1st", "TSEPrime"):
                continue
            k = (str(r["sec"]), str(r["st"]), str(r["en"]))
            if k not in first or str(r["pub"]) < str(first[k]["pub"]):
                first[k] = r                       # 訂正のあとの値は使わない（その時点で分かっていた値）
        wk = sorted(first.values(), key=lambda r: str(r["en"]))
        fnet = np.zeros(T)
        inet = np.zeros(T)
        cntw = np.zeros(T)
        for r in wk:
            pub = str(r["pub"] or "")[:10]
            if len(pub) != 10:
                pub = (pd.Timestamp(str(r["en"])[:10]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            k = t_after(pub)
            if k >= T:
                continue
            fnet[k] += (r["fbuy"] or 0.0) - (r["fsell"] or 0.0)
            inet[k] += (r["isell"] or 0.0) - (r["ibuy"] or 0.0)
            cntw[k] += 1
        idx = np.nonzero(cntw)[0]

        def lastn(netv, n):
            out = np.zeros(T, dtype=bool)
            if len(idx) == 0:
                return out
            cs = np.cumsum(netv[idx])
            for m, k in enumerate(idx):
                lo = cs[m - n] if m - n >= 0 else 0.0
                val = cs[m] - lo
                nxt = idx[m + 1] if m + 1 < len(idx) else T
                out[k:nxt] = (m + 1 >= n) and (val > 0)
            return out
        run_timing("INVF", "海外投資家の買い越し（東証一部・プライム）が直近 n 週の合計でプラスのとき持つ",
                   (("A", "4週", lastn(fnet, 4)), ("B", "13週", lastn(fnet, 13))))
        run_timing("INDV", "個人の売り越しが直近 n 週の合計でプラスのとき持つ（逆張り）",
                   (("A", "4週", lastn(inet, 4)), ("B", "13週", lastn(inet, 13))))
    # ── 業種別空売り比率（全業種の合計）
    if x_rows(xs, "srt"):
        ratio = daily("srt", lambda r: (r["s1"] + r["s2"]) / (r["sell"] + r["s1"] + r["s2"]) if (r["sell"] + r["s1"] + r["s2"]) > 0 else None)
        m20 = pd.Series(ratio).rolling(20, min_periods=15).mean().to_numpy()
        run_timing("SRAT", "全業種の空売りの比率（20日平均）が過去250日の中で高いとき持つ（逆張り）",
                   (("A", "中央値より上", lag1(np.nan_to_num(m20 > roll_q(m20, 250, 0.5)))),
                    ("B", "上から30％", lag1(np.nan_to_num(m20 > roll_q(m20, 250, 0.7))))))
    # ── 日経225オプション
    if x_rows(xs, "o225"):
        bv = daily("o225", lambda r: r["bv"])
        bvf = pd.Series(bv).ffill().to_numpy()
        run_timing("VOL225", "日経225オプションの基準ボラティリティ（いちばん近い限月）が過去250日の中で低いとき持つ",
                   (("A", "中央値より下", lag1(np.nan_to_num(bvf < roll_q(bvf, 250, 0.5)))),
                    ("B", "下から70％", lag1(np.nan_to_num(bvf < roll_q(bvf, 250, 0.7))))))
        pcr = daily("o225", lambda r: r["pvo"] / r["cvo"] if r["cvo"] else None)
        p20 = pd.Series(pcr).rolling(20, min_periods=15).mean().to_numpy()
        run_timing("PCR", "日経225オプションのプット ÷ コールの出来高（20日平均）が過去250日の中で高いとき持つ（逆張り）",
                   (("A", "中央値より上", lag1(np.nan_to_num(p20 > roll_q(p20, 250, 0.5)))),
                    ("B", "上から30％", lag1(np.nan_to_num(p20 > roll_q(p20, 250, 0.7))))))
    # ── オプション（全商品）
    opt = x_rows(xs, "opt")
    if opt:
        pv, cv = np.zeros(T), np.zeros(T)
        have = np.zeros(T, dtype=bool)
        prods = {}
        for r in opt:
            prods[r["prod"]] = prods.get(r["prod"], 0.0) + (r["pvo"] or 0.0) + (r["cvo"] or 0.0)
            k = t_on(r["date"])
            if k < T and cal_s[k] == str(r["date"])[:10]:
                pv[k] += r["pvo"] or 0.0
                cv[k] += r["cvo"] or 0.0
                have[k] = True
        log("  オプションの商品（出来高の多い順） … " + "・".join(f"{k} {v:,.0f}" for k, v in sorted(prods.items(), key=lambda x: -x[1])[:8]))
        pa = np.where(have & (cv > 0), pv / np.where(cv > 0, cv, 1.0), np.nan)
        a20 = pd.Series(pa).rolling(20, min_periods=15).mean().to_numpy()
        run_timing("PCRA", "オプション全商品のプット ÷ コールの出来高（20日平均）が過去250日の中で高いとき持つ（逆張り）",
                   (("A", "中央値より上", lag1(np.nan_to_num(a20 > roll_q(a20, 250, 0.5)))),
                    ("B", "上から30％", lag1(np.nan_to_num(a20 > roll_q(a20, 250, 0.7))))))
    # ── 先物（日経225・TOPIX の先物の建玉の合計）
    fut = x_rows(xs, "fut")
    if fut:
        tot = {}
        for r in fut:
            tot[r["prod"]] = tot.get(r["prod"], 0.0) + (r["oi"] or 0.0)
        log("  先物の商品（建玉の多い順） … " + "・".join(f"{k} {v:,.0f}" for k, v in sorted(tot.items(), key=lambda x: -x[1])[:10]))
        use = {k for k in tot if any(w in k.upper() for w in ("225", "TOPIX", "TPX"))} or set(tot)
        log("  建玉を足す商品 … " + "・".join(sorted(use)))
        oi = np.full(T, np.nan)
        for r in fut:
            if r["prod"] not in use:
                continue
            k = t_on(r["date"])
            if k < T and cal_s[k] == str(r["date"])[:10]:
                oi[k] = (0.0 if np.isnan(oi[k]) else oi[k]) + (r["oi"] or 0.0)
        oif = pd.Series(oi).ffill().to_numpy()

        def grew(n):
            prev = np.r_[np.full(n, np.nan), oif[:-n]]
            return lag1(np.nan_to_num(oif > prev))
        run_timing("FOI", "日経225・TOPIX の先物の建玉の合計が n 日前より多いとき持つ",
                   (("A", "20日", grew(20)), ("B", "60日", grew(60))))
    # ── 出来事の型（MALERT・LVH）
    traded = np.isfinite(C) & (C > 0)

    def elig(i, t):
        return 0 <= t < T and jm[t] >= 0 and bool(UNI[i, jm[t]]) and bool(traded[i, t])

    def pool_all(t):
        j = jm[t]
        return np.nonzero(UNI[:, j] & traded[:, t])[0] if j >= 0 else np.zeros(0, dtype=int)

    def placebo_run(cands, pool_fn, seed0):
        v, liq, s_ = ev_sim(cands, D, E["divpay"], t0)
        rnd, rnb = [], []
        for k in range(N_RAND):
            rg = np.random.default_rng(seed0 + k)
            pl = {}
            for t, cl in cands.items():
                pool = pool_fn(t)
                if len(pool) == 0:
                    continue
                pick = rg.choice(pool, size=min(len(cl), len(pool)), replace=False)
                pl[t] = [(float(rg.random()), int(i), t_out) for i, (_p, _i, t_out) in zip(pick, cl)]
            pv_, _l, ps_ = ev_sim(pl, D, E["divpay"], t0)
            rnd.append(half_stats(pv_, ev))
            rnb.append(ps_["n_buy"])
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
              "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        return {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "nb": nb, "held": float(np.mean(s_["held"])) if s_["held"] else 0.0,
                "fill": float(np.mean(rnb)) / max(1, s_["n_buy"]), "n_cand": sum(len(x) for x in cands.values())}
    res_e = {}

    def run_event(key, lab, variants, seed0):
        if ONLY and key not in ONLY:
            return
        r = {"lab": lab, "var": {}}
        for nm, vlab, cands in variants:
            r["var"][nm] = dict(placebo_run(cands, pool_all, seed0), vlab=vlab)
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda x: min(x["nb"].values()) < EV_MIN_BUYS or x["fill"] < RND_FILL_MIN)
        res_e[key] = r
        log(f"  {key} … {r['verdict']}")
    mal = x_rows(xs, "mal")
    if mal:
        seen = {}
        for r in mal:
            c = yb.norm_code(r["code"] or "")
            if c in ci and len(str(r["pub"] or "")) >= 10:
                seen.setdefault(ci[c], set()).add(t_after(r["pub"]))
        evs = []
        for i, ts in seen.items():
            prev = -10 ** 9
            for t in sorted(ts):
                if t - prev > 60:
                    evs.append((i, t))
                prev = t
        log(f"  日々公表の対象に入った最初の日 … {len(evs):,}回")
        rg = np.random.default_rng(13)

        def mk_mal(H):
            out = {}
            for i, t in evs:
                if t >= t0 and elig(i, t):
                    out.setdefault(t, []).append((float(rg.random()), i, t + H))
            return out
        run_event("MALERT", "日々公表信用取引残高の対象に入った最初の日の次の営業日に買う",
                  (("A", "20日持つ", mk_mal(20)), ("B", "5日持つ", mk_mal(5))), 13000)
    lvh = x_rows(xs, "lvh")
    if lvh:
        evs = []
        for r in lvh:
            c = yb.norm_code(r["code"] or "")
            if c in ci and str(r["typ"]) == "1" and str(r["doc"]) == "350" and len(str(r["sub"] or "")) >= 10:
                evs.append((ci[c], t_after(r["sub"]), r["ratio"] or 0.0))
        log(f"  新しく5％を超えた報告 … {len(evs):,}回")

        def mk_lvh(H):
            out = {}
            for i, t, ratio in evs:
                if t >= t0 and elig(i, t):
                    out.setdefault(t, []).append((ratio, i, t + H))
            return out
        run_event("LVH", "大量保有報告書（新しく5％を超えた報告）の次の営業日に買う",
                  (("A", "60日持つ", mk_lvh(60)), ("B", "20日持つ", mk_lvh(20))), 14000)
    # ── 報告（時期を見る型・出来事の型）
    HH = ("前半", "後半")
    print("\n【日本株の手法・13段目（J-Quants の残りのデータの型）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan13_2026-11-05）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    print(f"  時期を見る型 … 1306 を持つか現金にするか（片道{ETF_SLIP * 100:.2f}％・{TAX_TXT}）。でたらめは、持つ期間の長さをそのままに順番を入れ替えたもの。合図は前の営業日までのデータで作る")
    for res in (res_t, res_e):
        for key, r in res.items():
            print(f"     {key:<7}{r['lab']} … {r['verdict']}")
            for nm, x in r["var"].items():
                rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
                dd = " ".join(f"{r['detail'][(nm, h)][0]:+.1f}/{r['detail'][(nm, h)][1]:+.1f}・{r['detail'][(nm, h)][2]}回" for h in HH)
                extra = (f"持っていた日の割合 {x['held'] * 100:.0f}％" if res is res_t else
                         f"合図 {x['n_cand']:,}回・買い 前半 {x['nb']['前半']}回・後半 {x['nb']['後半']}回・平均の持ち数 {x['held']:.1f}・でたらめの買いの回数 {x['fill'] * 100:.0f}％")
                print(f"         {nm}（{x['vlab']}） {x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}（最大下落 {x['st']['通し']['mdd']:.0f}％）［{dd}］"
                      f" でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}・{extra}・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％")
    print("\n  判定のまとめ（時期を見る型・出来事の型）")
    allr = {**res_t, **res_e}
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in allr.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    # ── 毎月の点数の型（MSH・XSH）は1回目と同じ道具で
    extra = {}
    msh = x_rows(xs, "msh")
    if msh:
        by = {}
        for r in msh:
            c = yb.norm_code(r["code"] or "")
            if len(str(r["sub"] or "")) >= 10:
                by.setdefault(c, []).append((str(r["sub"])[:10], r["trust"]))
        for c in by:
            by[c].sort()

        def msh_fn(DD):
            out = np.full((len(DD["codes"]), len(DD["dates"])), np.nan)
            me = [d.strftime("%Y-%m-%d") for d in DD["dates"]]
            for i, c in enumerate(DD["codes"]):
                lst = by.get(c)
                if not lst or len(lst) < 2:
                    continue
                subs = [x[0] for x in lst]
                for j, m in enumerate(me):
                    k = int(np.searchsorted(subs, m, side="left")) - 1
                    if k >= 1:
                        out[i, j] = lst[k][1] - lst[k - 1][1]
            return out + np.random.default_rng(1305).random(len(DD["codes"]))[:, None] * 1e-9
        extra["MSH"] = (msh_fn, "大株主の「信託口」の比率の合計が、前の報告より増えた幅")
    xsh = x_rows(xs, "xsh")
    if xsh:
        byx = {}
        for r in xsh:
            c = yb.norm_code(r["code"] or "")
            if len(str(r["sub"] or "")) >= 10:
                byx.setdefault(c, []).append((str(r["sub"])[:10], r))
        for c in byx:
            byx[c].sort(key=lambda x: x[0])

        def xsh_fn(DD):
            out = np.full((len(DD["codes"]), len(DD["dates"])), np.nan)
            me = [d.strftime("%Y-%m-%d") for d in DD["dates"]]
            MC = DD["MCAP"]
            for i, c in enumerate(DD["codes"]):
                lst = byx.get(c)
                if not lst:
                    continue
                subs = [x[0] for x in lst]
                for j, m in enumerate(me):
                    k = int(np.searchsorted(subs, m, side="left")) - 1
                    if k < 0 or not (MC[i, j] > 0):
                        continue
                    if (pd.Timestamp(m) - pd.Timestamp(subs[k])).days > 450:
                        continue
                    r = lst[k][1]
                    dec = r["dec"] if r["dec"] is not None else 0.0          # 有価証券報告書があって、売った額の記載がなければ0（持ち合いがない・減らしていない）
                    out[i, j] = dec / MC[i, j]
            return out + np.random.default_rng(1306).random(len(DD["codes"]))[:, None] * 1e-12
        extra["XSH"] = (xsh_fn, "持ち合い株を減らして売った額 ÷ 時価総額（直近の有価証券報告書）")
        START_T["XSH"] = int(np.searchsorted(calD, np.datetime64("2021-07-30"), side="left"))
    if extra:
        global LAB_TITLE
        LAB_TITLE = "13段目（J-Quants の残りのデータ：毎月の点数）"
        lab1(store, extra=extra)
    return 0


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
    lab5_events(store, xs)
    # SSR の空売りの形：大口の空売りの報告が多い株を売る（LS・TXS。8段目と同じ道具・同じ判定）
    if (not ONLY or "SSR" in ONLY) and not xs.get("ssr", {}).get("bad"):
        lab8(store, extra={"SSR": (ssr_score(xs), "大口の空売りの報告")},
             head="【日本株の手法・5段目（残りの型：SSR の空売りの形）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan5_2026-11-04。"
                  "売る側は、大口の空売りの報告の比率の合計が大きい順）")
    return 0


CONFIRM_X_FROM = pd.Timestamp("2014-03-31")      # 11段目の確かめる段階：配当金情報が1年分たまってから比べ始める


def lab5_events(store: dict, xs: dict, fill: bool = False, topn: int | None = None, D: dict | None = None, keys: set | None = None) -> int:
    """5段目の出来事の型（DIVUP・E2C）。fill=True なら10段目（空いている資金を 1306 で持つ形）。
    topn を渡すと11段目（買う日の前の月末に、時価総額の順位が上から topn 位までの銘柄だけ。でたらめも同じ範囲から）。keys で回す型を絞る。"""
    stage = "11段目（出来事の型を大型株だけで ＋ 空いている資金は TOPIX）" if topn else ("10段目（出来事の型 ＋ 空いている資金は TOPIX）" if fill else "5段目の出来事の型")
    log(f"{stage}を回します" + (f"（上位{topn}社）" if topn else "") + "（指標を作っています）…")
    if D is None:
        D = build(store)
    dates, UNI, me_idx, cal = D["dates"], D["UNI"], D["me_idx"], D["cal"]
    if topn:
        UNI = UNI & topn_mask(D, topn)
    T = len(cal)
    C = D["C"]
    j0 = next((j for j, t in enumerate(me_idx) if t >= 252), None)
    if topn and LONG and STAGE == "confirm":
        j0 = next((j for j, t in enumerate(me_idx) if t >= 252 and dates[j] >= CONFIRM_X_FROM), None)
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

    sim_fn = ev_sim_fill if fill else ev_sim

    def placebo_run(cands, pool_fn):
        v, liq, s_ = sim_fn(cands, D, E["divpay"], t0)
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
            pv, _l, ps_ = sim_fn(pl, D, E["divpay"], t0)
            rnd.append(half_stats(pv, ev))
            rnb.append(ps_["n_buy"])
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
              "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        return {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "nb": nb, "held": float(np.mean(s_["held"])) if s_["held"] else 0.0,
                "fill": float(np.mean(rnb)) / max(1, s_["n_buy"]), "n_cand": sum(len(x) for x in cands.values()),
                "etf": float(np.mean(s_["etf_share"])) if s_.get("etf_share") else 0.0}

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
        if (ONLY and key not in ONLY) or (keys is not None and key not in keys):
            continue
        r = {"lab": lab, "var": {}}
        for nm, vlab, mk, pf in variants:
            r["var"][nm] = dict(placebo_run(mk(), pf), vlab=vlab)
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda x: min(x["nb"].values()) < EV_MIN_BUYS or x["fill"] < RND_FILL_MIN)
        res[key] = r
        log(f"  {key} … {r['verdict']}")
    HH = ("前半", "後半")
    if topn:
        print(f"\n【日本株の手法・11段目（出来事の型を時価総額の上位{topn}社だけで ＋ 空いている資金は TOPIX）（wide_lab）】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan11_2026-11-05）")
    elif fill:
        print("\n【日本株の手法・10段目（出来事の型 ＋ 空いている資金は TOPIX）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan10_2026-11-05）")
    else:
        print("\n【日本株の手法・5段目（残りの型：出来事）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan5_2026-11-04）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）。決まりと判定は2回目と同じ")
    if fill:
        print(f"  形 … 出来事の銘柄は5段目と同じ（1銘柄は資産の20分の1・20銘柄まで）。それ以外の資金はいつも 1306 で持つ（片道{ETF_SLIP * 100:.1f}％・分配金と売った益に税）。"
              "出来事の銘柄を買うお金が足りなければ 1306 を売る。でたらめの比べも同じ形")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    for key, r in res.items():
        print(f"     {key:<6}{r['lab']} … {r['verdict']}")
        for nm, x in r["var"].items():
            rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
            dd = " ".join(f"{r['detail'][(nm, h)][0]:+.1f}/{r['detail'][(nm, h)][1]:+.1f}・{r['detail'][(nm, h)][2]}回" for h in HH)
            print(f"         {nm}（{x['vlab']}） {x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}（最大下落 {x['st']['通し']['mdd']:.0f}％）［{dd}］"
                  f" でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}・合図 {x['n_cand']:,}回・買い 前半 {x['nb']['前半']}回・後半 {x['nb']['後半']}回・"
                  f"平均の持ち数 {x['held']:.1f}銘柄・" + (f"1306 の割合 {x['etf'] * 100:.0f}％・" if fill else "") +
                  f"でたらめの買いの回数 {x['fill'] * 100:.0f}％・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％")
    print("\n  判定のまとめ（出来事" + ("＋TOPIX" if fill else "") + (f"・上位{topn}社" if topn else "") + "）")
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in res.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    if topn and LONG and STAGE == "confirm":
        print("  ※ 確かめる段階：「前半」＝見ていない期間（2014年3月〜2018年7月）。前半で候補の基準を満たせば「確立」（wide_lab_plan11 の4）")
    return res


def topn_mask(D: dict, n: int) -> np.ndarray:
    """月末ごとに、範囲の中で時価総額の上から n 位までの印。"""
    UNI, MC = D["UNI"], D["MCAP"]
    out = np.zeros(UNI.shape, dtype=bool)
    for j in range(UNI.shape[1]):
        idx = np.nonzero(UNI[:, j] & np.isfinite(MC[:, j]))[0]
        if len(idx):
            out[idx[np.argsort(-MC[idx, j], kind="stable")[:n]], j] = True
    return out


def lab11(store: dict, xs: dict) -> int:
    """11段目：DIVUP・E2C を、時価総額の上位500社・1,000社だけで（空いている資金は 1306）。確かめる段階では --cfg（例 DIVUP/TOP1000）で選んだものだけ。"""
    sel = {}
    for x in CFG:
        k, _, u = x.partition("/")
        sel.setdefault(int(u.upper().replace("TOP", "")), set()).add(k.upper())
    if LONG and STAGE == "confirm" and not sel:
        print("\n■ 確かめる段階では、--cfg で候補（例 DIVUP/TOP1000）を指定してください。")
        return 2
    D = build(store)
    allres = {}
    for n in (500, 1000):
        if sel and n not in sel:
            continue
        allres[n] = lab5_events(store, xs, fill=True, topn=n, D=D, keys=sel.get(n))
    print("\n【11段目のまとめ】")
    for n, res in allres.items():
        for k, r in res.items():
            print(f"     {k}/TOP{n} … {r['verdict']}")
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


def lab9(store: dict, extra: dict | None = None) -> int:
    log("9段目（多くの型をまとめて学ばせる）を回します（指標を作っています）…" if not extra else "19段目（すべての点数でまとめて学ばせる）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, F, PXM, DIV = D["dates"], D["UNI"], D["F"], D["PXM"], D["DIV"]
    M = len(dates)
    keys = [k for k, _ in LAB1]
    for nm, (fn, _lab) in (extra or {}).items():       # 19段目：第1部・第2部の毎月の点数をすべて足す
        if nm not in F:
            F[nm] = fn(D)
        if nm not in keys:
            keys.append(nm)
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
    print("\n【日本株の手法・9段目（多くの型をまとめて学ばせる）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan9_2026-11-04）" if not extra else
          f"\n【日本株の手法・19段目（すべての点数でまとめて学ばせる：{len(keys)}の型）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan19_2026-11-05）")
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


# ───────────────────────── 15段目：出来事の型の組み立て方の探索 ─────────────────────────
# 計画は wide_lab_plan15_2026-11-05（結果を見る前に決めたもの）。出来事9つ × 持つ日数（または買う日）6 × 範囲3 × 持つ数3。
# どの形も、空いている資金はいつも 1306 で持つ（10段目の道具）。でたらめは、同じ日に同じ数だけ、同じ範囲の中から選ぶ
EV15_HOLDS = (5, 10, 20, 40, 60, 120)          # 持つ日数（DIVUP・PEADP・PEADN・SPLIT・IPO・MALERT・LVH）
EV15_LEADS = (2, 3, 5, 10, 15, 20)             # E2C：決算発表予定日の何営業日前に買うか（予定日の前の営業日に売る）
EV15_RLEADS = (5, 10, 20, 30, 40, 60)          # RECS：権利付き最終日の何営業日前に買うか（権利付き最終日に売る）
EV15_UNIS = (("ALL", None), ("TOP1000", 1000), ("TOP500", 500))
EV15_NMAX = (10, 20, 40)
EV15_PICK = 3                                  # 確かめる段階に持っていく数


def judge1(x: dict, ST_TX: dict) -> tuple:
    """1つの形の判定（A・B の組ではない）。両方の期間で TOPIX +2pt・でたらめ +2pt かつ20回のうち18回 → 候補。"""
    det = {}
    ok_tx, ok_rn = True, True
    for h in ("前半", "後半"):
        a = x["st"][h]["ann"]
        rnd = [y[h]["ann"] for y in x["rnd"]]
        d_tx, d_rn = a - ST_TX[h]["ann"], a - float(np.mean(rnd))
        beat = int(sum(1 for y in rnd if a > y))
        det[h] = (d_tx, d_rn, beat)
        ok_tx &= d_tx >= PT_TX
        ok_rn &= (d_rn >= PT_RND) and (beat >= BEAT_MIN)
    if min(x["nb"].values()) < EV_MIN_BUYS or x["fill"] < RND_FILL_MIN:
        return "保留（データ不足）", det
    return ("候補" if (ok_tx and ok_rn) else ("範囲の中では効く" if ok_rn else "効かない")), det


def lab15(store: dict, xs: dict, combo: bool = False) -> int:
    """15段目：出来事の型の組み立て方の探索（約470通り）。確かめる段階では --cfg（例 DIVUP/TOP1000/60/20）で選んだ形だけ。
    combo=True なら19段目の出来事の組み合わせ（DIVE2C・DIVBP。全銘柄・20銘柄・A と B の組で判定）。"""
    log("15段目（出来事の型の組み立て方の探索）を回します（指標を作っています）…" if not combo else "19段目（出来事の組み合わせ）を回します（指標を作っています）…")
    sel = set(x.upper() for x in CFG)
    if LONG and STAGE == "confirm" and not sel:
        print("\n■ 確かめる段階では、--cfg で回す形（例 DIVUP/TOP1000/60/20）を指定してください。")
        return 2
    D = build(store)
    dates, UNI0, me_idx, cal = D["dates"], D["UNI"], D["me_idx"], D["cal"]
    T = len(cal)
    C, CF = D["C"], D["CF"]
    TPc = D["tp"]["c"].astype(float)
    j0 = next((j for j, t in enumerate(me_idx) if t >= 252), None)
    if LONG and STAGE == "confirm":
        need_x = any(k.split("/")[0] in ("DIVUP", "E2C", "DIVE2C", "DIVBP") for k in sel)
        if need_x:
            j0 = next((j for j, t in enumerate(me_idx) if t >= 252 and dates[j] >= CONFIRM_X_FROM), None)
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
    S = len(codes)
    # ── 出来事を取り出す（範囲は、あとで形ごとに絞る）
    evs = {}
    # DIVUP（5段目と同じ取り出し方）
    divs = {}
    for r in x_rows(xs, "div"):
        c = yb.norm_code(r.get("code") or "")
        if c in ci:
            divs.setdefault(c, []).append(r)
    ev_div = []
    for c, lst in divs.items():
        i = ci[c]
        b = store["bars"][c]
        lst.sort(key=lambda r: (str(r.get("pub") or ""), str(r.get("time") or "")))
        known, result = {}, {}
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
            if str(r.get("cs") or "0") not in ("0", "", "None"):
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
                t_a = int(np.searchsorted(cal, np.datetime64(str(r.get("pub"))[:10]), side="right"))
                ev_div.append((i, t_a, v / prev - 1.0))
    if ev_div:
        evs["DIVUP"] = ev_div
    # E2C（5段目と同じ。予定日は、買う日より前に公表され、その日にまだ変わっていないものだけ）
    hist = {}
    for r in x_rows(xs, "ed"):
        c = yb.norm_code(r.get("code") or "")
        pub, sch = str(r.get("pub") or "")[:10], str(r.get("sch") or "")[:10]
        if c in ci and len(pub) == 10:
            hist.setdefault((ci[c], str(r.get("fq") or ""), str(r.get("fye") or "")), []).append((pub, sch))
    sched_recs = []
    SCHP = np.zeros((S, T + 1), dtype=bool)
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
    # 決算発表の反応（2回目の E1 と同じ）
    react = []
    for i, t_prev, t_a in E["annc"]:
        p0 = C[i, t_prev]
        if not (p0 == p0 and p0 > 0) or not traded[i, t_a]:
            continue
        react.append((i, t_prev, t_a, float(C[i, t_a] / p0) - float(TPc[t_a] / TPc[t_prev])))
    ANNP = np.zeros((S, T + 1), dtype=bool)
    for i, t_prev, _t_a in E["annc"]:
        ANNP[i, t_prev] = True
    # 権利付き最終日（2回目の E3 と同じ）
    RECM = np.zeros((S, T + 1), dtype=bool)
    DPSD = np.full((S, T), np.nan, dtype=np.float32)
    for i, t_c, dps, t_disc in sorted(E["recs"], key=lambda x: (x[0], x[3])):
        if 0 <= t_c < T:
            RECM[i, t_c] = True
        if t_disc + 1 < T:
            DPSD[i, t_disc + 1:] = dps
    for t_c, lst in E["divpay"].items():
        for i, _a in lst:
            if 0 <= t_c < T:
                RECM[i, t_c] = True
    # 上場直後：上場から21営業日たってから最初の月末（その月末に範囲に入っていれば買う）
    SEG = D["SEG"]
    ev_ipo = []
    seen_ipo = set()
    for t in me_idx:
        s_ = SEG[:, t]
        ok = np.nonzero((s_ < T) & (s_ >= 21) & ((t - s_) >= 21))[0]
        for i in ok:
            k = (int(i), int(s_[i]))
            if k not in seen_ipo:
                seen_ipo.add(k)
                ev_ipo.append((int(i), int(t)))
    # 日々公表の対象に入った最初の日・大量保有（13段目と同じ）
    calD = cal.astype("datetime64[D]")

    def t_after(dstr):
        return int(np.searchsorted(calD, np.datetime64(str(dstr)[:10]), side="right"))
    ev_mal = []
    mal = x_rows(xs, "mal")
    if mal:
        seen = {}
        for r in mal:
            c = yb.norm_code(r["code"] or "")
            if c in ci and len(str(r["pub"] or "")) >= 10:
                seen.setdefault(ci[c], set()).add(t_after(r["pub"]))
        for i, ts in seen.items():
            prev = -10 ** 9
            for t in sorted(ts):
                if t - prev > 60:
                    ev_mal.append((i, t))
                prev = t
    ev_lvh = []
    for r in x_rows(xs, "lvh"):
        c = yb.norm_code(r["code"] or "")
        if c in ci and str(r["typ"]) == "1" and str(r["doc"]) == "350" and len(str(r["sub"] or "")) >= 10:
            ev_lvh.append((ci[c], t_after(r["sub"]), r["ratio"] or 0.0))
    log(f"  出来事 … 増配の予想 {len(ev_div):,}・決算発表予定日 {len(sched_recs):,}・決算の反応 {len(react):,}・株式分割 {len(E['splits']):,}・"
        f"権利確定日の候補 {len(E['recs']):,}・上場 {len(ev_ipo):,}・日々公表の対象入り {len(ev_mal):,}・大量保有 {len(ev_lvh):,}")
    LAB = {"DIVUP": "増配の予想の発表の次の営業日に買う", "E2C": "決算発表予定日の前に買う（予定日の前の営業日に売る）",
           "PEADP": "決算発表の反応が TOPIX より +5％以上の日に買う", "PEADN": "決算発表の反応が TOPIX より −5％以下の日に買う（逆張り）",
           "SPLIT": "株式分割の効力発生日に買う", "RECS": f"権利付き最終日の前に買う（配当利回り{DY_TH * 100:.0f}％以上・権利付き最終日に売る）",
           "IPO": "上場から21営業日たってから最初の月末に買う", "MALERT": "日々公表の対象に入った最初の日の次の営業日に買う",
           "LVH": "大量保有（新しく5％を超えた報告）の次の営業日に買う"}
    have = {"DIVUP": bool(ev_div), "E2C": bool(sched_recs), "PEADP": bool(react), "PEADN": bool(react), "SPLIT": bool(E["splits"]),
            "RECS": bool(E["recs"]), "IPO": bool(ev_ipo), "MALERT": bool(ev_mal), "LVH": bool(ev_lvh)}
    unis, nmaxs = EV15_UNIS, EV15_NMAX
    if combo:
        LAB = {"DIVE2C": "増配の予想の発表 ＋ 決算発表予定日の前（両方の合図。同じ日は増配を先に）",
               "DIVBP": "増配の予想の発表のうち、純資産に対して安い上位半分（前の月末の範囲の中）だけ"}
        have = {"DIVE2C": bool(ev_div) and bool(sched_recs), "DIVBP": bool(ev_div)}
        unis, nmaxs = (("ALL", None),), (NMAX,)
    BPM = D["F"]["BP"]
    sim_fn = ev_sim_fill
    res = {}
    t_start = time.monotonic()
    for un, topn in unis:
        UNI = UNI0 & topn_mask(D, topn) if topn else UNI0

        def elig(i, t):
            return 0 <= t < T and jm[t] >= 0 and bool(UNI[i, jm[t]]) and bool(traded[i, t])

        def pool_all(t):
            j = jm[t]
            return np.nonzero(UNI[:, j] & traded[:, t])[0] if j >= 0 else np.zeros(0, dtype=int)
        ann_pool = {}
        for i, t_prev, t_a, _re in react:
            if t_a >= t0 and elig(i, t_a):
                ann_pool.setdefault(t_a, []).append(i)

        def pool_ann(t):
            return np.array(ann_pool.get(t, []), dtype=int)

        def pool_nosched(L):
            def f(t):
                el = pool_all(t)
                return el[~SCHP[el, t:t + L + 2].any(axis=1)] if len(el) else el
            return f

        def pool_rec(L):
            def f(t):
                el = pool_all(t)
                if not len(el):
                    return el
                with np.errstate(invalid="ignore", divide="ignore"):
                    y = DPSD[el, t] / CF[el, t]
                return el[(y >= DY_TH) & ~RECM[el, t:t + L + 1].any(axis=1)]
            return f

        def mk(key, p):
            out = {}
            if key == "DIVUP":
                for i, t_a, chg in ev_div:
                    if t_a >= t0 and elig(i, t_a):
                        out.setdefault(t_a, []).append((chg, i, t_a + p))
                return out, pool_all
            if key == "E2C":
                rg = np.random.default_rng(1)
                seen = set()
                for i, pub, t_s, nxt in sorted(sched_recs, key=lambda x: (x[2], x[0], x[1])):
                    t_in = t_s - 1 - p
                    if t_in < t0 or (i, t_s) in seen or not elig(i, t_in):
                        continue
                    day = str(cal[t_in])[:10]
                    if not (pub < day <= nxt):
                        continue
                    seen.add((i, t_s))
                    out.setdefault(t_in, []).append((float(rg.random()), i, t_s - 1))
                return out, pool_nosched(p)
            if key in ("PEADP", "PEADN"):
                for i, _tp, t_a, re in react:
                    if t_a >= t0 and elig(i, t_a) and ((re >= REACT_TH) if key == "PEADP" else (re <= -REACT_TH)):
                        out.setdefault(t_a, []).append((abs(re), i, t_a + p))
                return out, pool_ann
            if key == "SPLIT":
                for i, t_s in E["splits"]:
                    if t_s >= t0 and elig(i, t_s):
                        out.setdefault(t_s, []).append((0.0, i, t_s + p))
                return out, pool_all
            if key == "RECS":
                for i, t_c, dps, t_disc in E["recs"]:
                    t_in = t_c - p
                    if t_in < t0 or t_in <= t_disc or t_c >= T or not elig(i, t_in):
                        continue
                    y = dps / float(CF[i, t_in])
                    if y >= DY_TH:
                        out.setdefault(t_in, []).append((y, i, t_c))
                return out, pool_rec(p)
            if key == "IPO":
                for i, t in ev_ipo:
                    if t >= t0 and elig(i, t):
                        out.setdefault(t, []).append((0.0, i, t + p))
                return out, pool_all
            if key == "MALERT":
                rg = np.random.default_rng(13)
                for i, t in ev_mal:
                    if t >= t0 and elig(i, t):
                        out.setdefault(t, []).append((float(rg.random()), i, t + p))
                return out, pool_all
            if key == "LVH":
                for i, t, ratio in ev_lvh:
                    if t >= t0 and elig(i, t):
                        out.setdefault(t, []).append((ratio, i, t + p))
                return out, pool_all
            if key == "DIVE2C":                     # p＝"A"（増配 60日・決算予定日の10日前）／"B"（20日・5日前）
                hd, ld = (60, 10) if p == "A" else (20, 5)
                c1, _pf = mk("DIVUP", hd)
                c2, _pf = mk("E2C", ld)
                for t, cl in c1.items():
                    out.setdefault(t, []).extend((1.0 + pr, i, to) for pr, i, to in cl)
                for t, cl in c2.items():
                    out.setdefault(t, []).extend(cl)
                return out, pool_all
            if key == "DIVBP":                      # p＝"A"（60日持つ）／"B"（20日持つ）
                hd = 60 if p == "A" else 20
                cheap = {}

                def is_cheap(i, t):
                    j = jm[t]
                    if j not in cheap:
                        ok = UNI[:, j] & np.isfinite(BPM[:, j])
                        thr = float(np.median(BPM[ok, j])) if ok.sum() >= 30 else np.inf
                        cheap[j] = ok & (BPM[:, j] >= thr)
                    return bool(cheap[j][i])
                for i, t_a, chg in ev_div:
                    if t_a >= t0 and elig(i, t_a) and is_cheap(i, t_a):
                        out.setdefault(t_a, []).append((chg, i, t_a + hd))

                def pool_cheap(t):
                    el = pool_all(t)
                    return np.array([i for i in el if is_cheap(int(i), t)], dtype=int)
                return out, pool_cheap
            raise KeyError(key)
        for key in LAB:
            if not have[key] or (ONLY and key not in ONLY):
                continue
            params = ("A", "B") if combo else (EV15_LEADS if key == "E2C" else (EV15_RLEADS if key == "RECS" else EV15_HOLDS))
            for p in params:
                todo_n = [n for n in nmaxs if not sel or f"{key}/{un}/{p}/{n}" in sel]
                if not todo_n:
                    continue
                cands, pool_fn = mk(key, p)
                # でたらめの候補は、持つ数によらず同じものを使う（同じ日・同じ数・同じ範囲）
                pls = []
                for k in range(N_RAND):
                    rg = np.random.default_rng(15000 + k)
                    pl = {}
                    for t, cl in cands.items():
                        pool = pool_fn(t)
                        if len(pool) == 0:
                            continue
                        pick = rg.choice(pool, size=min(len(cl), len(pool)), replace=False)
                        pl[t] = [(float(rg.random()), int(i), t_out) for i, (_p, _i, t_out) in zip(pick, cl)]
                    pls.append(pl)
                for n in todo_n:
                    v, liq, s_ = sim_fn(cands, D, E["divpay"], t0, nmax=n)
                    rnd, rnb = [], []
                    for pl in pls:
                        pv, _l, ps_ = sim_fn(pl, D, E["divpay"], t0, nmax=n)
                        rnd.append(half_stats(pv, ev))
                        rnb.append(ps_["n_buy"])
                    nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
                          "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
                    x = {"st": half_stats(v, ev), "liq": liq, "rnd": rnd, "nb": nb, "fill": float(np.mean(rnb)) / max(1, s_["n_buy"]),
                         "held": float(np.mean(s_["held"])) if s_["held"] else 0.0, "n_cand": sum(len(c_) for c_ in cands.values()),
                         "etf": float(np.mean(s_["etf_share"])) if s_.get("etf_share") else 0.0, "skip": s_.get("skip_up", 0)}
                    x["verdict"], x["detail"] = judge1(x, ST_TX)
                    res[f"{key}/{un}/{p}/{n}"] = x
                    if sel and os.environ.get("WIDE_PATHS") == "1" and j0 + 12 < len(me_idx):
                        # 経路のばらつきの確かめ（plan15 の7）：始める月を1か月ずつずらした12通りで、同じ期間（12か月目の月末から最後まで）の年率を比べる
                        jc = j0 + 11
                        yc = max((dates[-1] - dates[jc]).days / 365.25, 0.5)
                        anns = []
                        for k in range(12):
                            t0k = me_idx[j0 + k]
                            ck = {t: cl for t, cl in cands.items() if t >= t0k}
                            vk, _l, _s = sim_fn(ck, D, E["divpay"], t0k, nmax=n)
                            if len(vk) > jc - (j0 + k):
                                anns.append(((vk[-1] / vk[jc - (j0 + k)]) ** (1 / yc) - 1) * 100)
                        txc = ((TXv[-1] / TXv[jc - j0]) ** (1 / yc) - 1) * 100
                        x["paths"] = (anns, txc, dates[jc])
            log(f"  {key}/{un} … 済（ここまで {len(res)}通り・{(time.monotonic() - t_start) / 60:.0f}分）")
    if combo:
        HH = ("前半", "後半")
        print("\n【日本株の手法・19段目（出来事の組み合わせ）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan19_2026-11-05）")
        stage_note()
        print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月）。形は15段目と同じ（20銘柄・空いている資金は 1306）。A と B の両方で基準を満たせば候補")
        print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
        if D.get("LIMU") is not None:
            print("  ストップ高・ストップ安 … ストップ高で引けた日は終値で買えない・ストップ安で引けた日は売れない（次の日に売る）。でたらめも同じ（15段目の6と同じ決まり）")
        out_v = {}
        for key in LAB:
            xa, xb = res.get(f"{key}/ALL/A/{NMAX}"), res.get(f"{key}/ALL/B/{NMAX}")
            if not xa or not xb:
                continue
            hold = any(x["verdict"].startswith("保留") for x in (xa, xb))
            ok_tx = all(x["detail"][h][0] >= PT_TX for x in (xa, xb) for h in HH)
            ok_rn = all(x["detail"][h][1] >= PT_RND and x["detail"][h][2] >= BEAT_MIN for x in (xa, xb) for h in HH)
            v = "保留（データ不足）" if hold else ("候補" if (ok_tx and ok_rn) else ("範囲の中では効く" if ok_rn else "効かない"))
            out_v[key] = v
            print(f"     {key:<7}{LAB[key]} … {v}")
            for nm, x in (("A", xa), ("B", xb)):
                dd = " ".join(f"{x['detail'][h][0]:+.1f}/{x['detail'][h][1]:+.1f}・{x['detail'][h][2]}回" for h in HH)
                rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
                print(f"         {nm} {x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}（最大下落 {x['st']['通し']['mdd']:.0f}％）［{dd}］ でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}"
                      f"・合図 {x['n_cand']:,}回・買い {x['nb']['前半']}/{x['nb']['後半']}・持ち {x['held']:.1f}・1306 {x['etf'] * 100:.0f}％"
                      + (f"・ストップ高で買えず {x['skip']}" if D.get("LIMU") is not None else ""))
                if x.get("paths"):
                    anns, txc, dc = x["paths"]
                    a_ = np.array(anns)
                    print(f"           経路のばらつき（判定には使わない）{dc.date()}〜 … 最小 {a_.min():.1f}・中央 {np.median(a_):.1f}・最大 {a_.max():.1f}"
                          f"（TOPIX {txc:.1f}。上回った {int((a_ > txc).sum())}/{len(a_)}・+2pt 以上 {int((a_ >= txc + PT_TX).sum())}/{len(a_)}）")
        for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
            ks = [k for k, vv in out_v.items() if vv == v]
            print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
        return 0
    # ── 報告
    HH = ("前半", "後半")
    print("\n【日本株の手法・15段目（出来事の型の組み立て方の探索）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan15_2026-11-05）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  形 … 記号は「出来事/範囲/持つ日数（E2C・RECS は何日前に買うか）/持つ数」。1銘柄は資産の（持つ数）分の1。空いている資金はいつも 1306（片道{ETF_SLIP * 100:.2f}％）。"
          "でたらめは同じ日に同じ数だけ、同じ範囲から（20回）。判定は両方の期間で TOPIX +2pt・でたらめ +2pt かつ18回以上")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    if D.get("LIMU") is not None:
        print("  ストップ高・ストップ安 … ストップ高で引けた日は終値で買えない・ストップ安で引けた日は売れない（次の日に売る）。でたらめも同じ（6段目と同じ決まり）")
    for key in LAB:
        ks = [k for k in res if k.startswith(key + "/")]
        if not ks:
            print(f"     {key:<7}{LAB[key]} … データがないので回していません")
            continue
        cnt = {}
        for k in ks:
            cnt[res[k]["verdict"]] = cnt.get(res[k]["verdict"], 0) + 1
        print(f"     {key:<7}{LAB[key]} … " + "・".join(f"{v} {n}" for v, n in cnt.items()))
    print("\n  1. すべての形（年率％「前半 ／ 後半」。［TOPIX との差／でたらめとの差・回数］）")
    for k, x in res.items():
        dd = " ".join(f"{x['detail'][h][0]:+.1f}/{x['detail'][h][1]:+.1f}・{x['detail'][h][2]}" for h in HH)
        rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
        print(f"     {k:<24}{x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}［{dd}］ でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}"
              f"・下落 {x['st']['通し']['mdd']:.0f}％・買い {x['nb']['前半']}/{x['nb']['後半']}・持ち {x['held']:.1f}・1306 {x['etf'] * 100:.0f}％"
              + (f"・ストップ高で買えず {x['skip']}" if D.get("LIMU") is not None else "") + f" … {x['verdict']}")
    pth = {k: x["paths"] for k, x in res.items() if x.get("paths")}
    if pth:
        print("\n  経路のばらつき（plan15 の7。判定には使わない）：始める月を1か月ずつずらした12通りの、同じ期間の年率％")
        for k, (anns, txc, dc) in pth.items():
            a_ = np.array(anns)
            print(f"     {k:<24}{dc.date()}〜 … 最小 {a_.min():.1f}・中央 {np.median(a_):.1f}・最大 {a_.max():.1f}（TOPIX {txc:.1f}。TOPIX を上回った {int((a_ > txc).sum())}/{len(a_)}・+2pt 以上 {int((a_ >= txc + PT_TX).sum())}/{len(a_)}）"
                  f" ／ 12通り：{'・'.join(f'{v:.1f}' for v in anns)}")
    cand = [k for k, x in res.items() if x["verdict"] == "候補"]
    near = [k for k, x in res.items() if x["verdict"] == "範囲の中では効く"]
    print("\n  2. 判定のまとめ")
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        print(f"     {v} … {sum(1 for x in res.values() if x['verdict'] == v)}通り")

    def worst(k):
        d = res[k]["detail"]
        return min(min(d[h][0] - PT_TX, d[h][1] - PT_RND) for h in HH)

    def tx_min(k):
        d = res[k]["detail"]
        return min(d[h][0] for h in HH)
    if not (LONG and STAGE == "confirm"):
        ok_conf = [k for k in (cand or near) if k.split("/")[0] != "LVH"]       # 大量保有は2021年7月からで、見ていない期間で確かめられない
        pick = sorted(ok_conf, key=lambda k: -(worst(k) if cand else tx_min(k)))[:EV15_PICK]
        print(f"  確かめる段階に持っていく形（{'候補' if cand else '準候補：範囲の中では効く形のうち、TOPIX との差の小さいほうが大きい順'}・{EV15_PICK}つまで） … "
              + ("・".join(pick) if pick else "なし"))
    else:
        print("  ※ 確かめる段階：「前半」＝見ていない期間。前半で候補の基準（TOPIX +2pt・でたらめ +2pt かつ18回以上）を満たせば「確立」")
    print(f"  （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")
    return 0



# ───────────────────────── 16〜19段目（第2部。計画は wide_program2_2026-11-05 と各段の plan） ─────────────────────────
def msh_score(xs: dict):
    """13段目の MSH と同じ（大株主の信託口の比率の合計が、前の報告より増えた幅）。"""
    by = {}
    for r in x_rows(xs, "msh"):
        c = yb.norm_code(r["code"] or "")
        if len(str(r["sub"] or "")) >= 10:
            by.setdefault(c, []).append((str(r["sub"])[:10], r["trust"]))
    for c in by:
        by[c].sort()

    def fn(DD):
        out = np.full((len(DD["codes"]), len(DD["dates"])), np.nan)
        me = [d.strftime("%Y-%m-%d") for d in DD["dates"]]
        for i, c in enumerate(DD["codes"]):
            lst = by.get(c)
            if not lst or len(lst) < 2:
                continue
            subs = [x[0] for x in lst]
            for j, m in enumerate(me):
                k = int(np.searchsorted(subs, m, side="left")) - 1
                if k >= 1:
                    out[i, j] = lst[k][1] - lst[k - 1][1]
        return out + np.random.default_rng(1305).random(len(DD["codes"]))[:, None] * 1e-9
    return fn


def xsh_score(xs: dict):
    """13段目の XSH と同じ（持ち合い株を減らして売った額 ÷ 時価総額。比べ始めるのは2021年7月末から）。"""
    byx = {}
    for r in x_rows(xs, "xsh"):
        c = yb.norm_code(r["code"] or "")
        if len(str(r["sub"] or "")) >= 10:
            byx.setdefault(c, []).append((str(r["sub"])[:10], r))
    for c in byx:
        byx[c].sort(key=lambda x: x[0])

    def fn(DD):
        out = np.full((len(DD["codes"]), len(DD["dates"])), np.nan)
        me = [d.strftime("%Y-%m-%d") for d in DD["dates"]]
        MC = DD["MCAP"]
        for i, c in enumerate(DD["codes"]):
            lst = byx.get(c)
            if not lst:
                continue
            subs = [x[0] for x in lst]
            for j, m in enumerate(me):
                k = int(np.searchsorted(subs, m, side="left")) - 1
                if k < 0 or not (MC[i, j] > 0):
                    continue
                if (pd.Timestamp(m) - pd.Timestamp(subs[k])).days > 450:
                    continue
                r = lst[k][1]
                dec = r["dec"] if r["dec"] is not None else 0.0
                out[i, j] = dec / MC[i, j]
        START_T["XSH"] = int(np.searchsorted(DD["cal"].astype("datetime64[D]"), np.datetime64("2021-07-30"), side="left"))
        return out + np.random.default_rng(1306).random(len(DD["codes"]))[:, None] * 1e-12
    return fn


def monthly_extra_all(store: dict, xs: dict, bd: dict | None) -> dict:
    """第1部の毎月の点数の型をすべて（1回目24・VQM・4回目2・5段目5・12段目3・13段目2）。16段目と19段目で使う。"""
    ex = {}
    for k, lab in LAB1:
        ex[k] = ((lambda D, k=k: D["F"][k]), lab)
    ex["VQM"] = ((lambda D: combo_score(D["F"], list(VQM), D["UNI"])), "割安・質・勢い（BP・ROE・MOM12 の順位の平均）")
    if bd and bd.get("complete"):
        ex["SHORT"] = (bd_score(bd, "SHORT"), "空売りの少ない株")
        ex["MBUY"] = (bd_score(bd, "MBUY"), "信用の新規買いの少ない株")
    ex["ACCR"] = (accr_score(store), "利益の質")
    ex["SECT"] = (sect_score, "業種の勢い")
    if not xs.get("ssr", {}).get("bad"):
        ex["SSR"] = (ssr_score(xs), "大口の空売りの報告がない株")
    if not xs.get("mi", {}).get("bad"):
        ex["MLONG"] = (mi_score(store, xs, "MLONG"), "信用買い残の軽さ")
        ex["MBAL"] = (mi_score(store, xs, "MBAL"), "信用の売り残の多さ")
    if x_rows(xs, "fsd2"):
        ex.update(lab12_scores(store, xs))
    if x_rows(xs, "msh"):
        ex["MSH"] = (msh_score(xs), "信託口の比率が増えた幅")
    if x_rows(xs, "xsh"):
        ex["XSH"] = (xsh_score(xs), "持ち合い株を売った額 ÷ 時価総額")
    return ex


W16 = (("CAP", "時価総額で加重"), ("VOL", "値動きの小ささで加重（過去1年の日々の値動きの逆数）"))
U16 = (("ALL", None, "全銘柄"), ("TOP500", 500, "時価総額の上位500社"))


def w16(kind: str):
    def fn(D):
        if kind == "CAP":
            return D["MCAP"].astype(float)
        lv = D["F"]["LOWVOL"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(lv) & (lv < 0), 1.0 / (-lv), np.nan)
    return fn


def lab16(store: dict, xs: dict, bd: dict | None) -> int:
    """16段目：加重の仕方（計画は wide_lab_plan16_2026-11-05）。毎月の点数の型37 × 加重2 × 範囲2。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE
    ex = monthly_extra_all(store, xs, bd)
    log(f"16段目（加重の仕方）を回します：型 {len(ex)}つ（{'・'.join(ex)}）")
    sel = {x.upper() for x in CFG}
    allres = {}
    for un, n, ulab in U16:
        for wk, wlab in W16:
            if sel and not any(c.startswith(f"{wk}/{un}") for c in sel):
                continue
            LAB_TITLE, LAB_PLAN = f"16段目（{wlab}・{ulab}）", "wide_lab_plan16_2026-11-05"
            LAB_NOTE = f"16段目：新しく買う銘柄の額を、上位20銘柄の中の重み（{wlab}。1銘柄は資産の{W_CAP * 100:.0f}％まで）で決める。範囲は{ulab}。でたらめも同じ加重・同じ範囲"
            exs = ex
            if sel:
                ks = {c.split("/")[2] for c in sel if c.startswith(f"{wk}/{un}/")}
                exs = {k: v for k, v in ex.items() if k in ks}
            lab1(store, extra=exs, uni_fn=(None if n is None else (lambda D, n=n: topn_mask(D, n))), w_fn=w16(wk))
            for k, r in (LAST_LAB1 or {}).items():
                allres[f"{wk}/{un}/{k}"] = r
    LAB_NOTE = ""
    print("\n【16段目のまとめ】（記号は「加重/範囲/型」）")
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in allres.items() if r["verdict"] == v]
        print(f"     {v} … {len(ks)}通り" + (f"：{'・'.join(ks)}" if ks and v in ("候補", "範囲の中では効く") else ""))

    def worst(r):
        return min(min(r["detail"][(nm, h)][0] - PT_TX, r["detail"][(nm, h)][1] - PT_RND) for nm, _ in KEEPS for h in ("前半", "後半"))

    def tx_min(r):
        return min(r["detail"][(nm, h)][0] for nm, _ in KEEPS for h in ("前半", "後半"))
    cand = [k for k, r in allres.items() if r["verdict"] == "候補" and k.split("/")[2] != "XSH"]
    near = [k for k, r in allres.items() if r["verdict"] == "範囲の中では効く" and k.split("/")[2] != "XSH"]
    if not (LONG and STAGE == "confirm"):
        pick = sorted(cand or near, key=lambda k: -(worst(allres[k]) if cand else tx_min(allres[k])))[:3]
        print("  確かめる段階に持っていく形（" + ("候補" if cand else "準候補：範囲の中では効く形のうち、TOPIX との差のいちばん小さいものが大きい順") + "・3つまで。XSH は見ていない期間のデータがないので除く） … "
              + ("・".join(pick) if pick else "なし"))
    return 0


# ── 18段目：毎月の点数の残り
def lab18_scores(store: dict, xs: dict) -> dict:
    """18段目の型（計画は wide_lab_plan18_2026-11-05）。どれも、その月末より前に開示された値だけを使う。"""
    tl12 = None
    if x_rows(xs, "fsd2"):
        tl12, _st = fsd_timeline(store, xs)
    cache = {}

    def mats(D):
        if "m" in cache:
            return cache["m"]
        codes, dates = D["codes"], D["dates"]
        S_, M_ = len(codes), len(dates)
        out = {k: np.full((S_, M_), np.nan) for k in ("DROE", "OPM", "PAYOUT", "DIVG", "GPM", "DGPM", "LEV")}
        me = [d.strftime("%Y-%m-%d") for d in dates]
        for i, c in enumerate(codes):
            fin = D["sector"].get(c, "") in yb.FIN_SECTORS
            b = store["bars"].get(c)
            if b is None:
                continue
            tl = fin_timeline(store["fins"].get(c))
            dd = [str(r["DiscDate"])[:10] for r in tl]
            for j, m in enumerate(me):
                k = int(np.searchsorted(dd, m, side="left")) - 1
                if k < 0:
                    continue
                fyd = {}
                for r in tl[:k + 1]:
                    if is_fy(r):
                        fyd[str(r.get("CurFYEn") or "")[:10]] = r
                fy = [fyd[e] for e in sorted(fyd) if e]
                if not fy:
                    continue
                r = fy[-1]
                npv, eq, sales, op = num(r.get("NP")), num(r.get("Eq")), num(r.get("Sales")), num(r.get("OP"))
                if len(fy) >= 2:
                    p = fy[-2]
                    np0, eq0 = num(p.get("NP")), num(p.get("Eq"))
                    if npv is not None and eq and eq > 0 and np0 is not None and eq0 and eq0 > 0:
                        out["DROE"][i, j] = npv / eq - np0 / eq0
                if not fin and op is not None and sales and sales > 0:
                    out["OPM"][i, j] = op / sales
                ann, eps = num(r.get("DivAnn")), num(r.get("EPS"))
                if eps is None and npv is not None and num(r.get("ShOutFY")):          # 1株利益がなければ、純利益 ÷ 株数（自己株を除く）
                    shn = num(r.get("ShOutFY")) - (num(r.get("TrShFY")) or 0.0)
                    eps = npv / shn if shn > 0 else None
                if ann is not None and eps and eps > 0:
                    out["PAYOUT"][i, j] = -(ann / eps)
                # 増配の続いた年数（分割のあとの基準にそろえて、前の年より増えた年が続いた数）
                dps = []
                for q in fy:
                    a = num(q.get("DivAnn"))
                    dps.append(None if a is None else a * fac_after(b["fd"], b["ff"], str(q["DiscDate"])[:10]))
                if dps[-1] is not None:
                    n_up = 0
                    for x0, x1 in zip(reversed(dps[:-1]), reversed(dps[1:])):
                        if x0 is None or x1 is None or not (x1 > x0 * 1.0001):
                            break
                        n_up += 1
                    out["DIVG"][i, j] = n_up
            if tl12 is not None and not fin and c in tl12:
                rows = tl12[c]
                av = [x["avail"] for x in rows]
                for j, m in enumerate(me):
                    k = int(np.searchsorted(av, m, side="left")) - 1
                    if k < 0:
                        continue
                    r = rows[k]
                    if r["gp"] is not None and r["sales"] and r["sales"] > 0:
                        out["GPM"][i, j] = r["gp"] / r["sales"]
                        if k >= 1:
                            q = rows[k - 1]
                            if q["gp"] is not None and q["sales"] and q["sales"] > 0:
                                out["DGPM"][i, j] = r["gp"] / r["sales"] - q["gp"] / q["sales"]
                    ta = r["ta"] if r["ta"] else r.get("ta_d")
                    if r["ltd"] is not None and ta and ta > 0:
                        out["LEV"][i, j] = -(r["ltd"] / ta)
        cache["m"] = out
        return out
    noise_c = {}

    def noise(D, seed):
        if seed not in noise_c:
            noise_c[seed] = np.random.default_rng(seed).random(len(D["codes"]))[:, None] * 1e-6
        return noise_c[seed]

    def mk(k, seed=None):
        return lambda D: mats(D)[k] + (noise(D, seed) if seed else 0.0)

    def lowpx(D):
        C = D["C"]
        return np.stack([-C[:, t].astype(float) for t in D["me_idx"]], axis=1)

    def shy(D):
        F = D["F"]
        return F["DY"] + np.nan_to_num(F["SHR"], nan=0.0)

    def ind(key):
        def fn(D):
            X, UNI = D["F"][key], D["UNI"]
            sec = np.array([D["sector"].get(c, "") for c in D["codes"]])
            out = np.full(X.shape, np.nan)
            for j in range(X.shape[1]):
                ok = UNI[:, j] & np.isfinite(X[:, j])
                for sname in set(sec[ok]):
                    m = ok & (sec == sname)
                    if m.sum() < 5:
                        continue
                    out[m, j] = pd.Series(X[m, j]).rank(pct=True).to_numpy()
            return out + noise(D, 1801)
        return fn

    def seas(D):
        PXM, DIV, dates, me_idx = D["PXM"], D["DIV"], D["dates"], D["me_idx"]
        M_ = len(dates)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.full(PXM.shape, np.nan)
            R[:, :-1] = (PXM[:, 1:] + DIV[:, 1:]) / PXM[:, :-1] - 1.0     # 月末 j から次の月末までの成績
        out = np.full(PXM.shape, np.nan)
        for j in range(M_):
            past = [j + 1 - 12 * k - 1 for k in range(1, 6) if j + 1 - 12 * k - 1 >= 0]   # 次の月と同じ月の、過去の年の成績（月末 j−12k から）
            if len(past) < 3:
                continue
            with np.errstate(invalid="ignore"):
                Z = R[:, past]
                cnt = np.isfinite(Z).sum(axis=1)
                mval = np.nanmean(np.where(np.isfinite(Z), Z, np.nan), axis=1)
            out[:, j] = np.where(cnt >= 3, mval, np.nan)
        j_ok = next((j for j in range(M_) if np.isfinite(out[:, j]).any()), M_ - 1)
        START_T["SEAS"] = int(me_idx[j_ok])
        return out
    optu = x_rows(xs, "optu")

    def ivlow(D):
        cal, me_idx, codes = D["cal"], D["me_idx"], D["codes"]
        calS = [str(x)[:10] for x in cal.astype("datetime64[D]")]
        ti = {d: k for k, d in enumerate(calS)}
        ci = {c: i for i, c in enumerate(codes)}
        IV = np.full((len(codes), len(cal)), np.nan)
        for r in optu:
            c = yb.norm_code(str(r["und"] or ""))
            k = ti.get(str(r["date"])[:10])
            if c in ci and k is not None and r["iv"] is not None:
                IV[ci[c], k] = r["iv"]
        out = np.full((len(codes), len(me_idx)), np.nan)
        for j, t in enumerate(me_idx):
            w = IV[:, max(0, t - 19):t + 1]
            cnt = np.isfinite(w).sum(axis=1)
            with np.errstate(invalid="ignore"):
                m = np.nanmean(w, axis=1)
            out[:, j] = np.where(cnt >= 15, -m, np.nan)
        n_ok = [int((np.isfinite(out[:, j]) & D["UNI"][:, j]).sum()) for j in range(len(me_idx))]
        log(f"  個別株オプション … 値動きの予想（IV）がある銘柄（範囲の中） 月の中央値 {int(np.median(n_ok)) if n_ok else 0}社（多い月 {max(n_ok) if n_ok else 0}社）")
        return out
    extra_iv = {"IVLOW": (ivlow, "個別株オプションの値動きの予想（IV。いちばん近い限月の中央値の20日平均）の低さ")} if optu else {}
    return {**extra_iv, "SEAS": (seas, "同じ月の過去の成績（過去3〜5年の、次の月と同じ月の成績の平均）"),
            "LOWPX": (lowpx, "株価の低さ（月末の株価の安い順）"),
            "DROE": (mk("DROE", 1802), "ROE の改善（直近の通期 − 前の通期）"),
            "OPM": (mk("OPM"), "営業利益率の高さ（金融を除く）"),
            "PAYOUT": (mk("PAYOUT", 1803), "配当性向の低さ（年間配当 ÷ 1株利益。赤字は除く）"),
            "DIVG": (mk("DIVG", 1804), "増配の続いた年数（多い順）"),
            "GPM": (mk("GPM"), "粗利益率の高さ（売上総利益 ÷ 売上。金融を除く）"),
            "DGPM": (mk("DGPM", 1805), "粗利益率の改善（金融を除く）"),
            "LEV": (mk("LEV", 1806), "長期の借入の少なさ（長期の借入と社債 ÷ 総資産。金融を除く）"),
            "SHY": (shy, "株主還元利回り（配当利回り ＋ 1年の株数の減り）"),
            "BPIND": (ind("BP"), "業種の中での純資産に対する安さ（業種の中の順位）"),
            "EPIND": (ind("EP"), "業種の中での利益に対する安さ（業種の中の順位）")}


# ── 17段目：暦と短期の残り
def lab17(store: dict) -> int:
    """17段目（計画は wide_lab_plan17_2026-11-05）。TOPIX を持つ時期の型5つと、個別株の短期の型2つ。夜だけ・昼だけは参考（費用の前の値も出す）。"""
    log("17段目（暦と短期の残り）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, me_idx, cal = D["dates"], D["UNI"], D["me_idx"], D["cal"]
    T = len(cal)
    C, CF = D["C"], D["CF"]
    j0 = next((j for j, t in enumerate(me_idx) if t >= 252), None)
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    E = ev_extract(store, D)
    calD = cal.astype("datetime64[D]")
    wd = (calD.astype("datetime64[D]").view("int64") - 4) % 7          # 0＝月曜 … 4＝金曜
    gap = np.r_[np.diff(calD).astype(int), 1]                          # 次の営業日までの日数
    wk = (calD.view("int64") - 4) // 7                                 # 週の番号（月曜はじまり）
    last_of_week = np.r_[wk[1:] != wk[:-1], True]
    res_t = {}

    def run_timing(key, lab, variants):
        if ONLY and key not in ONLY:
            return
        r = {"lab": lab, "var": {}}
        for nm, vlab, x in variants:
            x = np.asarray(x, dtype=bool)
            v, liq, s_ = cal_sim(x, D, t0)
            rnd = [half_stats(cal_sim(spells_placebo(x, t0, 17000 + k), D, t0)[0], ev) for k in range(N_RAND)]
            nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
                  "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
            r["var"][nm] = {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "vlab": vlab, "nb": nb,
                            "held": float(x[t0:].mean()), "n_cand": int(x[t0:].sum()), "fill": 1.0}
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda xx: xx["held"] <= 0.0)
        res_t[key] = r
        log(f"  {key} … {r['verdict']}")
    # WD：週末をまたがない ／ 週末だけ持つ
    run_timing("WD", "週末（土日）をまたいで持つかどうか",
               (("A", "週末をまたがない（週の最後の日の終値で売り、次の週の最初の日の終値で買う）", ~last_of_week),
                ("B", "週末だけ持つ（週の最後の日の終値で買い、次の営業日の終値で売る）", last_of_week)))
    # HOL：連休の前
    hol3 = (gap >= 3) & ~((wd == 4) & (gap == 3))
    hol4 = gap >= 4
    run_timing("HOL", "連休の前の日の終値で買い、連休明けの終値で売る",
               (("A", "週末だけの休みを除く3日以上の休みの前", hol3), ("B", "4日以上の休みの前（大型連休・年末年始など）", hol4)))
    # SQ：第2金曜を含む週
    ym = calD.astype("datetime64[M]")
    sqday = np.zeros(T, dtype=bool)
    for m_ in np.unique(ym):
        idx = np.nonzero(ym == m_)[0]
        d0 = np.datetime64(str(m_) + "-01")
        first_fri = d0 + ((4 - ((d0.view("int64") - 4) % 7)) % 7)
        sq = first_fri + 7
        k = int(np.searchsorted(calD, sq, side="right")) - 1           # 第2金曜（休みなら、その前の営業日）
        if k >= 0 and k in set(idx):
            sqday[k] = True
    sqweek = np.zeros(T, dtype=bool)
    for k in np.nonzero(sqday)[0]:
        w = wk[k]
        sqweek[(wk == w) & (np.arange(T) <= k)] = True
    hold_out = np.zeros(T, dtype=bool)                                   # SQ の週の値動き（前の週の最後の終値 → SQ の日の終値）を持たない
    for k in np.nonzero(sqday)[0]:
        ws = np.nonzero(wk == wk[k])[0][0]
        hold_out[max(0, ws - 1):k] = True
    run_timing("SQ", "SQ（第2金曜）の週",
               (("A", "SQ の週は持たない", ~hold_out), ("B", "SQ の週だけ持つ", hold_out)))
    # YE：年末年始
    mb = month_bounds(cal)
    mon = (cal.astype("datetime64[M]").astype(int) % 12) + 1

    def ye(pre, post):
        x = np.zeros(T, dtype=bool)
        for k in range(len(mb) - 1):
            s_, e_ = mb[k]
            if mon[e_] != 12:
                continue
            s2, e2 = mb[k + 1]
            x[max(s_, e_ - pre):min(e2, s2 + post - 1)] = True
        return x
    run_timing("YE", "年末年始だけ TOPIX を持つ",
               (("A", "12月の最後の日の10営業日前に買い、1月の5営業日目に売る", ye(10, 5)), ("B", "5営業日前に買い、1月の3営業日目に売る", ye(5, 3))))
    # DIVM：3月・9月の権利付き最終日の前
    def divm(L):
        x = np.zeros(T, dtype=bool)
        for _s, e_ in mb:
            if mon[e_] in (3, 9) and e_ >= 3:
                t_c = t_cum_of(cal, e_)
                x[max(0, t_c - L):t_c] = True
        return x
    run_timing("DIVM", "3月・9月の権利付き最終日の前に TOPIX を持つ（権利付き最終日に売る）",
               (("A", "20営業日前に買う", divm(20)), ("B", "10営業日前に買う", divm(10))))
    # ── 個別株の短期（出来事の道具。20銘柄まで）
    jm = np.searchsorted(np.array(me_idx), np.arange(T), side="left") - 1
    traded = np.isfinite(C) & (C > 0)
    codes = D["codes"]
    sec = np.array([D["sector"].get(c, "") for c in codes])
    secs = sorted(set(sec) - {""})
    MC = D["MCAP"]

    def uni_t(t):
        j = jm[t]
        return (UNI[:, j] & traded[:, t]) if j >= 0 else np.zeros(len(codes), dtype=bool)
    R5 = np.full(CF.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        R5[:, 5:] = CF[:, 5:] / CF[:, :-5] - 1.0
    srev_ev, smom_ev = {}, {}
    for t in range(max(t0, 5), T):
        u = uni_t(t) & np.isfinite(R5[:, t])
        if u.sum() < 50:
            continue
        rel = np.full(len(codes), np.nan)
        sm = {}
        for sname in secs:
            m = u & (sec == sname)
            if m.sum() >= 5:
                med = float(np.median(R5[m, t]))
                rel[m] = R5[m, t] - med
                sm[sname] = med
        lo = np.nonzero(np.isfinite(rel) & (rel <= -0.10))[0]
        if len(lo):
            srev_ev[t] = [(float(-rel[i]), int(i)) for i in lo]
        if last_of_week[t] and len(sm) >= 10:
            top = sorted(sm, key=lambda k: -sm[k])[:3]
            m = u & np.isin(sec, top)
            idx = np.nonzero(m)[0]
            jj = jm[t]
            smom_ev[t] = [(float(MC[i, jj]) if MC[i, jj] == MC[i, jj] else 0.0, int(i)) for i in idx]
    res_e = {}

    def pool_all(t):
        return np.nonzero(uni_t(t))[0]

    def run_ev(key, lab, variants):
        if ONLY and key not in ONLY:
            return
        r = {"lab": lab, "var": {}}
        for nm, vlab, evs, H in variants:
            cands = {t: [(p, i, t + H) for p, i in lst] for t, lst in evs.items()}
            v, liq, s_ = ev_sim(cands, D, E["divpay"], t0)
            rnd, rnb = [], []
            for k in range(N_RAND):
                rg = np.random.default_rng(17500 + k)
                pl = {}
                for t, cl in cands.items():
                    pool = pool_all(t)
                    if len(pool) == 0:
                        continue
                    pick = rg.choice(pool, size=min(len(cl), len(pool), NMAX * 3), replace=False)
                    pl[t] = [(float(rg.random()), int(i), t_out) for i, (_p, _i, t_out) in zip(pick, cl)]
                pv, _l, ps_ = ev_sim(pl, D, E["divpay"], t0)
                rnd.append(half_stats(pv, ev))
                rnb.append(ps_["n_buy"])
            nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
                  "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
            r["var"][nm] = {"st": half_stats(v, ev), "liq": liq, "s": s_, "rnd": rnd, "vlab": vlab, "nb": nb,
                            "held": float(np.mean(s_["held"])) if s_["held"] else 0.0, "n_cand": sum(len(x) for x in cands.values()),
                            "fill": float(np.mean(rnb)) / max(1, s_["n_buy"])}
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda x: min(x["nb"].values()) < EV_MIN_BUYS or x["fill"] < RND_FILL_MIN)
        res_e[key] = r
        log(f"  {key} … {r['verdict']}")
    run_ev("SREV", "業種の中での数日の戻り（5営業日の値動きが、同じ業種の中央値より10％以上悪い株を、その日の終値で買う）",
           (("A", "5日持つ", srev_ev, 5), ("B", "10日持つ", srev_ev, 10)))
    run_ev("SMOM", "業種の1週間の勢い（週の最後の日に、5営業日の上げが上位3業種の株を時価総額の大きい順に買う）",
           (("A", "5日持つ（1週間）", smom_ev, 5), ("B", "10日持つ（2週間）", smom_ev, 10)))
    # ── 参考：夜だけ持つ・昼だけ持つ（1306。始値があるときだけ）
    tp = D["tp"]
    ref = None
    if "o" in tp and np.isfinite(np.asarray(tp["o"], dtype=float)[t0:]).mean() > 0.9:
        o = np.asarray(tp["o"], dtype=float)
        c_ = np.asarray(tp["c"], dtype=float)
        on = o[t0 + 1:] / c_[t0:-1]
        idr = c_[t0 + 1:] / o[t0 + 1:]
        ok = np.isfinite(on) & np.isfinite(idr) & (on > 0) & (idr > 0)
        n_ = int(ok.sum())
        ann_on = (np.prod(on[ok]) ** (252 / n_) - 1) * 100 if n_ else float("nan")
        ann_id = (np.prod(idr[ok]) ** (252 / n_) - 1) * 100 if n_ else float("nan")
        ref = (ann_on, ann_id, n_)
    # ── 報告
    HH = ("前半", "後半")
    print("\n【日本株の手法・17段目（暦と短期の残り）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan17_2026-11-05）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}")
    print(f"  時期を見る型 … 1306 を持つか現金にするか（片道{ETF_SLIP * 100:.2f}％・{TAX_TXT}）。でたらめは、持つ期間の長さをそのままに順番を入れ替えたもの")
    print(f"  個別株の短期 … 2回目と同じ道具（20銘柄まで・1銘柄は資産の20分の1・片道{SLIP * 100:.1f}％）。でたらめは同じ日に同じ数だけ範囲から")
    for res in (res_t, res_e):
        for key, r in res.items():
            print(f"     {key:<6}{r['lab']} … {r['verdict']}")
            for nm, x in r["var"].items():
                rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
                dd = " ".join(f"{r['detail'][(nm, h)][0]:+.1f}/{r['detail'][(nm, h)][1]:+.1f}・{r['detail'][(nm, h)][2]}回" for h in HH)
                extra = (f"持っていた日の割合 {x['held'] * 100:.0f}％" if res is res_t else
                         f"合図 {x['n_cand']:,}回・買い 前半 {x['nb']['前半']}回・後半 {x['nb']['後半']}回・平均の持ち数 {x['held']:.1f}・でたらめの買いの回数 {x['fill'] * 100:.0f}％")
                print(f"         {nm}（{x['vlab']}） {x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}（最大下落 {x['st']['通し']['mdd']:.0f}％）［{dd}］"
                      f" でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}・{extra}・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％")
    if ref:
        rt = 252 * 2 * ETF_SLIP * 100
        print(f"  参考（判定には入れない）：1306 を夜だけ持つ（終値で買い、次の日の始値で売る）と、費用の前で年 {ref[0]:+.1f}％・昼だけ持つ（始値で買い、終値で売る）と年 {ref[1]:+.1f}％"
              f"（{ref[2]}日）。どちらも毎日売り買いするので、費用だけで年 約{rt:.0f}％かかる")
    else:
        print("  参考：1306 の始値がないので、夜だけ・昼だけは計算していない（20年分のデータで回すと出る）")
    print("\n  判定のまとめ")
    allr = {**res_t, **res_e}
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in allr.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))
    print(f"  （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")
    return 0


# ── 19段目：組み合わせと空売り側
def mals_score(xs: dict):
    """日々公表の対象に入ってから20営業日以内の株を −1、それ以外を0（8段目の道具で、−1 の株を売る）。"""
    rows = x_rows(xs, "mal")

    def fn(D):
        cal, me_idx, codes = D["cal"], D["me_idx"], D["codes"]
        calD = cal.astype("datetime64[D]")
        ci = {c: i for i, c in enumerate(codes)}
        seen = {}
        for r in rows:
            c = yb.norm_code(r["code"] or "")
            if c in ci and len(str(r["pub"] or "")) >= 10:
                seen.setdefault(ci[c], set()).add(int(np.searchsorted(calD, np.datetime64(str(r["pub"])[:10]), side="right")))
        first = []
        for i, ts in seen.items():
            prev = -10 ** 9
            for t in sorted(ts):
                if t - prev > 60:
                    first.append((i, t))
                prev = t
        out = np.zeros((len(codes), len(me_idx)))
        me = np.array(me_idx)
        for i, t in first:
            j = int(np.searchsorted(me, t, side="left"))         # 対象に入った日（以降の最初の月末）
            if j < len(me) and me[j] - t <= 20:
                out[i, j] = -1.0
        return out + np.random.default_rng(1901).random(len(codes))[:, None] * 1e-6
    return fn


# ── 20段目：市況（日経平均・原油・金利・為替・VIX）で切り替える（計画は wide_lab_plan20_2026-11-05）
EXT_FRED = {"NK": "NIKKEI225", "OIL": "DCOILWTICO", "US10": "DGS10", "YEN": "DEXJPUS", "VIX": "VIXCLS"}
EXT_MOF = ("https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv",
           "https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv")
EXT_LAB = {"NK": "日経平均", "OIL": "原油（WTI）", "US10": "米国の10年金利", "JP10": "日本の10年金利", "YEN": "ドル円", "VIX": "VIX"}
EXT_LAG_DAYS = 7          # 月末より7日以上前の値だけを使う（公表の遅れのため）
EXT_A_LAG, EXT_B_WIN = 63, 250


def _mof_date(s: str):
    """財務省の基準日（例 R6.4.1・H30.12.28・S49.9.24）を日付にする。"""
    s = str(s).strip()
    if not s or s[0] not in "SHR":
        try:
            return np.datetime64(pd.Timestamp(s).date())
        except Exception:
            return None
    base = {"S": 1925, "H": 1988, "R": 2018}[s[0]]
    try:
        y, m, d = (int(x) for x in s[1:].split("."))
        return np.datetime64(f"{base + y:04d}-{m:02d}-{d:02d}")
    except Exception:
        return None


EXT_ALT = {     # FRED が取れないときの代わり（順に試す）。どれも日々の終値。値の大きさの単位は局面の決め方（上がったか・平均より上か）に効かない
    "NK": (("yahoo", "^N225"), ("stooq", "^nkx")),
    "OIL": (("yahoo", "CL=F"), ("stooq", "cl.f")),
    "US10": (("yahoo", "^TNX"), ("stooq", "10usy.b")),
    "YEN": (("yahoo", "JPY=X"), ("stooq", "usdjpy")),
    "VIX": (("cboe", "VIX"), ("yahoo", "^VIX")),
}
_UA = {"User-Agent": "Mozilla/5.0 (research script)"}


def _ext_get(src: str, sid: str):
    """1つの出どころから日々の値を取る。戻り値は（日付の並び, 値の並び）。"""
    import io
    import requests
    if src == "fred":
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid}, timeout=25, headers=_UA)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        d, v = pd.to_datetime(df[df.columns[0]], errors="coerce"), pd.to_numeric(df[df.columns[-1]], errors="coerce")
    elif src == "yahoo":
        from urllib.parse import quote
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(sid, safe='')}",
                         params={"period1": 1104537600, "period2": int(time.time()), "interval": "1d"}, timeout=25, headers=_UA)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        off = int(res.get("meta", {}).get("gmtoffset") or 0)
        d = pd.to_datetime(np.asarray(res["timestamp"], dtype="int64") + off, unit="s").normalize()
        v = pd.Series(res["indicators"]["quote"][0]["close"], dtype="float64")
        d = pd.Series(d)
    elif src == "stooq":
        r = requests.get("https://stooq.com/q/d/l/", params={"s": sid, "i": "d"}, timeout=25, headers=_UA)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        d, v = pd.to_datetime(df["Date"], errors="coerce"), pd.to_numeric(df["Close"], errors="coerce")
    elif src == "cboe":
        r = requests.get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sid}_History.csv", timeout=25, headers=_UA)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [str(c).strip().upper() for c in df.columns]
        d, v = pd.to_datetime(df["DATE"], errors="coerce", format="%m/%d/%Y"), pd.to_numeric(df["CLOSE"], errors="coerce")
    else:
        raise KeyError(src)
    d, v = pd.Series(d).reset_index(drop=True), pd.Series(v).reset_index(drop=True)
    ok = d.notna() & v.notna()
    s_ = pd.Series(v[ok].to_numpy(dtype=float), index=pd.DatetimeIndex(d[ok])).groupby(level=0).last().sort_index()
    if len(s_) < 1000 or s_.index[0] > pd.Timestamp("2007-06-30"):
        raise ValueError(f"短すぎる（{len(s_)}日・{s_.index[0].date() if len(s_) else '-'}から）")
    return s_.index.to_numpy(dtype="datetime64[D]"), s_.to_numpy(dtype=float)


def fetch_ext(have: dict | None = None) -> dict:
    """市況のデータを取る（plan20 の1と5の追記）。まだ持っていない系列だけを、FRED → 代わりの出どころの順に試す。日本の10年金利は財務省。"""
    import io
    import requests
    out = dict((have or {}).get("series") or {})
    src_used = dict((have or {}).get("src") or {})
    bad = {}
    for k, sid in EXT_FRED.items():
        if k in out:
            continue
        errs = []
        for src, s_id in (("fred", sid),) + EXT_ALT.get(k, ()):
            try:
                out[k] = _ext_get(src, s_id)
                src_used[k] = f"{src}:{s_id}"
                log(f"  {EXT_LAB[k]}（{src} {s_id}） … {len(out[k][0]):,}日（{str(out[k][0][0])}〜{str(out[k][0][-1])}）")
                break
            except Exception as e:
                errs.append(f"{src} {s_id}: {type(e).__name__}: {str(e)[:70]}")
                log(f"  {EXT_LAB[k]}（{src} {s_id}） … 取れませんでした（{type(e).__name__}）")
        if k not in out:
            bad[k] = " ／ ".join(errs)
    if "JP10" not in out:
        rows = {}
        errs = []
        for url in EXT_MOF:
            try:
                r = requests.get(url, timeout=60, headers=_UA)
                r.raise_for_status()
                txt = r.content.decode("cp932", errors="replace")
                df = pd.read_csv(io.StringIO(txt), skiprows=1)
                col = next((c for c in df.columns if str(c).strip() == "10年"), None)
                if col is None:
                    raise KeyError("10年の列がない")
                for s0, v in zip(df[df.columns[0]], pd.to_numeric(df[col], errors="coerce")):
                    d = _mof_date(s0)
                    if d is not None and v == v:
                        rows[d] = float(v)
            except Exception as e:
                errs.append(f"{url.rsplit('/', 1)[1]}: {type(e).__name__}: {str(e)[:60]}")
        if len(rows) > 1000:
            ds = np.array(sorted(rows), dtype="datetime64[D]")
            out["JP10"] = (ds, np.array([rows[d] for d in ds], dtype=float))
            src_used["JP10"] = "mof"
            log(f"  {EXT_LAB['JP10']}（財務省） … {len(ds):,}日（{str(ds[0])}〜{str(ds[-1])}）" + (f"（一部取れず：{' ／ '.join(errs)}）" if errs else ""))
        else:
            bad["JP10"] = " ／ ".join(errs) or "行が少なすぎる"
            log(f"  {EXT_LAB['JP10']}（財務省） … 取れませんでした（{bad['JP10'][:120]}）")
    return {"series": out, "bad": bad, "src": src_used, "at": str(date.today())}


def ext_flags(ext: dict, dates) -> dict:
    """月末ごとの「上昇局面」（A＝63営業日前より高い・B＝250営業日の平均より高い）。値は 1.0／0.0、決められない月は空。"""
    out = {}
    for k, (ds, vs) in ext.get("series", {}).items():
        fa = np.full(len(dates), np.nan)
        fb = np.full(len(dates), np.nan)
        for j, d in enumerate(dates):
            cut = np.datetime64(pd.Timestamp(d).date()) - np.timedelta64(EXT_LAG_DAYS, "D")
            i = int(np.searchsorted(ds, cut, side="right")) - 1
            if i >= EXT_A_LAG:
                fa[j] = 1.0 if vs[i] > vs[i - EXT_A_LAG] else 0.0
            if i >= EXT_B_WIN - 1:
                fb[j] = 1.0 if vs[i] > float(np.mean(vs[i - EXT_B_WIN + 1:i + 1])) else 0.0
        out[k] = {"A": fa, "B": fb}
    return out


def mp_sim(on: np.ndarray, D: dict, j0: int, pick_on, pick_off=None):
    """月末ごとに切り替える運用（20段目）。on[j] が真の月は pick_on(j) の株（なければ 1306）、偽の月は pick_off(j)（なければ 1306）。
    局面が続くあいだは同じ株を持ち、配当と現金は月末に今の持ち株へ比率どおりに足す（資産の1％を超えるとき）。"""
    PXM, DIV, cal, me_idx, tp = D["PXM"], D["DIV"], D["cal"], D["me_idx"], D["tp"]
    M = len(me_idx)
    px_e = np.array([float(tp["c"][t]) for t in me_idx])
    dist = [(np.datetime64(pd.Timestamp(d).date()), a / 100.0) for d, a in yb.ETF_DIST.get("1306", ())]
    cash, loss = float(CAP), 0.0
    pos = {}                    # 資産（株の番号か "E"）→ [口数, 取得の額]
    leg = None
    vals, st = [], {"switch": 0, "on_m": 0, "tax": 0.0, "n_hold": []}

    def price(a, j):
        return px_e[j] if a == "E" else float(PXM[a, j])

    def sell(a, j, frac=1.0):
        nonlocal cash, loss
        u, b = pos[a]
        p = price(a, j)
        slip = ETF_SLIP if a == "E" else SLIP
        pr = u * frac * p * (1.0 - slip)
        g = pr - b * frac
        cash += pr
        if g > 0:
            tx = max(0.0, g - loss) * TAX
            loss = max(0.0, loss - g)
            cash -= tx
            st["tax"] += tx
        else:
            loss += -g
        if frac >= 1.0:
            del pos[a]
        else:
            pos[a] = [u * (1 - frac), b * (1 - frac)]

    def buy(a, j, amt):
        nonlocal cash
        p = price(a, j)
        if not (p == p and p > 0) or amt <= 0:
            return
        slip = ETF_SLIP if a == "E" else SLIP
        u = amt / (p * (1.0 + slip))
        cash -= amt
        if a in pos:
            pos[a][0] += u
            pos[a][1] += amt
        else:
            pos[a] = [u, amt]
    for j in range(j0, M):
        if j > j0:
            d0, d1 = cal[me_idx[j - 1]], cal[me_idx[j]]
            for a in list(pos):
                if a == "E":
                    dd = sum(x for d, x in dist if d0 < d <= d1)
                    if dd > 0:
                        g = pos[a][0] * dd
                        cash += g * (1.0 - TAX)
                        st["tax"] += g * TAX
                    continue
                p = float(PXM[a, j])
                dv = float(DIV[a, j]) if DIV[a, j] == DIV[a, j] else 0.0
                if dv > 0:
                    g = pos[a][0] * dv
                    cash += g * (1.0 - TAX)
                    st["tax"] += g * TAX
                if not (p == p and p > 0):          # 上場をやめた：前の月末の値で現金にする
                    u, b = pos.pop(a)
                    pr = u * float(PXM[a, j - 1]) * (1.0 - SLIP)
                    cash += pr
                    if pr - b > 0:
                        tx = max(0.0, pr - b - loss) * TAX
                        loss = max(0.0, loss - (pr - b))
                        cash -= tx
                    else:
                        loss += b - pr
        if j == M - 1:
            vals.append(cash + sum(u * price(a, j) for a, (u, _b) in pos.items()))
            break
        want = bool(on[j])
        if leg is not None and leg != "E" and (leg == "ON") == want:
            sel = "keep"                            # 同じ局面が続く：同じ株を持ち続ける
        else:
            sel = (pick_on(j) if want else (pick_off(j) if pick_off else None))
        new_leg = leg if sel == "keep" else ("E" if sel is None else ("ON" if want else "OFF"))
        if leg is None or new_leg != leg:
            for a in list(pos):
                sell(a, j)
            if leg is not None:
                st["switch"] += 1
            tot = cash
            if sel is None:
                buy("E", j, tot)
            else:
                for i, w in zip(*sel):
                    buy(int(i), j, tot * float(w))
            leg = new_leg
        else:
            tot = cash + sum(u * price(a, j) for a, (u, _b) in pos.items())
            if pos and cash > 0.01 * tot:
                hv = {a: u * price(a, j) for a, (u, _b) in pos.items()}
                sv = sum(hv.values())
                c0 = cash
                for a, v in hv.items():
                    buy(a, j, c0 * v / sv)
        if leg == "ON":
            st["on_m"] += 1
        st["n_hold"].append(len(pos))
        vals.append(cash + sum(u * price(a, j) for a, (u, _b) in pos.items()))
    jl = M - 1
    liq = cash
    for a, (u, b) in pos.items():
        slip = ETF_SLIP if a == "E" else SLIP
        pr = u * price(a, jl) * (1.0 - slip)
        liq += pr - TAX * max(0.0, pr - b - loss)
    return np.array(vals), liq, st


def lab20(store: dict, xs: dict) -> int:
    """20段目（計画は wide_lab_plan20_2026-11-05）。T1〜T7：市況で 1306 を持つか現金か。R1〜R5：市況で業種の大型株か 1306 か。"""
    log("20段目（市況で切り替える）を回します（指標を作っています）…")
    ext = xs.get("ext") or {}
    if not ext.get("series"):
        print("\n■ 市況のデータがありません")
        return 2
    D = build(store)
    dates, UNI, me_idx, cal = D["dates"], D["UNI"], D["me_idx"], D["cal"]
    T = len(cal)
    M = len(me_idx)
    j0 = next((j for j, t in enumerate(me_idx) if t >= 252), None)
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    FL = ext_flags(ext, dates)
    for k in EXT_LAB:
        if k not in FL:
            log(f"  {EXT_LAB[k]} … データがないので、この指標を使う型は回しません")
        else:
            for v in ("A", "B"):
                f = FL[k][v][j0:]
                log(f"  {EXT_LAB[k]}（{v}） … 上昇局面 {int(np.nansum(f))}か月 ／ 決められない {int(np.isnan(f).sum())}か月（比べる期間 {len(f)}か月）")

    def daily(mf: np.ndarray) -> np.ndarray:
        x = np.zeros(T, dtype=bool)
        for j in range(M):
            e_ = me_idx[j + 1] if j + 1 < M else T
            x[me_idx[j]:e_] = bool(mf[j])
        return x

    def mplac(mf: np.ndarray, seed: int) -> np.ndarray:
        return spells_placebo(np.asarray(mf, dtype=bool), j0, seed)
    res = {}
    # ── T：1306 を持つか現金か（月末に決める）
    hold_rule = {"T1": ("NK", True), "T2": ("OIL", False), "T3": ("US10", False), "T4": ("JP10", False), "T5": ("YEN", True), "T6": ("VIX", False)}
    tlab = {"T1": "日経平均が上昇局面なら TOPIX、下落局面なら現金", "T2": "原油が上昇局面なら現金", "T3": "米国の金利が上昇局面なら現金",
            "T4": "日本の金利が上昇局面なら現金", "T5": "円高局面（ドル円が下落局面）なら現金", "T6": "VIX が上昇局面なら現金",
            "T7": "T1〜T6 のうち4つ以上が「現金」なら現金（多数決）"}
    holds = {}
    for key, (s, up_hold) in hold_rule.items():
        if s not in FL:
            continue
        holds[key] = {}
        for v in ("A", "B"):
            f = FL[s][v]
            holds[key][v] = np.where(np.isnan(f), 1.0, (f if up_hold else 1.0 - f)) > 0.5      # 決められない月は TOPIX を持つ
    if len(holds) == 6:
        holds["T7"] = {v: (sum((~holds[k][v]).astype(int) for k in hold_rule) < 4) for v in ("A", "B")}
    for key, hv in holds.items():
        if ONLY and key not in ONLY:
            continue
        r = {"lab": tlab[key], "var": {}, "kind": "T"}
        for v in ("A", "B"):
            mf = hv[v]
            vv, liq, s_ = cal_sim(daily(mf), D, t0)
            rnd = [half_stats(cal_sim(daily(mplac(mf, 20000 + k)), D, t0)[0], ev) for k in range(N_RAND)]
            sw = int(np.sum(mf[j0 + 1:] != mf[j0:-1]))
            r["var"][v] = {"st": half_stats(vv, ev), "liq": liq, "rnd": rnd, "vlab": "3か月前と比べる" if v == "A" else "1年の平均と比べる",
                           "held": float(mf[j0:].mean()), "sw": sw}
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda xx: xx["sw"] == 0)
        res[key] = r
        log(f"  {key} … {r['verdict']}")
    # ── R：業種の大型株か 1306 か
    codes = D["codes"]
    sec = np.array([D["sector"].get(c, "") for c in codes])
    MC, BPM, PXM = D["MCAP"], D["F"]["BP"], D["PXM"]
    G = {"FIN": {"銀行業", "保険業", "証券、商品先物取引業", "その他金融業"}, "RES": {"鉱業", "石油・石炭製品", "卸売業"},
         "EXP": {"輸送用機器", "電気機器", "機械", "精密機器"}, "DOM": {"食料品", "医薬品", "電気・ガス業", "陸運業", "小売業", "情報・通信業"},
         "DEF": {"食料品", "医薬品", "電気・ガス業", "陸運業"}}
    pick_cache = {}

    def pick(grp):
        def f(j):
            ck = (grp, j)
            if ck in pick_cache:
                return pick_cache[ck]
            ok = UNI[:, j] & np.isfinite(PXM[:, j]) & (PXM[:, j] > 0) & np.isfinite(MC[:, j]) & (MC[:, j] > 0)
            if grp == "VAL":
                idx = np.nonzero(ok)[0]
                top = idx[np.argsort(-MC[idx, j])][:500]
                b = BPM[top, j]
                fin = np.isfinite(b)
                m = np.zeros(len(codes), dtype=bool)
                if fin.sum() >= 30:
                    m[top[fin & (b >= np.median(b[fin]))]] = True
                ok = ok & m
            else:
                ok = ok & np.isin(sec, list(G[grp]))
            idx = np.nonzero(ok)[0]
            if len(idx) < 5:
                pick_cache[ck] = None
                return None
            sel = idx[np.argsort(-MC[idx, j])][:NMAX]
            out = (sel, cap_weights(MC[sel, j], W_CAP))
            pick_cache[ck] = out
            return out
        return f
    rot = {"R1": ("JP10", "FIN", None, "日本の金利の上昇局面は金融の大型株、それ以外は TOPIX"),
           "R2": ("US10", "VAL", None, "米国の金利の上昇局面は割安株（上位500社のうち純資産に対して安い上位半分）、それ以外は TOPIX"),
           "R3": ("OIL", "RES", None, "原油の上昇局面は資源と商社の大型株、それ以外は TOPIX"),
           "R4": ("YEN", "EXP", "DOM", "円安局面は輸出の大型株、円高局面は内需の大型株"),
           "R5": ("VIX", "DEF", None, "VIX の上昇局面は守りの業種の大型株、それ以外は TOPIX")}
    ref = {}
    for key, (s, g_on, g_off, lab) in rot.items():
        if (ONLY and key not in ONLY) or s not in FL:
            continue
        p_on, p_off = pick(g_on), (pick(g_off) if g_off else None)
        r = {"lab": lab, "var": {}, "kind": "R"}
        for v in ("A", "B"):
            mf = np.nan_to_num(FL[s][v], nan=0.0) > 0.5
            vv, liq, s_ = mp_sim(mf, D, j0, p_on, p_off)
            rnd = [half_stats(mp_sim(mplac(mf, 21000 + k), D, j0, p_on, p_off)[0], ev) for k in range(N_RAND)]
            sw = int(np.sum(mf[j0 + 1:] != mf[j0:-1]))
            r["var"][v] = {"st": half_stats(vv, ev), "liq": liq, "rnd": rnd, "vlab": "3か月前と比べる" if v == "A" else "1年の平均と比べる",
                           "held": float(mf[j0:].mean()), "sw": sw, "tax": s_["tax"], "nh": float(np.mean(s_["n_hold"])) if s_["n_hold"] else 0.0,
                           "on_m": s_["on_m"] / max(1, len(s_["n_hold"]))}
        r["verdict"], r["detail"] = judge2(r["var"], ST_TX, lambda xx: xx["sw"] == 0 or xx["on_m"] <= 0)
        res[key] = r
        # 参考：業種の株をずっと持つ（R4 は輸出と内需の両方）
        for g in ((g_on,) + ((g_off,) if g_off else ())):
            if g not in ref:
                vv, liq, _s = mp_sim(np.ones(M, dtype=bool), D, j0, pick(g))
                ref[g] = (half_stats(vv, ev), liq)
        log(f"  {key} … {r['verdict']}")
    # ── 報告
    HH = ("前半", "後半")
    print("\n【日本株の手法・20段目（市況で切り替える）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan20_2026-11-05）")
    stage_note()
    print(f"  期間 … 比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev)}か月。前半 〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半 {SPLIT.date()}〜）")
    print(f"  局面 … 毎月末に、その日より{EXT_LAG_DAYS}日以上前の最後の値で決める。A＝{EXT_A_LAG}営業日前より高ければ上昇局面・B＝{EXT_B_WIN}営業日の平均より高ければ上昇局面")
    print(f"  T … 1306 を持つか現金か（片道{ETF_SLIP * 100:.2f}％・{TAX_TXT}）。R … 局面に合う業種の時価総額の大きい{NMAX}銘柄（時価総額で加重・1銘柄{W_CAP * 100:.0f}％まで。局面のあいだは同じ株）か 1306（株 片道{SLIP * 100:.1f}％）")
    print("  でたらめ … 局面の続いた長さをそのままに、順番を入れ替えたもの（20回）。判定は A・B の両方・前半と後半の両方で TOPIX +2pt・でたらめ +2pt かつ18回以上")
    print(f"  TOPIX（税あり） … {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}（通し {ST_TX['通し']['ann']:.1f}・最大下落 {ST_TX['通し']['mdd']:.0f}％・清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")
    for k, (ds, vs) in ext["series"].items():
        print(f"  データ {EXT_LAB.get(k, k)} … {str(ds[0])}〜{str(ds[-1])}（{len(ds):,}日・{(ext.get('src') or {}).get(k, '財務省' if k == 'JP10' else 'FRED')}）")
    if ext.get("bad"):
        print(f"  取れなかったデータ … {ext['bad']}")
    print("\n  1. 型ごとの成績（年率％。「前半 ／ 後半」。［TOPIX との差／でたらめとの差・でたらめを上回った回数］）")
    for key, r in res.items():
        print(f"     {key:<4}{r['lab']} … {r['verdict']}")
        for nm, x in r["var"].items():
            rn = {h: float(np.mean([y[h]['ann'] for y in x['rnd']])) for h in HH}
            dd = " ".join(f"{r['detail'][(nm, h)][0]:+.1f}/{r['detail'][(nm, h)][1]:+.1f}・{r['detail'][(nm, h)][2]}回" for h in HH)
            extra = (f"TOPIX を持った月 {x['held'] * 100:.0f}％" if r["kind"] == "T" else
                     f"上昇局面の月 {x['held'] * 100:.0f}％・業種の株を持った月 {x['on_m'] * 100:.0f}％・平均の持ち数 {x['nh']:.1f}")
            print(f"         {nm}（{x['vlab']}） {x['st']['前半']['ann']:6.1f} ／ {x['st']['後半']['ann']:6.1f}（最大下落 {x['st']['通し']['mdd']:.0f}％）［{dd}］"
                  f" でたらめ {rn['前半']:.1f} ／ {rn['後半']:.1f}・{extra}・切り替え {x['sw']}回・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％")
    if ref:
        names = {"FIN": "金融", "VAL": "割安株", "RES": "資源と商社", "EXP": "輸出", "DOM": "内需", "DEF": "守りの業種"}
        print("\n  参考（判定には使わない）：業種の大型株をずっと持った成績（前半 ／ 後半・清算後の通し）")
        for g, (sx, liq) in ref.items():
            print(f"     {names[g]} … {sx['前半']['ann']:.1f} ／ {sx['後半']['ann']:.1f}（最大下落 {sx['通し']['mdd']:.0f}％・清算後の通し {((liq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")
    print("\n  2. 判定のまとめ" + ("（見ていない期間。plan20 の9：探す段階で効かなかったので参考。確立とは呼ばない）" if (LONG and STAGE == "confirm") else ""))
    for v in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, r in res.items() if r["verdict"] == v]
        print(f"     {v} … " + ("・".join(ks) if ks else "なし"))

    def worst(k):
        d = res[k]["detail"]
        return min(min(d[(nm, h)][0] - PT_TX, d[(nm, h)][1] - PT_RND) for nm in res[k]["var"] for h in HH)

    def worst_tx(k):
        d = res[k]["detail"]
        return min(d[(nm, h)][0] for nm in res[k]["var"] for h in HH)
    cand = sorted([k for k, r in res.items() if r["verdict"] == "候補"], key=lambda k: -worst(k))[:3]
    near = sorted([k for k, r in res.items() if r["verdict"] == "範囲の中では効く"], key=lambda k: -worst_tx(k))[:3]
    if LONG and STAGE == "confirm":
        return 0
    print("\n  3. 確かめる段階に持っていくもの（3つまで）")
    if cand:
        print("     候補 … " + "・".join(cand))
    elif near:
        print("     準候補（候補がないので、範囲の中では効く形を TOPIX との差の小さいほうが大きい順） … " + "・".join(near))
    else:
        print("     なし")
    return 0


# ───────────────────────── 21段目：上岡式（2016年の本）の答え合わせ ─────────────────────────
# 計画は wide_lab_plan21_2026-11-08（結果を見る前に決めたもの）。
# 上岡正明『うねりチャート底値買い投資術』（2016年12月）の決まりを、本人以外のまとめ（ねほり.com 2017年・ダイヤモンド・ザイ 2017年・
# 書店サイトのレビュー）から決まりに直したもの。公開は2016年12月なので、9年分のデータ（2017年10月〜）はすべて公開のあと。
K_CACHE = OUT / "wide_k.pkl"           # 業種別空売り比率（スタンダード）。Actions のキャッシュの名前は wide-lab-k-v1-
K_MKT = {"0101", "0102", "0111", "0112"}   # 一部・二部・プライム・スタンダード（新興市場＝マザーズ・JASDAQ・グロースは除く）
K_PRIME = {"0101", "0111"}             # 騰落レシオを数える市場（一部・プライム）
K_WIN = 250                            # うねりを見る期間（営業日。約1年）
K_LO_Q, K_HI_Q = 10, 90                # 箱の下と上：その期間の終値の10％点と90％点
K_WMIN, K_WMAX = 0.15, 0.60            # 箱の幅（上 ÷ 下 − 1）
K_ZONE = 0.25                          # 箱の下から幅の25％までを「底の近く」、上から25％までを「天井の近く」
K_TRANS = 4                            # 1年で底の近くと天井の近くを行き来した回数の下限（4回＝2往復＝周期はおよそ6か月以内）
K_TAIL = 0.05                          # 箱からはみ出さないこと：終値の2％点が箱の下の5％下より上、98％点が箱の上の5％上より下
K_WATCH = 150                          # 見張る銘柄の数（ザイ 2017年：約150銘柄）
K_BELOW = 0.95                         # 箱の下を5％より下に割っていたら、箱が崩れたとみなして買わない
K_ADR_DAYS = 25                        # 騰落レシオ：25営業日の値上がり銘柄数の合計 ÷ 値下がり銘柄数の合計 × 100
K_SR_TH = 0.30                         # 空売り比率の下限
K_TP = 0.10                            # 平均の取得単価（費用込み）から +10％で全部売る
K_STEP_DN = 0.05                       # 次の分は、前に買った値段より5％以上下がってから
K_SLOTS = 20                           # 同時に持つ銘柄の数（1銘柄の予算は、最初に買う日の資産の20分の1）
K_MIN_BUYS = 20                        # 前半・後半のどちらかで、最初の買いがこれ未満なら保留
K_FORMS = (
    ("K1", "基本：騰落レシオ80以下・空売り比率30％以上の日に、底の近くの見張り銘柄を買い始める。3分割（前の値段から5％下がり、20営業日あけて）。+10％で全部売る。損切りなし",
     {"adr": 80, "split": (1 / 3, 1 / 3, 1 / 3), "gap": 20, "stop": None, "mkt": True, "fill": False}),
    ("K2", "K1 の騰落レシオを70以下にする",
     {"adr": 70, "split": (1 / 3, 1 / 3, 1 / 3), "gap": 20, "stop": None, "mkt": True, "fill": False}),
    ("K3", "K1 の分け方を「1・2・1・2」（4回。60営業日あけて）にする（ねほり.com のまとめ）",
     {"adr": 80, "split": (1 / 6, 2 / 6, 1 / 6, 2 / 6), "gap": 60, "stop": None, "mkt": True, "fill": False}),
    ("K4", "K1 に損切りを足す（最初に買った値段から15％下で全部売る）",
     {"adr": 80, "split": (1 / 3, 1 / 3, 1 / 3), "gap": 20, "stop": 0.15, "mkt": True, "fill": False}),
    ("K5", "参考：市場の合図なし（いつでも底の近くで買い始める）",
     {"adr": None, "split": (1 / 3, 1 / 3, 1 / 3), "gap": 20, "stop": None, "mkt": False, "fill": False}),
    ("K6", "参考：K1 で、空いている資金をいつも 1306 で持つ",
     {"adr": 80, "split": (1 / 3, 1 / 3, 1 / 3), "gap": 20, "stop": None, "mkt": True, "fill": True}),
)


def k_xs_load() -> dict:
    """21段目の空売り比率は、5段目のデータ（プレミアムの時期に取ったもの）とは別のファイルに持つ。"""
    global X_CACHE
    X_CACHE = K_CACHE
    return x_load()


def k_market_codes(store: dict, D: dict) -> np.ndarray:
    """月末ごとの市場コード（行＝銘柄・列＝月末。一覧にない月は空）。"""
    ci = {c: i for i, c in enumerate(D["codes"])}
    MK = np.full((len(D["codes"]), len(D["dates"])), "", dtype=object)
    for j, d in enumerate(D["dates"]):
        for c, mk, _sc, _s33, _nm in store["months"].get(d.strftime("%Y-%m"), ()):
            i = ci.get(c)
            if i is not None:
                MK[i, j] = mk
    return MK


def k_adr(D: dict, MK: np.ndarray) -> np.ndarray:
    """騰落レシオ（一部・プライムの普通株。25営業日）。値上がり＝その日の終値が前に売買があった日の終値より高い。数えられない日は NaN。"""
    C, CF, me_idx = D["C"], D["CF"], D["me_idx"]
    S, T = C.shape
    jc = np.minimum(np.searchsorted(np.array(me_idx), np.arange(T), side="left"), len(me_idx) - 1)   # その日を含む月の月末
    prime = np.isin(MK, list(K_PRIME))
    nu, nd = np.zeros(T), np.zeros(T)
    with np.errstate(invalid="ignore"):
        for t in range(1, T):
            pm = prime[:, jc[t]] & np.isfinite(C[:, t]) & np.isfinite(CF[:, t - 1])
            nu[t] = float(np.sum(pm & (C[:, t] > CF[:, t - 1])))
            nd[t] = float(np.sum(pm & (C[:, t] < CF[:, t - 1])))
    out = np.full(T, np.nan)
    cu, cd = np.cumsum(nu), np.cumsum(nd)
    for t in range(K_ADR_DAYS, T):
        u, d = cu[t] - cu[t - K_ADR_DAYS], cd[t] - cd[t - K_ADR_DAYS]
        if d > 0:
            out[t] = u / d * 100.0
    return out


def k_short_ratio(xs: dict, cal: np.ndarray) -> np.ndarray:
    """空売り比率（全業種の合計）＝（価格規制ありの空売り＋なしの空売り）÷（実注文の売り＋空売り）。データのない日は前の日の値。"""
    T = len(cal)
    sr = np.full(T, np.nan)
    for r in xs.get("srt", {}).get("rows", []):
        d, sell, s1, s2 = r
        tot = (sell or 0.0) + (s1 or 0.0) + (s2 or 0.0)
        if tot <= 0:
            continue
        t = int(np.searchsorted(cal, np.datetime64(str(d)[:10])))
        if t < T and cal[t] == np.datetime64(str(d)[:10]):
            sr[t] = ((s1 or 0.0) + (s2 or 0.0)) / tot
    return pd.Series(sr).ffill().to_numpy()


def k_watch(D: dict, MK: np.ndarray) -> list:
    """月末ごとの見張り銘柄：{銘柄の番号: (箱の下, 箱の上, 行き来の回数)}。範囲（共通の範囲 × 新興市場を除く）の中で、
    1年の終値が箱（10％点〜90％点）の中を4回以上行き来し、幅が15〜60％で、箱から大きくはみ出さない（2％点と98％点が箱の5％の外まで）銘柄を、
    行き来の多い順・売買代金の多い順に150まで。"""
    CF, C, SEG, LIQ, UNI, me_idx = D["CF"], D["C"], D["SEG"], D["LIQ"], D["UNI"], D["me_idx"]
    mk_ok = np.isin(MK, list(K_MKT))
    out = []
    for j, t in enumerate(me_idx):
        if t < K_WIN - 1:
            out.append({})
            continue
        t1 = t - K_WIN + 1
        ok = UNI[:, j] & mk_ok[:, j] & (SEG[:, t] <= t1) & (np.isfinite(C[:, t1:t + 1]).sum(axis=1) >= 200)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            out.append({})
            continue
        w = CF[idx, t1:t + 1].astype(float)
        lo = np.nanpercentile(w, K_LO_Q, axis=1)
        hi = np.nanpercentile(w, K_HI_Q, axis=1)
        p2 = np.nanpercentile(w, 2, axis=1)
        p98 = np.nanpercentile(w, 98, axis=1)
        rng = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            width = hi / lo - 1.0
            st = np.where(w <= (lo + K_ZONE * rng)[:, None], -1.0, np.where(w >= (hi - K_ZONE * rng)[:, None], 1.0, np.nan))
        st = pd.DataFrame(st).ffill(axis=1).to_numpy()
        trans = np.sum(np.isfinite(st[:, 1:]) & np.isfinite(st[:, :-1]) & (st[:, 1:] != st[:, :-1]), axis=1)
        q = (lo > 0) & (width >= K_WMIN) & (width <= K_WMAX) & (trans >= K_TRANS) & (p2 >= lo * (1.0 - K_TAIL)) & (p98 <= hi * (1.0 + K_TAIL))
        sel = np.nonzero(q)[0]
        liq = np.nan_to_num(LIQ[idx[sel], j])
        order = sel[np.lexsort((-liq, -trans[sel]))][:K_WATCH]
        out.append({int(idx[k]): (float(lo[k]), float(hi[k]), int(trans[k])) for k in order})
    return out


def k_sim(cands: dict, D: dict, divpay: dict, t0: int, cfg: dict, last_t: np.ndarray, cap: float = CAP, slip: float = SLIP,
          tax: float = TAX, eslip: float = ETF_SLIP):
    """上岡式の運用。cands … {合図の日の番号: [最初に買う候補の銘柄の番号（優先の順）]}。合図はその日の終値で決め、次の営業日の終値で売り買いする。
    毎日 ①上場をやめた銘柄を最後の値段で売る ②前の日に決めた売りと買い（買い足しが先・新しい銘柄はそのあと）③配当 ④（fill なら）空いた資金で 1306
    ⑤その日の終値で、次の日の売り（+10％・損切り）と買い足し・新しい買いを決める。1銘柄の予算は最初に買う日の資産の20分の1で、split の割合ずつ買う。
    新しい買いは cands の順（見張り銘柄の順）に、空いている枠の数まで。"""
    CF, C, me_idx, cal = D["CF"], D["C"], D["me_idx"], D["cal"]
    tpc = np.asarray(D["tp"]["c"], dtype=float)
    T = CF.shape[1]
    split, gap, stop, fill = cfg["split"], cfg["gap"], cfg["stop"], cfg["fill"]
    me_set = set(me_idx)
    dist: dict = {}
    if fill:
        for d_, a_ in yb.ETF_DIST.get("1306", ()):
            tt = int(np.searchsorted(cal, np.datetime64(pd.Timestamp(d_).date()), side="left"))
            if tt < T:
                dist[tt] = dist.get(tt, 0.0) + a_ / 100.0
    cash, loss = float(cap), 0.0
    eu, ecost = 0, 0.0
    pos: dict = {}            # 銘柄 → {sh, cost, budget, k, last_px, last_t, first_px, first_t}
    pend_buy: list = []
    pend_sell: set = set()
    vals = []
    st = {"n_new": 0, "n_add": 0, "buy_t": [], "closed": [], "tax": 0.0, "held": [], "cash_share": [], "etf_share": [], "delist": 0}

    def px(i, t):
        v = CF[i, t]
        return float(v) if (v == v and v > 0) else 0.0

    def realize(g):
        nonlocal cash, loss
        if g > 0:
            tx = max(0.0, g - loss) * tax
            loss = max(0.0, loss - g)
            cash -= tx
            st["tax"] += tx
        else:
            loss += -g

    def close_pos(i, t, price):
        nonlocal cash
        p = pos.pop(i)
        eff = price * (1.0 - slip)
        cash += p["sh"] * eff
        realize(p["sh"] * eff - p["cost"])
        st["closed"].append((p["first_t"], t, p["sh"] * eff / p["cost"] - 1.0, p["k"]))

    def etf_sell(units, t):
        nonlocal eu, ecost, cash
        units = min(int(units), eu)
        if units <= 0:
            return
        pr = tpc[t] * (1.0 - eslip)
        c_part = ecost * units / eu
        cash += units * pr
        eu -= units
        ecost -= c_part
        realize(units * pr - c_part)

    def value_at(t):
        return cash + eu * tpc[t] + sum(p["sh"] * px(k, t) for k, p in pos.items())
    for t in range(t0, T):
        ep = float(tpc[t])
        if t in dist and eu > 0:
            g = eu * dist[t]
            cash += g * (1.0 - tax)
            st["tax"] += g * tax
        # ① 上場をやめた銘柄（最後の売買の日から5営業日たっても売買がない）は、最後の値段で売る
        for i in [k for k in pos if last_t[k] < t - 5]:
            close_pos(i, t, px(i, last_t[i]))
            pend_sell.discard(i)
            st["delist"] += 1
        # ② 前の日に決めた売り
        for i in sorted(pend_sell):
            v = C[i, t]
            if i in pos and v == v and v > 0:
                close_pos(i, t, float(v))
                pend_sell.discard(i)
            elif i not in pos:
                pend_sell.discard(i)
        # ② 前の日に決めた買い（買い足しが先）
        if pend_buy:
            value = value_at(t)
            for i, kind in pend_buy:
                v = C[i, t]
                if not (v == v and v > 0):
                    continue
                if kind == "new":
                    if i in pos or len(pos) >= K_SLOTS:
                        continue
                    budget = value / K_SLOTS
                    amt = budget * split[0]
                else:
                    p = pos.get(i)
                    if p is None or p["k"] >= len(split):
                        continue
                    amt = p["budget"] * split[p["k"]]
                pb = float(v) * (1.0 + slip)
                if fill and cash < amt and eu > 0:
                    etf_sell(np.ceil((amt - cash) / (ep * (1.0 - eslip))), t)
                sh = int(min(amt, cash) / pb)
                if sh < 1:
                    continue
                cash -= sh * pb
                if kind == "new":
                    pos[i] = {"sh": sh, "cost": sh * pb, "budget": budget, "k": 1, "last_px": float(v), "last_t": t,
                              "first_px": float(v), "first_t": t}
                    st["n_new"] += 1
                    st["buy_t"].append(t)
                else:
                    p["sh"] += sh
                    p["cost"] += sh * pb
                    p["k"] += 1
                    p["last_px"], p["last_t"] = float(v), t
                    st["n_add"] += 1
            pend_buy = []
        # ③ 配当（権利付き最終日の終わりに持っていれば受け取る）
        for i, amt in divpay.get(t, ()):
            if i in pos:
                g = pos[i]["sh"] * amt
                cash += g * (1.0 - tax)
                st["tax"] += g * tax
        # ④ 空いた資金で 1306
        if fill and ep > 0:
            value = value_at(t)
            if cash > 0.01 * value:
                u = int(cash / (ep * (1.0 + eslip)))
                if u > 0:
                    cash -= u * ep * (1.0 + eslip)
                    eu += u
                    ecost += u * ep * (1.0 + eslip)
        if t in me_set:
            value = value_at(t)
            vals.append(value)
            st["held"].append(len(pos))
            st["cash_share"].append(cash / value if value > 0 else 0.0)
            st["etf_share"].append(eu * ep / value if value > 0 else 0.0)
        if t >= T - 1:
            break
        # ⑤ その日の終値で、次の日の売り買いを決める
        for i, p in pos.items():
            v = C[i, t]
            if not (v == v and v > 0):
                continue
            if v >= p["cost"] / p["sh"] * (1.0 + K_TP):
                pend_sell.add(i)
                continue
            if stop is not None and v < p["first_px"] * (1.0 - stop):
                pend_sell.add(i)
                continue
            if p["k"] < len(split) and t - p["last_t"] >= gap and v <= p["last_px"] * (1.0 - K_STEP_DN):
                pend_buy.append((i, "add"))
        for i in cands.get(t, ()):
            if i not in pos and i not in pend_sell:
                pend_buy.append((i, "new"))
    liq, gain = cash, 0.0
    for k, p in pos.items():
        eff = px(k, T - 1) * (1.0 - slip)
        liq += p["sh"] * eff
        gain += p["sh"] * eff - p["cost"]
    if eu > 0:
        eff = tpc[T - 1] * (1.0 - eslip) * eu
        liq += eff
        gain += eff - ecost
    liq -= tax * max(0.0, gain - loss)
    st["open"] = [(p["first_t"], px(k, T - 1) * p["sh"] / p["cost"] - 1.0, px(k, T - 1) * p["sh"]) for k, p in pos.items()]
    return np.array(vals), liq, st


def k_judge(x: dict, ST_TX: dict) -> tuple:
    """判定：TOPIX と でたらめ（範囲から）は judge1 と同じ。でたらめ（見張り銘柄から）は「底で買う効き目」の向きだけを見る。"""
    det = {}
    ok_tx, ok_ra, rb_up, rb_dn = True, True, True, True
    for h in ("前半", "後半"):
        a = x["st"][h]["ann"]
        ra = [y[h]["ann"] for y in x["ra"]]
        rb = [y[h]["ann"] for y in x["rb"]]
        d_tx, d_ra, d_rb = a - ST_TX[h]["ann"], a - float(np.mean(ra)), a - float(np.mean(rb))
        b_ra, b_rb = int(sum(1 for y in ra if a > y)), int(sum(1 for y in rb if a > y))
        det[h] = (d_tx, d_ra, b_ra, d_rb, b_rb)
        ok_tx &= d_tx >= PT_TX
        ok_ra &= (d_ra >= PT_RND) and (b_ra >= BEAT_MIN)
        rb_up &= (d_rb >= PT_RND) and (b_rb >= BEAT_MIN)
        rb_dn &= (d_rb <= -PT_RND) and (b_rb <= N_RAND - BEAT_MIN)
    rb_lab = "ある" if rb_up else ("裏目" if rb_dn else "見えない")
    if min(x["nb"].values()) < K_MIN_BUYS or x["fill_ra"] < RND_FILL_MIN or x["fill_rb"] < RND_FILL_MIN:
        return "保留（データ不足）", rb_lab, det
    return ("候補" if (ok_tx and ok_ra) else ("範囲の中では効く" if ok_ra else "効かない")), rb_lab, det


def lab21(store: dict, xs: dict) -> int:
    """21段目：上岡式（2016年の本）の答え合わせ。計画は wide_lab_plan21_2026-11-08。"""
    log("21段目（上岡式の答え合わせ）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, me_idx, cal, codes = D["dates"], D["UNI"], D["me_idx"], D["cal"], D["codes"]
    T = len(cal)
    C = D["C"]
    MK = k_market_codes(store, D)
    mk_ok = np.isin(MK, list(K_MKT))
    j0 = next((j for j, t in enumerate(me_idx) if t >= K_WIN), None)
    if j0 is None:
        print("\n■ 株価の期間が短すぎます（1年分のうねりを見る期間が取れません）")
        return 2
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    E = ev_extract(store, D)
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    traded = np.isfinite(C) & (C > 0)
    last_t = np.array([int(np.nonzero(traded[i])[0][-1]) if traded[i].any() else -1 for i in range(len(codes))])
    jm = np.searchsorted(np.array(me_idx), np.arange(T), side="left") - 1      # その日より前の最後の月末（その日の月末の日は前の月末）
    log("  騰落レシオ・空売り比率・見張り銘柄を作っています…")
    adr = k_adr(D, MK)
    sr_ok = bool(xs.get("srt", {}).get("rows")) and not xs.get("srt", {}).get("bad")
    sr = k_short_ratio(xs, cal) if sr_ok else np.full(T, np.nan)
    W = k_watch(D, MK)
    # 底の近くにいる見張り銘柄（合図の日ごと。見張り銘柄の順＝うねりのきれいな順。W[j] は行き来の多い順・売買代金の多い順に並んでいる）
    zone_c: dict = {}
    for t in range(t0, T - 1):
        j = jm[t]
        if j < 0 or not W[j]:
            continue
        ids = np.array(list(W[j].keys()))
        lo = np.array([W[j][i][0] for i in ids])
        hi = np.array([W[j][i][1] for i in ids])
        v = C[ids, t]
        ok = np.isfinite(v) & (v >= lo * K_BELOW) & (v <= lo + K_ZONE * (hi - lo))
        if ok.any():
            zone_c[t] = [int(x) for x in ids[ok]]

    def mkt_on(t, th):
        return (adr[t] == adr[t] and adr[t] <= th) and (not sr_ok or (sr[t] == sr[t] and sr[t] >= K_SR_TH))

    def pool_ra(t):
        j = jm[t]
        return np.nonzero(UNI[:, j] & mk_ok[:, j] & traded[:, t])[0] if j >= 0 else np.zeros(0, dtype=int)

    def pool_rb(t):
        j = jm[t]
        if j < 0 or not W[j]:
            return np.zeros(0, dtype=int)
        ids = np.array(list(W[j].keys()))
        return ids[traded[ids, t]]

    # ── 市場の合図と見張り銘柄の様子（判定の前に見る） ──
    tt = np.arange(t0, T)
    sr_v = sr[tt][np.isfinite(sr[tt])]
    adr_v = adr[tt][np.isfinite(adr[tt])]
    print("\n" + "=" * 100)
    print("【日本株の手法・21段目（上岡式〈2016年の本〉の答え合わせ）（wide_lab）：東証の一部・二部・プライム・スタンダード】")
    print("  （基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan21_2026-11-08。決まりは本人以外のまとめから直したもの）")
    print(f"  比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev) - 1}か月）。前半＝〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半＝{SPLIT.date()}〜")
    print(f"  費用と税 {COST}：株 片道{SLIP * 100:.1f}％・1306 片道{ETF_SLIP * 100:.2f}％・{TAX_TXT}（損は同じ運用の中で繰り越す）。元本{CAP / 1e4:,.0f}万円")
    print("\n  0. 市場の合図と見張り銘柄")
    if sr_ok:
        yr_min = pd.Series(sr[tt], index=pd.DatetimeIndex(cal[tt])).groupby(lambda x: x.year).min()
        print(f"     空売り比率 … 30％以上の日 {np.mean(sr_v >= K_SR_TH) * 100:.1f}％（{len(sr_v)}日）・いちばん低い日 {np.min(sr_v) * 100:.1f}％・"
              f"5％点 {np.percentile(sr_v, 5) * 100:.1f}％・中央 {np.median(sr_v) * 100:.1f}％・年ごとのいちばん低い日 "
              + "・".join(f"{y} {v * 100:.0f}％" for y, v in yr_min.items()))
    else:
        print(f"     空売り比率 … 取れませんでした（{xs.get('srt', {}).get('bad', 'データなし')}）。市場の合図は騰落レシオだけで見ます（計画の4）")
    for th in (80, 70):
        on = np.array([mkt_on(t, th) for t in tt])
        runs = int(np.sum(on[1:] & ~on[:-1]) + (1 if on[0] else 0))
        print(f"     騰落レシオ{th}以下 … {np.mean(adr_v <= th) * 100:.1f}％の日（{int(np.sum(adr_v <= th))}日）・空売り比率とあわせた合図の日 {int(on.sum())}日"
              f"（1年あたり {on.sum() / yrs:.0f}日・ひと続きの回数 {runs}回）")
    nW = [len(W[j]) for j in range(j0, len(W))]
    allq = [np.mean([v[2] for v in W[j].values()]) if W[j] else np.nan for j in range(j0, len(W))]
    print(f"     見張り銘柄 … 月平均 {np.mean(nW):.0f}銘柄（{min(nW)}〜{max(nW)}）・行き来の回数の平均 {np.nanmean(allq):.1f}回・"
          f"底の近くにいる銘柄 1日平均 {np.mean([len(zone_c.get(t, ())) for t in tt]):.1f}銘柄")
    for jj in sorted({j0, (j0 + len(W) - 1) // 2, len(W) - 1}):
        top = sorted(W[jj].items(), key=lambda kv: (-kv[1][2], -np.nan_to_num(D["LIQ"][kv[0], jj])))[:8]
        print(f"     例（{dates[jj].date()}）：" + "・".join(f"{codes[i]} {D['names'].get(codes[i], '')[:8]}（{v[2]}回・幅{(v[1] / v[0] - 1) * 100:.0f}％）" for i, v in top))
    res = {}
    for key, lab, cfg in K_FORMS:
        if ONLY and key not in ONLY:
            continue
        log(f"  {key} を回しています（でたらめ 20回 × 2）…")
        if cfg["mkt"]:
            cands = {t: zone_c[t] for t in zone_c if mkt_on(t, cfg["adr"])}
        else:
            cands = dict(zone_c)
        v, liq, s_ = k_sim(cands, D, E["divpay"], t0, cfg, last_t)
        ra, rb, nra, nrb = [], [], [], []
        n_c = max(1, sum(len(cl) for cl in cands.values()))
        for nm, pool_fn, seed, out_, nout in (("ra", pool_ra, 7000, ra, nra), ("rb", pool_rb, 8000, rb, nrb)):
            pools = {t: pool_fn(t) for t in cands}
            nout.append(sum(min(len(cl), len(pools[t])) for t, cl in cands.items()) / n_c)    # 候補の数だけ、でたらめに選べた割合
            for k in range(N_RAND):
                rg = np.random.default_rng(seed + k)
                pl = {}
                for t, cl in cands.items():
                    pool = pools[t]
                    if len(pool):
                        pl[t] = [int(x) for x in rg.choice(pool, size=min(len(cl), len(pool)), replace=False)]
                pv, _l, ps_ = k_sim(pl, D, E["divpay"], t0, cfg, last_t)
                out_.append(half_stats(pv, ev))
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
              "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        x = {"st": half_stats(v, ev), "liq": liq, "s": s_, "ra": ra, "rb": rb, "nb": nb,
             "fill_ra": nra[0], "fill_rb": nrb[0]}
        x["ci"] = boot_ci(v, TXv, liq, TXliq, yrs)      # 30段目：ぶれの幅
        x["verdict"], x["rb_lab"], x["det"] = k_judge(x, ST_TX)
        res[key] = (lab, x)
    print(f"\n  1. 結果（年率％「前半 ／ 後半」。TOPIX（税あり）{ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}。"
          "差は「TOPIX との差／でたらめ（範囲から）との差・上回った回数／でたらめ（見張り銘柄から）との差・上回った回数」）")
    for key, (lab, x) in res.items():
        a, s_ = x["st"], x["s"]
        cl = s_["closed"]
        win = np.mean([r[2] > 0 for r in cl]) * 100 if cl else float("nan")
        gw = [r[2] for r in cl if r[2] > 0]
        gl = [r[2] for r in cl if r[2] <= 0]
        hold = [(r[1] - r[0]) for r in cl]
        opn = s_["open"]
        old = [o for o in opn if o[0] < T - 1 - 250]
        ov = sum(o[2] for o in opn)
        ra_m = [np.mean([y[h]["ann"] for y in x["ra"]]) for h in ("前半", "後半")]
        rb_m = [np.mean([y[h]["ann"] for y in x["rb"]]) for h in ("前半", "後半")]
        print(f"\n  {key} {lab}")
        print(f"     {ci_txt(x['ci'])}")
        print(f"     年率 {a['前半']['ann']:6.1f} ／ {a['後半']['ann']:6.1f}（最大下落 {a['通し']['mdd']:.0f}％・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％）"
              f"［{x['det']['前半'][0]:+.1f}／{x['det']['前半'][1]:+.1f}・{x['det']['前半'][2]}／{x['det']['前半'][3]:+.1f}・{x['det']['前半'][4]} ｜ "
              f"{x['det']['後半'][0]:+.1f}／{x['det']['後半'][1]:+.1f}・{x['det']['後半'][2]}／{x['det']['後半'][3]:+.1f}・{x['det']['後半'][4]}］"
              f" … {x['verdict']}・底で買う効き目 {x['rb_lab']}")
        print(f"     でたらめの平均：範囲から {ra_m[0]:.1f} ／ {ra_m[1]:.1f}・見張り銘柄から {rb_m[0]:.1f} ／ {rb_m[1]:.1f}")
        print(f"     最初の買い {s_['n_new']}回（前半 {x['nb']['前半']}・後半 {x['nb']['後半']}・1年あたり {s_['n_new'] / yrs:.0f}回）・買い足し {s_['n_add']}回・"
              f"売り {len(cl)}回（勝ち {win:.0f}％・勝ちの平均 {(f'{np.mean(gw) * 100:+.1f}％') if gw else '—'}・負けの平均 {(f'{np.mean(gl) * 100:+.1f}％') if gl else '—'}・"
              f"持った日数の中央 {np.median(hold) if hold else float('nan'):.0f}営業日）・上場をやめて売った {s_['delist']}回")
        print(f"     平均の持ち数 {np.mean(s_['held']):.1f}銘柄・現金の割合 平均 {np.mean(s_['cash_share']) * 100:.0f}％"
              + (f"・1306 の割合 平均 {np.mean(s_['etf_share']) * 100:.0f}％" if key == "K6" else "")
              + f"・最後に持っていた {len(opn)}銘柄（含み損益 {np.mean([o[1] for o in opn]) * 100 if opn else 0:+.1f}％・1年以上持っている {len(old)}銘柄"
              f"＝評価額の {sum(o[2] for o in old) / ov * 100 if ov > 0 else 0:.0f}％）・でたらめに候補の数だけ選べた割合 {x['fill_ra'] * 100:.0f}％／{x['fill_rb'] * 100:.0f}％")
    print("\n  2. 判定のまとめ")
    for vv in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, (_l, x) in res.items() if x["verdict"] == vv]
        print(f"     {vv} … " + ("・".join(ks) if ks else "なし"))
    for vv in ("ある", "見えない", "裏目"):
        ks = [k for k, (_l, x) in res.items() if x["rb_lab"] == vv]
        print(f"     底で買う効き目 {vv} … " + ("・".join(ks) if ks else "なし"))
    if "K1" in res and "K5" in res:
        d = [res["K1"][1]["st"][h]["ann"] - res["K5"][1]["st"][h]["ann"] for h in ("前半", "後半")]
        lab_m = "市場の売られすぎを待つと良い" if min(d) >= PT_RND else ("待つと悪い" if max(d) <= -PT_RND else "差は見えない")
        print(f"     市場の合図の効き目（K1 − K5）… {d[0]:+.1f} ／ {d[1]:+.1f}pt → {lab_m}")
    if "K1" in res and "K6" in res:
        d = [res["K6"][1]["st"][h]["ann"] - res["K1"][1]["st"][h]["ann"] for h in ("前半", "後半")]
        print(f"     空いた資金を 1306 で持つ効き目（K6 − K1・参考）… {d[0]:+.1f} ／ {d[1]:+.1f}pt")
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")
    return 0


# ───────────────────────── 22段目：DUKE。（新高値ブレイク投資術）と CAN SLIM の日本版 ─────────────────────────
# 計画は wide_lab_plan22_2026-11-09（結果を見る前に決めたもの）。DUKE。『1勝4敗でもしっかり儲ける新高値ブレイク投資術』（2016年7月）。
# 9年分のデータ（2017年10月〜）はすべて本が出たあと。9年分には始値・高値・安値がないので、ストップ高で引けた日は終値の動きから見つける。
D_MCAP_MAX = 5e10        # 時価総額500億円未満
D_HI_WIN = 250           # 新高値＝その日までの250営業日の終値の最高値を、終値で超えた日
D_RUN_GAP = 10           # 新高値のない日が10営業日続いたあとの新高値を「新しいブレイク」とする
D_MAX_RUNS = 3           # 過去250営業日のブレイクが3回目まで（「2回目のブレイクくらいまでなら初動」）
D_STOP = 0.10            # 平均の取得単価から −10％で損切り
D_BE = 0.20              # 最初に買った値段から +20％に届いたら、損切りの線を取得単価まで上げる
D_MA_TRAIL = 50          # +20％に届いたあとは、終値が50日移動平均を割ったら売る
D_ADD_STEP = 0.05        # 5分の1ルール：前に買った値段から +5％ごとに、5分の1ずつ買い増す
D_MKT_MA = 25            # 市況：1306 の終値が25日移動平均より下なら、新しく買わない
D_MIN_BUYS = 20
D_FORMS = (
    ("D1", "DUKE。基本：新高値ブレイク（過去1年で3回目まで）・直近2四半期の利益が前年比+20％以上と売上が+10％以上・時価総額500億円未満・5銘柄・"
           "−10％か箱に戻ったら売る・+20％で損切りを取得単価へ・その後は50日線割れで売る",
     {"filt": "duke", "slots": 5, "pyr": False, "mkt": False}),
    ("D2", "D1 を5分の1ルールにする（最初は枠の5分の1、+5％ごとに5分の1ずつ買い増す）", {"filt": "duke", "slots": 5, "pyr": True, "mkt": False}),
    ("D3", "D1 に市況を足す（1306 が25日線より下なら新しく買わない）", {"filt": "duke", "slots": 5, "pyr": False, "mkt": True}),
    ("D4", "参考：D1 を20銘柄に広げる（運の幅を小さくした形）", {"filt": "duke", "slots": 20, "pyr": False, "mkt": False}),
    ("D5", "参考：D1 で、箱に戻っても売らない（売りは −10％の損切りと、+20％のあとの取得単価・50日線だけ）",
     {"filt": "duke", "slots": 5, "pyr": False, "mkt": False, "nobox": True}),
    ("C1", "CAN SLIM の日本版：直近の四半期の利益が前年比+40％以上・通期の利益の3年の年率+25％以上（ほかは D1 と同じ）",
     {"filt": "canslim", "slots": 5, "pyr": False, "mkt": False}),
    ("B0", "参考：業績の条件なし（新高値ブレイクだけ。ほかは D1 と同じ）", {"filt": "none", "slots": 5, "pyr": False, "mkt": False}),
)


def limit_masks_close(store: dict, codes: list, cal: np.ndarray, C: np.ndarray, CF: np.ndarray):
    """始値・高値・安値がないときのストップ高・ストップ安で引けた日：前の日の終値から、制限値幅いっぱいまで終値が動いた日
    （制限値幅いっぱいの値段は、その日の高値〈安値〉でもある）。値幅は分割の調整をはずした株価で決める（limit_masks と同じ）。"""
    S, T = C.shape
    LIMU = np.zeros((S, T), dtype=bool)
    LIMD = np.zeros((S, T), dtype=bool)
    calD = cal.astype("datetime64[D]")
    for i, c in enumerate(codes):
        b = store["bars"][c]
        fd, ff = b["fd"], b["ff"]
        if len(fd):
            suf = np.r_[np.cumprod(ff[::-1])[::-1], 1.0]
            fac = suf[np.searchsorted(fd, calD, side="right")]
        else:
            fac = np.ones(T)
        cl = C[i]
        ok = np.isfinite(cl) & (cl > 0)
        ok[0] = False
        if not ok.any():
            continue
        t = np.nonzero(ok)[0]
        base = CF[i, t - 1] / fac[t]
        close = cl[t] / fac[t]
        w = price_limit(base)
        tol = 1e-4 * np.maximum(base, 1.0)
        LIMU[i, t] = close >= base + w - tol
        LIMD[i, t] = close <= base - w + tol
    return LIMU, LIMD


def d_breakouts(C: np.ndarray, CF: np.ndarray, SEG: np.ndarray) -> tuple:
    """新高値ブレイクの合図（行＝銘柄・列＝日）と、その日より前の250営業日の最高値。
    新高値＝終値が、その日より前の250営業日の終値の最高値を超えた日（同じ売買の続きの中で250営業日以上たってから）。
    新高値のない日が10営業日続いたあとの新高値を「ブレイク」とし、その日を含む250営業日のブレイクが3回目までのときだけ合図にする。"""
    S, T = C.shape
    traded = np.isfinite(C) & (C > 0)
    with np.errstate(invalid="ignore"):
        prev_hi = pd.DataFrame(CF.T).rolling(D_HI_WIN, min_periods=200).max().shift(1).to_numpy().T
        newhi = traded & np.isfinite(prev_hi) & (C > prev_hi) & (SEG <= np.arange(T)[None, :] - D_HI_WIN)
    idx = np.arange(T)
    csp = np.concatenate([np.zeros((S, 1)), np.cumsum(newhi, axis=1)], axis=1)        # csp[:, k] ＝ newhi[:, :k] の合計
    prev_cnt = csp[:, idx] - csp[:, np.maximum(idx - D_RUN_GAP, 0)]                   # その日の前の10営業日の新高値の数
    run_start = newhi & (prev_cnt == 0)
    rsp = np.concatenate([np.zeros((S, 1)), np.cumsum(run_start, axis=1)], axis=1)
    runs_win = rsp[:, idx + 1] - rsp[:, np.maximum(idx + 1 - D_HI_WIN, 0)]           # その日を含む250営業日のブレイクの数
    return run_start & (runs_win <= D_MAX_RUNS), prev_hi


def d_fy_back(fy: str) -> str:
    return f"{int(fy[:4]) - 1}-{fy[5:7]}"


def d_fin_status(rows: list) -> list:
    """開示ごとの業績の判定：[(開示日, DUKE の条件, CAN SLIM の条件, 直近の四半期の利益の伸び)]。
    四半期の値は、同じ年度の累計の差で作る（第1四半期はそのまま）。利益は経常利益（IFRS などで経常利益のない会社は、すべて営業利益）。"""
    tl = [r for r in fin_timeline(rows) if "FinancialStatements" in str(r.get("DocType") or "")
          and str(r.get("CurPerType") or "") in ("1Q", "2Q", "3Q", "FY")]
    if not tl:
        return []
    use_od = all(num(r.get("OdP")) is not None for r in tl)
    pk = "OdP" if use_od else "OP"
    qn = {"1Q": 1, "2Q": 2, "3Q": 3, "FY": 4}
    cum: dict = {}            # (年度末の年月, 四半期) → (売上の累計, 利益の累計)
    out = []
    for r in tl:
        fy = str(r.get("CurFYEn") or "")[:7]
        if len(fy) != 7:
            continue
        cum[(fy, qn[r["CurPerType"]])] = (num(r.get("Sales")), num(r.get(pk)))

        def qv(f, q):
            a = cum.get((f, q))
            if a is None:
                return None, None
            if q == 1:
                return a
            b = cum.get((f, q - 1))
            if b is None:
                return None, None
            s = a[0] - b[0] if (a[0] is not None and b[0] is not None) else None
            p = a[1] - b[1] if (a[1] is not None and b[1] is not None) else None
            return s, p

        def yoy(f, q):
            s1, p1 = qv(f, q)
            s0, p0 = qv(d_fy_back(f), q)
            gs = (s1 / s0 - 1.0) if (s1 is not None and s0 is not None and s0 > 0) else None
            gp = (p1 / p0 - 1.0) if (p1 is not None and p0 is not None and p0 > 0) else None
            return gs, gp
        lf, lq = max(cum)                                  # いちばん新しい四半期
        pf, pq = (lf, lq - 1) if lq > 1 else (d_fy_back(lf), 4)
        gs1, gp1 = yoy(lf, lq)
        gs0, gp0 = yoy(pf, pq)
        duke = all(x is not None for x in (gs1, gp1, gs0, gp0)) and gs1 >= 0.10 and gp1 >= 0.20 and gs0 >= 0.10 and gp0 >= 0.20
        fys = sorted(f for (f, q) in cum if q == 4)
        can = False
        if gp1 is not None and gp1 >= 0.40 and fys:
            f0 = fys[-1]
            f3 = f"{int(f0[:4]) - 3}-{f0[5:7]}"
            a0, a3 = cum.get((f0, 4)), cum.get((f3, 4))
            if a0 and a3 and a0[1] is not None and a3[1] is not None and a3[1] > 0 and a0[1] > 0:
                can = (a0[1] / a3[1]) ** (1 / 3) - 1.0 >= 0.25
        out.append((str(r.get("DiscDate"))[:10], duke, can, gp1 if gp1 is not None else -9.0))
    return out


def d_sim(cands: dict, D: dict, divpay: dict, t0: int, cfg: dict, last_t: np.ndarray, LIMU: np.ndarray, LIMD: np.ndarray,
          MA: np.ndarray, cap: float = CAP, slip: float = SLIP, tax: float = TAX):
    """新高値ブレイクの運用。cands … {合図の日: [(銘柄, ブレイクした高値), …]（優先の順）}。合図はその日の終値で決め、次の営業日の終値で売り買いする。
    ストップ高で引けた日は買えず（合図は消える）、ストップ安で引けた日は売れない（次の日に持ち越す）。"""
    CF, C, me_idx = D["CF"], D["C"], D["me_idx"]
    T = CF.shape[1]
    slots, pyr = cfg["slots"], cfg["pyr"]
    me_set = set(me_idx)
    cash, loss = float(cap), 0.0
    pos: dict = {}
    pend_buy: list = []
    pend_sell: dict = {}
    vals = []
    st = {"n_new": 0, "n_add": 0, "buy_t": [], "closed": [], "skip_lim": 0, "held": [], "cash_share": [], "delist": 0,
          "why": {"損切り": 0, "箱に戻った": 0, "取得単価": 0, "50日線": 0, "上場をやめた": 0}}

    def px(i, t):
        v = CF[i, t]
        return float(v) if (v == v and v > 0) else 0.0

    def realize(g):
        nonlocal cash, loss
        if g > 0:
            tx = max(0.0, g - loss) * tax
            loss = max(0.0, loss - g)
            cash -= tx
        else:
            loss += -g

    def close_pos(i, t, price, why):
        nonlocal cash
        p = pos.pop(i)
        eff = price * (1.0 - slip)
        cash += p["sh"] * eff
        realize(p["sh"] * eff - p["cost"])
        st["closed"].append((p["first_t"], t, p["sh"] * eff / p["cost"] - 1.0, why))
        st["why"][why] += 1

    for t in range(t0, T):
        for i in [k for k in pos if last_t[k] < t - 5]:
            close_pos(i, t, px(i, last_t[i]), "上場をやめた")
            pend_sell.pop(i, None)
            st["delist"] += 1
        for i in sorted(pend_sell):
            v = C[i, t]
            if i not in pos:
                pend_sell.pop(i)
            elif v == v and v > 0 and not LIMD[i, t]:
                close_pos(i, t, float(v), pend_sell.pop(i))
        if pend_buy:
            value = cash + sum(p["sh"] * px(k, t) for k, p in pos.items())
            for i, kind, lvl in pend_buy:
                v = C[i, t]
                if not (v == v and v > 0):
                    continue
                if LIMU[i, t]:
                    if kind == "new":
                        st["skip_lim"] += 1
                    continue
                if kind == "new":
                    if i in pos or len(pos) >= slots:
                        continue
                    budget = value / slots
                    amt = budget / 5.0 if pyr else budget
                else:
                    p = pos.get(i)
                    if p is None or p["k"] >= 5:
                        continue
                    amt = p["budget"] / 5.0
                pb = float(v) * (1.0 + slip)
                sh = int(min(amt, cash) / pb)
                if sh < 1:
                    continue
                cash -= sh * pb
                if kind == "new":
                    pos[i] = {"sh": sh, "cost": sh * pb, "budget": budget, "k": 1, "entry": float(v), "last_px": float(v),
                              "level": lvl, "be": False, "first_t": t}
                    st["n_new"] += 1
                    st["buy_t"].append(t)
                else:
                    p["sh"] += sh
                    p["cost"] += sh * pb
                    p["k"] += 1
                    p["last_px"] = float(v)
                    st["n_add"] += 1
            pend_buy = []
        for i, amt in divpay.get(t, ()):
            if i in pos:
                g = pos[i]["sh"] * amt
                cash += g * (1.0 - tax)
        if t in me_set:
            value = cash + sum(p["sh"] * px(k, t) for k, p in pos.items())
            vals.append(value)
            st["held"].append(len(pos))
            st["cash_share"].append(cash / value if value > 0 else 0.0)
        if t >= T - 1:
            break
        for i, p in pos.items():
            v = C[i, t]
            if not (v == v and v > 0) or i in pend_sell:
                continue
            avg = p["cost"] / p["sh"]
            if v >= p["entry"] * (1.0 + cfg.get("be", D_BE)):
                p["be"] = True
            if not p["be"]:
                if v <= avg * (1.0 - cfg.get("stop", D_STOP)):        # 27段目は cfg で損切りの幅を変える（なければ22段目の −10％）
                    pend_sell[i] = "損切り"
                    continue
                if v < p["level"] and not cfg.get("nobox"):
                    pend_sell[i] = "箱に戻った"
                    continue
            else:
                if v <= avg:
                    pend_sell[i] = "取得単価"
                    continue
                m = MA[i, t]
                if m == m and v < m:
                    pend_sell[i] = "50日線"
                    continue
            if pyr and p["k"] < 5 and v >= p["last_px"] * (1.0 + D_ADD_STEP):
                pend_buy.append((i, "add", 0.0))
        for i, lvl in cands.get(t, ()):
            if i not in pos and i not in pend_sell:
                pend_buy.append((i, "new", lvl))
    liq, gain = cash, 0.0
    for k, p in pos.items():
        eff = px(k, T - 1) * (1.0 - slip)
        liq += p["sh"] * eff
        gain += p["sh"] * eff - p["cost"]
    liq -= tax * max(0.0, gain - loss)
    st["open"] = [(p["first_t"], px(k, T - 1) * p["sh"] / p["cost"] - 1.0) for k, p in pos.items()]
    return np.array(vals), liq, st


def lab22(store: dict) -> int:
    """22段目：DUKE。（新高値ブレイク投資術）と CAN SLIM の日本版。計画は wide_lab_plan22_2026-11-09。"""
    log("22段目（DUKE。・CAN SLIM）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, MC, me_idx, cal, codes = D["dates"], D["UNI"], D["MCAP"], D["me_idx"], D["cal"], D["codes"]
    C, CF, SEG = D["C"], D["CF"], D["SEG"]
    S, T = C.shape
    j0 = next((j for j, t in enumerate(me_idx) if t >= D_HI_WIN + 5), None)
    if j0 is None:
        print("\n■ 株価の期間が短すぎます")
        return 2
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    E = ev_extract(store, D)
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    traded = np.isfinite(C) & (C > 0)
    last_t = np.array([int(np.nonzero(traded[i])[0][-1]) if traded[i].any() else -1 for i in range(S)])
    jm = np.searchsorted(np.array(me_idx), np.arange(T), side="left") - 1
    LIMU, LIMD = D.get("LIMU"), D.get("LIMD")
    if LIMU is None:
        LIMU, LIMD = limit_masks_close(store, codes, cal, C, CF)
    log("  新高値・業績の条件を作っています…")
    sig, prev_hi = d_breakouts(C, CF, SEG)
    with np.errstate(invalid="ignore"):
        MA = pd.DataFrame(CF.T).rolling(D_MA_TRAIL, min_periods=D_MA_TRAIL).mean().to_numpy().T
    tp = np.asarray(D["tp"]["c"], dtype=float)
    tma = pd.Series(tp).rolling(D_MKT_MA).mean().to_numpy()
    # 業績の条件（開示の翌日から使う：開示日がその日より前のもの）
    fst = {}
    for i, c in enumerate(codes):
        s_ = d_fin_status(store["fins"].get(c))
        if s_:
            fst[i] = ([x[0] for x in s_], s_)
    calS = [str(pd.Timestamp(d).date()) for d in cal]
    uni_s = np.zeros((S, len(me_idx)), dtype=bool)
    for j in range(len(me_idx)):
        uni_s[:, j] = UNI[:, j] & np.isfinite(MC[:, j]) & (MC[:, j] < D_MCAP_MAX)
    base: dict = {}        # 合図の日 → [(銘柄, 高値, DUKE, CAN SLIM, 伸び, ブレイクの強さ)]
    n_sig = 0
    for t in range(t0, T - 1):
        j = jm[t]
        if j < 0:
            continue
        ids = np.nonzero(sig[:, t] & uni_s[:, j])[0]
        if not len(ids):
            continue
        lst = []
        for i in ids:
            f = fst.get(int(i))
            du = ca = False
            g = -9.0
            if f:
                k = bisect.bisect_left(f[0], calS[t]) - 1
                if k >= 0:
                    _d, du, ca, g = f[1][k]
            lst.append((int(i), float(prev_hi[i, t]), du, ca, g, float(C[i, t] / prev_hi[i, t] - 1.0)))
        base[t] = lst
        n_sig += len(lst)

    def pool_ra(t):
        j = jm[t]
        return np.nonzero(uni_s[:, j] & traded[:, t])[0] if j >= 0 else np.zeros(0, dtype=int)

    print("\n" + "=" * 100)
    print("【日本株の手法・22段目（DUKE。新高値ブレイク投資術〈2016年7月〉と CAN SLIM の日本版）（wide_lab）：時価総額500億円未満】")
    print("  （基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan22_2026-11-09）")
    print(f"  比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev) - 1}か月）。前半＝〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半＝{SPLIT.date()}〜")
    print(f"  費用と税 {COST}：株 片道{SLIP * 100:.1f}％・{TAX_TXT}（損は同じ運用の中で繰り越す）。元本{CAP / 1e4:,.0f}万円。合図の次の営業日の終値で売り買い"
          + ("（始値・高値・安値つき）" if D.get("LIMU") is not None else "（9年分は始値がないので終値。ストップ高は終値の動きから見つける）"))
    nd = sum(1 for v in base.values() for x in v if x[2])
    nc = sum(1 for v in base.values() for x in v if x[3])
    print(f"\n  0. 合図：新高値ブレイク（過去1年で3回目まで・範囲の中）{n_sig}回（1年あたり {n_sig / yrs:.0f}回）・うち DUKE の業績の条件 {nd}回（{nd / yrs:.0f}回／年）・"
          f"CAN SLIM の条件 {nc}回（{nc / yrs:.0f}回／年）・ストップ高で引けた日 {int(LIMU[:, t0:].sum())}（銘柄×日）")
    res = {}
    for key, lab, cfg in D_FORMS:
        if ONLY and key not in ONLY:
            continue
        log(f"  {key} を回しています（でたらめ 20回 × 2）…")
        cands: dict = {}
        for t, lst in base.items():
            if cfg["mkt"] and not (tp[t] == tp[t] and tma[t] == tma[t] and tp[t] >= tma[t]):
                continue
            if cfg["filt"] == "duke":
                sel = sorted([x for x in lst if x[2]], key=lambda x: -x[4])
            elif cfg["filt"] == "canslim":
                sel = sorted([x for x in lst if x[3]], key=lambda x: -x[4])
            else:
                sel = sorted(lst, key=lambda x: -x[5])
            if sel:
                cands[t] = [(x[0], x[1]) for x in sel]
        v, liq, s_ = d_sim(cands, D, E["divpay"], t0, cfg, last_t, LIMU, LIMD, MA)
        ra, rb, fill = [], [], []
        n_c = max(1, sum(len(cl) for cl in cands.values()))
        for nm, seed, out_ in (("ra", 7100, ra), ("rb", 8100, rb)):
            pools = {}
            for t in cands:
                if nm == "ra":
                    pools[t] = pool_ra(t)
                else:
                    pools[t] = np.array([x[0] for x in base[t]], dtype=int)
            fill.append(sum(min(len(cl), len(pools[t])) for t, cl in cands.items()) / n_c)
            for k in range(N_RAND):
                rg = np.random.default_rng(seed + k)
                pl = {}
                for t, cl in cands.items():
                    pool = pools[t]
                    if len(pool):
                        pick = rg.choice(pool, size=min(len(cl), len(pool)), replace=False)
                        # 範囲からのでたらめはブレイクしていないので、箱に戻ったら売る決まりは使わない（損切り・取得単価・50日線は同じ）
                        pl[t] = [(int(i), float(prev_hi[i, t]) if nm == "rb" else 0.0) for i in pick]
                pv, _l, _s = d_sim(pl, D, E["divpay"], t0, cfg, last_t, LIMU, LIMD, MA)
                out_.append(half_stats(pv, ev))
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
              "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        x = {"st": half_stats(v, ev), "liq": liq, "s": s_, "ra": ra, "rb": rb, "nb": nb, "fill_ra": fill[0], "fill_rb": fill[1]}
        x["ci"] = boot_ci(v, TXv, liq, TXliq, yrs)      # 30段目：ぶれの幅
        x["verdict"], x["rb_lab"], x["det"] = d_judge(x, ST_TX, cfg["filt"] == "none")
        res[key] = (lab, x)
    print(f"\n  1. 結果（年率％「前半 ／ 後半」。TOPIX（税あり）{ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}。"
          "差は「TOPIX との差／でたらめ（範囲から）との差・上回った回数／でたらめ（同じ日の新高値ブレイクから）との差・上回った回数」）")
    for key, (lab, x) in res.items():
        a, s_ = x["st"], x["s"]
        cl = s_["closed"]
        win = np.mean([r[2] > 0 for r in cl]) * 100 if cl else float("nan")
        gw = [r[2] for r in cl if r[2] > 0]
        gl = [r[2] for r in cl if r[2] <= 0]
        hold = [r[1] - r[0] for r in cl]
        dd = x["det"]
        print(f"\n  {key} {lab}")
        print(f"     {ci_txt(x['ci'])}")
        print(f"     年率 {a['前半']['ann']:6.1f} ／ {a['後半']['ann']:6.1f}（最大下落 {a['通し']['mdd']:.0f}％・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％）"
              f"［{dd['前半'][0]:+.1f}／{dd['前半'][1]:+.1f}・{dd['前半'][2]}／{dd['前半'][3]:+.1f}・{dd['前半'][4]} ｜ "
              f"{dd['後半'][0]:+.1f}／{dd['後半'][1]:+.1f}・{dd['後半'][2]}／{dd['後半'][3]:+.1f}・{dd['後半'][4]}］ … {x['verdict']}"
              + ("" if key == "B0" else f"・業績で絞る効き目 {x['rb_lab']}"))
        ra_m = [np.mean([y[h]["ann"] for y in x["ra"]]) for h in ("前半", "後半")]
        rb_m = [np.mean([y[h]["ann"] for y in x["rb"]]) for h in ("前半", "後半")]
        print(f"     でたらめの平均：範囲から {ra_m[0]:.1f} ／ {ra_m[1]:.1f}・同じ日の新高値ブレイクから {rb_m[0]:.1f} ／ {rb_m[1]:.1f}")
        print(f"     最初の買い {s_['n_new']}回（前半 {x['nb']['前半']}・後半 {x['nb']['後半']}・1年あたり {s_['n_new'] / yrs:.0f}回）・ストップ高で買えず {s_['skip_lim']}回・"
              f"買い増し {s_['n_add']}回・売り {len(cl)}回（勝ち {win:.0f}％・勝ちの平均 {(f'{np.mean(gw) * 100:+.1f}％') if gw else '—'}・"
              f"負けの平均 {(f'{np.mean(gl) * 100:+.1f}％') if gl else '—'}・持った日数の中央 {np.median(hold) if hold else float('nan'):.0f}営業日）")
        print("     売った理由 … " + "・".join(f"{k} {v}" for k, v in s_["why"].items())
              + f"・平均の持ち数 {np.mean(s_['held']):.1f}銘柄・現金の割合 平均 {np.mean(s_['cash_share']) * 100:.0f}％・"
              f"でたらめに候補の数だけ選べた割合 {x['fill_ra'] * 100:.0f}％／{x['fill_rb'] * 100:.0f}％")
    print("\n  2. 判定のまとめ")
    for vv in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, (_l, x) in res.items() if x["verdict"] == vv]
        print(f"     {vv} … " + ("・".join(ks) if ks else "なし"))
    for vv in ("ある", "見えない", "裏目"):
        ks = [k for k, (_l, x) in res.items() if x["rb_lab"] == vv and k != "B0"]
        print(f"     業績で絞る効き目 {vv} … " + ("・".join(ks) if ks else "なし"))
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")
    return 0


# ───────────────────────── 23段目：成長×割安の手法（弐億貯男・井村俊哉・PEG）と、みきまる・清原達郎の近い形 ─────────────────────────
# 計画は wide_lab_plan23_2026-11-09（結果を見る前に決めたもの）。第1部と同じ月末の運用の形（20銘柄・売る決まり A・B・でたらめな20銘柄）で、
# 条件に合う銘柄だけに点数をつける。条件に合わなくなった銘柄は、その月末に売る。
G_TYPES = (
    ("NIOKU", "弐億貯男：2期続けて増収増益・PER15倍以下（実績か予想の低いほう）・配当性向30％以上・配当利回り3％以上・時価総額1,000億円未満（PER の低い順）"),
    ("IMURA", "井村俊哉：通期の売上+30％以上・営業利益+30％以上（粗利の代わり）・PER が売上の伸び率（％）より小さい（伸び率÷PER の大きい順）"),
    ("IMURA_S", "井村俊哉（売上だけ）：通期の売上+30％以上・PER が売上の伸び率（％）より小さい（伸び率÷PER の大きい順）"),
    ("PEG", "リンチの PEG：予想 PER ÷ 会社予想の利益の伸び率（％）が1未満（PEG の小さい順）"),
    ("MIKI", "みきまる：PER×PBR が11.25未満・配当利回り3％以上（優待は入れない）・時価総額300億円以下（PER×PBR の小さい順）"),
    ("KIYO", "清原達郎（近い形）：（現金同等物 −（総資産 − 純資産））÷ 時価総額が1以上・黒字・時価総額500億円未満・金融を除く（比率の大きい順）"),
)


def g_panel(store: dict, D: dict) -> dict:
    """月末ごとの基礎の値（行＝銘柄・列＝月末）：実績と予想の PER・PBR・配当利回り・配当性向・通期の売上と利益の伸び・
    会社予想の利益の伸び・現金から負債を引いた額。開示は月末の日より前のものだけ（build と同じ。売買が長く途切れたあとは、そのあとの開示だけ）。"""
    codes, me_idx, MC, CF, SEG, seg_first = D["codes"], D["me_idx"], D["MCAP"], D["CF"], D["SEG"], D["seg_first"]
    T = len(D["cal"])
    cal_i = D["cal"].astype("datetime64[D]")
    S, M = len(codes), len(me_idx)
    me_np = np.array([np.datetime64(d.date()) for d in D["dates"]])
    keys = ("per_a", "per_f", "pbr", "dy", "payout", "gs1", "gp1", "gs0", "gp0", "gop1", "ge_f", "netcash", "np_a",
            "np_f", "eq", "eq_up3", "div_nocut3", "div_up3", "stable3",      # この6つは24段目（ヘムの増配株）で足したもの
            "eqr", "opm", "roa", "psr",                                       # この4つは25段目（たーちゃん）で足したもの
            "up3",                                                             # 26段目（なのなの）：3期で売上・営業利益が増え、3期とも黒字
            "sal_up2", "np_up2", "cash_up", "cash_cov", "div_cut2",            # 32段目（バク）：売上・純利益が2期続けて増加・現金の増加と配当の何年分か・2期続けて減配
            "sal_cagr", "op_cagr", "opm_up", "np_g1",                          # 33段目（児玉）：売上・営業利益の年平均の伸び（最大5年）・営業利益率の改善・純利益の伸び
            "op_avg3",                                                         # 34段目（山口）：営業利益の前々期・前期・今期予想の平均
            "gs_f", "roe", "roic", "sal_cagr3")                                # 35段目（機関投資家の手口）：売上の伸びの予想・ROE・ROIC の近い値・売上の3年の年平均の伸び
    P = {k: np.full((S, M), np.nan) for k in keys}
    for i, c in enumerate(codes):
        tl = fin_timeline(store["fins"].get(c))
        if not tl:
            continue
        b = store["bars"][c]
        fd, ff = b["fd"], b["ff"]
        dd = np.array([np.datetime64(str(r["DiscDate"])[:10]) for r in tl])
        dpost = div_post_map(c, fd, ff, tl)
        fys_all = [r for r in tl if is_fy(r)]
        use_od = bool(fys_all) and all(num(r.get("OdP")) is not None for r in fys_all)
        pk = "OdP" if use_od else "OP"          # 利益は経常利益（経常利益のない会社は、すべての期を営業利益で。22段目と同じ）
        for j in range(M):
            mc = MC[i, j]
            px = CF[i, me_idx[j]]
            if not (mc == mc and mc > 0 and px == px and px > 0):
                continue
            k = int(np.searchsorted(dd, cal_i[me_idx[j]], side="left")) - 1      # 月末の最後の営業日より前の開示だけ（build と同じ）
            if k < 0:
                continue
            sg = int(SEG[i, me_idx[j]])
            k_lo = int(np.searchsorted(dd, cal_i[sg], side="left")) if (sg < T and sg > seg_first[i]) else 0
            if k_lo > k:
                continue
            sub = tl[k_lo:k + 1]
            by_end = {}
            for r in sub:
                if is_fy(r):
                    by_end[str(r.get("CurFYEn") or "")[:10]] = r
            fy = [by_end[e] for e in sorted(by_end)]
            if not fy:
                continue
            r = fy[-1]
            fy_end = str(r.get("CurFYEn") or "")[:10]
            nc = "NonConsolidated" in str(r.get("DocType") or "")      # 単体だけの会社は、単体の予想も見る（build と同じ）
            np_a = num(r.get("NP"))
            P["np_a"][i, j] = np_a if np_a is not None else np.nan
            if np_a and np_a > 0:
                P["per_a"][i, j] = mc / np_a
            # 会社予想の当期の純利益：いちばん新しい本決算の短信の「来期」の予想。そのあとに当期の四半期の短信が出ていれば、その「当期」の予想
            np_f, dps_f, dps_f_d = None, None, None
            op_f = None
            sal_f = None
            for q in reversed(sub):
                if is_fy(q):
                    if np_f is None and str(q.get("CurFYEn") or "")[:10] == fy_end:
                        np_f = first_num(q, ("NxFNp", "NxFNP") + (("NxFNCNP", "NxFNCNp") if nc else ()))
                    if op_f is None and str(q.get("CurFYEn") or "")[:10] == fy_end:
                        op_f = num(q.get("NxFOP"))
                    if sal_f is None and str(q.get("CurFYEn") or "")[:10] == fy_end:
                        sal_f = num(q.get("NxFSales"))
                    break
                if str(q.get("CurFYEn") or "")[:10] > fy_end:
                    if op_f is None:
                        op_f = num(q.get("FOP"))
                    if sal_f is None:
                        sal_f = num(q.get("FSales"))
                    if np_f is None:
                        np_f = first_num(q, ("FNP",) + (("FNCNP",) if nc else ()))
                    if dps_f is None and num(q.get("FDivAnn")) is not None:
                        dps_f, dps_f_d = num(q.get("FDivAnn")), str(q["DiscDate"])[:10]
            if np_f is not None:
                P["np_f"][i, j] = np_f
            # 34段目：営業利益の前々期・前期の実績と今期の予想の平均（そろわなければ、あるものの平均。2つ以上あるとき）
            ops = [num(x.get("OP")) for x in fy[-2:]] + [op_f]
            ops = [v for v in ops if v is not None]
            if len(ops) >= 2:
                P["op_avg3"][i, j] = float(np.mean(ops))
            if np_f and np_f > 0:
                P["per_f"][i, j] = mc / np_f
            if np_f is not None and np_a and np_a > 0:
                P["ge_f"][i, j] = np_f / np_a - 1.0
            eq_r = next((q for q in reversed(sub) if num(q.get("Eq")) is not None), None)
            eq = num(eq_r.get("Eq")) if eq_r else None
            if eq and eq > 0:
                P["pbr"][i, j] = mc / eq
                P["eq"][i, j] = eq
            ta_r = next((q for q in reversed(sub) if num(q.get("TA")) is not None), None)
            cash_r = next((q for q in reversed(sub) if num(q.get("CashEq")) is not None), None)
            if eq is not None and ta_r is not None and cash_r is not None:
                P["netcash"][i, j] = (num(cash_r.get("CashEq")) - (num(ta_r.get("TA")) - eq)) / mc
            # 25段目：自己資本比率（純資産 ÷ 総資産）・営業利益率・ROA（通期の純利益 ÷ 総資産）・PSR（時価総額 ÷ 通期の売上）
            ta = num(ta_r.get("TA")) if ta_r is not None else None
            if eq is not None and ta and ta > 0:
                P["eqr"][i, j] = eq / ta
                if np_a is not None:
                    P["roa"][i, j] = np_a / ta
            s_l, o_l = num(r.get("Sales")), num(r.get("OP"))
            # 35段目：売上の伸びの予想（今期の会社予想 ÷ いちばん新しい通期の実績 − 1）・ROE（通期の純利益 ÷ 純資産）・
            #         ROIC の近い値（営業利益 ×（1 − 30％）÷（総資産 − 現預金）。有利子負債が要約にないので、投下資本を大きめに見る＝低めに出る）
            if sal_f is not None and s_l and s_l > 0:
                P["gs_f"][i, j] = sal_f / s_l - 1.0
            if np_a is not None and eq and eq > 0:
                P["roe"][i, j] = np_a / eq
            if o_l is not None and ta and cash_r is not None and ta - num(cash_r.get("CashEq")) > 0:
                P["roic"][i, j] = o_l * 0.7 / (ta - num(cash_r.get("CashEq")))
            if s_l and s_l > 0:
                P["psr"][i, j] = mc / s_l
                if o_l is not None:
                    P["opm"][i, j] = o_l / s_l
            # 配当利回り：当期の四半期の短信の予想の年間配当。まだなければ、いちばん新しい通期の実績（来期の配当の予想はキャッシュにない）
            ann = num(r.get("DivAnn"))
            if dps_f is not None:
                P["dy"][i, j] = dps_f * fac_after(fd, ff, dps_f_d) / float(px)
            elif ann is not None:
                P["dy"][i, j] = dps_now(fd, ff, r, ann, fy_end, dpost.get(fy_end, False)) / float(px)
            # 配当性向：いちばん新しい通期の実績の年間配当 ÷ 1株利益（1株利益がなければ、純利益 ÷ 自己株を除く株数）
            eps = num(r.get("EPS"))
            shr = (num(r.get("ShOutFY")) or 0.0) - (num(r.get("TrShFY")) or 0.0)
            if ann is not None and np_a and np_a > 0:
                # 1株利益は、分割が期末のあと開示の前なら分割後に直して出す決まり（会計基準）。配当はいまの基準にそろえてから割る
                dps_c = dps_now(fd, ff, r, ann, fy_end, dpost.get(fy_end, False))
                if eps and eps > 0:
                    P["payout"][i, j] = dps_c / (eps * fac_after(fd, ff, str(r["DiscDate"])[:10]))
                elif shr > 0:
                    P["payout"][i, j] = dps_c * (mc / float(px)) / np_a
            if len(fy) >= 2:
                s1, s0 = num(fy[-1].get("Sales")), num(fy[-2].get("Sales"))
                p1, p0 = num(fy[-1].get(pk)), num(fy[-2].get(pk))
                o1, o0 = num(fy[-1].get("OP")), num(fy[-2].get("OP"))
                if s1 is not None and s0 and s0 > 0:
                    P["gs1"][i, j] = s1 / s0 - 1.0
                if p1 is not None and p0 is not None:
                    P["gp1"][i, j] = (p1 - p0) / abs(p0) if p0 != 0 else (np.inf if p1 > 0 else np.nan)
                if o1 is not None and o0 and o0 > 0:
                    P["gop1"][i, j] = o1 / o0 - 1.0
                # 33段目：営業利益率の改善・純利益の伸び（前期比）・売上と営業利益の年平均の伸び（いちばん古くて5期前まで）
                if s1 and s0 and s1 > 0 and s0 > 0 and o1 is not None and o0 is not None:
                    P["opm_up"][i, j] = float(o1 / s1 > o0 / s0)
                n1_, n0_ = num(fy[-1].get("NP")), num(fy[-2].get("NP"))
                if n1_ is not None and n0_ and n0_ > 0:
                    P["np_g1"][i, j] = n1_ / n0_ - 1.0
                kk = min(6, len(fy))
                sa, sb = num(fy[-kk].get("Sales")), num(fy[-1].get("Sales"))
                oa, ob = num(fy[-kk].get("OP")), num(fy[-1].get("OP"))
                if kk >= 3 and sa and sb and sa > 0 and sb > 0:
                    P["sal_cagr"][i, j] = (sb / sa) ** (1.0 / (kk - 1)) - 1.0
                if kk >= 3 and oa and ob and oa > 0 and ob > 0:
                    P["op_cagr"][i, j] = (ob / oa) ** (1.0 / (kk - 1)) - 1.0
                if len(fy) >= 4:          # 35段目：売上の3年の年平均の伸び（3期前 → いちばん新しい通期）
                    s3a, s3b = num(fy[-4].get("Sales")), num(fy[-1].get("Sales"))
                    if s3a and s3b and s3a > 0 and s3b > 0:
                        P["sal_cagr3"][i, j] = (s3b / s3a) ** (1.0 / 3.0) - 1.0
                c1, c0 = num(fy[-1].get("CashEq")), num(fy[-2].get("CashEq"))
                if c1 is not None and c0 is not None:
                    P["cash_up"][i, j] = float(c1 > c0)
            # 32段目：現金が配当の何年分か（いちばん新しい通期の現金同等物 ÷（年間配当 × 自己株を除く株数）。どれも同じ短信の数字）
            c_l = num(r.get("CashEq"))
            if c_l is not None and ann and ann > 0 and shr > 0:
                P["cash_cov"][i, j] = c_l / (ann * shr)
            if len(fy) >= 3:
                # 24段目：3期の自己資本の積み上げ・減配なし・業績の安定（3期とも黒字・コロナの期も黒字・売上が2期前以上）
                e3 = [num(x.get("Eq")) for x in fy[-3:]]
                P["eq_up3"][i, j] = float(all(v is not None for v in e3) and e3[0] < e3[1] < e3[2])
                d3 = [num(x.get("DivAnn")) for x in fy[-3:]]
                if all(v is not None for v in d3):
                    d3 = [dps_now(fd, ff, x, v, str(x.get("CurFYEn") or "")[:10], dpost.get(str(x.get("CurFYEn") or "")[:10], False))
                          for v, x in zip(d3, fy[-3:])]
                    P["div_nocut3"][i, j] = float(d3[2] > 0 and d3[0] <= d3[1] * 1.0001 and d3[1] <= d3[2] * 1.0001)
                    P["div_up3"][i, j] = float(d3[2] > 0 and d3[0] <= d3[1] * 1.0001 and d3[1] <= d3[2] * 1.0001 and d3[2] > d3[0] * 1.0001)
                    P["div_cut2"][i, j] = float(d3[1] < d3[0] * 0.9999 and d3[2] < d3[1] * 0.9999)
                n3 = [num(x.get("NP")) for x in fy[-3:]]
                s3 = [num(x.get("Sales")) for x in fy[-3:]]
                P["sal_up2"][i, j] = float(all(v is not None and v > 0 for v in s3) and s3[0] < s3[1] < s3[2])
                P["np_up2"][i, j] = float(all(v is not None and v > 0 for v in n3) and n3[0] < n3[1] < n3[2])
                cov = [num(x.get("NP")) for x in fy if "2020-04-01" <= str(x.get("CurFYEn") or "")[:10] <= "2021-03-31"]
                sl = [num(x.get("Sales")) for x in (fy[-3], fy[-1])]
                P["stable3"][i, j] = float(all(v is not None and v > 0 for v in n3) and all(v is not None and v > 0 for v in cov)
                                           and sl[0] is not None and sl[1] is not None and sl[1] >= sl[0])
                op2 = [num(x.get("OP")) for x in (fy[-3], fy[-1])]
                P["up3"][i, j] = float(all(v is not None and v > 0 for v in n3) and sl[0] is not None and sl[1] is not None and sl[1] >= sl[0]
                                       and op2[0] is not None and op2[1] is not None and op2[1] >= op2[0])
                s0, sm = num(fy[-2].get("Sales")), num(fy[-3].get("Sales"))
                p0, pm = num(fy[-2].get(pk)), num(fy[-3].get(pk))
                if s0 is not None and sm and sm > 0:
                    P["gs0"][i, j] = s0 / sm - 1.0
                if p0 is not None and pm is not None:
                    P["gp0"][i, j] = (p0 - pm) / abs(pm) if pm != 0 else (np.inf if p0 > 0 else np.nan)
    return P


def g_scores(store: dict, D: dict) -> dict:
    """23段目の型の点数（条件に合う銘柄だけ。合わない銘柄は空）。"""
    P = g_panel(store, D)
    MC = D["MCAP"]
    fin = np.array([D["sector"].get(c, "") in yb.FIN_SECTORS for c in D["codes"]])[:, None]
    per = np.where(np.isfinite(P["per_f"]), P["per_f"], P["per_a"])            # 予想があれば予想、なければ実績
    per_min = np.fmin(P["per_a"], P["per_f"])                                  # 実績か予想の低いほう（弐億）
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = ((P["gs1"] > 0) & (P["gs0"] > 0) & (P["gp1"] > 0) & (P["gp0"] > 0) & (per_min > 0) & (per_min <= 15)
              & (P["payout"] >= 0.30) & (P["dy"] >= 0.03) & (MC < 1e11))
        out["NIOKU"] = np.where(ok, -per_min, np.nan)
        g = P["gs1"] * 100.0
        ok = (P["gs1"] >= 0.30) & (P["gop1"] >= 0.30) & (per > 0) & (per < g)
        out["IMURA"] = np.where(ok, g / per, np.nan)
        ok = (P["gs1"] >= 0.30) & (per > 0) & (per < g)
        out["IMURA_S"] = np.where(ok, g / per, np.nan)
        peg = P["per_f"] / (P["ge_f"] * 100.0)
        ok = (P["per_f"] > 0) & (P["ge_f"] > 0) & (peg < 1.0)
        out["PEG"] = np.where(ok, -peg, np.nan)
        mix = per * P["pbr"]
        ok = (per > 0) & (P["pbr"] > 0) & (mix < 11.25) & (P["dy"] >= 0.03) & (MC <= 3e10)
        out["MIKI"] = np.where(ok, -mix, np.nan)
        ok = (P["netcash"] >= 1.0) & (P["np_a"] > 0) & (MC < 5e10) & ~fin
        out["KIYO"] = np.where(ok, P["netcash"], np.nan)
    return out


def lab23(store: dict) -> int:
    """23段目：成長×割安の手法。第1部の lab1 の運用の形を、そのまま使う（条件に合う銘柄だけに点数）。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE
    LAB_TITLE, LAB_PLAN = "23段目（手法が公開されている投資家の型：弐億貯男・井村俊哉・PEG・みきまる・清原達郎）", "wide_lab_plan23_2026-11-09"
    LAB_NOTE = ("23段目：条件に合う銘柄だけに点数をつけ、合う銘柄が20より少ない月は、空いた分を現金で持つ。条件に合わなくなった銘柄は、その月末に売る。"
                "保留は「合格した銘柄が1か月あたり平均5未満」（点数のある割合の決まりの代わり）。決算短信の要約だけで作る（優待・粗利・流動資産は使えない）。"
                "参考の「でたらめ（同じ数）」は、毎月の合格した数だけ範囲からでたらめに選んだもの（現金の割合をそろえて、選び方だけを比べる）")
    cache: dict = {}

    def mk(name):
        def fn(D):
            if "sc" not in cache:
                log("  基礎の値（PER・PBR・配当・伸び・現金）を作っています…")
                cache["sc"] = g_scores(store, D)
            return cache["sc"][name]
        return fn
    extra = {nm: (mk(nm), lab) for nm, lab in G_TYPES}
    return lab1(store, extra=extra, filt=True)


# ───────────────────────── 24段目：ヘム『「増配」株投資』（KADOKAWA、2025年1月27日）の数値の決まり ─────────────────────────
# 計画は wide_lab_plan24_2026-11-09（結果を見る前に決めたもの）。23段目と同じ月末の運用の形（条件に合う銘柄だけに点数・20銘柄・売る決まり A・B）。
H_PUB = pd.Timestamp("2025-01-31")       # 出版（2025年1月27日）のあとの最初の月末。この次の月からを「出版後」として別に出す
H_TYPES = (
    ("H1", "基本条件：PER12倍以下・PBR1.0倍以下・予想ROE8％以上・配当利回り3％以上・配当性向40％以下・安全域2.0倍以上・時価総額1,000億円以下・"
           "自己資本が3期続けて増加・業績の安定・減配なし（安全域の大きい順）"),
    ("H2", "小型割安株ファンド：PER10倍以下・PBR0.8倍以下・予想ROE5％以上・配当利回り2.5％以上・配当性向40％以下・安全域2.0倍以上・時価総額700億円以下・業績の安定・減配なし"),
    ("H3", "増配テーマの近い形：PER13倍以下・PBR1.2倍以下・予想ROE5％以上・配当利回り2.7％以上・配当性向45％以下・安全域1.8倍以上・業績の安定・3期で増配（減配なし）"),
    ("H0", "参考：基本条件の指標だけ（自己資本・業績の安定・減配なしの条件を外す）"),
    ("H1K", "参考：基本条件で買い、売りは本の決まり（配当性向60％超・株価が簡易理論株価より上・赤字・3年の間に一度も条件に合わなくなったとき）"),
)


def h_scores(store: dict, D: dict) -> dict:
    P = g_panel(store, D)
    MC = D["MCAP"]
    per = np.where(np.isfinite(P["per_f"]), P["per_f"], P["per_a"])            # 予想があれば予想、なければ実績（23段目と同じ）
    npb = np.where(np.isfinite(P["np_f"]), P["np_f"], P["np_a"])               # 安全域・予想ROE に使う純利益（同じく予想を優先）
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        roe = npb / P["eq"]
        safe = (P["eq"] + 10.0 * npb) / MC                                         # (BPS + EPS×10) ÷ 株価 ＝ (純資産 + 純利益×10) ÷ 時価総額
        ok_eq, ok_st, ok_dn, ok_du = P["eq_up3"] == 1, P["stable3"] == 1, P["div_nocut3"] == 1, P["div_up3"] == 1

        def base(per_mx, pbr_mx, roe_mn, dy_mn, po_mx, sf_mn, mc_mx):
            return ((per > 0) & (per <= per_mx) & (P["pbr"] > 0) & (P["pbr"] <= pbr_mx) & (roe >= roe_mn) & (P["dy"] >= dy_mn)
                    & (P["payout"] > 0) & (P["payout"] <= po_mx) & (safe >= sf_mn) & (MC <= mc_mx))
        b1 = base(12, 1.0, 0.08, 0.03, 0.40, 2.0, 1e11)
        e1 = b1 & ok_eq & ok_st & ok_dn
        out["H1"] = np.where(e1, safe, np.nan)
        out["H2"] = np.where(base(10, 0.8, 0.05, 0.025, 0.40, 2.0, 7e10) & ok_st & ok_dn, safe, np.nan)
        out["H3"] = np.where(base(13, 1.2, 0.05, 0.027, 0.45, 1.8, np.inf) & ok_st & ok_du, safe, np.nan)
        out["H0"] = np.where(b1, safe, np.nan)
        # H1K：36か月以内に H1 の条件に合ったことがあり、配当性向60％以下・安全域1.0以上・黒字なら持ち続けてよい（買う順は H1 に合う銘柄が先）
        keep = (P["payout"] > 0) & (P["payout"] <= 0.60) & (safe >= 1.0) & (P["np_a"] > 0)
        recent = np.zeros_like(e1)
        last = np.full(e1.shape[0], -10 ** 6)
        for j in range(e1.shape[1]):
            last = np.where(e1[:, j], j, last)
            recent[:, j] = (j - last) <= 36
        out["H1K"] = np.where(e1, safe, np.where(recent & keep, safe - 1000.0, np.nan))
    return out


def lab24(store: dict) -> int:
    """24段目：ヘムの増配株の数値の決まり。23段目と同じ形で回し、出版のあとの期間の成績を別に出す（判定には使わない）。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE, PUB_FROM
    LAB_TITLE, LAB_PLAN = "24段目（ヘム『「増配」株投資』2025年1月の数値の決まり）", "wide_lab_plan24_2026-11-09"
    LAB_NOTE = ("24段目：23段目と同じく、条件に合う銘柄だけに点数をつけ、合う銘柄が20より少ない月は現金。条件に合わなくなった銘柄は、その月末に売る（H1K だけ本の売りの決まり）。"
                "保留は「合格した銘柄が1か月あたり平均5未満」。**前半・後半とも本の出版（2025年1月27日）より前**なので、判定は「本人が見ていた期間で効いたか」の参考。"
                "出版後（2025年2月〜）は別に出し、判定はしない。配当政策（DOE・累進）・優待・定性分析・リーマンの時期の業績は使えない")
    PUB_FROM = H_PUB
    cache: dict = {}

    def mk(name):
        def fn(D):
            if "sc" not in cache:
                log("  基礎の値（PER・PBR・ROE・安全域・配当・自己資本・業績）を作っています…")
                cache["sc"] = h_scores(store, D)
            return cache["sc"][name]
        return fn
    extra = {nm: (mk(nm), lab) for nm, lab in H_TYPES}
    rc = lab1(store, extra=extra, filt=True)
    if rc == 0 and LAST_LAB1:
        print("\n  3. 出版のあと（2025年2月〜。判定はしない。年率％） … A ／ B ／ TOPIX（税あり） ／ でたらめな20銘柄の平均 A ／ でたらめ（同じ数・A）の平均")
        for k, r in LAST_LAB1.items():
            if "出版後" not in r["A"]:
                continue
            ra = [x["出版後"]["ann"] for x in r["rnd"]["A"] if "出版後" in x]
            rn = [x["出版後"]["ann"] for x in r.get("rnd_n", []) if "出版後" in x]
            print(f"     {k:<4} {r['A']['出版後']['ann']:6.1f} ／ {r['B']['出版後']['ann']:6.1f} ／ {r['tx']['出版後']['ann']:6.1f} ／ "
                  f"{np.mean(ra) if ra else float('nan'):6.1f}（上回った回数 {sum(1 for x in ra if r['A']['出版後']['ann'] > x)}） ／ "
                  f"{np.mean(rn) if rn else float('nan'):6.1f}（{sum(1 for x in rn if r['A']['出版後']['ann'] > x)}）"
                  f" ・ 最大下落 A {r['A']['出版後']['mdd']:.0f}％")
    PUB_FROM = None
    return rc


# ───────────────────────── 25段目：たーちゃん『50万円を50億円に増やした』（ダイヤモンド社、2025年7月3日）の数値の決まり ─────────────────────────
# 計画は wide_lab_plan25_2026-11-10（結果を見る前に決めたもの）。24段目と同じ形。
T_PUB = pd.Timestamp("2025-07-31")
T_TYPES = (
    ("T1", "資産バリュー：PBR0.5倍以下・自己資本比率60％以上（PBR の低い順）"),
    ("T1P", "資産バリュー＋収益：T1 に PER12倍以下（黒字）を足す（PBR の低い順）"),
    ("T2", "収益バリュー：営業利益率10％以上・PER10倍以下・PBR1.5倍以下（PER×PBR22.5以下なら PBR は問わない）・ROA7％以上・時価総額300億円以下（PER×PBR の小さい順）"),
    ("T2L", "収益バリュー（緩めた形）：営業利益率8％以上・PER15倍以下・ほかは T2 と同じ"),
    ("T3", "シクリカル（近い形）：通期が赤字・PSR0.5倍以下・純資産がプラス・金融を除く（PSR の小さい順）"),
)


def t_scores(store: dict, D: dict) -> dict:
    P = g_panel(store, D)
    MC = D["MCAP"]
    fin = np.array([D["sector"].get(c, "") in yb.FIN_SECTORS for c in D["codes"]])[:, None]
    per = np.where(np.isfinite(P["per_f"]), P["per_f"], P["per_a"])
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        pbr = P["pbr"]
        a = (pbr > 0) & (pbr <= 0.5) & (P["eqr"] >= 0.60)
        out["T1"] = np.where(a, -pbr, np.nan)
        out["T1P"] = np.where(a & (per > 0) & (per <= 12), -pbr, np.nan)
        mix = per * pbr

        def t2(opm_mn, per_mx):
            return ((P["opm"] >= opm_mn) & (per > 0) & (per <= per_mx) & (pbr > 0) & ((pbr <= 1.5) | (mix <= 22.5))
                    & (P["roa"] >= 0.07) & (MC <= 3e10))
        out["T2"] = np.where(t2(0.10, 10), -mix, np.nan)
        out["T2L"] = np.where(t2(0.08, 15), -mix, np.nan)
        out["T3"] = np.where((P["np_a"] < 0) & (P["psr"] > 0) & (P["psr"] <= 0.5) & (pbr > 0) & ~fin, -P["psr"], np.nan)
    return out


def lab25(store: dict) -> int:
    """25段目：たーちゃんの3つの投資法の数値の決まり。24段目と同じく、出版のあとの期間の成績を別に出す（判定には使わない）。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE, PUB_FROM
    LAB_TITLE, LAB_PLAN = "25段目（たーちゃん『50万円を50億円に増やした』2025年7月の数値の決まり）", "wide_lab_plan25_2026-11-10"
    LAB_NOTE = ("25段目：24段目と同じく、条件に合う銘柄だけに点数をつけ、合う銘柄が20より少ない月は現金。条件に合わなくなった銘柄は、その月末に売る。"
                "保留は「合格した銘柄が1か月あたり平均5未満」。前半・後半とも本の出版（2025年7月3日）より前なので、判定は「本人が見ていた期間で効いたか」の参考。"
                "出版後（2025年8月〜）は別に出し、判定はしない。土地・有価証券の含み益・黒字化のシナリオ・出来高の急増での売りは使えない")
    PUB_FROM = T_PUB
    cache: dict = {}

    def mk(name):
        def fn(D):
            if "sc" not in cache:
                log("  基礎の値（PBR・自己資本比率・PER・営業利益率・ROA・PSR）を作っています…")
                cache["sc"] = t_scores(store, D)
            return cache["sc"][name]
        return fn
    rc = lab1(store, extra={nm: (mk(nm), lab) for nm, lab in T_TYPES}, filt=True)
    if rc == 0 and LAST_LAB1:
        print("\n  3. 出版のあと（2025年8月〜。判定はしない。年率％） … A ／ B ／ TOPIX（税あり） ／ でたらめな20銘柄の平均 A ／ でたらめ（同じ数・A）の平均")
        for k, r in LAST_LAB1.items():
            if "出版後" not in r["A"]:
                continue
            ra = [x["出版後"]["ann"] for x in r["rnd"]["A"] if "出版後" in x]
            rn = [x["出版後"]["ann"] for x in r.get("rnd_n", []) if "出版後" in x]
            print(f"     {k:<4} {r['A']['出版後']['ann']:6.1f} ／ {r['B']['出版後']['ann']:6.1f} ／ {r['tx']['出版後']['ann']:6.1f} ／ "
                  f"{np.mean(ra) if ra else float('nan'):6.1f}（上回った回数 {sum(1 for x in ra if r['A']['出版後']['ann'] > x)}） ／ "
                  f"{np.mean(rn) if rn else float('nan'):6.1f}（{sum(1 for x in rn if r['A']['出版後']['ann'] > x)}）"
                  f" ・ 最大下落 A {r['A']['出版後']['mdd']:.0f}％")
    PUB_FROM = None
    return rc


# ───────────────────────── 26段目：なのなの『「爆配当」株投資』（KADOKAWA、2023年12月26日）の数値の決まり ─────────────────────────
# 計画は wide_lab_plan26_2026-11-10（結果を見る前に決めたもの）。24・25段目と同じ形。
N_PUB = pd.Timestamp("2023-12-31")
N_TYPES = (
    ("N1", "買いの4条件：配当利回り4.0％以上・PER18倍以下・3期で減配なし・3期で売上と営業利益が増え3期とも黒字（利回りの高い順）"),
    ("N2", "基本のスクリーニング：配当利回り3.8％以上・PER18倍以下（利回りの高い順）"),
    ("N3", "強いスクリーニング：N2 に自己資本比率50％以上・ROE12％以上・営業利益率10％以上を足す（利回りの高い順）"),
    ("N1S", "参考：N1 で買い、売りは本の決まりに近い形（利回り2.5％未満で売る。減配して PER20倍以上になったら売る。それまでは持つ）"),
)


def n_scores(store: dict, D: dict) -> dict:
    P = g_panel(store, D)
    per = np.where(np.isfinite(P["per_f"]), P["per_f"], P["per_a"])
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        dy = P["dy"]
        roe = P["np_a"] / P["eq"]
        e1 = (dy >= 0.04) & (per > 0) & (per <= 18) & (P["div_nocut3"] == 1) & (P["up3"] == 1)
        b2 = (dy >= 0.038) & (per > 0) & (per <= 18)
        out["N1"] = np.where(e1, dy, np.nan)
        out["N2"] = np.where(b2, dy, np.nan)
        out["N3"] = np.where(b2 & (P["eqr"] >= 0.50) & (roe >= 0.12) & (P["opm"] >= 0.10), dy, np.nan)
        broke = (P["div_nocut3"] == 0) & ((per >= 20) | ~(per > 0))
        keep = (dy >= 0.025) & ~broke
        held = np.zeros_like(e1)
        last = np.full(e1.shape[0], -10 ** 6)
        for j in range(e1.shape[1]):
            last = np.where(e1[:, j], j, last)
            held[:, j] = (j - last) <= 36
        out["N1S"] = np.where(e1, dy, np.where(held & keep, dy - 1.0, np.nan))
    return out


def lab26(store: dict) -> int:
    """26段目：なのなのの高配当株の数値の決まり。出版のあとの期間（2024年1月〜）の成績を別に出す（判定には使わない）。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE, PUB_FROM
    LAB_TITLE, LAB_PLAN = "26段目（なのなの『「爆配当」株投資』2023年12月の数値の決まり）", "wide_lab_plan26_2026-11-10"
    LAB_NOTE = ("26段目：24・25段目と同じく、条件に合う銘柄だけに点数をつけ、合う銘柄が20より少ない月は現金。条件に合わなくなった銘柄は、その月末に売る（N1S だけ本の売りの決まりに近い形）。"
                "保留は「合格した銘柄が1か月あたり平均5未満」。前半（〜2022年10月）は出版より前。後半（2022年11月〜）のうち2024年1月からは出版のあと。"
                "出版後（2024年1月〜）は別にも出す（判定はしない）。外国人持株比率・優待・定性の判断は使わない")
    PUB_FROM = N_PUB
    cache: dict = {}

    def mk(name):
        def fn(D):
            if "sc" not in cache:
                log("  基礎の値（配当利回り・PER・減配・業績・自己資本比率・ROE・営業利益率）を作っています…")
                cache["sc"] = n_scores(store, D)
            return cache["sc"][name]
        return fn
    rc = lab1(store, extra={nm: (mk(nm), lab) for nm, lab in N_TYPES}, filt=True)
    if rc == 0 and LAST_LAB1:
        print("\n  3. 出版のあと（2024年1月〜。判定はしない。年率％） … A ／ B ／ TOPIX（税あり） ／ でたらめな20銘柄の平均 A ／ でたらめ（同じ数・A）の平均")
        for k, r in LAST_LAB1.items():
            if "出版後" not in r["A"]:
                continue
            ra = [x["出版後"]["ann"] for x in r["rnd"]["A"] if "出版後" in x]
            rn = [x["出版後"]["ann"] for x in r.get("rnd_n", []) if "出版後" in x]
            print(f"     {k:<4} {r['A']['出版後']['ann']:6.1f} ／ {r['B']['出版後']['ann']:6.1f} ／ {r['tx']['出版後']['ann']:6.1f} ／ "
                  f"{np.mean(ra) if ra else float('nan'):6.1f}（上回った回数 {sum(1 for x in ra if r['A']['出版後']['ann'] > x)}） ／ "
                  f"{np.mean(rn) if rn else float('nan'):6.1f}（{sum(1 for x in rn if r['A']['出版後']['ann'] > x)}）"
                  f" ・ 最大下落 A {r['A']['出版後']['mdd']:.0f}％")
    PUB_FROM = None
    return rc


# ───────────────────────── 32段目：投資家バク『バクでも稼げる高配当・増配株投資』（ぱる出版、2023年6月）の数値の決まり ─────────────────────────
# 計画は wide_lab_plan32_2026-11-10（結果を見る前に決めたもの）。24〜26段目と同じ月末の運用の形（条件に合う銘柄だけに点数・20銘柄・売る決まり A・B）。
B_PUB = pd.Timestamp("2023-06-30")      # 電子書籍の制作日（2023年6月1日）のあとの最初の月末。この次の月からを「出版後」として別に出す
B_WIN, B_MINOBS = 36, 24                 # ⑨⑩：その銘柄の過去3年（36か月）の中での位置。24か月以上あるときだけ
B_TYPES = (
    ("K1", "秘伝の法則（数値にできる①〜⑩）：売上とEPS（純利益）が2期続けて増加・営業利益率8％以上・ROE8％以上・3期で増配・配当性向30〜50％・"
           "自己資本比率40％以上（金融は問わない）・現金が前期より増え配当の2年分以上・配当利回り4％以上・利回りが過去3年の上から2割以内・PER が過去3年の下半分（利回りの高い位置の順）"),
    ("K2", "K1 から割安のタイミング（⑨⑩：過去3年の中の位置）を外す（配当利回りの高い順）"),
    ("K3", "K1 の配当利回りの下限を4％から3％にする（利回りの高い位置の順）"),
    ("K4", "参考：業績と財務の条件だけ（①〜⑧。配当利回りとタイミングを外す。配当利回りの高い順）"),
    ("K1S", "参考：K1 で買い、売りは本の決まりに近い形（2期続けて減配したら売る。それまでは持つ）"),
)


def own_pct(X: np.ndarray, win: int = B_WIN, min_obs: int = B_MINOBS) -> np.ndarray:
    """月末ごとに、その銘柄の過去 win か月（その月を含む）の値の中で、いまの値以下の割合（0〜1）。値が min_obs 個未満なら空。"""
    S, M = X.shape
    out = np.full((S, M), np.nan)
    for j in range(M):
        w = X[:, max(0, j - win + 1):j + 1]
        x = X[:, j]
        ok = np.isfinite(w)
        n = ok.sum(axis=1)
        with np.errstate(invalid="ignore"):
            le = (np.where(ok, w, np.inf) <= x[:, None]).sum(axis=1)
        out[:, j] = np.where((n >= min_obs) & np.isfinite(x), le / np.maximum(n, 1), np.nan)
    return out


def b_scores(store: dict, D: dict) -> dict:
    P = g_panel(store, D)
    fin = np.array([D["sector"].get(c, "") in yb.FIN_SECTORS for c in D["codes"]])[:, None]
    per = np.where(np.isfinite(P["per_f"]), P["per_f"], P["per_a"])
    dy = P["dy"]
    dy_p = own_pct(dy)
    per_p = own_pct(np.where(per > 0, per, np.nan))
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        roe = P["np_a"] / P["eq"]
        perf = ((P["sal_up2"] == 1) & (P["np_up2"] == 1) & (P["opm"] >= 0.08) & (roe >= 0.08) & (P["div_up3"] == 1)
                & (P["payout"] >= 0.30) & (P["payout"] <= 0.50) & ((P["eqr"] >= 0.40) | fin) & (P["cash_up"] == 1) & (P["cash_cov"] >= 2.0))
        timing = (dy_p >= 0.80) & (per_p <= 0.50)
        k1 = perf & (dy >= 0.04) & timing
        out["K1"] = np.where(k1, dy_p, np.nan)
        out["K2"] = np.where(perf & (dy >= 0.04), dy, np.nan)
        out["K3"] = np.where(perf & (dy >= 0.03) & timing, dy_p, np.nan)
        out["K4"] = np.where(perf, dy, np.nan)
        held = np.zeros_like(k1)
        last = np.full(k1.shape[0], -10 ** 6)
        for j in range(k1.shape[1]):
            last = np.where(k1[:, j], j, last)
            held[:, j] = (j - last) <= 36
        keep = (P["div_cut2"] != 1) & np.isfinite(dy)
        out["K1S"] = np.where(k1, dy_p, np.where(held & keep, dy_p - 1.0, np.nan))
    return out


def lab32(store: dict) -> int:
    """32段目：投資家バクの高配当・増配株の数値の決まり。出版のあとの期間（2023年7月〜）の成績を別に出す（判定には使わない）。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE, PUB_FROM
    LAB_TITLE, LAB_PLAN = "32段目（投資家バク『バクでも稼げる高配当・増配株投資』2023年6月の数値の決まり）", "wide_lab_plan32_2026-11-10"
    LAB_NOTE = ("32段目：24〜26段目と同じく、条件に合う銘柄だけに点数をつけ、合う銘柄が20より少ない月は現金。条件に合わなくなった銘柄は、その月末に売る（K1S だけ本の売りの決まりに近い形）。"
                "保留は「合格した銘柄が1か月あたり平均5未満」。前半（〜2022年10月）は出版より前。後半（2022年11月〜）のうち2023年7月からは出版のあと（40か月）。"
                "出版後（2023年7月〜）は別にも出す（判定はしない）。10年の業績・強み・将来性・業種の偏り（⑪⑫）は使わない（データが9年分で3期ずつ）")
    PUB_FROM = B_PUB
    cache: dict = {}

    def mk(name):
        def fn(D):
            if "sc" not in cache:
                log("  基礎の値（売上・EPS・営業利益率・ROE・配当・配当性向・自己資本比率・現金・過去3年の中の位置）を作っています…")
                cache["sc"] = b_scores(store, D)
            return cache["sc"][name]
        return fn
    rc = lab1(store, extra={nm: (mk(nm), lab) for nm, lab in B_TYPES}, filt=True)
    if rc == 0 and LAST_LAB1:
        print("\n  3. 出版のあと（2023年7月〜。判定はしない。年率％） … A ／ B ／ TOPIX（税あり） ／ でたらめな20銘柄の平均 A ／ でたらめ（同じ数・A）の平均")
        for k, r in LAST_LAB1.items():
            if "出版後" not in r["A"]:
                continue
            ra = [x["出版後"]["ann"] for x in r["rnd"]["A"] if "出版後" in x]
            rn = [x["出版後"]["ann"] for x in r.get("rnd_n", []) if "出版後" in x]
            print(f"     {k:<4} {r['A']['出版後']['ann']:6.1f} ／ {r['B']['出版後']['ann']:6.1f} ／ {r['tx']['出版後']['ann']:6.1f} ／ "
                  f"{np.mean(ra) if ra else float('nan'):6.1f}（上回った回数 {sum(1 for x in ra if r['A']['出版後']['ann'] > x)}） ／ "
                  f"{np.mean(rn) if rn else float('nan'):6.1f}（{sum(1 for x in rn if r['A']['出版後']['ann'] > x)}）"
                  f" ・ 最大下落 A {r['A']['出版後']['mdd']:.0f}％")
    PUB_FROM = None
    return rc


# ───────────────────────── 33段目：児玉一希『高配当10倍株投資』（KADOKAWA、2024年3月22日）の数値の決まり ─────────────────────────
# 計画は wide_lab_plan33_2026-11-10（結果を見る前に決めたもの）。24〜26・32段目と同じ月末の運用の形（条件に合う銘柄だけに点数・20銘柄・売る決まり A・B）。
J_PUB = pd.Timestamp("2024-03-31")      # 出版（2024年3月22日）のあとの最初の月末。この次の月からを「出版後」として別に出す
J_ADR = 80.0                             # G2：その月のうちに騰落レシオ（25営業日・一部とプライム）が80以下になった月だけ入れ替える
J_TYPES = (
    ("G1", "スクリーニング4STEP＋α：時価総額500億円以上・自己資本比率40％以上・売上と営業利益の年平均の伸び（最大5年）がどちらも3％以上・配当性向30％未満（配当あり）・"
           "PER が過去3年の下半分（売上の伸びの大きい順）"),
    ("G2", "G1 を、騰落レシオ80以下になった月だけ入れ替える（暴落の買いのタイミング）"),
    ("G3", "G1 に成功率を上げるポイント（売上が2期続けて増加・純利益が前期比+10％以上・営業利益率が前期より改善・直近1年の安値がその前の1年の安値より高い）を足す"),
    ("G4", "参考：G1 から PER の条件（α）を外す"),
    ("G5", "参考：G1 に配当利回り3％以上を足す"),
)


def j_scores(store: dict, D: dict) -> dict:
    P = g_panel(store, D)
    MC, CF, me_idx = D["MCAP"], D["CF"], D["me_idx"]
    per = np.where(np.isfinite(P["per_f"]), P["per_f"], P["per_a"])
    per_p = own_pct(np.where(per > 0, per, np.nan))
    S, M = MC.shape
    hl = np.zeros((S, M), dtype=bool)          # 直近250営業日の安値 > その前の250営業日の安値（下値の切り上がり）
    for j, t in enumerate(me_idx):
        if t >= 500:
            with np.errstate(invalid="ignore"):
                hl[:, j] = np.nanmin(CF[:, t - 249:t + 1], axis=1) > np.nanmin(CF[:, t - 499:t - 249], axis=1)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = ((MC >= 5e10) & (P["eqr"] >= 0.40) & (P["sal_cagr"] >= 0.03) & (P["op_cagr"] >= 0.03)
              & (P["payout"] > 0) & (P["payout"] < 0.30) & (per > 0) & (per_p <= 0.50))
        g4 = ((MC >= 5e10) & (P["eqr"] >= 0.40) & (P["sal_cagr"] >= 0.03) & (P["op_cagr"] >= 0.03) & (P["payout"] > 0) & (P["payout"] < 0.30))
        sg = P["sal_cagr"]
        out["G1"] = np.where(g1, sg, np.nan)
        out["G3"] = np.where(g1 & (P["sal_up2"] == 1) & (P["np_g1"] >= 0.10) & (P["opm_up"] == 1) & hl, sg, np.nan)
        out["G4"] = np.where(g4, sg, np.nan)
        out["G5"] = np.where(g1 & (P["dy"] >= 0.03), sg, np.nan)
        # G2：騰落レシオ80以下になった月だけ入れ替え、ほかの月は直前に入れ替えた月の合格をそのまま使う（持ち株を変えない）
        adr = k_adr(D, k_market_codes(store, D))
        sig = np.zeros(M, dtype=bool)
        for j in range(M):
            t1 = me_idx[j]
            t0 = me_idx[j - 1] + 1 if j > 0 else 0
            w = adr[t0:t1 + 1]
            sig[j] = bool(np.any(w[np.isfinite(w)] <= J_ADR)) if np.isfinite(w).any() else False
        g2 = np.full((S, M), np.nan)
        last = None
        for j in range(M):
            if sig[j]:
                last = j
            if last is not None:
                g2[:, j] = out["G1"][:, last]
        out["G2"] = g2
        out["_sig"] = sig
    return out


def lab33(store: dict) -> int:
    """33段目：児玉一希『高配当10倍株投資』の数値の決まり。出版のあとの期間（2024年4月〜）の成績を別に出す（判定には使わない）。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE, PUB_FROM
    LAB_TITLE, LAB_PLAN = "33段目（児玉一希『高配当10倍株投資』2024年3月の数値の決まり）", "wide_lab_plan33_2026-11-10"
    LAB_NOTE = ("33段目：24〜26・32段目と同じく、条件に合う銘柄だけに点数をつけ、合う銘柄が20より少ない月は現金。条件に合わなくなった銘柄は、その月末に売る。"
                "保留は「合格した銘柄が1か月あたり平均5未満」。前半（〜2022年10月）は出版より前。後半（2022年11月〜）のうち2024年4月からは出版のあと（31か月）。"
                "出版後（2024年4月〜）は別にも出す（判定はしない）。従業員数・設備投資・VIX指数・時代の流れ・ビジネスモデル（定性）は使わない")
    PUB_FROM = J_PUB
    cache: dict = {}

    def mk(name):
        def fn(D):
            if "sc" not in cache:
                log("  基礎の値（時価総額・自己資本比率・売上と営業利益の伸び・配当性向・PER の位置・騰落レシオ）を作っています…")
                cache["sc"] = j_scores(store, D)
                sig = cache["sc"]["_sig"]
                log(f"  騰落レシオ{J_ADR:.0f}以下になった月 … {int(sig.sum())}か月（{', '.join(d.strftime('%Y-%m') for d, s in zip(D['dates'], sig) if s)}）")
            return cache["sc"][name]
        return fn
    rc = lab1(store, extra={nm: (mk(nm), lab) for nm, lab in J_TYPES}, filt=True)
    if rc == 0 and LAST_LAB1:
        print("\n  3. 出版のあと（2024年4月〜。判定はしない。年率％） … A ／ B ／ TOPIX（税あり） ／ でたらめな20銘柄の平均 A ／ でたらめ（同じ数・A）の平均")
        for k, r in LAST_LAB1.items():
            if "出版後" not in r["A"]:
                continue
            ra = [x["出版後"]["ann"] for x in r["rnd"]["A"] if "出版後" in x]
            rn = [x["出版後"]["ann"] for x in r.get("rnd_n", []) if "出版後" in x]
            print(f"     {k:<4} {r['A']['出版後']['ann']:6.1f} ／ {r['B']['出版後']['ann']:6.1f} ／ {r['tx']['出版後']['ann']:6.1f} ／ "
                  f"{np.mean(ra) if ra else float('nan'):6.1f}（上回った回数 {sum(1 for x in ra if r['A']['出版後']['ann'] > x)}） ／ "
                  f"{np.mean(rn) if rn else float('nan'):6.1f}（{sum(1 for x in rn if r['A']['出版後']['ann'] > x)}）"
                  f" ・ 最大下落 A {r['A']['出版後']['mdd']:.0f}％")
    PUB_FROM = None
    return rc


# ───────────────────────── 34段目：山口揚平『知ってそうで知らなかった ほんとうの株のしくみ』（PHP文庫、2013年）の5分でできる企業価値の算定 ─────────────────────────
# 計画は wide_lab_plan34_2026-11-10（結果を見る前に決めたもの）。24〜26・32・33段目と同じ月末の運用の形（条件に合う銘柄だけに点数・20銘柄・売る決まり A・B）。
Y_TYPES = (
    ("Y1", "本の形（保守的）：事業価値（営業利益の前々期・前期・今期予想の平均×10）＋現預金 − 負債の合計 が時価総額以上（安全域1倍以上）。金融を除く。安全域の大きい順"),
    ("Y2", "本の形（甘め）：財産価値 − 負債 を純資産で代用（事業価値＋純資産）。安全域1倍以上。安全域の大きい順"),
    ("Y3", "Y1 の安全域を2倍以上にする（株価が価値の半分以下）"),
    ("Y4", "参考：事業価値だけ（営業利益の3期平均×10 ÷ 時価総額）。1倍以上。大きい順"),
)


def y_scores(store: dict, D: dict) -> dict:
    P = g_panel(store, D)
    MC = D["MCAP"]
    fin = np.array([D["sector"].get(c, "") in yb.FIN_SECTORS for c in D["codes"]])[:, None]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        biz = 10.0 * P["op_avg3"]
        net_cash = P["netcash"] * MC                          # 現預金 −（総資産 − 純資産）
        s1 = (biz + net_cash) / MC
        s2 = (biz + P["eq"]) / MC
        s4 = biz / MC
        ok = (P["op_avg3"] > 0) & ~fin
        out["Y1"] = np.where(ok & (s1 >= 1.0), s1, np.nan)
        out["Y2"] = np.where(ok & (s2 >= 1.0), s2, np.nan)
        out["Y3"] = np.where(ok & (s1 >= 2.0), s1, np.nan)
        out["Y4"] = np.where(ok & (s4 >= 1.0), s4, np.nan)
    return out


def lab34(store: dict) -> int:
    """34段目：山口揚平の5分でできる企業価値の算定。9年分はすべて出版のあと。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE
    LAB_TITLE, LAB_PLAN = "34段目（山口揚平『ほんとうの株のしくみ』2013年の企業価値の算定）", "wide_lab_plan34_2026-11-10"
    LAB_NOTE = ("34段目：24〜26・32・33段目と同じく、条件に合う銘柄だけに点数をつけ、合う銘柄が20より少ない月は現金。条件に合わなくなった銘柄（株価が価値に追いついた銘柄）は、その月末に売る。"
                "保留は「合格した銘柄が1か月あたり平均5未満」。9年分はすべて本の出版（2013年）のあと。流動資産・流動負債・投資その他の資産は決算短信の要約にないので、"
                "財産価値 − 負債 を「現預金 − 負債の合計」（保守的）と「純資産」（甘め）の2通りで代用する")
    cache: dict = {}

    def mk(name):
        def fn(D):
            if "sc" not in cache:
                log("  基礎の値（営業利益の3期平均・現預金・負債・純資産）を作っています…")
                cache["sc"] = y_scores(store, D)
            return cache["sc"][name]
        return fn
    return lab1(store, extra={nm: (mk(nm), lab) for nm, lab in Y_TYPES}, filt=True)


# ───────────────────────── 36段目 B：組み立て方を本人のやり方に寄せる点検（持つ数・大型株だけ・時価総額で加重） ─────────────────────────
# 計画は wide_lab_plan36_2026-11-11（結果を見る前に決めたもの）。良い形を探すのではなく、「組み立て方のせいで負けていたか」を見る点検
R36_FORMS = (("H0", "h"), ("H3", "h"), ("H2", "h"), ("NIOKU", "g"), ("MIKI", "g"), ("N1", "n"), ("T1P", "t"), ("Y1", "y"),
             ("IMURA", "g"), ("G1", "j"))
R36_VARS = (("BASE", "20銘柄・同じ額・共通の範囲（これまでと同じ）", 20, None, False),
            ("N10", "10銘柄（売る決まり A＝10位・B＝30位より下）", 10, None, False),
            ("N5", "5銘柄（A＝5位・B＝15位より下）", 5, None, False),
            ("TOP500", "範囲を時価総額の上位500社に絞る（20銘柄・同じ額）", 20, 500, False),
            ("CAPW", "時価総額で加重（1銘柄は資産の15％まで。20銘柄）", 20, None, True),
            ("T500W", "上位500社 × 時価総額で加重（TOPIX にいちばん近い組み立て）", 20, 500, True))


def lab36(store: dict) -> int:
    global LAB_TITLE, LAB_PLAN, LAB_NOTE, NMAX, KEEPS
    LAB_PLAN = "wide_lab_plan36_2026-11-11"
    fns = {"h": h_scores, "g": g_scores, "n": n_scores, "t": t_scores, "y": y_scores, "j": j_scores}
    labels = {}
    for tbl in (H_TYPES, G_TYPES, N_TYPES, T_TYPES, Y_TYPES, J_TYPES):
        labels.update({k: v for k, v in tbl})
    sc_cache: dict = {}

    def mk(nm, kind):
        def fn(D):
            if kind not in sc_cache:
                log(f"  基礎の値（{kind}）を作っています…")
                sc_cache[kind] = fns[kind](store, D)
            return sc_cache[kind][nm]
        return fn
    only = ONLY
    vsel = [v for v in R36_VARS if not CFG36 or v[0] in CFG36]
    table = {}
    n0, k0 = NMAX, KEEPS
    try:
        for vk, vlab, n, topn, capw in vsel:
            NMAX, KEEPS = n, (("A", n), ("B", 3 * n))
            LAB_TITLE = f"36段目 B（組み立て方の点検）：{vk} … {vlab}"
            LAB_NOTE = f"36段目 B：{vlab}。本の条件と並べる順は各段のまま。判定の基準と、でたらめ（同じ数・同じ入れ替え）は各段と同じ作り方"
            log(f"■ {vk}：{vlab}")
            extra = {nm: (mk(nm, kind), labels.get(nm, nm)) for nm, kind in R36_FORMS if not only or nm in only}
            rc = lab1(store, extra=extra, filt=True, uni_fn=(None if topn is None else (lambda D, n_=topn: topn_mask(D, n_))),
                      w_fn=(w16("CAP") if capw else None))
            if rc != 0:
                return rc
            for nm, r in LAST_LAB1.items():
                table[(vk, nm)] = (r["A_ci"][0], r["B_ci"][0], r["verdict"], float(np.mean(r.get("npass", [0]))),
                                   r["A"]["前半"]["ann"] - r["tx"]["前半"]["ann"], r["A"]["後半"]["ann"] - r["tx"]["後半"]["ann"])
    finally:
        NMAX, KEEPS = n0, k0
    print("\n【36段目 B のまとめ：清算後の通しの TOPIX との差（pt。A ／ B）】")
    forms = [nm for nm, _ in R36_FORMS if not only or nm in only]
    print("形 | " + " | ".join(v[0] for v in vsel))
    for nm in forms:
        cells = []
        for v in vsel:
            t = table.get((v[0], nm))
            cells.append("—" if t is None else f"{t[0]:+.1f} ／ {t[1]:+.1f}（{t[3]:.0f}）")
        print(f"{nm} | " + " | ".join(cells))
    if "BASE" in [v[0] for v in vsel]:
        print("\nBASE からの変わり方（A。10の形の中央値・良くなった形の数）")
        for v in vsel:
            if v[0] == "BASE":
                continue
            dif = [table[(v[0], nm)][0] - table[("BASE", nm)][0] for nm in forms if (v[0], nm) in table and ("BASE", nm) in table]
            if dif:
                print(f"  {v[0]} … 中央値 {float(np.median(dif)):+.1f}pt・良くなった {sum(1 for x in dif if x > 0)}／{len(dif)}")
    return 0


# ───────────────────────── 35段目：日本個別株デューデリジェンスセンター『機関投資家の「手口」を知れば、株価の「なぜ」はすべて読める』（2026年2月18日）の需給の条件 ─────────────────────────
# 計画は wide_lab_plan35_2026-11-10（結果を見る前に決めたもの）。24〜26・32〜34段目と同じ月末の運用の形（条件に合う銘柄だけに点数・20銘柄・売る決まり A・B）。
# 範囲は小型株を含む範囲（cfg=SMALL：売買代金の中央値1,000万円以上・時価総額30億円以上）で回す。信用取引の週末残高（スタンダード）を週ごとに取り、data/wide_k.pkl に持つ
Q_TYPES = (
    ("Q1", "品薄・好業績の初動狙い（浮動株比率を除く）：時価総額50〜500億円・売上の伸びの予想20％以上・ROE10％以上・25日線からの乖離0〜+5％。売上の伸びの予想の大きい順"),
    ("Q2", "Q1 に出来高の急増（直近20営業日の売買代金の平均が、その前の60営業日の平均の1.5倍以上）を足す"),
    ("Q3", "踏み上げ予備軍：信用倍率1倍以下（売り残あり）・信用倍率が前の週より低下・終値が年初来高値の95％以上。信用倍率の低い順"),
    ("Q4", "機関投資家のユニバース入り候補：時価総額300〜500億円・ROIC8％以上（近い値）・営業利益率10％以上・売買代金が増加傾向（直近20営業日の平均＞その前の60営業日の平均）。ROIC の大きい順"),
    ("Q5", "機関に発見される前（第7章）：時価総額50〜200億円・売上の3年の年平均の伸び20％以上・営業利益率10％以上。売上の伸びの大きい順"),
    ("Q6", "空売りの踏み上げ狙い（第6章）：信用倍率0.5倍以下・終値が250営業日の安値の1.2倍以下・直近3か月に会社予想の上方修正（5％以上）。信用倍率の低い順"),
)
Q_REV_DAYS = 92          # 上方修正を見る期間（月末の日から何日前まで）
Q_REV_MIN = 0.05         # 上方修正とみなす幅（前の予想より5％以上高い）


def q_margin(store: dict, xs: dict, D: dict) -> tuple:
    """月末ごとの信用倍率（買い残 ÷ 売り残）と、その1つ前の週の信用倍率。
    残高は、月末の7日前までの日付の最新のもの（月末から37日より前なら使わない。5段目と同じ）。1つ前の週は、その前の記録（14日以内）。
    売り残が0の銘柄は信用倍率を出さない（貸借銘柄でない株など）。倍率は株数どうしの比なので、分割で直さなくてよい。"""
    rows = x_rows(xs, "mi")
    codes, dates = D["codes"], D["dates"]
    ci = {c: i for i, c in enumerate(codes)}
    S, M = len(codes), len(dates)
    mr, mp = np.full((S, M), np.nan), np.full((S, M), np.nan)
    by = {}
    for r in rows:
        c = yb.norm_code(r.get("code") or "")
        lv, sv = num(r.get("long")), num(r.get("short"))
        if c in ci and lv is not None and sv is not None:
            by.setdefault(c, {})[str(r.get("date") or "")[:10]] = (lv, sv)
    for c, mp_ in by.items():
        lst = sorted(mp_.items())
        ds = np.array([x[0] for x in lst])
        i = ci[c]
        for j, d in enumerate(dates):
            hi = (d - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            lo = (d - pd.Timedelta(days=37)).strftime("%Y-%m-%d")
            k = int(np.searchsorted(ds, hi, side="right")) - 1
            if k < 0 or ds[k] < lo:
                continue
            lv, sv = lst[k][1]
            if sv > 0:
                mr[i, j] = lv / sv
            if k >= 1 and (np.datetime64(str(ds[k])) - np.datetime64(str(ds[k - 1]))).astype(int) <= 14:
                lv0, sv0 = lst[k - 1][1]
                if sv0 > 0:
                    mp[i, j] = lv0 / sv0
    return mr, mp


def q_revision(store: dict, D: dict) -> np.ndarray:
    """月末ごとに、直近 Q_REV_DAYS 日に会社予想の上方修正があったか（1・0）。
    同じ期の予想（本決算の短信は来期の予想、四半期の短信と予想の修正は当期の予想）が、その前の開示の予想より Q_REV_MIN 以上高い。
    比べるのは営業利益（前と今の両方にあるとき）、なければ純利益。前の予想がプラスのときだけ。開示は月末の最後の営業日より前のものだけ（build と同じ）。"""
    codes, me_idx = D["codes"], D["me_idx"]
    cal_i = D["cal"].astype("datetime64[D]")
    S, M = len(codes), len(me_idx)
    out = np.zeros((S, M))
    for i, c in enumerate(codes):
        tl = fin_timeline(store["fins"].get(c))
        if not tl:
            continue
        last = {}
        ups = []
        for r in tl:
            d = str(r.get("DiscDate") or "")[:10]
            fe = str(r.get("CurFYEn") or "")[:10]
            if len(fe) != 10:
                continue
            if is_fy(r):
                tgt = f"{int(fe[:4]) + 1}{fe[4:]}"
                op, npf = num(r.get("NxFOP")), first_num(r, ("NxFNp", "NxFNP"))
            else:
                tgt = fe
                op, npf = num(r.get("FOP")), num(r.get("FNP"))
            if op is None and npf is None:
                continue
            prev = last.get(tgt)
            if prev is not None:
                if op is not None and prev[0] is not None and prev[0] > 0:
                    if op >= prev[0] * (1 + Q_REV_MIN):
                        ups.append(np.datetime64(d))
                elif npf is not None and prev[1] is not None and prev[1] > 0:
                    if npf >= prev[1] * (1 + Q_REV_MIN):
                        ups.append(np.datetime64(d))
            last[tgt] = (op if op is not None else (prev[0] if prev else None), npf if npf is not None else (prev[1] if prev else None))
        if not ups:
            continue
        ua = np.array(sorted(ups))
        for j in range(M):
            t_me = cal_i[me_idx[j]]
            k1 = int(np.searchsorted(ua, t_me, side="left"))
            k0 = int(np.searchsorted(ua, t_me - np.timedelta64(Q_REV_DAYS, "D"), side="left"))
            out[i, j] = float(k1 > k0)
    return out


def q_price(D: dict) -> dict:
    """月末ごとの値動きの値：25日線からの乖離・年初来高値に対する位置・250営業日の安値に対する位置・売買代金の20日平均 ÷ その前の60日平均。"""
    CF, VA, me_idx, cal = D["CF"], D["VA"], D["me_idx"], D["cal"]
    S, M = CF.shape[0], len(me_idx)
    out = {k: np.full((S, M), np.nan) for k in ("dev25", "ytd", "lo250", "va_r")}
    yr = pd.DatetimeIndex(cal).year.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for j, t in enumerate(me_idx):
            if t >= 24:
                out["dev25"][:, j] = CF[:, t] / np.nanmean(CF[:, t - 24:t + 1], axis=1) - 1.0
            y0 = int(np.searchsorted(yr, yr[t], side="left"))
            out["ytd"][:, j] = CF[:, t] / np.nanmax(CF[:, y0:t + 1], axis=1)
            if t >= 249:
                out["lo250"][:, j] = CF[:, t] / np.nanmin(CF[:, t - 249:t + 1], axis=1)
            if t >= 79:
                a20 = VA[:, t - 19:t + 1].mean(axis=1)
                a60 = VA[:, t - 79:t - 19].mean(axis=1)
                out["va_r"][:, j] = np.where(a60 > 0, a20 / a60, np.nan)
    return out


def q_scores(store: dict, xs: dict, D: dict) -> dict:
    P = g_panel(store, D)
    X = q_price(D)
    mr, mp = q_margin(store, xs, D)
    rev = q_revision(store, D)
    MC = D["MCAP"]
    fin = np.array([D["sector"].get(c, "") in yb.FIN_SECTORS for c in D["codes"]])[:, None]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        q1 = ((MC >= 5e9) & (MC <= 5e10) & (P["gs_f"] >= 0.20) & (P["roe"] >= 0.10) & (X["dev25"] >= 0.0) & (X["dev25"] <= 0.05))
        out["Q1"] = np.where(q1, P["gs_f"], np.nan)
        out["Q2"] = np.where(q1 & (X["va_r"] >= 1.5), P["gs_f"], np.nan)
        q3 = (mr <= 1.0) & (mr < mp) & (X["ytd"] >= 0.95)
        out["Q3"] = np.where(q3, -mr, np.nan)
        q4 = (MC >= 3e10) & (MC <= 5e10) & (P["roic"] >= 0.08) & (P["opm"] >= 0.10) & (X["va_r"] > 1.0) & ~fin
        out["Q4"] = np.where(q4, P["roic"], np.nan)
        q5 = (MC >= 5e9) & (MC <= 2e10) & (P["sal_cagr3"] >= 0.20) & (P["opm"] >= 0.10)
        out["Q5"] = np.where(q5, P["sal_cagr3"], np.nan)
        q6 = (mr <= 0.5) & (X["lo250"] <= 1.2) & (rev == 1)
        out["Q6"] = np.where(q6, -mr, np.nan)
        cov = np.isfinite(mr) & D["UNI"]
        out["_mcov"] = cov.sum(axis=0) / np.maximum(1, D["UNI"].sum(axis=0))
    return out


def lab35(store: dict, xs: dict) -> int:
    """35段目：機関投資家の手口（需給）の本の数値の決まり。9年分はすべて出版（2026年2月）の前。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE
    LAB_TITLE, LAB_PLAN = "35段目（日本個別株デューデリジェンスセンター『機関投資家の「手口」を知れば…』2026年2月の需給の条件）", "wide_lab_plan35_2026-11-10"
    LAB_NOTE = ("35段目：24〜26・32〜34段目と同じく、条件に合う銘柄だけに点数をつけ、合う銘柄が20より少ない月は現金。条件に合わなくなった銘柄は、その月末に売る（本の「前提が崩れたら売る」）。"
                "保留は「合格した銘柄が1か月あたり平均5未満」。9年分はすべて本の出版（2026年2月18日）の前。浮動株比率・逆日歩・大量保有報告書・自社株買いの発表・板と歩み値は使えない（データがない）。"
                "損切り（1〜2％）は月末の運用では入れない")
    if MIN_MCAP >= 1e10:
        log("  ※ 35段目は小型株を含む範囲（cfg=SMALL）で回す計画です。いまは共通の範囲です")
    cache: dict = {}

    def mk(name):
        def fn(D):
            if "sc" not in cache:
                log("  基礎の値（売上の伸びの予想・ROE・ROIC・営業利益率・信用倍率・上方修正・25日線・年初来高値・売買代金の増え方）を作っています…")
                cache["sc"] = q_scores(store, xs, D)
                cv = cache["sc"]["_mcov"]
                log(f"  範囲の中で信用倍率が出せた割合 … 中央値 {float(np.median(cv)) * 100:.0f}％（最小 {float(cv.min()) * 100:.0f}％）")
            return cache["sc"][name]
        return fn
    return lab1(store, extra={nm: (mk(nm), lab) for nm, lab in Q_TYPES}, filt=True)


# ───────────────────────── 27段目：ミネルヴィニ『ミネルヴィニの成長株投資法』（原書2013年）の数値の決まり ─────────────────────────
# 計画は wide_lab_plan27_2026-11-10（結果を見る前に決めたもの）。売り買いは22段目と同じ毎日の道具（d_sim）を使う。
M_RS_MIN = 0.70          # レラティブストレングスの順位（範囲の中で上から何割か）70以上
M_PIVOT = 25             # ブレイクの基準：その前の25営業日（約5週）の終値の高値を超えた日
M_TIGHT = 0.15           # M2：その25営業日の値幅（高値 ÷ 安値 − 1）が15％以下（ボラティリティの縮小の近い形）
M_FORMS = (
    ("M1", "トレンドテンプレート（8つ）・RS70以上・25営業日の高値を超えた日に買う・5銘柄・−7.5％で損切り・+20％で損切りを取得単価へ・その後は50日線割れで売る",
     {"filt": "none", "slots": 5, "pyr": False, "mkt": False, "nobox": True, "stop": 0.075}),
    ("M2", "M1 に「その前の25営業日の値幅が15％以下」（ボラティリティの縮小）を足す", {"filt": "tight", "slots": 5, "pyr": False, "mkt": False, "nobox": True, "stop": 0.075}),
    ("M3", "M1 に業績（直近2四半期の利益が前年比+20％以上・売上+10％以上）を足す", {"filt": "fin", "slots": 5, "pyr": False, "mkt": False, "nobox": True, "stop": 0.075}),
    ("M4", "参考：M1 を20銘柄に広げる（運の幅を小さくした形）", {"filt": "none", "slots": 20, "pyr": False, "mkt": False, "nobox": True, "stop": 0.075}),
)


def m_signals(D: dict) -> tuple:
    """毎日のトレンドテンプレート・RS の順位・25営業日の高値のブレイク・値幅。どれもその日の終値までで決める。"""
    CF, UNI, me_idx = D["CF"], D["UNI"], D["me_idx"]
    S, T = CF.shape
    df = pd.DataFrame(CF.T)
    with np.errstate(invalid="ignore"):
        ma50 = df.rolling(50, min_periods=50).mean().to_numpy().T
        ma150 = df.rolling(150, min_periods=150).mean().to_numpy().T
        ma200 = df.rolling(200, min_periods=200).mean().to_numpy().T
        lo52 = df.rolling(250, min_periods=250).min().to_numpy().T
        hi52 = df.rolling(250, min_periods=250).max().to_numpy().T
        ma200_prev = np.full_like(ma200, np.nan)
        ma200_prev[:, 21:] = ma200[:, :-21]
        p = CF
        tt = ((p > ma150) & (p > ma200) & (ma150 > ma200) & (ma200 > ma200_prev) & (ma50 > ma150) & (ma50 > ma200) & (p > ma50)
              & (p >= lo52 * 1.30) & (p >= hi52 * 0.75))
        prev_hi = df.shift(1).rolling(M_PIVOT, min_periods=M_PIVOT).max().to_numpy().T
        prev_lo = df.shift(1).rolling(M_PIVOT, min_periods=M_PIVOT).min().to_numpy().T
        brk = (p > prev_hi)
        tight = (prev_hi / prev_lo - 1.0) <= M_TIGHT

        def lag(k):
            out = np.full_like(CF, np.nan)
            out[:, k:] = CF[:, :-k]
            return out
        rs_raw = 0.4 * (p / lag(63) - 1) + 0.2 * (p / lag(126) - 1) + 0.2 * (p / lag(189) - 1) + 0.2 * (p / lag(252) - 1)
    jm = np.searchsorted(np.array(me_idx), np.arange(T), side="left") - 1
    rs = np.full((S, T), np.nan)
    for t in range(T):
        j = jm[t]
        if j < 0:
            continue
        idx = np.nonzero(UNI[:, j] & np.isfinite(rs_raw[:, t]))[0]
        if len(idx) > 1:
            r = np.argsort(np.argsort(rs_raw[idx, t], kind="stable"), kind="stable")
            rs[idx, t] = r / (len(idx) - 1)
    return tt, rs, brk, tight, prev_hi, ma50, jm


def lab27(store: dict) -> int:
    """27段目：ミネルヴィニのトレンドテンプレートとブレイク。計画は wide_lab_plan27_2026-11-10。"""
    log("27段目（ミネルヴィニ）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, me_idx, cal, codes = D["dates"], D["UNI"], D["me_idx"], D["cal"], D["codes"]
    C, CF = D["C"], D["CF"]
    S, T = C.shape
    j0 = next((j for j, t in enumerate(me_idx) if t >= 260), None)
    if j0 is None:
        print("\n■ 株価の期間が短すぎます")
        return 2
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    E = ev_extract(store, D)
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    traded = np.isfinite(C) & (C > 0)
    last_t = np.array([int(np.nonzero(traded[i])[0][-1]) if traded[i].any() else -1 for i in range(S)])
    LIMU, LIMD = D.get("LIMU"), D.get("LIMD")
    if LIMU is None:
        LIMU, LIMD = limit_masks_close(store, codes, cal, C, CF)
    log("  トレンドテンプレート・RS・ブレイクを作っています…")
    tt, rs, brk, tight, prev_hi, MA, jm = m_signals(D)
    fst = {}
    for i, c in enumerate(codes):
        s_ = d_fin_status(store["fins"].get(c))
        if s_:
            fst[i] = ([x[0] for x in s_], s_)
    calS = [str(pd.Timestamp(d).date()) for d in cal]
    base: dict = {}      # 合図の日 → [(銘柄, ブレイクの基準, 値幅が狭い, 業績, RS)]
    allbrk: dict = {}    # でたらめ（同じ日のブレイクから）の元：範囲の中で25営業日の高値を超えた銘柄（トレンドテンプレートは問わない）
    n_sig = n_tt = 0
    for t in range(t0, T - 1):
        j = jm[t]
        if j < 0:
            continue
        b_ids = np.nonzero(brk[:, t] & UNI[:, j] & traded[:, t])[0]
        if not len(b_ids):
            continue
        allbrk[t] = b_ids
        n_sig += len(b_ids)
        ids = [int(i) for i in b_ids if tt[i, t] and rs[i, t] == rs[i, t] and rs[i, t] >= M_RS_MIN]
        if not ids:
            continue
        lst = []
        for i in ids:
            f = fst.get(i)
            du = False
            if f:
                k = bisect.bisect_left(f[0], calS[t]) - 1
                if k >= 0:
                    du = bool(f[1][k][1])
            lst.append((i, float(prev_hi[i, t]), bool(tight[i, t]), du, float(rs[i, t])))
        base[t] = lst
        n_tt += len(lst)

    def pool_ra(t):
        j = jm[t]
        return np.nonzero(UNI[:, j] & traded[:, t])[0] if j >= 0 else np.zeros(0, dtype=int)

    print("\n" + "=" * 100)
    print("【日本株の手法・27段目（ミネルヴィニ『ミネルヴィニの成長株投資法』原書2013年）（wide_lab）：東証全銘柄】")
    print("  （基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan27_2026-11-10。9年分はすべて本が出たあと）")
    print(f"  比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev) - 1}か月）。前半＝〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半＝{SPLIT.date()}〜")
    print(f"  費用と税 {COST}：株 片道{SLIP * 100:.1f}％・{TAX_TXT}（損は同じ運用の中で繰り越す）。元本{CAP / 1e4:,.0f}万円。合図の次の営業日の終値で売り買い（9年分は始値がないので終値）")
    print(f"\n  0. 合図：範囲の中で25営業日の高値を超えた日 {n_sig}回（{n_sig / yrs:.0f}回／年）・そのうちトレンドテンプレートと RS70以上 {n_tt}回（{n_tt / yrs:.0f}回／年）")
    res = {}
    for key, lab, cfg in M_FORMS:
        if ONLY and key not in ONLY:
            continue
        log(f"  {key} を回しています（でたらめ 20回 × 2）…")
        cands: dict = {}
        for t, lst in base.items():
            if cfg["filt"] == "tight":
                sel = [x for x in lst if x[2]]
            elif cfg["filt"] == "fin":
                sel = [x for x in lst if x[3]]
            else:
                sel = lst
            sel = sorted(sel, key=lambda x: -x[4])          # RS の高い順
            if sel:
                cands[t] = [(x[0], x[1]) for x in sel]
        v, liq, s_ = d_sim(cands, D, E["divpay"], t0, cfg, last_t, LIMU, LIMD, MA)
        ra, rb, fill = [], [], []
        n_c = max(1, sum(len(cl) for cl in cands.values()))
        for nm, seed, out_ in (("ra", 7200, ra), ("rb", 8200, rb)):
            pools = {t: (pool_ra(t) if nm == "ra" else allbrk.get(t, np.zeros(0, dtype=int))) for t in cands}
            fill.append(sum(min(len(cl), len(pools[t])) for t, cl in cands.items()) / n_c)
            for k in range(N_RAND):
                rg = np.random.default_rng(seed + k)
                pl = {}
                for t, cl in cands.items():
                    pool = pools[t]
                    if len(pool):
                        pick = rg.choice(pool, size=min(len(cl), len(pool)), replace=False)
                        pl[t] = [(int(i), 0.0) for i in pick]
                pv, _l, _s = d_sim(pl, D, E["divpay"], t0, cfg, last_t, LIMU, LIMD, MA)
                out_.append(half_stats(pv, ev))
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT),
              "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        x = {"st": half_stats(v, ev), "liq": liq, "s": s_, "ra": ra, "rb": rb, "nb": nb, "fill_ra": fill[0], "fill_rb": fill[1]}
        x["ci"] = boot_ci(v, TXv, liq, TXliq, yrs)      # 30段目：ぶれの幅
        x["verdict"], x["rb_lab"], x["det"] = d_judge(x, ST_TX, False)
        res[key] = (lab, x)
        log(f"  {key} … {x['verdict']}")
    print(f"\n  1. 結果（年率％「前半 ／ 後半」。TOPIX（税あり）{ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}。"
          "差は「TOPIX との差／でたらめ（範囲から）との差・上回った回数／でたらめ（同じ日に25営業日の高値を超えた銘柄から）との差・上回った回数」）")
    for key, (lab, x) in res.items():
        a, s_ = x["st"], x["s"]
        cl = s_["closed"]
        win = np.mean([r[2] > 0 for r in cl]) * 100 if cl else float("nan")
        gw = [r[2] for r in cl if r[2] > 0]
        gl = [r[2] for r in cl if r[2] <= 0]
        hold = [r[1] - r[0] for r in cl]
        dd = x["det"]
        print(f"\n  {key} {lab}")
        print(f"     {ci_txt(x['ci'])}")
        print(f"     年率 {a['前半']['ann']:6.1f} ／ {a['後半']['ann']:6.1f}（最大下落 {a['通し']['mdd']:.0f}％・清算後の通し {((x['liq'] / CAP) ** (1 / yrs) - 1) * 100:.1f}％）"
              f"［{dd['前半'][0]:+.1f}／{dd['前半'][1]:+.1f}・{dd['前半'][2]}／{dd['前半'][3]:+.1f}・{dd['前半'][4]} ｜ "
              f"{dd['後半'][0]:+.1f}／{dd['後半'][1]:+.1f}・{dd['後半'][2]}／{dd['後半'][3]:+.1f}・{dd['後半'][4]}］ … {x['verdict']}・選ぶ効き目（同じ日のブレイクと比べて） {x['rb_lab']}")
        print(f"     最初の買い {s_['n_new']}（前半 {x['nb']['前半']}・後半 {x['nb']['後半']}）・売り {len(cl)}回（勝ち {win:.0f}％・勝ちの平均 {np.mean(gw) * 100 if gw else float('nan'):+.1f}％・"
              f"負けの平均 {np.mean(gl) * 100 if gl else float('nan'):+.1f}％・持った日数の中央 {np.median(hold) if hold else float('nan'):.0f}日）・"
              f"売った理由 " + "・".join(f"{k}{n}" for k, n in s_["why"].items() if n) + f"・現金の割合 {np.mean(s_['cash_share']) * 100:.0f}％・ストップ高で買えず {s_['skip_lim']}")
        ram = [np.mean([y[h]["ann"] for y in x["ra"]]) for h in ("前半", "後半")]
        rbm = [np.mean([y[h]["ann"] for y in x["rb"]]) for h in ("前半", "後半")]
        print(f"     でたらめの平均（範囲から ／ 同じ日のブレイクから） {ram[0]:.1f}・{ram[1]:.1f} ／ {rbm[0]:.1f}・{rbm[1]:.1f}・選べた割合 {x['fill_ra'] * 100:.0f}％・{x['fill_rb'] * 100:.0f}％")
    print("\n  2. 判定のまとめ")
    for vv in ("候補", "範囲の中では効く", "効かない", "保留（データ不足）"):
        ks = [k for k, (_l, x) in res.items() if x["verdict"] == vv]
        print(f"     {vv} … " + ("・".join(ks) if ks else "なし"))
    print(f"     （TOPIX の清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")
    return 0


# ───────────────────────── 28段目：ALFA『高配当株で小金持ちになる方法』の往復銘柄の売り買い ─────────────────────────
# 計画は wide_lab_plan28_2026-11-10（結果を見る前に決めたもの）。毎日の売り買い。損切りはしない（本のとおり）。
R_MEAN = 250             # 「ここ数年の平均値」の代わり：250営業日の終値の平均
R_STEP = 0.03            # 買い下がり：前に買った値段から3％下がるごとに
R_TRANCHE = 4            # 資金を4回に分ける
R_TARGET = 0.06          # 売り：終値が平均の +6％以上（本の例：平均1,650円・売り1,750円）。平均より上で買ったでたらめは、最初に買った値段の +6％
R_FORMS = (
    ("R1", "東証一部・プライムの大型株（範囲の時価総額の上から500位）・配当利回り4％以上・1日の出来高100万株以上。終値が250日平均以下で買い、3％下がるごとに買い増し（4回まで）、平均の+6％で売る。損切りなし。5銘柄",
     {"slots": 5, "yld": 0.04}),
    ("R2", "参考：R1 を20銘柄に広げる", {"slots": 20, "yld": 0.04}),
    ("R0", "参考：R1 から配当利回りの条件を外す（往復の売り買いだけの効き目）", {"slots": 5, "yld": 0.0}),
)


def r_sim(cands: dict, D: dict, divpay: dict, t0: int, cfg: dict, last_t: np.ndarray, MEAN: np.ndarray, rand_hold: bool = False,
          cap: float = CAP, slip: float = SLIP, tax: float = TAX):
    """往復銘柄の運用。cands … {日: [銘柄, …]（優先の順）}。合図はその日の終値で決め、次の営業日の終値で売り買いする。"""
    CF, C, me_idx = D["CF"], D["C"], D["me_idx"]
    T = CF.shape[1]
    slots = cfg["slots"]
    me_set = set(me_idx)
    cash, loss = float(cap), 0.0
    pos: dict = {}
    pend: list = []
    vals = []
    st = {"n_new": 0, "n_add": 0, "buy_t": [], "closed": [], "held": [], "cash_share": [], "delist": 0, "open_days": []}

    def px(i, t):
        v = CF[i, t]
        return float(v) if (v == v and v > 0) else 0.0

    def realize(g):
        nonlocal cash, loss
        if g > 0:
            tx = max(0.0, g - loss) * tax
            loss = max(0.0, loss - g)
            cash -= tx
        else:
            loss += -g

    def close_pos(i, t, price):
        nonlocal cash
        p_ = pos.pop(i)
        eff = price * (1.0 - slip)
        cash += p_["sh"] * eff
        realize(p_["sh"] * eff - p_["cost"])
        st["closed"].append((p_["first_t"], t, p_["sh"] * eff / p_["cost"] - 1.0))

    for t in range(t0, T):
        for i in [k for k in pos if last_t[k] < t - 5]:
            close_pos(i, t, px(i, last_t[i]))
            st["delist"] += 1
        if pend:
            value = cash + sum(p_["sh"] * px(k, t) for k, p_ in pos.items())
            for i, kind in pend:
                v = C[i, t]
                if not (v == v and v > 0):
                    continue
                if kind == "sell":
                    if i in pos:
                        close_pos(i, t, float(v))
                    continue
                if kind == "new":
                    if i in pos or len(pos) >= slots:
                        continue
                    budget = value / slots
                else:
                    p_ = pos.get(i)
                    if p_ is None or p_["k"] >= R_TRANCHE:
                        continue
                    budget = p_["budget"]
                pb = float(v) * (1.0 + slip)
                sh = int(min(budget / R_TRANCHE, cash) / pb)
                if sh < 1:
                    continue
                cash -= sh * pb
                if kind == "new":
                    pos[i] = {"sh": sh, "cost": sh * pb, "budget": budget, "k": 1, "last": float(v), "first": float(v), "first_t": t}
                    st["n_new"] += 1
                    st["buy_t"].append(t)
                else:
                    p_["sh"] += sh
                    p_["cost"] += sh * pb
                    p_["k"] += 1
                    p_["last"] = float(v)
                    st["n_add"] += 1
            pend = []
        for i, amt in divpay.get(t, ()):
            if i in pos:
                cash += pos[i]["sh"] * amt * (1.0 - tax)
        if t in me_set:
            value = cash + sum(p_["sh"] * px(k, t) for k, p_ in pos.items())
            vals.append(value)
            st["held"].append(len(pos))
            st["cash_share"].append(cash / value if value > 0 else 0.0)
        if t >= T - 1:
            break
        for i, p_ in pos.items():
            v, m = C[i, t], MEAN[i, t]
            if not (v == v and v > 0 and m == m):
                continue
            if v >= max(m, p_["first"]) * (1.0 + R_TARGET):      # 平均の+6％（平均より上で買ったでたらめは、最初に買った値段の+6％。平均以下で買う本の形では平均の+6％と同じ）
                pend.append((i, "sell"))
            elif p_["k"] < R_TRANCHE and v <= p_["last"] * (1.0 - R_STEP):
                pend.append((i, "add"))
        for i in cands.get(t, ()):
            if i not in pos:
                pend.append((i, "new"))
    liq, gain = cash, 0.0
    for k, p_ in pos.items():
        eff = px(k, T - 1) * (1.0 - slip)
        liq += p_["sh"] * eff
        gain += p_["sh"] * eff - p_["cost"]
    liq -= tax * max(0.0, gain - loss)
    st["open_days"] = [T - 1 - p_["first_t"] for p_ in pos.values()]
    return np.array(vals), liq, st


def lab28(store: dict) -> int:
    """28段目：往復銘柄の売り買い（平均以下で買い、買い下がり、平均の上で売る。損切りなし）。計画は wide_lab_plan28_2026-11-10。"""
    log("28段目（ALFA の往復銘柄）を回します（指標を作っています）…")
    D = build(store)
    dates, UNI, me_idx, cal, codes, MC = D["dates"], D["UNI"], D["me_idx"], D["cal"], D["codes"], D["MCAP"]
    C, CF = D["C"], D["CF"]
    S, T = C.shape
    j0 = next((j for j, t in enumerate(me_idx) if t >= R_MEAN + 5), None)
    if j0 is None:
        print("\n■ 株価の期間が短すぎます")
        return 2
    t0 = me_idx[j0]
    ev = dates[j0:]
    yrs = max((ev[-1] - ev[0]).days / 365.25, 0.5)
    E = ev_extract(store, D)
    TXv, TXliq = topix_taxed(D, j0)
    ST_TX = half_stats(TXv, ev)
    traded = np.isfinite(C) & (C > 0)
    last_t = np.array([int(np.nonzero(traded[i])[0][-1]) if traded[i].any() else -1 for i in range(S)])
    MK = k_market_codes(store, D)
    P = g_panel(store, D)
    with np.errstate(invalid="ignore", divide="ignore"):
        MEAN = pd.DataFrame(CF.T).rolling(R_MEAN, min_periods=R_MEAN).mean().to_numpy().T
    # 1日の出来高（株数）＝売買代金 ÷ 終値。20営業日の中央値
    SHV = np.full((S, T), np.nan)
    cal_i = D["cal"].astype("datetime64[D]")
    for i, c in enumerate(codes):
        b = store["bars"][c]
        d_i = np.asarray(b["d"]).astype("datetime64[D]")
        idx = np.searchsorted(cal_i, d_i)
        ok = (idx < T)
        ok[ok] &= cal_i[idx[ok]] == d_i[ok]
        sv = np.full(T, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            sv[idx[ok]] = np.asarray(b["va"], dtype=float)[ok] / np.asarray(b["c"], dtype=float)[ok]
        SHV[i] = pd.Series(sv).rolling(20, min_periods=10).median().to_numpy()
    jm = np.searchsorted(np.array(me_idx), np.arange(T), side="left") - 1
    rank_mc = np.full(UNI.shape, np.inf)
    for j in range(len(me_idx)):
        idx = np.nonzero(UNI[:, j])[0]
        o = idx[np.argsort(-MC[idx, j])]
        rank_mc[o, j] = np.arange(len(o))
    big = (rank_mc < 500) & np.isin(MK, ["0101", "0111"])
    res = {}
    print("\n" + "=" * 100)
    print("【日本株の手法・28段目（ALFA『高配当株で小金持ちになる方法』の往復銘柄）（wide_lab）：一部・プライムの大型株】")
    print("  （基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan28_2026-11-10）")
    print(f"  比べる期間 {ev[0].date()}〜{ev[-1].date()}（{len(ev) - 1}か月）。前半＝〜{(SPLIT - pd.Timedelta(days=1)).date()}・後半＝{SPLIT.date()}〜")
    print(f"  費用と税 {COST}：株 片道{SLIP * 100:.1f}％・{TAX_TXT}（損は同じ運用の中で繰り越す）。元本{CAP / 1e4:,.0f}万円。合図の次の営業日の終値で売り買い。損切りなし")
    for key, lab, cfg in R_FORMS:
        if ONLY and key not in ONLY:
            continue
        log(f"  {key} を回しています（でたらめ 20回）…")
        elig = {}
        cands = {}
        n_c = 0
        for t in range(t0, T - 1):
            j = jm[t]
            if j < 0:
                continue
            ok = big[:, j] & traded[:, t] & (SHV[:, t] >= 1e6)
            if cfg["yld"] > 0:
                ok &= P["dy"][:, j] >= cfg["yld"]
            ids = np.nonzero(ok)[0]
            if not len(ids):
                continue
            elig[t] = ids
            sig = [i for i in ids if C[i, t] <= MEAN[i, t]]
            if sig:
                cands[t] = sorted(sig, key=lambda i: -(P["dy"][i, j] if P["dy"][i, j] == P["dy"][i, j] else 0))
                n_c += len(sig)
        v, liq, s_ = r_sim(cands, D, E["divpay"], t0, cfg, last_t, MEAN)
        ra = []
        for k in range(N_RAND):
            rg = np.random.default_rng(9100 + k)
            pl = {}
            for t, cl in cands.items():
                pool = elig[t]
                pl[t] = [int(i) for i in rg.choice(pool, size=min(len(cl), len(pool)), replace=False)]
            pv, _l, _s = r_sim(pl, D, E["divpay"], t0, cfg, last_t, MEAN)
            ra.append(half_stats(pv, ev))
        st_ = half_stats(v, ev)
        det = {}
        ok_tx = ok_ra = True
        for h in ("前半", "後半"):
            a = st_[h]["ann"]
            rr = [y[h]["ann"] for y in ra]
            d_tx, d_ra, b_ra = a - ST_TX[h]["ann"], a - float(np.mean(rr)), int(sum(1 for y in rr if a > y))
            det[h] = (d_tx, d_ra, b_ra, float(np.mean(rr)))
            ok_tx &= d_tx >= PT_TX
            ok_ra &= (d_ra >= PT_RND) and (b_ra >= BEAT_MIN)
        nb = {"前半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) < SPLIT), "後半": sum(1 for t in s_["buy_t"] if pd.Timestamp(cal[t]) >= SPLIT)}
        verdict = "保留（データ不足）" if min(nb.values()) < 20 else ("候補" if (ok_tx and ok_ra) else ("範囲の中では効く" if ok_ra else "効かない"))
        res[key] = verdict
        cl = s_["closed"]
        win = np.mean([r[2] > 0 for r in cl]) * 100 if cl else float("nan")
        hold = [r[1] - r[0] for r in cl]
        log(f"  {key} … {verdict}")
        print(f"\n  {key} {lab}")
        print(f"     {ci_txt(boot_ci(v, TXv, liq, TXliq, yrs))}")
        print(f"     年率 {st_['前半']['ann']:6.1f} ／ {st_['後半']['ann']:6.1f}（最大下落 {st_['通し']['mdd']:.0f}％・清算後の通し {((liq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）"
              f"［TOPIX との差 {det['前半'][0]:+.1f} ／ {det['後半'][0]:+.1f}・でたらめとの差 {det['前半'][1]:+.1f}・{det['前半'][2]}回 ／ {det['後半'][1]:+.1f}・{det['後半'][2]}回］ … {verdict}")
        print(f"     合図 {n_c}回（銘柄×日）・最初の買い {s_['n_new']}（前半 {nb['前半']}・後半 {nb['後半']}）・買い増し {s_['n_add']}・売り {len(cl)}回（勝ち {win:.0f}％・"
              f"平均 {np.mean([r[2] for r in cl]) * 100 if cl else float('nan'):+.1f}％・持った日数の中央 {np.median(hold) if hold else float('nan'):.0f}日）・"
              f"最後に持ったまま {len(s_['open_days'])}銘柄（持った日数の中央 {np.median(s_['open_days']) if s_['open_days'] else 0:.0f}日）・現金の割合 {np.mean(s_['cash_share']) * 100:.0f}％・上場をやめた {s_['delist']}")
        print(f"     でたらめ（同じ日に、条件に合う銘柄から同じ数だけでたらめに選ぶ。売り買いの決まりは同じ）の平均 {det['前半'][3]:.1f} ／ {det['後半'][3]:.1f}")
    print(f"\n  TOPIX（税あり） {ST_TX['前半']['ann']:.1f} ／ {ST_TX['後半']['ann']:.1f}（清算後の通し {((TXliq / CAP) ** (1 / yrs) - 1) * 100:.1f}％）")
    print("  判定のまとめ … " + "・".join(f"{k} {v}" for k, v in res.items()))
    return 0


def d_judge(x: dict, ST_TX: dict, no_rb: bool) -> tuple:
    """判定は k_judge と同じ（でたらめ〈範囲から〉で判定・でたらめ〈同じ日の新高値ブレイクから〉は業績で絞る効き目の向き）。最初の買いの下限は D_MIN_BUYS。"""
    det = {}
    ok_tx, ok_ra, rb_up, rb_dn = True, True, True, True
    for h in ("前半", "後半"):
        a = x["st"][h]["ann"]
        ra = [y[h]["ann"] for y in x["ra"]]
        rb = [y[h]["ann"] for y in x["rb"]]
        d_tx, d_ra, d_rb = a - ST_TX[h]["ann"], a - float(np.mean(ra)), a - float(np.mean(rb))
        b_ra, b_rb = int(sum(1 for y in ra if a > y)), int(sum(1 for y in rb if a > y))
        det[h] = (d_tx, d_ra, b_ra, d_rb, b_rb)
        ok_tx &= d_tx >= PT_TX
        ok_ra &= (d_ra >= PT_RND) and (b_ra >= BEAT_MIN)
        rb_up &= (d_rb >= PT_RND) and (b_rb >= BEAT_MIN)
        rb_dn &= (d_rb <= -PT_RND) and (b_rb <= N_RAND - BEAT_MIN)
    rb_lab = "—" if no_rb else ("ある" if rb_up else ("裏目" if rb_dn else "見えない"))
    if min(x["nb"].values()) < D_MIN_BUYS or x["fill_ra"] < RND_FILL_MIN or x["fill_rb"] < RND_FILL_MIN:
        return "保留（データ不足）", rb_lab, det
    return ("候補" if (ok_tx and ok_ra) else ("範囲の中では効く" if ok_ra else "効かない")), rb_lab, det


# ───────────────────────── 29段目：外部の数字との答え合わせ ─────────────────────────
# 計画は wide_lab_plan29_2026-11-10（結果を見る前に決めたもの）。道具の土台（株価・分割・株数・配当・TOPIX の比べる相手）が、
# 公表されている数字と合うかを確かめる。売り買いのエンジンそのものは、別のレビューで確かめる
X_JPX_ASOF = "2026-09-30"
X_JPX_TR = ((1, 33.91), (3, 89.75), (5, 128.72))   # JPX の TOPIX（配当込み）の累積騰落率（％）。2026年9月30日現在のファクトシート
X_TOL_TR = 1.0          # ① 1306＋分配金と JPX の年率の差の許容（pt）
X_COR_REC = 0.98        # ② 作り直した TOPIX と 1306＋分配金の、月ごとの値動きの相関の下限
X_TOL_REC = 2.0         # ② 年率の差の許容（pt。前半・後半・通しのそれぞれ）
X_COR_FF = 0.70         # ③ Fama-French の日本の要因との、月ごとの相関の下限
FF_ROOT = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"


def x_tr_daily(D: dict) -> np.ndarray:
    """1306 の終値に分配金を足し戻した日ごとの値（税・費用なし）。分配金は、決算日より前の最後の営業日（権利落ちの日の近く）に再投資する。"""
    px = np.asarray(D["tp"]["c"], dtype=float)
    calD = D["cal"].astype("datetime64[D]")
    add = np.zeros(len(px))
    for d, a in yb.ETF_DIST.get("1306", ()):
        k = int(np.searchsorted(calD, np.datetime64(pd.Timestamp(d).date()), side="left")) - 1
        if 1 <= k < len(px):
            add[k] += a / 100.0
    g = np.ones(len(px))
    g[1:] = (px[1:] + add[1:]) / px[:-1]
    return np.cumprod(g)


def x_ann(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        return float("nan")
    return (float(np.prod(1.0 + r)) ** (12.0 / len(r)) - 1.0) * 100.0


def x_ff_fetch() -> tuple:
    """Kenneth French のデータ館から、日本の3要因（SMB・HML）と勢い（WML）の月次（％・米ドル建て）を取る。取れなければ空。"""
    import io
    import re
    import zipfile
    import requests
    names = {"3F": "Japan_3_Factors_CSV.zip", "MOM": "Japan_Mom_Factor_CSV.zip"}
    try:
        h = requests.get(FF_ROOT + "data_library.html", timeout=40, headers=_UA).text
        links = re.findall(r'href="(ftp/[^"]+?_CSV\.zip)"', h)
        a = next((l for l in links if l.endswith("Japan_3_Factors_CSV.zip")), None)
        b = next((l for l in links if re.search(r"Japan_Mo?m_Factor_CSV\.zip$", l, re.I)), None)
        if a:
            names["3F"] = a.split("/")[-1]
        if b:
            names["MOM"] = b.split("/")[-1]
    except Exception as e:
        log(f"  データ館のページが取れませんでした（{e}）。決まった名前で取ります")
    out, src = {}, []
    for key, nm in names.items():
        try:
            z = requests.get(FF_ROOT + "ftp/" + nm, timeout=60, headers=_UA).content
            zf = zipfile.ZipFile(io.BytesIO(z))
            txt = zf.read(zf.namelist()[0]).decode("latin-1")
        except Exception as e:
            log(f"  {nm} が取れませんでした（{e}）")
            continue
        src.append(nm)
        head = None
        for line in txt.splitlines():
            if "Annual" in line:
                break
            cells = [x.strip() for x in line.split(",")]
            if len(cells) < 2:
                continue
            if head is None and cells[0] == "" and any(re.search(r"[A-Za-z]", x) for x in cells[1:]):
                head = cells[1:]
            elif head is not None and re.fullmatch(r"\d{6}", cells[0]):
                for h_, v in zip(head, cells[1:]):
                    try:
                        out.setdefault(h_, {})[cells[0]] = float(v) / 100.0
                    except ValueError:
                        pass
    if "WML" not in out:
        for k in ("Mom", "MOM", "UMD"):
            if k in out:
                out["WML"] = out[k]
    return out, src


def x_ff_ours(D: dict, MK: np.ndarray) -> dict:
    """同じ作り方に近い形で、こちらのデータから SMB・HML・WML を作る（月ごとに組み直す・時価総額で加重）。
    大型＝時価総額の合計の上から90％までの銘柄。HML は純資産 ÷ 時価総額（直近の開示）、WML は12か月の勢い（直近1か月を除く）で、
    大型株の30％・70％の点で分ける。中身は東証の全市場の普通株（範囲の条件はかけない）。"""
    PXM, DIV, MC, BP, MOM = D["PXM"], D["DIV"], D["MCAP"], D["F"]["BP"], D["F"]["MOM12"]
    S, M = PXM.shape
    res = {"SMB": {}, "HML": {}, "WML": {}}
    for j in range(M - 1):
        p0, p1 = PXM[:, j], PXM[:, j + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (p1 + DIV[:, j + 1]) / p0 - 1.0
        mc = MC[:, j]
        ok = (MK[:, j] != "") & np.isfinite(mc) & (mc > 0) & np.isfinite(r) & (p0 > 0)
        if ok.sum() < 200:
            continue
        idx = np.nonzero(ok)[0]
        o = idx[np.argsort(-mc[idx])]
        cs = np.cumsum(mc[o]) / mc[o].sum()
        big = np.zeros(S, dtype=bool)
        big[o[(cs - mc[o] / mc[o].sum()) < 0.90]] = True
        ym = D["dates"][j + 1].strftime("%Y%m")

        def vw(m):
            w = mc[m]
            return float((w * r[m]).sum() / w.sum()) if m.any() and w.sum() > 0 else float("nan")

        def two_by_three(x):
            okx = ok & np.isfinite(x)
            xb = x[okx & big]
            if len(xb) < 30 or (okx & ~big).sum() < 30:
                return None
            lo, hi = np.percentile(xb, 30), np.percentile(xb, 70)
            g = {}
            for sz, m_sz in (("S", okx & ~big), ("B", okx & big)):
                g[sz + "L"] = vw(m_sz & (x <= lo))
                g[sz + "M"] = vw(m_sz & (x > lo) & (x < hi))
                g[sz + "H"] = vw(m_sz & (x >= hi))
            return g
        bpj = np.where(BP[:, j] > 0, BP[:, j], np.nan)
        g = two_by_three(bpj)
        if g is not None:
            res["HML"][ym] = 0.5 * (g["SH"] + g["BH"]) - 0.5 * (g["SL"] + g["BL"])
            res["SMB"][ym] = (g["SL"] + g["SM"] + g["SH"]) / 3.0 - (g["BL"] + g["BM"] + g["BH"]) / 3.0
        g = two_by_three(MOM[:, j])
        if g is not None:
            res["WML"][ym] = 0.5 * (g["SH"] + g["BH"]) - 0.5 * (g["SL"] + g["BL"])
    for k in res:                      # 組が空で作れなかった月は除く
        res[k] = {m: v for m, v in res[k].items() if np.isfinite(v)}
    return res


def lab29(store: dict) -> int:
    """29段目：外部の数字との答え合わせ。計画は wide_lab_plan29_2026-11-10。"""
    log("29段目（外部の数字との答え合わせ）を回します（指標を作っています）…")
    D = build(store)
    dates, me_idx, cal = D["dates"], D["me_idx"], D["cal"]
    MK = k_market_codes(store, D)
    PXM, DIV, MC, UNI = D["PXM"], D["DIV"], D["MCAP"], D["UNI"]
    S, M = PXM.shape
    j0 = next((j for j, t in enumerate(me_idx) if t >= 252), None)
    TR = x_tr_daily(D)
    calD = cal.astype("datetime64[D]")
    verdict = {}
    print("\n【日本株の手法・29段目（外部の数字との答え合わせ）（wide_lab）】")
    print("  （基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan29_2026-11-10）")
    print(f"  データ … {str(calD[0])}〜{str(calD[-1])}（{len(D['codes'])}銘柄）。税と費用は入れない（公表値と同じ土台で比べるため）")

    # ① 比べる相手の TOPIX（1306＋分配金）が、JPX の公表値（TOPIX 配当込み）と合うか
    print(f"\n  ① 比べる相手の TOPIX（1306 の終値＋分配金）と、JPX の TOPIX（配当込み）の公表値（{X_JPX_ASOF} 現在のファクトシート）")
    t_end = int(np.searchsorted(calD, np.datetime64(X_JPX_ASOF), side="right")) - 1
    ok1 = True
    for n, jpx in X_JPX_TR:
        d_st = (pd.Timestamp(X_JPX_ASOF) - pd.DateOffset(years=n)).date()
        t_st = int(np.searchsorted(calD, np.datetime64(d_st), side="right")) - 1
        if t_st < 0 or t_end <= t_st:
            print(f"     {n}年 … データが足りません")
            ok1 = False
            continue
        ours = (TR[t_end] / TR[t_st] - 1.0) * 100.0
        a_o = ((1 + ours / 100) ** (1 / n) - 1) * 100
        a_j = ((1 + jpx / 100) ** (1 / n) - 1) * 100
        good = abs(a_o - a_j) <= X_TOL_TR
        ok1 &= good
        print(f"     {n}年（{str(calD[t_st])}〜{str(calD[t_end])}） … こちら {ours:+.2f}％（年率 {a_o:.2f}） ／ JPX {jpx:+.2f}％（年率 {a_j:.2f}）"
              f" … 年率の差 {a_o - a_j:+.2f}pt … {'合う' if good else '合わない'}")
    # 分配金の利回りの桁（分配金 ÷ その日の終値）
    ys = []
    for d, a in yb.ETF_DIST.get("1306", ()):
        k = int(np.searchsorted(calD, np.datetime64(pd.Timestamp(d).date()), side="left")) - 1
        if 0 <= k < len(TR):
            ys.append((d[:4], a / 100.0 / float(D["tp"]["c"][k]) * 100.0))
    print("     分配金 ÷ 終値（％） … " + "・".join(f"{y} {v:.2f}" for y, v in ys))
    verdict["①"] = ok1

    # ② 銘柄ごとのデータから TOPIX を作り直す（一部・プライムの普通株を、時価総額で加重。浮動株の調整はしない）
    tr_m = np.array([TR[t] for t in me_idx])
    rows = []
    cover = []
    for j in range(max(1, j0 or 1), M):
        p0, p1 = PXM[:, j - 1], PXM[:, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (p1 + DIV[:, j]) / p0 - 1.0
        mc = MC[:, j - 1]
        prime_all = np.isin(MK[:, j - 1], ["0101", "0111"])
        okp = prime_all & np.isfinite(r) & np.isfinite(mc) & (mc > 0) & (p0 > 0)
        cover.append(okp.sum() / max(1, prime_all.sum()))
        w = mc[okp]
        cap = float((w * r[okp]).sum() / w.sum())
        oku = UNI[:, j - 1] & np.isfinite(r) & np.isfinite(mc) & (mc > 0)
        ew = float(r[oku].mean()) if oku.any() else np.nan
        cw = float((mc[oku] * r[oku]).sum() / mc[oku].sum()) if oku.any() else np.nan
        contrib = w * r[okp] / w.sum()
        top = np.argsort(-np.abs(contrib))[:3]
        ii = np.nonzero(okp)[0][top]
        rows.append({"d": dates[j], "tx": tr_m[j] / tr_m[j - 1] - 1.0, "cap": cap, "ew": ew, "cw": cw,
                     "top": [(D["codes"][i], float(r[i]) * 100, float(mc[i] / w.sum()) * 100) for i in ii]})
    R = pd.DataFrame(rows).set_index("d")
    print(f"\n  ② 銘柄ごとのデータ（株価・分割・株数・配当）から作り直した TOPIX（一部・プライムの普通株を時価総額で加重。浮動株の調整なし）と、①の TOPIX")
    print(f"     比べる期間 {R.index[0]:%Y-%m}〜{R.index[-1]:%Y-%m}（{len(R)}か月）。時価総額が分かった銘柄の割合（一部・プライムのうち）平均 {np.mean(cover) * 100:.0f}％・最少 {np.min(cover) * 100:.0f}％")
    ok2 = True
    for h, m in (("前半", R.index < SPLIT), ("後半", R.index >= SPLIT), ("通し", np.ones(len(R), dtype=bool))):
        a, b = x_ann(R["cap"][m]), x_ann(R["tx"][m])
        cor = float(np.corrcoef(R["cap"][m], R["tx"][m])[0, 1])
        te = float(np.std(R["cap"][m] - R["tx"][m]) * np.sqrt(12) * 100)
        good = (cor >= X_COR_REC) and abs(a - b) <= X_TOL_REC
        ok2 &= good
        print(f"     {h} … 作り直し {a:.1f}％ ／ TOPIX {b:.1f}％（年率の差 {a - b:+.1f}pt）・相関 {cor:.3f}・ずれの大きさ（年率） {te:.1f}％ … {'合う' if good else '合わない'}")
    dif = (R["cap"] - R["tx"]).abs().sort_values(ascending=False)
    print("     ずれの大きい月（作り直し − TOPIX。その月に効いた銘柄：コード 騰落率％ 重み％）")
    for d in dif.index[:3]:
        x = R.loc[d]
        print(f"       {d:%Y-%m} … {x['cap'] * 100:+.1f}％ ／ {x['tx'] * 100:+.1f}％ … " +
              "・".join(f"{c} {rr:+.0f}％ {ww:.1f}％" for c, rr, ww in x["top"]))
    verdict["②"] = ok2
    print(f"\n  参考（判定しない）：範囲（売買代金5,000万円以上・時価総額100億円以上）を、同じ額ずつ持つ形と時価総額で加重する形（税・費用なし。毎月組み直す）")
    for h, m in (("前半", R.index < SPLIT), ("後半", R.index >= SPLIT), ("通し", np.ones(len(R), dtype=bool))):
        print(f"     {h} … 同じ額ずつ {x_ann(R['ew'][m]):.1f}％ ／ 時価総額で加重 {x_ann(R['cw'][m]):.1f}％ ／ TOPIX {x_ann(R['tx'][m]):.1f}％")

    # ③ Fama-French の日本の要因と、こちらで作った要因の突き合わせ
    print("\n  ③ Kenneth French のデータ館の日本の要因（月次・米ドル建て）と、こちらのデータで作った要因（円建て。月ごとに組み直す）")
    ff, src = x_ff_fetch()
    ours = x_ff_ours(D, MK)
    if not ff:
        print("     データ館から取れませんでした … 判定できない")
        verdict["③"] = None
    else:
        print(f"     取ったファイル … {'・'.join(src)}")
        ok3 = True
        for k in ("SMB", "HML", "WML"):
            a, b = ff.get(k, {}), ours.get(k, {})
            ks = sorted(x for x in set(a) & set(b) if np.isfinite(a[x]) and np.isfinite(b[x]))
            ks = [x for x in ks if x >= f"{dates[j0]:%Y%m}"] if j0 is not None else ks
            if len(ks) < 24:
                print(f"     {k} … 重なる月が {len(ks)}か月しかありません … 判定できない")
                ok3 = False
                continue
            xa, xb = np.array([a[x] for x in ks]), np.array([b[x] for x in ks])
            cor = float(np.corrcoef(xa, xb)[0, 1])
            good = cor >= X_COR_FF
            ok3 &= good
            print(f"     {k}（{ks[0][:4]}-{ks[0][4:]}〜{ks[-1][:4]}-{ks[-1][4:]}・{len(ks)}か月） … 相関 {cor:.2f}・年率 French {x_ann(xa):+.1f}％ ／ こちら {x_ann(xb):+.1f}％"
                  f" … {'合う' if good else '合わない'}")
        verdict["③"] = ok3
    sd = {k: len(v) for k, v in SPLIT_DIAG.items()}
    print(f"\n  ④ 直したあとの診断（判定しない）：期末のまわり（期末の7日前〜開示の日）に分割・併合があり、基準があいまいだったもの … "
          f"通期の配当 {sd['div']}件（うち分割後に直してあると判断 {sd['div_post']}件）・株数 {sd['sh']}件（うち {sd['sh_post']}件）")
    txt = {True: "合う", False: "合わない", None: "判定できない"}
    print("\n  判定のまとめ … " + "・".join(f"{k} {txt[v]}" for k, v in verdict.items()))
    return 0


# ───────────────────────── 31段目：ヘムの H3・H0 の並走記録（h_forward_plan_2026-11-10） ─────────────────────────
# 決まりは h_forward_plan_2026-11-10（記録を始める前に決めたもの）。24段目の H3・H0 の決まりを変えずに、2026年11月末の終値で組み、
# 12月からの値動きを記録する。生のデータ（株価・要約）は Actions のキャッシュに置き、リポジトリには data/h_track.json（結果だけ）を入れる
HT_START_YM = "2026-11"            # この月の最後の営業日の終値で最初に組む
HT_FORMS = (("H3", "増配テーマの近い形"), ("H0", "基本条件の指標だけ"), ("Y1", "山口揚平の保守的な安全域（11月10日に足した）"))
HT_OUT = OUT / "h_track.json"
HT_NW_LAG = 3                       # 月ごとの超過の t（Newey-West・ずらし3か月）
HT_SUMM_KEY = ("Code", "DiscDate", "DiscTime", "DocType", "CurPerType", "CurFYEn")


def _append_bar(b: dict, d, c, va, f, o=None, h=None, l_=None, v=None) -> None:
    """1銘柄に1日分を足す。分割・併合の係数（その日が権利落ち）があれば、それより前の株価をいまの基準に直してから足す。"""
    if f is not None and f == f and f > 0 and abs(f - 1.0) > 1e-12:
        b["c"] = (b["c"].astype(np.float64) * f).astype(np.float32)
        for k in ("o", "h", "l"):
            if k in b:
                b[k] = (b[k].astype(np.float64) * f).astype(np.float32)
        if "v" in b:
            b["v"] = (b["v"].astype(np.float64) / f).astype(np.float32)
        b["fd"] = np.append(b["fd"], np.datetime64(d, "D"))
        b["ff"] = np.append(b["ff"], float(f))
    b["d"] = np.append(b["d"], np.datetime64(d, "D"))
    b["c"] = np.append(b["c"], np.float32(np.nan if c is None else c))
    b["va"] = np.append(b["va"], np.float32(np.nan if va is None else va))
    for k, x in (("o", o), ("h", h), ("l", l_), ("v", v)):
        if k in b:
            b[k] = np.append(b[k], np.float32(np.nan if x is None else x))


def update_by_date(jq, store: dict) -> bool:
    """31段目：キャッシュの最後の日の次から今日までを、日付ごとに取り足す（全銘柄の株価・決算短信の要約・月末の銘柄一覧）。
    株価はその日の終値（調整前）を足し、権利落ちの日の係数でそれより前をいまの基準に直す（pack_bars と同じ基準になる）。"""
    tp = store["etf1306"]
    last = pd.Timestamp(tp["d"][-1])
    today = pd.Timestamp(date.today())
    hol = set(store.setdefault("upd_holidays", []))
    sdone = set(store.setdefault("upd_summ_done", []))
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(last + pd.Timedelta(days=1), today) if d.strftime("%Y-%m-%d") not in hol]
    if days:
        log(f"株価（全銘柄・日付ごと）を取り足しています（{days[0]}〜・{len(days)}日）…")
    bars, fins = store["bars"], store["fins"]
    n_new = n_split = 0
    for ds in days:
        time.sleep(yb.API_SLEEP)
        rows = jq.get("/v2/equities/bars/daily", {"date": ds})
        if not rows:
            if ds < today.strftime("%Y-%m-%d"):
                hol.add(ds)
            continue
        for r in rows:
            c = yb.norm_code(r.get("Code") or "")
            if not c:
                continue
            cr = num(r.get("C", r.get("Close")))
            va = num(r.get("Va", r.get("TurnoverValue")))
            if va is None:
                vo = num(r.get("Vo", r.get("Volume")))
                va = vo * cr if (vo is not None and cr is not None) else None
            f = num(r.get("AdjFactor", r.get("AdjustmentFactor"))) or 1.0
            ohlv = (num(r.get("O", r.get("Open"))), num(r.get("H", r.get("High"))), num(r.get("L", r.get("Low"))), num(r.get("Vo", r.get("Volume"))))
            if c == "1306":
                _append_bar(tp, ds, cr, va, f, *ohlv)
                continue
            b = bars.get(c)
            if b is None:
                b = {"d": np.array([], dtype="datetime64[D]"), "c": np.array([], dtype=np.float32), "va": np.array([], dtype=np.float32),
                     "fd": np.array([], dtype="datetime64[D]"), "ff": np.array([], dtype=float)}
                bars[c] = b
                fins.setdefault(c, [])
                n_new += 1
            elif len(b["d"]) and b["d"][-1] >= np.datetime64(ds, "D"):
                continue
            if abs(f - 1.0) > 1e-12:
                n_split += 1
            _append_bar(b, ds, cr, va, f, *ohlv)
        if store.get("done") is not None:
            store["done"] = sorted(set(store["done"]) | set(bars))
    # 決算短信の要約：株価のある日すべて（キャッシュの最後の日の翌日から）。今日の分は、夜の開示があるので次の実行でも取り直す
    s_days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(last + pd.Timedelta(days=1), today)
              if d.strftime("%Y-%m-%d") not in sdone and d.strftime("%Y-%m-%d") not in hol]
    n_s = 0
    for ds in s_days:
        time.sleep(yb.API_SLEEP)
        rows = jq.get("/v2/fins/summary", {"date": ds})
        for r in rows or []:
            c = yb.norm_code(r.get("Code") or "")
            if not c:
                continue
            pr = prune_fins([r])[0]
            lst = fins.setdefault(c, [])
            key = tuple(str(pr.get(k)) for k in HT_SUMM_KEY)
            if any(tuple(str(x.get(k)) for k in HT_SUMM_KEY) == key for x in lst[-12:]):
                continue
            lst.append(pr)
            n_s += 1
        if ds < today.strftime("%Y-%m-%d"):
            sdone.add(ds)
    store["upd_holidays"], store["upd_summ_done"] = sorted(hol), sorted(sdone)
    ok = fetch_months(jq, store)
    log(f"  取り足した … 株価 {len(days)}日（新しい銘柄 {n_new}・分割・併合 {n_split}）・要約 {n_s}件・いちばん新しい日 {str(tp['d'][-1])}")
    return ok


def ht_nw(x: np.ndarray, lag: int = HT_NW_LAG) -> tuple:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return (float(x.mean()) if n else float("nan")), float("nan")
    m = float(x.mean())
    u = x - m
    s = float(np.sum(u * u))
    for L in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - L / (lag + 1.0)) * float(np.sum(u[L:] * u[:-L]))
    se = np.sqrt(max(s, 0.0)) / n
    return m, (m / se if se > 0 else float("nan"))


def htrack(store: dict) -> int:
    """31段目：H3・H0 の並走記録を作り直して data/h_track.json に書く。決まりは h_forward_plan_2026-11-10。"""
    log("31段目（ヘムの H3・H0 の並走記録）を作っています…")
    D = build(store)
    dates, UNI, PXM, DIV, codes, names = D["dates"], D["UNI"], D["PXM"], D["DIV"], D["codes"], D["names"]
    M = len(dates)
    sc = dict(h_scores(store, D))
    sc["Y1"] = y_scores(store, D)["Y1"]          # 34段目（2026年11月10日に足した。記録を始める前）
    ym = [d.strftime("%Y-%m") for d in dates]
    out = {"generated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(timespec="seconds"), "plan": "h_forward_plan_2026-11-10",
           "data_last": str(D["cal"][-1]), "start_month_end": HT_START_YM, "forms": {}}
    jl = M - 1
    print("\n【31段目：ヘムの H3・H0 の並走記録（wide_lab）】（決まりは記録を始める前に決めたもの。プロジェクトの h_forward_plan_2026-11-10）")
    print(f"  データ … いちばん新しい日 {out['data_last']}・最後の月末 {dates[jl].date()}・範囲 {int(UNI[:, jl].sum())}銘柄")
    if HT_START_YM not in ym:
        out["status"] = "記録前"
        for k, lab in HT_FORMS:
            ok = UNI[:, jl] & np.isfinite(sc[k][:, jl])
            idx = np.nonzero(ok)[0]
            idx = idx[np.argsort(-sc[k][idx, jl], kind="stable")]
            out["forms"][k] = {"label": lab, "n_pass_last": int(len(idx)),
                               "top20_last": [{"code": codes[i], "name": names.get(codes[i], ""), "score": round(float(sc[k][i, jl]), 3)} for i in idx[:20]]}
            print(f"  {k} {lab} … 記録前（{HT_START_YM} 末から）。いちばん新しい月末の合格 {len(idx)}銘柄・上位：" +
                  "・".join(f"{codes[i]} {names.get(codes[i], '')}" for i in idx[:5]))
        HT_OUT.parent.mkdir(parents=True, exist_ok=True)
        HT_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return 0
    j0 = ym.index(HT_START_YM)
    ev = dates[j0:]
    tv, tl = topix_taxed(D, j0)
    out["status"] = "記録中"
    out["dates"] = [d.strftime("%Y-%m-%d") for d in ev]
    out["topix"] = {"vals": [round(float(x)) for x in tv], "liq": round(float(tl))}
    for k, lab in HT_FORMS:
        s = sc[k]
        rk = ranks_from(s, UNI)
        fo = {"label": lab}
        for nm, kp in KEEPS:
            v, liq, st = sim(rk, PXM, DIV, j0, kp)
            hold = [{"code": codes[i], "name": names.get(codes[i], ""), "shares": int(sh)} for i, (sh, _c) in sorted(st["pos"].items())]
            fo[nm] = {"vals": [round(float(x)) for x in v], "liq": round(float(liq)), "n_buy": st["n_buy"], "n_sell": st["n_sell"], "holdings": hold}
        # 判定に使う測り方：月末ごとに合格したすべての銘柄（同じ額ずつ・配当を含む・費用なし）の翌月の値上がり − 範囲の全銘柄の平均
        ex, npass = [], []
        for j in range(j0, M - 1):
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (PXM[:, j + 1] + DIV[:, j + 1]) / PXM[:, j] - 1.0
            u = UNI[:, j] & np.isfinite(r)
            p = u & np.isfinite(s[:, j])
            npass.append(int(p.sum()))
            if p.sum() >= 1 and u.sum() >= 50:
                ex.append(float(r[p].mean() - r[u].mean()))
        m, t = ht_nw(np.array(ex))
        fo["excess"] = {"monthly": [round(x, 5) for x in ex], "mean": None if m != m else round(m, 5), "t": None if t != t else round(t, 2),
                        "n_months": len(ex), "n_pass": npass}
        out["forms"][k] = fo
        a = fo["A"]
        print(f"\n  {k} {lab}")
        print(f"     紙の上の運用（A・20銘柄）… 評価額 {a['vals'][-1]:,} 円（{ev[0].date()} から）・清算後 {a['liq']:,} 円 ／ TOPIX（税あり）{out['topix']['vals'][-1]:,} 円・清算後 {out['topix']['liq']:,} 円")
        print(f"     判定に使う超過（合格した銘柄すべて − 範囲の平均。月ごと）… {len(ex)}か月・平均 "
              + ("—" if m != m else f"{m * 100:+.2f}％／月") + "・t " + ("—" if t != t else f"{t:.2f}") + "（途中。判定は計画の時点で）")
        print("     いまの持ち株（A）… " + "・".join(f"{h['code']} {h['name']}" for h in a["holdings"][:20]))
    HT_OUT.parent.mkdir(parents=True, exist_ok=True)
    HT_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"  {HT_OUT} に書きました")
    return 0


def run_mode(mode: str, store: dict, args) -> int:
    """1つの mode を回す（データはそろっている前提で、足りなければ取る）。14段目の lab14 は、これを続けて呼ぶ。"""
    global LAB_TITLE, LAB_PLAN, LAB_NOTE
    LAB_TITLE, LAB_PLAN = "1回目", "wide_lab_plan_2026-11-04"
    if mode == "lab3":
        LAB_TITLE, LAB_PLAN = "3回目（組み合わせ）", "wide_lab_plan3_2026-11-04"
    elif mode == "lab4":
        LAB_TITLE, LAB_PLAN = "4回目（売買の内訳）", "wide_lab_plan4_2026-11-04"
    elif mode == "lab5":
        LAB_TITLE, LAB_PLAN = "5段目（残りの型：毎月の点数）", "wide_lab_plan5_2026-11-04"
    elif mode == "lab12":
        LAB_TITLE, LAB_PLAN = "12段目（財務の明細の型）", "wide_lab_plan12_2026-11-05"
    elif mode == "lab13":
        LAB_TITLE, LAB_PLAN = "13段目（J-Quants の残りのデータ）", "wide_lab_plan13_2026-11-05"
    if mode == "lab22":
        return lab22(store)
    if mode == "lab23":
        return lab23(store)
    if mode == "lab24":
        return lab24(store)
    if mode == "lab25":
        return lab25(store)
    if mode == "lab26":
        return lab26(store)
    if mode == "lab27":
        return lab27(store)
    if mode == "lab28":
        return lab28(store)
    if mode == "lab29":
        return lab29(store)
    if mode == "lab32":
        return lab32(store)
    if mode == "lab33":
        return lab33(store)
    if mode == "lab34":
        return lab34(store)
    if mode == "lab36":
        return lab36(store)
    if mode == "lab35":
        # 35段目：信用取引の週末残高（スタンダード）を週ごとに取り、21段目と同じ別のファイル（data/wide_k.pkl）に持つ
        xs = k_xs_load()
        if not x_complete(store, xs, ("mi",)):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_x(yb.JQ(key), store, xs, ("mi",)):
                print("\n■ まだ終わっていません（信用取引の週末残高を取っている途中です）。同じ設定のまま、もう一度押してください。")
                return EXIT_UNFINISHED
        st = xs.get("mi", {})
        log(f"  信用取引の週末残高 … {len(st.get('rows', [])):,}件・取れた週 {len(st.get('done', ())):,}・取れなかった週 {len(st.get('fail', ()))}"
            + (f"・取るのをやめた：{st['bad']}" if st.get("bad") else ""))
        if st.get("bad"):
            print("\n■ 信用取引の週末残高が取れなかったので、Q3・Q6 は保留になります")
        return lab35(store, xs)
    if mode == "lab21":
        # 21段目：上岡式の答え合わせ。業種別空売り比率（スタンダード）を日ごとに取り、別のファイル（data/wide_k.pkl）に持つ
        xs = k_xs_load()
        if not x_complete(store, xs, ("srt",)):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_x(yb.JQ(key), store, xs, ("srt",)):
                print("\n■ まだ終わっていません（業種別空売り比率を取っている途中です）。同じ設定のまま、もう一度押してください。")
                return EXIT_UNFINISHED
        st = xs.get("srt", {})
        log(f"  業種別空売り比率 … {len(st.get('rows', [])):,}日・取れなかった日 {len(st.get('fail', ()))}日"
            + (f"・取るのをやめた：{st['bad']}" if st.get("bad") else ""))
        return lab21(store, xs)
    if mode == "lab7":
        return lab7(store)
    if mode == "lab6":
        return lab6(store)
    if mode == "lab9":
        return lab9(store)
    if mode == "lab20":
        xs = x_load()
        if not (xs.get("ext") or {}).get("series") or len(xs["ext"]["series"]) < len(EXT_LAB):
            log("市況のデータ（FRED・財務省。取れなければ代わりの出どころ）を取ります…")
            ext = fetch_ext(xs.get("ext"))
            if ext["series"]:
                xs["ext"] = ext
                x_save(xs, f"市況のデータ {len(ext['series'])}種類を保存しました")
        return lab20(store, xs)
    if mode == "lab19" and LONG:
        # plan19 の5：20年分では出来事の組み合わせ（DIVE2C・DIVBP）だけを回す（まとめて学ばせる型と MALS は9年分で準候補なし・20年分のデータもない）
        xs = x_load()
        if STAGE != "confirm" and not x_complete(store, xs, ("div", "ed")):
            print("\n■ 20年分の配当金情報・決算発表予定日がそろっていません。先に years=20 で lab15 か xfetch を回してください。")
            return 2
        return lab15(store, xs, combo=True)
    if mode in ("lab16", "lab18", "lab19"):
        xs = x_load()
        bd = bd_load()
        if not (LONG and STAGE == "confirm") and not x_complete(store, xs, ("div", "ed", "ssr", "mi", "fsd2", "mal", "msh", "xsh")):
            print("\n■ 第1部のデータ（5・12・13段目）がそろっていません。先に lab5・lab12・lab13 を回してください。")
            return 2
        if mode == "lab16":
            return lab16(store, xs, bd)
        if mode == "lab18":
            if not (LONG and STAGE == "confirm") and not x_complete(store, xs, ("optu",)):
                key = os.environ.get("J_QUANTS_API_KEY")
                if not key:
                    print("J_QUANTS_API_KEY が設定されていません")
                    return 2
                if not fetch_x(yb.JQ(key), store, xs, ("optu",)):
                    print("\n■ まだ終わっていません（個別株オプションを取っている途中です）。同じ設定のまま、もう一度押してください。")
                    return EXIT_UNFINISHED
            log(f"  個別株オプション … {len(xs.get('optu', {}).get('rows', [])):,}件" + (f"・取るのをやめた：{xs['optu']['bad']}" if xs.get("optu", {}).get("bad") else ""))
            LAB_TITLE, LAB_PLAN = "18段目（毎月の点数の残り）", "wide_lab_plan18_2026-11-05"
            if "M36" in [c.upper() for c in CFG]:
                # plan18 の5：第2部の計画（wide_program2）にあった「3年の勢い」を足す（直近1か月を除く36か月の成績の大きい順）
                LAB_TITLE = "18段目の追記（3年の勢い）"
                LAB_NOTE = "18段目の追記：第2部の計画にあった「3年の勢い」（MOM36）。3年分の株価がたまってから比べ始める（前半が短い）"

                def mom36(D):
                    PXM, me_idx = D["PXM"], D["me_idx"]
                    out = np.full(PXM.shape, np.nan)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        for j in range(36, PXM.shape[1]):
                            out[:, j] = PXM[:, j - 1] / PXM[:, j - 36] - 1.0
                    out[~np.isfinite(out)] = np.nan
                    j_ok = next((j for j in range(PXM.shape[1]) if np.isfinite(out[:, j]).sum() >= 100), PXM.shape[1] - 1)
                    START_T["MOM36"] = int(me_idx[j_ok])
                    return out
                rc = lab1(store, extra={"MOM36": (mom36, "3年の勢い（直近1か月を除く36か月の成績の大きい順）")})
                LAB_NOTE = ""
                return rc
            ex18 = lab18_scores(store, xs)
            if "IVU" in [c.upper() for c in CFG] and "IVLOW" in ex18:
                # plan18 の4：IVLOW を、IV がある銘柄だけを範囲にして回す（でたらめも同じ範囲）
                LAB_TITLE = "18段目の追記（IVLOW を個別株オプションのある銘柄だけで）"
                LAB_NOTE = "18段目の追記：範囲を、その月末に IV（いちばん近い限月の中央値の20日平均）がある銘柄だけに絞った。でたらめも同じ範囲"
                ivc = {}
                fn_iv = ex18["IVLOW"][0]

                def iv_mat(D):
                    if "m" not in ivc:
                        ivc["m"] = fn_iv(D)
                    return ivc["m"]
                rc = lab1(store, extra={"IVLOW": ((lambda D: iv_mat(D)), ex18["IVLOW"][1])}, uni_fn=lambda D: np.isfinite(iv_mat(D)))
                LAB_NOTE = ""
                return rc
            log("18段目（毎月の点数の残り）を回します…")
            return lab1(store, extra=ex18)
        # 19段目：①すべての毎月の点数でまとめて学ばせる ②出来事の組み合わせ ③日々公表の対象入りを売る形
        ex = monthly_extra_all(store, xs, bd)
        ex.update(lab18_scores(store, xs))
        log(f"19段目：まとめて学ばせる型 {len(ex)}つ")
        lab9(store, extra={k: v for k, v in ex.items() if k not in dict(LAB1)})
        lab15(store, xs, combo=True)
        if not (ONLY and "MALS" not in ONLY):
            lab8(store, extra={"MALS": (mals_score(xs), "日々公表の対象に入ってから20営業日以内の株")},
                 head="【日本株の手法・19段目（日々公表の対象入りを売る形）（wide_lab）：東証全銘柄】（基準は結果を見る前に決めたもの。計画はプロジェクトの wide_lab_plan19_2026-11-05。"
                      "売る側は、その月末までの20営業日に日々公表の対象に入った株）")
        return 0
    if mode == "lab17":
        return lab17(store)
    if mode == "lab15":
        xs = x_load()
        need15 = ("div", "ed") if LONG else ("div", "ed", "mal", "lvh")     # 20年分では、日々公表と大量保有は取っていない（その2つは回さない）
        if not (LONG and STAGE == "confirm") and not x_complete(store, xs, need15):
            print("\n■ 15段目のデータ（配当金情報・決算発表予定日・日々公表・大量保有）がそろっていません。先に lab5 と lab13 を回してください。")
            return 2
        return lab15(store, xs)
    if mode == "lab13":
        xs = x_load()
        if not x_complete(store, xs, X13):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_x(yb.JQ(key), store, xs, X13):
                print("\n■ まだ終わっていません（13段目のデータを取っている途中です）。同じ設定のまま、もう一度押してください。")
                return EXIT_UNFINISHED
        for n in X13:
            st = xs.get(n, {})
            log(f"  {n} … {len(st.get('rows', [])):,}件・取れた日 {len(st.get('done', ()))}日・取れなかった日 {len(st.get('fail', ()))}日"
                + (f"・取るのをやめた：{st['bad']}" if st.get("bad") else ""))
        return lab13(store, xs)
    if mode == "lab12":
        xs = x_load()
        if not x_complete(store, xs, ("fsd2",)):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_x(yb.JQ(key), store, xs, ("fsd2",)):
                print("\n■ まだ終わっていません（財務の明細を取っている途中です）。同じ設定のまま、もう一度押してください。")
                return EXIT_UNFINISHED
        if xs.get("fsd2", {}).get("bad"):
            print(f"\n■ 財務の明細が取れませんでした（{xs['fsd2']['bad']}）。12段目は保留です。")
            return 2
        log("12段目（財務の明細の型）を回します…")
        return lab1(store, extra=lab12_scores(store, xs))
    if mode in ("lab10", "lab11", "xfetch"):
        xs = x_load()
        need = ("div", "ed")              # 10・11段目で使うのは、配当金情報と決算発表予定日の履歴だけ
        if not x_complete(store, xs, need):
            if mode == "lab10":
                print("\n■ 5段目のデータ（配当金情報・決算発表予定日）がそろっていません。先に lab5 か xfetch を回してください。")
                return 2
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_x(yb.JQ(key), store, xs, need):
                print("\n■ まだ終わっていません（配当金情報などを取っている途中です）。同じ設定のまま、もう一度押してください。")
                return EXIT_UNFINISHED
        if mode == "xfetch":
            for n in need:
                st = xs.get(n, {})
                d = sorted(st.get("done", ()))
                print(f"  {n} … {len(st.get('rows', [])):,}件・取れた日 {len(d)}日（{d[0] if d else '-'}〜{d[-1] if d else '-'}）・取れなかった日 {len(st.get('fail', ()))}日" + ("・取るのをやめた：" + str(st['bad']) if st.get('bad') else ""))
            print("\n■ 配当金情報と決算発表予定日の履歴がそろっています（成績は計算していません）。")
            return 0
        if mode == "lab11":
            return lab11(store, xs)
        lab5_events(store, xs, fill=True)
        return 0
    if mode == "lab5":
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
    if mode == "lab8":
        if not all(k in store.get("mrgn", {}) for k in store["months"]):
            key = os.environ.get("J_QUANTS_API_KEY")
            if not key:
                print("J_QUANTS_API_KEY が設定されていません")
                return 2
            if not fetch_margin(yb.JQ(key), store):
                print("\n■ まだ終わっていません（貸借銘柄を取っている途中です）。もう一度押してください。")
                return EXIT_UNFINISHED
        return lab8(store)
    if mode == "lab4":
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
    if mode == "lab3":
        sel = [x.strip().upper() for x in args.combo.split(",") if x.strip()]
        bad = [k for k in sel if k not in dict(LAB1)]
        if bad:
            print(f"\n■ 組み合わせに使えない記号があります：{'・'.join(bad)}")
            return 2
        combos = {"VQM": (VQM, "割安・質・勢い（固定）")}
        if len(sel) >= 2:
            combos["SEL"] = (tuple(sel), "1回目で効いた型の組み合わせ")
        return lab1(store, combos)
    return lab1(store) if mode == "lab1" else lab2(store)


LAB14_MODES = ("lab1", "lab3", "lab2", "lab4", "lab5", "lab8", "lab9", "lab10", "lab11", "lab12", "lab13")


def lab14(store: dict, args) -> int:
    """14段目：費用と税の前提（--cost の S1・S2）で、1〜13段目のうち軽い段をまとめて回す（6段目と7段目は別に回す）。
    3回目の SEL は、第1部と同じ決まりで、この前提での1回目の結果から作る。計画は wide_lab_plan14_2026-11-05。"""
    log(f"14段目：費用と税の前提 {COST}（株 片道{SLIP * 100:.1f}％・1306 片道{ETF_SLIP * 100:.2f}％・{TAX_TXT}）で、{'・'.join(LAB14_MODES)} を続けて回します")
    out = {}
    for m in LAB14_MODES:
        if elapsed() > float(os.environ.get("LAB14_LIMIT_MIN", "150")):
            log(f"  ■ 時間が足りないので {m} 以降は回しません")
            out[m] = "時間切れ"
            break
        if m == "lab3":
            res1 = LAST_LAB1 or {}
            sel = [k for k, r in res1.items() if r.get("verdict") in ("候補", "範囲の中では効く") and k in dict(LAB1)]
            args.combo = ",".join(sel)
            log(f"  3回目の SEL … {'・'.join(sel) if len(sel) >= 2 else 'なし（範囲の中では効く以上が2つ未満）'}")
        print(f"\n\n======== 14段目（{COST}）：{m} ========")
        try:
            rc = run_mode(m, store, args)
        except Exception as e:          # 1つの段が落ちても、残りは回す
            import traceback
            traceback.print_exc()
            rc = f"失敗（{type(e).__name__}: {str(e)[:120]}）"
        out[m] = rc
        log(f"  {m} … 終わり（{rc}・開始から{elapsed():.0f}分）")
    print(f"\n【14段目（{COST}）でまとめて回した段】 " + "・".join(f"{k}={v}" for k, v in out.items()))
    return 0 if all(v in (0, None) for v in out.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fetch", "lab1", "lab2", "lab3", "lab4", "lab5", "lab6", "lab7", "lab8", "lab9", "lab10", "lab11", "lab12", "lab13", "lab14", "lab15", "lab16", "lab17", "lab18", "lab19", "lab20", "lab21", "lab22", "lab23", "lab24", "lab25", "lab26", "lab27", "lab28", "lab29", "lab32", "lab33", "lab34", "lab35", "lab36", "htrack", "xfetch"], default="lab1")
    ap.add_argument("--combo", default="", help="3回目：1回目で「範囲の中では効く」以上だった型の記号（カンマ区切り）。2つ以上なら SEL として回す")
    ap.add_argument("--only", default="", help="回す型の記号（カンマ区切り。空欄なら全部）。確かめる段階で候補だけを回すとき")
    ap.add_argument("--cfg", default="", help="7段目の確かめる段階で回す組み立て（例: VQM/TOP500/20/Q,NXG/TOP500/20/Q）")
    args = ap.parse_args()
    global ONLY, CFG
    ONLY = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    CFG = [x.strip() for x in args.cfg.split(",") if x.strip()]
    global CFG36
    CFG36 = {c.upper() for c in CFG} & {v[0] for v in R36_VARS}     # 36段目 B：回す組み立て（空なら全部）
    if "SMALL" in [c.upper() for c in CFG]:      # 30段目：小型株を含む範囲（売買代金の中央値1,000万円以上・時価総額30億円以上）
        global MIN_LIQ, MIN_MCAP
        MIN_LIQ, MIN_MCAP = 1e7, 3e9
        log(f"  範囲：{uni_txt()}")
    if LONG:
        yb.LONG_MODE = True          # 株価を「20年前まで」で取る。取れなければプレミアムではないので止まる
        yb.API_SLEEP = float(os.environ.get("LONG_API_SLEEP", "0.25"))    # プレミアムは呼び出しの上限が大きい
        log(f"確かめる段階（{YEARS}年分・データは {CACHE}）")
    store = load()
    if args.mode == "htrack":         # 31段目：キャッシュを日付ごとに取り足してから、並走記録を作り直す
        key = os.environ.get("J_QUANTS_API_KEY")
        if not store.get("complete"):
            print("■ 9年分のキャッシュがありません（wide-lab の fetch のあとに動かす）")
            return 2
        if key:
            update_by_date(yb.JQ(key), store)
            save(store, "取り足した分を保存しました")
        else:
            log("  J_QUANTS_API_KEY がないので、取り足さずに作ります")
        return htrack(store)
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
    if KEEP_OHLC and store.get("etf1306") and "o" not in store["etf1306"]:
        # 17段目の参考（夜だけ・昼だけ）のため：11月4日夜より前に取った 1306 には始値がないので、1306 だけ取り直す（1回の呼び出し）
        key = os.environ.get("J_QUANTS_API_KEY")
        if key:
            try:
                b = pack_bars(yb.fetch_bars(yb.JQ(key), "1306", YEARS))
                if b and "o" in b and len(b["d"]) >= len(store["etf1306"]["d"]) * 0.95:
                    store["etf1306"] = b
                    save(store)
                    log(f"  1306 を始値つきで取り直しました（{len(b['d'])}日）")
            except Exception as e:          # 取れなくても、ほかの計算は続ける
                log(f"  1306 の取り直しに失敗しました（{type(e).__name__}）")
    n = len(store.get("bars", {}))
    log(f"データ … 銘柄一覧 {len(store.get('months', {}))}か月・株価 {n}銘柄・株価が取れなかった {len(store.get('nodata', []))}銘柄")
    if args.mode == "fetch":
        print("\n■ 東証全銘柄のデータがそろっています。")
        return 0
    if elapsed() > COMPUTE_LIMIT_MIN:
        print("\n■ まだ終わっていません（データは取り終えましたが、計算の時間が残っていません）。もう一度押してください。")
        return EXIT_UNFINISHED
    if args.mode == "lab14":
        return lab14(store, args)
    return run_mode(args.mode, store, args)


if __name__ == "__main__":
    sys.exit(main())

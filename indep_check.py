#!/usr/bin/env python3
"""indep_check.py -- independent backtest (TOPIX proxy / H0 / M1) written from SPEC.md only.

usage: python indep_check.py DATA.pkl OUT.json
stdlib + numpy + pandas only.  See DECISIONS.md for every judgement call.
"""
import sys, re, json, time, math, pickle, datetime, warnings
from functools import lru_cache
import numpy as np
import pandas as pd

TAX = 0.20315
FEE = 0.002
INIT = 10_000_000.0
TSE_MKT = {"0101", "0102", "0104", "0106", "0107", "0111", "0112", "0113"}
FIN_SECTORS = {"銀行業", "保険業", "証券、商品先物取引業", "その他金融業"}
END_MONTH = "2026-10"
SPLIT_MONTH = "2022-10"
WARMUP = 252
LIQ_DAYS, LIQ_MIN = 60, 5e7
MCAP_MIN = 1e10
ETF_DIST = [("2017-07-10", 2.60), ("2018-07-10", 3.05), ("2019-07-10", 3.33), ("2020-07-10", 3.29),
            ("2021-07-10", 3.66), ("2022-07-10", 4.64), ("2023-07-10", 5.21), ("2024-07-10", 5.79),
            ("2025-07-10", 6.95), ("2026-07-10", 7.90)]
T2_START = int(np.datetime64("2019-07-16", "D").astype(np.int64))   # T+2 settlement start
NUM_FIELDS = ["Sales", "OP", "NP", "EPS", "BPS", "TA", "Eq", "CashEq", "DivAnn", "Div1Q", "Div2Q", "Div3Q",
              "DivFY", "FNP", "FNCNP", "NxFNp", "NxFNP", "NxFNCNP", "NxFNCNp", "FDivAnn", "FDiv2Q",
              "ShOutFY", "TrShFY"]
QIDX = {"1Q": 1, "2Q": 2, "3Q": 3, "FY": 4}
REV_RE = re.compile(r"revision|forecast", re.I)
NONCOMMON_RE = re.compile(r"優先株|種優先|優先出資|ETF|ＥＴＦ|投資法人|REIT|ＲＥＩＴ|受益証券|上場投信|ETN|ＥＴＮ")
NAN = float("nan")


# ----------------------------------------------------------------------------- parsing helpers
def fnum(x):
    if x is None or isinstance(x, bool):
        return NAN
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else NAN
    s = str(x).strip().replace(",", "").replace("，", "")
    if s in ("", "-", "－", "—", "―", "nan", "NaN", "None", "null"):
        return NAN
    try:
        v = float(s)
        return v if math.isfinite(v) else NAN
    except ValueError:
        return NAN


def fday(x):
    """-> int days since epoch, or None"""
    if x is None:
        return None
    try:
        if isinstance(x, np.datetime64):
            if np.isnat(x):
                return None
            return int(x.astype("datetime64[D]").astype(np.int64))
        if isinstance(x, (pd.Timestamp, datetime.date, datetime.datetime)):
            return int(np.datetime64(pd.Timestamp(x).date(), "D").astype(np.int64))
        s = str(x).strip()[:10].replace("/", "-")
        if len(s) < 8:
            return None
        return int(np.datetime64(s, "D").astype(np.int64))
    except Exception:
        return None


def dstr(di):
    return str(np.datetime64(int(di), "D"))


@lru_cache(maxsize=None)
def add_months(di, k):
    d = np.datetime64(int(di), "D")
    m = d.astype("datetime64[M]")
    dom = int((d - m.astype("datetime64[D]")).astype(np.int64))
    last = (m + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    m2 = m + np.timedelta64(int(k), "M")
    first2 = m2.astype("datetime64[D]")
    last2 = (m2 + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    if d == last:
        return int(last2.astype(np.int64))
    return int(min(first2 + dom, last2).astype(np.int64))


def canon(code):
    c = str(code).strip()
    if len(c) == 4:
        c = c + "0"
    return c


def is_common(code, name):
    c = canon(code)
    if len(c) == 5 and c[4] != "0":
        return False
    if name and NONCOMMON_RE.search(str(name)):
        return False
    return True


# ----------------------------------------------------------------------------- splits / calendar
class Splits:
    """fd = ex-date (J-Quants AdjustmentFactor date); basis 'b' of a number = all splits with fd <= b reflected."""

    def __init__(self, fd, ff):
        fd = np.asarray(fd if fd is not None else [], dtype="datetime64[D]").astype(np.int64)
        ff = np.asarray(ff if ff is not None else [], dtype=float)
        n = min(len(fd), len(ff))
        fd, ff = fd[:n], ff[:n]
        ok = np.isfinite(ff) & (ff > 0) & (np.abs(ff - 1.0) > 1e-12)
        o = np.argsort(fd[ok], kind="stable")
        self.fd = fd[ok][o]
        self.ff = ff[ok][o]
        self.cp = np.concatenate([[1.0], np.cumprod(self.ff)])
        self.F = float(self.cp[-1])

    def cumf(self, d):
        if len(self.fd) == 0:
            return 1.0
        return float(self.cp[np.searchsorted(self.fd, d, "right")])

    def window(self, a, b):
        """splits with a < fd <= b -> (product of ff, list of fd)"""
        if len(self.fd) == 0 or b <= a:
            return 1.0, []
        i = np.searchsorted(self.fd, a, "right")
        j = np.searchsorted(self.fd, b, "right")
        return float(self.cp[j] / self.cp[i]), list(self.fd[i:j])

    def ps_to_fetch(self, v, b):      # per-share value in basis b -> fetch basis (same basis as bars c)
        return v * self.F / self.cumf(b)

    def sh_to_fetch(self, s, b):      # share count in basis b -> fetch basis
        return s * self.cumf(b) / self.F


class Cal:
    def __init__(self, days_int):
        self.d = days_int
        self.last = int(days_int[-1])

    def cum(self, R):
        """record date R -> (index of last cum-rights trading day or None, its day)"""
        if R > self.last:
            return None, R - 4
        i = int(np.searchsorted(self.d, R, "right")) - 1
        k = 2 if R >= T2_START else 3
        j = i - k
        if j < 0:
            return None, R - 4
        return j, int(self.d[j])


# ----------------------------------------------------------------------------- fins
def parse_rows(rows):
    out = []
    for n, r in enumerate(rows or []):
        if not isinstance(r, dict):
            continue
        disc = fday(r.get("DiscDate"))
        if disc is None:
            continue
        doc = str(r.get("DocType") or "")
        per = str(r.get("CurPerType") or "").strip().upper()
        fyen = fday(r.get("CurFYEn"))
        kind = "rev" if REV_RE.search(doc) else ("stmt" if per in QIDX else "other")
        vals = {}
        for f in NUM_FIELDS:
            if f in r:
                v = fnum(r[f])
                if v == v:
                    vals[f] = v
        pe = fday(r.get("CurPerEn"))
        out.append((disc, str(r.get("DiscTime") or ""), n, kind, per, fyen, vals, pe))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


class StockFins:
    """Processes one stock's disclosures in time order, emitting the point-in-time state after each disclosure day."""

    def __init__(self, rows, sp, cal):
        self.sp, self.cal = sp, cal
        self.rows = parse_rows(rows)
        self.P = {}            # (fyen, q) -> {"v": {field: value}, "d": {field: disc}, "pe": period end}
        self.npc = []          # NP forecast candidates (disc, target_fyen, value)
        self.dvc = []          # dividend forecast candidates (disc, target_fyen, FDivAnn, FDiv2Q)
        self.Lfy = self.Lsh = self.Leq = None
        self.memo = {}
        self.stats = {"div_post_basis": 0, "sh_pre_basis": 0, "split_window_div": 0, "split_window_sh": 0}

    def period_end(self, key, rec):
        if rec.get("pe") is not None:
            return rec["pe"]
        return add_months(key[0], -3 * (4 - key[1]))

    def apply(self, row):
        disc, _t, _n, kind, per, fyen, vals, pe = row
        if kind == "stmt":
            if fyen is None:
                return
            q = QIDX[per]
            key = (fyen, q)
            rec = self.P.setdefault(key, {"v": {}, "d": {}, "pe": None})
            if pe is not None:
                rec["pe"] = pe
            for f, v in vals.items():
                rec["v"][f] = v
                rec["d"][f] = disc
            if q == 4 and (self.Lfy is None or key >= self.Lfy):
                self.Lfy = key
            if "ShOutFY" in vals and (self.Lsh is None or key >= self.Lsh):
                self.Lsh = key
            if "Eq" in vals and (self.Leq is None or key >= self.Leq):
                self.Leq = key
            if q < 4:
                v = vals.get("FNP", vals.get("FNCNP"))
                if v is not None:
                    self.npc.append((disc, fyen, v))
                if "FDivAnn" in vals:
                    self.dvc.append((disc, fyen, vals["FDivAnn"], vals.get("FDiv2Q")))
            else:
                for f in ("NxFNp", "NxFNP", "NxFNCNP", "NxFNCNp"):
                    if f in vals:
                        self.npc.append((disc, add_months(fyen, 12), vals[f]))
                        break
        elif kind == "rev":
            if per not in ("FY", ""):
                return          # half-year (2Q) forecast revisions are not full-year numbers
            tg = fyen
            if tg is None:
                if self.Lfy is None:
                    return
                tg = add_months(self.Lfy[0], 12)
            v = vals.get("FNP", vals.get("FNCNP"))
            if v is not None:
                self.npc.append((disc, tg, v))
            if "FDivAnn" in vals:
                self.dvc.append((disc, tg, vals["FDivAnn"], vals.get("FDiv2Q")))

    # --- split-basis decisions ---------------------------------------------------------------
    def divann_basis(self, fyen, rec):
        disc = rec["d"]["DivAnn"]
        mk = ("dv", fyen, disc)
        if mk in self.memo:
            return self.memo[mk]
        _ci, cumd = self.cal.cum(fyen)
        Fw, fds = self.sp.window(cumd, disc)
        basis = cumd                       # default: basis of the record date (pre-split for FY-end-record splits)
        if fds:
            self.stats["split_window_div"] += 1
            prev = None
            for (d, tg, v, _x) in self.dvc:
                if tg == fyen and d < fds[0] and v > 0:
                    prev = (d, v)
            dv = rec["v"]["DivAnn"]
            if prev is not None and dv > 0:
                prev_rec = prev[1] * self.sp.cumf(cumd) / self.sp.cumf(prev[0])
                r = dv / prev_rec
                if abs(math.log(r) - math.log(Fw)) < abs(math.log(r)):
                    basis = disc           # reported in post-split basis
                    self.stats["div_post_basis"] += 1
        self.memo[mk] = basis
        return basis

    def shares_basis(self, key, rec):
        disc = rec["d"]["ShOutFY"]
        mk = ("sh", key, disc)
        if mk in self.memo:
            return self.memo[mk]
        pe = self.period_end(key, rec)
        _ci, cumd = self.cal.cum(pe)
        Fw, fds = self.sp.window(cumd, disc)
        basis = disc                       # default: post-split basis (all splits up to disclosure)
        if fds:
            self.stats["split_window_sh"] += 1
            v = rec["v"]
            S = v["ShOutFY"] - v.get("TrShFY", 0.0)
            npv, eps = v.get("NP"), v.get("EPS")
            if npv and eps and npv * eps > 0 and S > 0:
                ratio = S / (npv / eps)
                if abs(math.log(ratio) - math.log(Fw)) < abs(math.log(ratio)):
                    basis = cumd
                    self.stats["sh_pre_basis"] += 1
        self.memo[mk] = basis
        return basis

    # --- point-in-time state ------------------------------------------------------------------
    def state(self):
        sp = self.sp
        fyrec = self.P[self.Lfy] if self.Lfy else None
        Lend = self.Lfy[0] if self.Lfy else None
        Ldisc = min(fyrec["d"].values()) if fyrec and fyrec["d"] else None
        sh = NAN
        if self.Lsh:
            rec = self.P[self.Lsh]
            S = rec["v"]["ShOutFY"] - rec["v"].get("TrShFY", 0.0)
            sh = sp.sh_to_fetch(S, self.shares_basis(self.Lsh, rec))
        eq = self.P[self.Leq]["v"]["Eq"] if self.Leq else NAN
        npu, src = NAN, 0
        for (d, tg, v) in reversed(self.npc):
            if Lend is None or tg > Lend:
                npu, src = v, 1
                break
            if Ldisc is not None and d <= Ldisc:
                break
        if src == 0 and fyrec and "NP" in fyrec["v"]:
            npu, src = fyrec["v"]["NP"], 2
        dps = NAN
        for (d, tg, v, _x) in reversed(self.dvc):
            if Lend is None or tg > Lend:
                dps = sp.ps_to_fetch(v, d)
                break
            if Ldisc is not None and d <= Ldisc:
                break
        po = NAN
        if fyrec and "DivAnn" in fyrec["v"]:
            dv_f = sp.ps_to_fetch(fyrec["v"]["DivAnn"], self.divann_basis(self.Lfy[0], fyrec))
            if dps != dps:
                dps = dv_f
            if "EPS" in fyrec["v"]:
                eps_f = sp.ps_to_fetch(fyrec["v"]["EPS"], fyrec["d"]["EPS"])
                po = dv_f / eps_f if eps_f > 0 else math.inf
        return (sh, npu, eq, dps, po, float(src))

    def run(self):
        ev_d, ev_s = [], []
        rows, i, n = self.rows, 0, len(self.rows)
        while i < n:
            day = rows[i][0]
            while i < n and rows[i][0] == day:
                self.apply(rows[i])
                i += 1
            ev_d.append(day)
            ev_s.append(self.state())
        return np.array(ev_d, dtype=np.int64), np.array(ev_s, dtype=float).reshape(-1, 6)

    # --- dividends actually received (for P&L) -------------------------------------------------
    def dividend_events(self):
        """-> list of (record_day, cum_idx, amount per fetch-basis share, estimated?)"""
        sp, cal, P = self.sp, self.cal, self.P
        out = []
        fyens = sorted(set(k[0] for k in P) | set(tg for (_d, tg, _v, _x) in self.dvc))
        got = {}     # fyen -> {q: fetch amount}
        for fyen in fyens:
            fy = P.get((fyen, 4))
            amts = {}
            raw_interim = 0.0
            for q, fld in ((1, "Div1Q"), (2, "Div2Q"), (3, "Div3Q")):
                R = add_months(fyen, -3 * (4 - q))
                ci, cumd = cal.cum(R)
                val, est = None, False
                if fy and fld in fy["v"]:
                    val = fy["v"][fld]
                    raw_interim += val
                else:
                    for qq in (q, q + 1, 3):
                        if qq <= 3 and (fyen, qq) in P and fld in P[(fyen, qq)]["v"]:
                            val = P[(fyen, qq)]["v"][fld]
                            break
                if val is not None:
                    a = sp.ps_to_fetch(val, cumd)
                else:
                    reported_after = fy is not None or any((fyen, qq) in P for qq in range(q, 4))
                    if reported_after or R > cal.last:
                        a = 0.0
                    else:   # period ended, no report yet in data -> estimate
                        a, est = 0.0, True
                        if q == 2:
                            for (d, tg, v, f2) in reversed(self.dvc):
                                if tg == fyen and f2 is not None:
                                    a = sp.ps_to_fetch(f2, d)
                                    break
                            else:
                                a = got.get(add_months(fyen, -12), {}).get(q, 0.0)
                        else:
                            a = got.get(add_months(fyen, -12), {}).get(q, 0.0)
                amts[q] = a
                if a > 0 and ci is not None:
                    out.append((R, ci, a, est))
            # year-end
            ci, cumd = cal.cum(fyen)
            a, est = 0.0, False
            if fy and "DivAnn" in fy["v"]:
                ye = fy["v"]["DivFY"] if "DivFY" in fy["v"] else fy["v"]["DivAnn"] - raw_interim
                a = max(sp.ps_to_fetch(ye, self.divann_basis(fyen, fy)), 0.0)
            elif fy is None and fyen <= cal.last:
                est = True
                fc = None
                for (d, tg, v, _x) in reversed(self.dvc):
                    if tg == fyen:
                        fc = sp.ps_to_fetch(v, d)
                        break
                if fc is not None:
                    a = max(fc - sum(amts.values()), 0.0)
                else:
                    a = got.get(add_months(fyen, -12), {}).get(4, 0.0)
            amts[4] = a
            got[fyen] = amts
            if a > 0 and ci is not None:
                out.append((fyen, ci, a, est))
        return out


# ----------------------------------------------------------------------------- tax
def tax_apply(net_gain, carry):
    """per-day net realised gain -> (tax, new carry). Losses carried forward indefinitely."""
    if net_gain > 0:
        off = min(net_gain, carry)
        return (net_gain - off) * TAX, carry - off
    return 0.0, carry - net_gain


def ann(v1, v0, months):
    if months <= 0 or not (v0 > 0) or not (v1 > 0):
        return NAN
    return ((v1 / v0) ** (12.0 / months) - 1.0) * 100.0


def maxdd(vals):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return NAN
    return float(np.min(v / np.maximum.accumulate(v) - 1.0) * 100.0)


# ----------------------------------------------------------------------------- data build
def build(store, log):
    etf = store["etf1306"]
    cal_d = np.asarray(etf["d"]).astype("datetime64[D]")
    o = np.argsort(cal_d, kind="stable")
    cal_d = cal_d[o]
    keep = np.r_[True, cal_d[1:] != cal_d[:-1]]
    cal_d = cal_d[keep]
    etf_c = pd.Series(np.asarray(etf["c"], dtype=float)[o][keep]).ffill().to_numpy()
    cal_i = cal_d.astype(np.int64)
    T = len(cal_d)
    cal = Cal(cal_i)
    ym = cal_d.astype("datetime64[M]")
    is_me = np.r_[ym[1:] != ym[:-1], True]
    me_all = np.nonzero(is_me)[0]
    me_month = [str(ym[i]) for i in me_all]
    s_pos = int(np.nonzero(me_all >= WARMUP)[0][0])
    e_cands = [p for p, m in enumerate(me_month) if m <= END_MONTH]
    e_pos = e_cands[-1]
    split_pos = me_month.index(SPLIT_MONTH)
    me_idx = me_all[s_pos:e_pos + 1]
    me_mon = me_month[s_pos:e_pos + 1]
    split_j = split_pos - s_pos

    months = store.get("months", {}) or {}
    mkeys = sorted(months.keys())
    month_lists = []
    for m in me_mon:
        k = m if m in months else next((x for x in reversed(mkeys) if x <= m), None)
        month_lists.append(months.get(k, []) if k else [])

    bars = store.get("bars", {}) or {}
    fins = store.get("fins", {}) or {}
    bar_by = {}
    for k in bars:
        bar_by.setdefault(canon(k), k)
    fin_by = {}
    for k in fins:
        fin_by.setdefault(canon(k), k)

    listed_codes = set()
    missing_bars = set()
    for lst in month_lists:
        for ent in lst:
            if not ent:
                continue
            c = canon(ent[0])
            if c in bar_by:
                listed_codes.add(c)
            else:
                missing_bars.add(c)
    codes = sorted(listed_codes)
    N = len(codes)
    cix = {c: j for j, c in enumerate(codes)}
    log(f"calendar {dstr(cal_i[0])}..{dstr(cal_i[-1])} T={T}; period MEs {len(me_idx)}; stocks N={N} "
        f"(listed codes without bars: {len(missing_bars)})")

    C = np.full((T, N), np.nan)
    VA = np.zeros((T, N))
    TR = np.zeros((T, N), dtype=bool)
    lastbar = np.full(N, -1, dtype=np.int64)
    dropped = 0
    splits = []
    for j, c in enumerate(codes):
        b = bars[bar_by[c]]
        d = np.asarray(b.get("d", []), dtype="datetime64[D]")
        cc = np.asarray(b.get("c", []), dtype=float)
        n = min(len(d), len(cc))
        d, cc = d[:n], cc[:n]
        va_raw = b.get("va")
        va = np.asarray(va_raw, dtype=float)[:n] if va_raw is not None else np.full(n, np.nan)
        if len(va) < n:
            va = np.r_[va, np.full(n - len(va), np.nan)]
        pos = np.searchsorted(cal_d, d)
        ok = pos < T
        ok[ok] = cal_d[pos[ok]] == d[ok]
        dropped += int((~ok).sum())
        good = ok & np.isfinite(cc) & (cc > 0)
        p = pos[good]
        C[p, j] = cc[good]
        VA[p, j] = np.nan_to_num(va[good], nan=0.0)
        TR[p, j] = True
        if len(p):
            lastbar[j] = p.max()
        splits.append(Splits(b.get("fd"), b.get("ff")))
    idx = np.where(TR, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    Cff = C[idx, np.arange(N)[None, :]]          # forward-filled close; NaN before the first bar
    first = np.where(TR.any(0), TR.argmax(0), T)
    Cff[np.arange(T)[:, None] < first[None, :]] = np.nan
    del idx
    log(f"bars matrices built; bar rows dropped (date not in 1306 calendar): {dropped}")
    return dict(cal=cal, cal_i=cal_i, cal_d=cal_d, etf_c=etf_c, T=T, N=N, codes=codes, cix=cix,
                me_idx=me_idx, me_mon=me_mon, split_j=split_j, month_lists=month_lists, C=C, Cff=Cff,
                VA=VA, TR=TR, lastbar=lastbar, splits=splits, fins=fins, fin_by=fin_by,
                missing_bars=len(missing_bars), dropped=dropped)


def build_fins(D, log):
    me_days = D["cal_i"][D["me_idx"]]
    M, N = len(me_days), D["N"]
    ST = np.full((6, M, N), np.nan)
    div_ev = [[] for _ in range(N)]
    stats = {"div_post_basis": 0, "sh_pre_basis": 0, "split_window_div": 0, "split_window_sh": 0,
             "no_fins": 0, "est_div_events": 0}
    for j, c in enumerate(D["codes"]):
        fk = D["fin_by"].get(c)
        rows = D["fins"].get(fk) if fk is not None else None
        if not rows:
            stats["no_fins"] += 1
            continue
        sf = StockFins(rows, D["splits"][j], D["cal"])
        ev_d, ev_s = sf.run()
        if len(ev_d):
            ii = np.searchsorted(ev_d, me_days, "left") - 1      # strictly before ME date
            ok = ii >= 0
            ST[:, ok, j] = ev_s[ii[ok]].T
        evs = sf.dividend_events()
        div_ev[j] = [(ci, a) for (_R, ci, a, _e) in evs]
        stats["est_div_events"] += sum(1 for e in evs if e[3] and D["me_idx"][0] <= e[1] <= D["me_idx"][-1])
        for k in ("div_post_basis", "sh_pre_basis", "split_window_div", "split_window_sh"):
            stats[k] += sf.stats[k]
    log(f"fins processed: {stats}")
    return ST, div_ev, stats


def universe(D, ST):
    me_idx, N = D["me_idx"], D["N"]
    M = len(me_idx)
    U = np.zeros((M, N), dtype=bool)
    Ualt = np.zeros((M, N), dtype=bool)
    FIN = np.zeros((M, N), dtype=bool)
    VAn = None
    for j, i in enumerate(me_idx):
        listed = np.zeros(N, dtype=bool)
        for ent in D["month_lists"][j]:
            if not ent or len(ent) < 2:
                continue
            c = canon(ent[0])
            k = D["cix"].get(c)
            if k is None:
                continue
            mkt = str(ent[1]).strip().zfill(4)
            name = ent[4] if len(ent) > 4 else ""
            sect = str(ent[3]).strip() if len(ent) > 3 else ""
            if mkt in TSE_MKT and is_common(ent[0], name):
                listed[k] = True
                FIN[j, k] = sect in FIN_SECTORS
        lo = i - LIQ_DAYS + 1
        w = D["VA"][max(lo, 0):i + 1]
        liq = np.median(w, axis=0) if lo >= 0 else np.zeros(N)
        wt = D["TR"][max(lo, 0):i + 1]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            liq_alt = np.nanmedian(np.where(wt, w, np.nan), axis=0) if wt.any() else np.zeros(N)
        tr5 = D["TR"][i - 4:i + 1].any(0)
        mc = D["Cff"][i] * ST[0, j]
        base = listed & tr5 & (mc >= MCAP_MIN)
        U[j] = base & (liq >= LIQ_MIN)
        Ualt[j] = base & (np.nan_to_num(liq_alt) >= LIQ_MIN)
    return U, Ualt, FIN


# ----------------------------------------------------------------------------- H0
def h0_candidates(D, ST, U, exclude=None):
    out, npass = [], []
    codes_rank = np.arange(D["N"])
    for j, i in enumerate(D["me_idx"]):
        p = D["Cff"][i]
        sh, npu, eq, dps, po = ST[0, j], ST[1, j], ST[2, j], ST[3, j], ST[4, j]
        with np.errstate(all="ignore"):
            mc = p * sh
            ok = U[j] & (npu > 0) & (eq > 0) & (mc > 0)
            ok &= (mc / npu <= 12.0) & (mc / eq <= 1.0) & (npu / eq >= 0.08) & (dps / p >= 0.03)
            ok &= (po <= 0.40) & ((eq + 10.0 * npu) / mc >= 2.0) & (mc <= 1e11)
            if exclude is not None:
                ok &= ~exclude[j]
            saf = (eq + 10.0 * npu) / mc
        ks = np.nonzero(ok)[0]
        order = np.lexsort((codes_rank[ks], -saf[ks]))
        out.append([int(k) for k in ks[order]])
        npass.append(len(ks))
    return out, npass


def sim_h0(D, cands, limit, div_ev, nslots=20):
    Cff, me_idx = D["Cff"], D["me_idx"]
    cash, carry, nbuy = INIT, 0.0, 0
    hold = {}
    vals, nh = [], []
    taxes = divs = 0.0
    vliq = NAN
    for j, i in enumerate(me_idx):
        if j > 0:
            p_ = me_idx[j - 1]
            for k, h in hold.items():
                for (ci, a) in div_ev[k]:
                    if p_ <= ci <= i - 1:
                        g = h["sh"] * a
                        cash += g * (1 - TAX)
                        divs += g
        if j == len(me_idx) - 1:
            V = cash + sum(h["sh"] * Cff[i, k] for k, h in hold.items())
            vals.append(V)
            nh.append(len(hold))
            net = 0.0
            for k, h in hold.items():
                pr = h["sh"] * Cff[i, k] * (1 - FEE)
                cash += pr
                net += pr - h["cost"]
            tx, carry = tax_apply(net, carry)
            taxes += tx
            vliq = cash - tx
            break
        rank = {k: r + 1 for r, k in enumerate(cands[j])}
        sold = [k for k in hold if rank.get(k, 10 ** 9) > limit]
        net = 0.0
        for k in sold:
            h = hold.pop(k)
            pr = h["sh"] * Cff[i, k] * (1 - FEE)
            cash += pr
            net += pr - h["cost"]
        tx, carry = tax_apply(net, carry)
        cash -= tx
        taxes += tx
        V = cash + sum(h["sh"] * Cff[i, k] for k, h in hold.items())
        tgt = V / nslots
        sold_set = set(sold)
        for k in cands[j]:
            if len(hold) >= nslots:
                break
            if k in hold or k in sold_set:
                continue
            unit = Cff[i, k] * (1 + FEE)
            n = math.floor(min(tgt, cash) / unit)
            if n >= 1:
                hold[k] = {"sh": n, "cost": n * unit, "bi": i}
                cash -= n * unit
                nbuy += 1
        vals.append(cash + sum(h["sh"] * Cff[i, k] for k, h in hold.items()))
        nh.append(len(hold))
    vals[0] = INIT
    return vals, vliq, nbuy, nh, taxes, divs


# ----------------------------------------------------------------------------- M1
def m1_signals(D, U, log):
    Cff, TR, T, N = D["Cff"], D["TR"], D["T"], D["N"]
    me_idx = D["me_idx"]
    t0 = time.time()
    df = pd.DataFrame(Cff)
    s50 = df.rolling(50, min_periods=50).mean().to_numpy()
    s150 = df.rolling(150, min_periods=150).mean().to_numpy()
    s200 = df.rolling(200, min_periods=200).mean().to_numpy()
    c = Cff
    with np.errstate(invalid="ignore"):
        base = (c > s150) & (c > s200) & (s150 > s200) & (s50 > s150) & (s50 > s200) & (c > s50)
        s200_21 = np.full_like(s200, np.nan)
        s200_21[21:] = s200[:-21]
        base &= s200 > s200_21
        del s150, s200, s200_21
        lo = df.rolling(250, min_periods=250).min().to_numpy()
        base &= c >= 1.3 * lo
        del lo
        hi = df.rolling(250, min_periods=250).max().to_numpy()
        base &= c >= 0.75 * hi
        del hi
        pmax = df.rolling(25, min_periods=25).max().shift(1).to_numpy()
        base &= (c > pmax) & TR
        del pmax
        rs = np.zeros_like(c)
        for n, w in ((63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2)):
            prev = np.full_like(c, np.nan)
            prev[n:] = c[:-n]
            rs += w * (c / prev - 1.0)
            del prev
    log(f"M1 indicator matrices: {time.time() - t0:.1f}s")
    sig = {}
    rsrank_store = {}
    s, e = int(me_idx[0]), int(me_idx[-1])
    for j in range(len(me_idx) - 1):
        a, b = int(me_idx[j]), int(me_idx[j + 1])     # days a..b-1 use ME j's universe; b uses ME j+1
        u = np.nonzero(U[j])[0]
        if len(u) == 0:
            continue
        sub = rs[a:b][:, u]
        rk = pd.DataFrame(sub).rank(axis=1, method="average").to_numpy()
        cnt = np.isfinite(sub).sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(cnt > 1, (rk - 1.0) / (cnt - 1.0), np.where(cnt == 1, 1.0, np.nan))
        ok = base[a:b][:, u] & (pct >= 0.70)
        for r in range(b - a):
            t = a + r
            if t >= e:
                continue
            ks = np.nonzero(ok[r])[0]
            if len(ks):
                kk = u[ks]
                order = np.lexsort((kk, -rs[t, kk]))
                sig[t] = [int(x) for x in kk[order]]
        rsrank_store[j] = (u, pct)
    return sig, s50, base, rs, rsrank_store


def sim_m1(D, sig, s50, div_ev):
    Cff, TR, me_idx, lastbar = D["Cff"], D["TR"], D["me_idx"], D["lastbar"]
    s, e = int(me_idx[0]), int(me_idx[-1])
    divmap = [dict() for _ in range(D["N"])]
    for k, evs in enumerate(div_ev):
        for (ci, a) in evs:
            divmap[k][ci] = divmap[k].get(ci, 0.0) + a
    cash, carry, nbuy = INIT, 0.0, 0
    pos, trades, daily = {}, [], {}
    pending = []
    vliq = NAN
    for t in range(s, e + 1):
        if t > s:
            for k, p in pos.items():
                a = divmap[k].get(t - 1)
                if a:
                    cash += p["sh"] * a * (1 - TAX)
        if t == e:
            daily[t] = cash + sum(p["sh"] * Cff[t, k] for k, p in pos.items())
            net = 0.0
            for k, p in pos.items():
                pr = p["sh"] * Cff[t, k] * (1 - FEE)
                cash += pr
                net += pr - p["cost"]
                trades.append((k, p["bi"], t, p["cost"], pr))
            tx, carry = tax_apply(net, carry)
            vliq = cash - tx
            break
        sells = []
        for k, p in pos.items():
            px = Cff[t, k]
            if t > lastbar[k]:
                sells.append(k)
                continue
            stop = p["px"] if p["hit"] else p["px"] * 0.925
            if px <= stop or (p["hit"] and px < s50[t, k]):
                sells.append(k)
                continue
            if px >= 1.2 * p["px"]:
                p["hit"] = True
        net = 0.0
        for k in sells:
            p = pos.pop(k)
            pr = p["sh"] * Cff[t, k] * (1 - FEE)
            cash += pr
            net += pr - p["cost"]
            trades.append((k, p["bi"], t, p["cost"], pr))
        tx, carry = tax_apply(net, carry)
        cash -= tx
        if pending and len(pos) < 5:
            V = cash + sum(p["sh"] * Cff[t, k] for k, p in pos.items())
            tgt = V / 5.0
            for k in pending:
                if len(pos) >= 5:
                    break
                if k in pos or not TR[t, k]:
                    continue
                unit = Cff[t, k] * (1 + FEE)
                n = math.floor(min(tgt, cash) / unit)
                if n >= 1:
                    pos[k] = {"sh": n, "cost": n * unit, "px": Cff[t, k], "bi": t, "hit": False}
                    cash -= n * unit
                    nbuy += 1
        pending = sig.get(t, [])
        daily[t] = cash + sum(p["sh"] * Cff[t, k] for k, p in pos.items())
    me_vals = [daily[int(i)] for i in me_idx]
    me_vals[0] = INIT
    dvals = [INIT] + [daily[t] for t in range(s, e + 1)]
    return me_vals, vliq, nbuy, trades, dvals


# ----------------------------------------------------------------------------- TOPIX
def sim_topix(D):
    c, me_idx, cal_i = D["etf_c"], D["me_idx"], D["cal_i"]
    dist = [(fday(d), a) for d, a in ETF_DIST]
    i0 = me_idx[0]
    unit = c[i0] * (1 + FEE)
    units = math.floor(INIT / unit)
    cash = INIT - units * unit
    cost = units * unit
    vals, rec = [INIT], 0
    for j in range(1, len(me_idx)):
        i, p_ = me_idx[j], me_idx[j - 1]
        for (dd, a) in dist:
            if cal_i[p_] < dd <= cal_i[i]:
                cash += units * a * (1 - TAX)
                rec += 1
                n = math.floor(cash / (c[i] * (1 + FEE)))
                units += n
                cash -= n * c[i] * (1 + FEE)
                cost += n * c[i] * (1 + FEE)
        vals.append(units * c[i] + cash)
    e = me_idx[-1]
    pr = units * c[e] * (1 - FEE)
    gain = pr - cost
    vliq = cash + pr - max(gain, 0.0) * TAX
    return vals, vliq, rec


# ----------------------------------------------------------------------------- main
def main(argv):
    t_start = time.time()
    if len(argv) < 3:
        print(__doc__)
        return 2
    lines = []

    def log(msg):
        print(f"[{time.time() - t_start:6.1f}s] {msg}", flush=True)

    with open(argv[1], "rb") as f:
        store = pickle.load(f)
    log("pickle loaded")
    D = build(store, log)
    del store
    ST, div_ev, fstats = build_fins(D, log)
    D.pop("fins", None)
    U, Ualt, FIN = universe(D, ST)
    me_idx, split_j = D["me_idx"], D["split_j"]
    M = len(me_idx)
    m_front, m_back, m_tot = split_j, M - 1 - split_j, M - 1
    cal_i = D["cal_i"]

    def pack(vals, vliq):
        return {"front": ann(vals[split_j], vals[0], m_front), "back": ann(vals[-1], vals[split_j], m_back),
                "liq": ann(vliq, INIT, m_tot), "maxdd": maxdd(vals)}

    tv, tliq, trec = sim_topix(D)
    topix = pack(tv, tliq)
    log("TOPIX done")

    cands, npass = h0_candidates(D, ST, U)
    H0 = {}
    for rule, lim in (("A", 20), ("B", 60)):
        vals, vliq, nbuy, nh, taxes, divs = sim_h0(D, cands, lim, div_ev)
        r = pack(vals, vliq)
        r.update({"n_buy": nbuy, "avg_hold": float(np.mean(nh)), "final_value": vals[-1], "liq_value": vliq,
                  "dividends_gross": divs, "taxes_on_gains": taxes})
        H0[rule] = r
    H0.update({"avg_pass": float(np.mean(npass)), "min_pass": int(np.min(npass)), "max_pass": int(np.max(npass)),
               "avg_universe": float(U.sum(1).mean())})
    # variant: financials excluded (diagnostic only)
    cands_x, npass_x = h0_candidates(D, ST, U, exclude=FIN)
    H0x = {"avg_pass": float(np.mean(npass_x))}
    for rule, lim in (("A", 20), ("B", 60)):
        vals, vliq, nbuy, nh, _t, _d = sim_h0(D, cands_x, lim, div_ev)
        H0x[rule] = {k: v for k, v in pack(vals, vliq).items()}
        H0x[rule]["n_buy"] = nbuy
    log("H0 done")

    sig, s50, _base, _rs, _rr = m1_signals(D, U, log)
    mv, mliq, mbuy, trades, dvals = sim_m1(D, sig, s50, div_ev)
    wins = [1 for tr in trades if tr[4] > tr[3]]
    M1 = pack(mv, mliq)
    M1.update({"n_buy": mbuy, "win_rate": (len(wins) / len(trades) * 100.0) if trades else NAN,
               "avg_hold_days": float(np.mean([tr[2] - tr[1] for tr in trades])) if trades else NAN,
               "maxdd_daily": maxdd(dvals), "n_trades_closed": len(trades), "signal_days": len(sig)})
    log("M1 done")

    def rnd(x):
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, float):
            return None if not math.isfinite(x) else round(x, 4)
        return x

    notes = [
        "annualisation: (V_end/V_start)^(12/months)-1 on month-end valuations; V_start = 10,000,000 before buying",
        "month-end valuation = after that month-end's trades (costs, taxes paid), last month-end = before liquidation",
        "maxdd (%, negative) on month-end valuations; M1.maxdd_daily on daily valuations",
        "tax: realised gains/losses netted per day, then carry-forward; dividends taxed 20.315% at credit",
        "dividends credited at the first rebalancing point after the cum-rights date (T+2 from 2019-07-16, T+3 before)",
        "liquidity: median of 60 calendar trading days of turnover, no-trade day = 0 yen",
        f"universe with liquidity median over traded days only: avg {float(Ualt.sum(1).mean()):.1f}",
        f"H0 variant excluding financial sectors: avg_pass {H0x['avg_pass']:.2f}, "
        f"A liq {rnd(H0x['A']['liq'])}, B liq {rnd(H0x['B']['liq'])}",
        f"listed codes without bars: {D['missing_bars']}; stocks without fins: {fstats['no_fins']}; "
        f"bar rows outside 1306 calendar: {D['dropped']}",
        f"split-basis: DivAnn rows with split after record date: {fstats['split_window_div']} "
        f"(judged post-split basis: {fstats['div_post_basis']}); share rows with split after period end: "
        f"{fstats['split_window_sh']} (judged pre-split basis: {fstats['sh_pre_basis']})",
        f"estimated (not yet disclosed) dividend events in period: {fstats['est_div_events']}",
        f"TOPIX distributions received: {trec}",
    ]
    res = {"period": {"start": dstr(cal_i[me_idx[0]]), "split": dstr(cal_i[me_idx[split_j]]),
                      "end": dstr(cal_i[me_idx[-1]]), "months": M},
           "topix": rnd(topix), "H0": rnd(H0), "M1": rnd(M1), "notes": notes,
           "extra": rnd({"H0_exfin": H0x, "avg_universe_liq_traded_days_only": float(Ualt.sum(1).mean()),
                         "runtime_sec": time.time() - t_start})}
    with open(argv[2], "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    P = res["period"]
    print("=" * 72)
    print(f"period {P['start']} .. {P['split']} .. {P['end']}  ({P['months']} month-ends)")
    print(f"universe avg {H0['avg_universe']:.1f} | H0 pass avg {H0['avg_pass']:.2f} "
          f"min {H0['min_pass']} max {H0['max_pass']}")
    print(f"{'':10s}{'front%':>9s}{'back%':>9s}{'liq%':>9s}{'maxdd%':>9s}{'n_buy':>7s}  other")
    def row(name, r, other=""):
        f = lambda v: f"{v:9.2f}" if isinstance(v, (int, float)) and v is not None else f"{'n/a':>9s}"
        print(f"{name:10s}{f(r['front'])}{f(r['back'])}{f(r['liq'])}{f(r['maxdd'])}"
              f"{r.get('n_buy', ''):>7}  {other}")
    row("TOPIX", res["topix"])
    row("H0-A", res["H0"]["A"], f"avg_hold {res['H0']['A']['avg_hold']}")
    row("H0-B", res["H0"]["B"], f"avg_hold {res['H0']['B']['avg_hold']}")
    row("M1", res["M1"], f"win {res['M1']['win_rate']}% hold {res['M1']['avg_hold_days']}d")
    print(f"H0 ex-financials: A liq {rnd(H0x['A']['liq'])}  B liq {rnd(H0x['B']['liq'])}")
    print(f"runtime {time.time() - t_start:.1f}s")
    print("=== JSON ===")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

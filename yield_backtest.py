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
  python yield_backtest.py --years 7            過去7年で全ルールを比較
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


def fetch_all(jq: JQ, years: int, limit: int) -> dict:
    """株価と財務をまとめて取得する。時間がかかるのでキャッシュします。"""
    log.info("銘柄一覧を取得中…")
    info = jq.get("/v2/equities/master", {})
    uni = []
    for row in info:
        if row.get("Mkt") != MARKET_PRIME:
            continue
        if (row.get("ScaleCat") or "") not in SCALE_TARGETS:
            continue
        code = norm_code(row.get("Code", ""))
        if code:
            uni.append({"code": code,
                        "name": row.get("CoName", "") or row.get("CoNameEn", ""),
                        "sector": row.get("S33Nm", "")})
    if limit:
        uni = uni[:limit]
    log.info("対象 %d銘柄", len(uni))

    today = date.today()
    frm = (today - timedelta(days=365 * (years + 5) + 60)).isoformat()
    to = today.isoformat()

    store = {"universe": uni, "quotes": {}, "stmts": {}}
    for i, u in enumerate(uni, 1):
        time.sleep(API_SLEEP)
        try:
            q = jq.get("/v2/equities/bars/daily",
                       {"code": u["code"], "from": frm, "to": to})
        except Exception as e:
            log.warning("株価取得失敗 %s: %s", u["code"], e)
            continue
        if not q:
            continue
        time.sleep(API_SLEEP)
        try:
            st = jq.get("/v2/fins/summary", {"code": u["code"]})
        except Exception:
            st = []
        store["quotes"][u["code"]] = q
        store["stmts"][u["code"]] = st
        if i % 50 == 0:
            log.info("  取得 %d/%d", i, len(uni))
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


def build_panel(store: dict, years: int) -> pd.DataFrame:
    """月末ごとの「その時点で分かっていた」利回りの表を作る。"""
    frames = []
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

        f = pd.DataFrame({"date": m.index, "code": code,
                          "price": m.values, "dps": d.values, "yield": y.values})
        # 増配率（過去3年）。これも過去だけを見る
        f["dps_growth"] = f["dps"].pct_change(36) * 100.0 / 3.0
        frames.append(f.dropna(subset=["yield"]))

    if not frames:
        sys.exit("パネルを作れませんでした。データ取得をご確認ください。")
    panel = pd.concat(frames, ignore_index=True)

    # 各時点で、過去5年ぶんの分布における自分の位置（未来は見ない）
    panel = panel.sort_values(["code", "date"])
    panel["pct_own"] = (
        panel.groupby("code")["yield"]
        .transform(lambda s: s.rolling(60, min_periods=24)
                   .apply(lambda w: (w.iloc[-1] >= w[:-1]).mean() * 100, raw=False))
    )
    # その日の市場内での順位
    panel["pct_cross"] = panel.groupby("date")["yield"].rank(pct=True) * 100.0

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
             max_names: int = 15) -> dict:
    """月末ごとに判定して売買する。等金額・分割建玉。"""
    entry, exits = cfg["entry"], cfg["exit"]
    dyn, cross = cfg.get("dynamic", False), cfg.get("cross", False)
    n_tr = len(entry)

    dates = sorted(panel["date"].unique())
    cash = capital
    pos: dict[str, dict] = {}          # code -> {"units": n, "shares": [], "cost": x}
    curve, trades = [], []
    # 診断：可変ルールが実際に判定を変えた回数。
    # 保有上限で先に枠が埋まると、閾値を緩めても結果が変わらない。
    # 「効いていないのに複雑にしている」状態を見つけるために数える。
    diag = {"adjusted": 0, "changed": 0, "full_slots": 0, "months": 0}

    # 市場全体の利回り水準（局面判定に使う）
    mkt = panel.groupby("date")["yield"].median()
    mkt_z = (mkt - mkt.rolling(36, min_periods=12).mean()) / \
            mkt.rolling(36, min_periods=12).std()

    for dt in dates:
        day = panel[panel["date"] == dt].set_index("code")
        day = day.assign(mkt_yield_z=float(mkt_z.get(dt, 0) or 0))

        # ── 保有の評価と売り判定 ──
        for code in list(pos.keys()):
            if code not in day.index:
                continue
            row = day.loc[code]
            p = row["pct_cross"] if cross else row["pct_own"]
            st = pos[code]
            # 売り: 分位が exit を下回った段階数だけ手放す
            want_units = n_tr - sum(1 for e in exits if p <= e)
            while st["units"] > want_units and st["units"] > 0:
                sh = st["shares"].pop()
                cash += sh * row["price"]
                trades.append({"code": code, "date": dt, "side": "sell",
                               "price": row["price"], "shares": sh})
                st["units"] -= 1
            if st["units"] == 0:
                del pos[code]

        # ── 買い判定 ──
        total = cash + sum(sum(s["shares"]) * day.loc[c, "price"]
                           for c, s in pos.items() if c in day.index)
        unit_size = total / max_names / n_tr

        cands = []
        diag["months"] += 1
        for code, row in day.iterrows():
            p = row["pct_cross"] if cross else row["pct_own"]
            need = [required_pct(e, row, dyn) for e in entry]
            want = sum(1 for nd in need if p >= nd)
            want_base = sum(1 for e in entry if p >= e)
            if dyn and need != list(map(float, entry)):
                diag["adjusted"] += 1
                if want != want_base:
                    diag["changed"] += 1
            have = pos.get(code, {}).get("units", 0)
            if want > have:
                cands.append((p, code, want - have, row["price"]))
        cands.sort(reverse=True)
        # 実際の制約は候補数ではなく保有枠。枠が埋まっていれば、
        # 閾値をいくら緩めても新しい銘柄は入らない。
        if len(pos) >= max_names:
            diag["full_slots"] += 1

        for p, code, add, price in cands:
            if len(pos) >= max_names and code not in pos:
                continue
            for _ in range(add):
                if cash < unit_size or unit_size < price:
                    break
                sh = int(unit_size // price)
                if sh <= 0:
                    break
                cash -= sh * price
                st = pos.setdefault(code, {"units": 0, "shares": []})
                st["shares"].append(sh)
                st["units"] += 1
                trades.append({"code": code, "date": dt, "side": "buy",
                               "price": price, "shares": sh})

        val = cash + sum(sum(s["shares"]) * day.loc[c, "price"]
                         for c, s in pos.items() if c in day.index)
        curve.append({"date": dt, "value": val, "cash": cash, "names": len(pos)})

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
    return {"総リターン": total * 100, "年率": cagr * 100,
            "最大下落": dd * 100, "シャープ": sharpe,
            "売買回数": len(t), "平均保有銘柄": float(c["names"].mean()),
            "判定変化": d.get("changed", 0), "枠飽和率": saturated}


# ══════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=7, help="検証する年数")
    ap.add_argument("--limit", type=int, default=0, help="試し実行。先頭N銘柄")
    ap.add_argument("--only", default="", help="比較するルールをカンマ区切りで指定")
    ap.add_argument("--capital", type=float, default=3_000_000)
    ap.add_argument("--max-names", type=int, default=15)
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
        store = fetch_all(JQ(key), args.years, args.limit)
        pd.to_pickle(store, CACHE)
        log.info("キャッシュに保存しました: %s", CACHE)

    log.info("パネルを作成中…")
    panel = build_panel(store, args.years)
    log.info("判定できる時点: %d件 / 銘柄 %d / 期間 %s〜%s",
             len(panel), panel["code"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    names = [x.strip() for x in args.only.split(",") if x.strip()] or list(VARIANTS)
    rows = []
    for name in names:
        if name not in VARIANTS:
            log.warning("未定義のルール: %s", name)
            continue
        cfg = VARIANTS[name]
        res = simulate(panel, cfg, args.capital, args.max_names)
        m = metrics(res)
        if m:
            rows.append({"ルール": cfg["label"], **m})
            log.info("  %s 完了", cfg["label"])

    if not rows:
        print("結果がありません。")
        return 1

    df = pd.DataFrame(rows).sort_values("年率", ascending=False)
    print(f"\n■ 比較結果（{panel['date'].min().date()} 〜 "
          f"{panel['date'].max().date()} / 元本 {args.capital:,.0f}円）\n")
    print(f"{'ルール':<22}{'総リターン':>10}{'年率':>8}{'最大下落':>9}"
          f"{'シャープ':>9}{'売買':>7}{'判定変化':>9}")
    print("-" * 76)
    for _, x in df.iterrows():
        print(f"{x['ルール']:<22}{x['総リターン']:>9.1f}%{x['年率']:>7.1f}%"
              f"{x['最大下落']:>8.1f}%{x['シャープ']:>9.2f}"
              f"{x['売買回数']:>7.0f}{x['判定変化']:>9.0f}")

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
    print("\n※ 過去の成績であり、将来の結果を保証するものではありません。"
          "\n※ 手数料・税金・約定のずれは含めていません。実際の成績はこれより下がります。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

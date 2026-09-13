#!/usr/bin/env python3
"""投資できる金額から、買うべき銘柄と株数を出す。

その日の results.json を読み、実運用と同じルールで配分する。
  ・スクリーニングを通過している
  ・BUYシグナル（現在利回り ≧ 過去3年分布のQ75）
  ・緊急撤退に当たっていない
  ・Tier順（S→A→B）、同Tier内は利回りの高い順
  ・1銘柄の予算 ＝ 金額 ÷ 15 × （Tierの重み ÷ 平均）
  ・100株単位に切り捨て

相場を読んで買い控えることはしない（検証で11案すべて失敗したため）。
使い切れないのは「条件を満たす銘柄が足りないとき」だけ。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = DATA / "results.json"

TARGET_HOLDINGS = 15
TIER_WEIGHT = {"S": 2.0, "A": 1.5, "B": 1.0}
MAX_HOLDINGS = 20


def yen(v: float) -> str:
    return f"{v:,.0f}円"


def load_results() -> dict:
    if not RESULTS.exists():
        sys.exit(f"{RESULTS} がありません。"
                 "先に「Generate Dividend Signals」を実行してください。")
    with RESULTS.open(encoding="utf-8") as f:
        return json.load(f)


def pick(data: dict) -> tuple[list, dict]:
    """買いの候補を、実運用と同じ順番で並べて返す。"""
    rows = data.get("stocks") or data.get("results") or []
    stat = {"全体": len(rows), "スクリーニング通過": 0, "BUY": 0,
            "緊急撤退で除外": 0, "価格なし": 0}
    cands = []
    for r in rows:
        if not r.get("screening_pass", r.get("pass", False)):
            continue
        stat["スクリーニング通過"] += 1
        if str(r.get("signal", "")).upper() != "BUY":
            continue
        stat["BUY"] += 1
        if r.get("emergency_exit"):
            stat["緊急撤退で除外"] += 1
            continue
        px = r.get("price") or r.get("close")
        if not px or px <= 0:
            stat["価格なし"] += 1
            continue
        cands.append({
            "code": str(r.get("code", "")),
            "name": r.get("name", ""),
            "tier": r.get("tier", "B"),
            "price": float(px),
            "yield": float(r.get("current_yield") or r.get("yield") or 0),
            "pct": float(r.get("yield_percentile") or r.get("percentile") or 0),
            "sector": r.get("sector", ""),
        })
    order = {"S": 0, "A": 1, "B": 2}
    cands.sort(key=lambda x: (order.get(x["tier"], 9), -x["yield"]))
    return cands, stat


def fit_target(cands: list, budget: float, target: int) -> int:
    """金額が小さいときは、目標銘柄数を減らす。

    1銘柄あたりの枠が小さすぎると、株価の高い銘柄が100株も買えず、
    「候補はあるのに1つも買えない」という結果になってしまう。
    上位の銘柄が最低3つは買える水準まで、目標を下げる。
    """
    if not cands:
        return target
    avg_w = sum(TIER_WEIGHT.values()) / len(TIER_WEIGHT)
    top = cands[:max(3, min(8, len(cands)))]
    for t in range(target, 0, -1):
        ok = 0
        for c in top:
            unit = budget / t * (TIER_WEIGHT.get(c["tier"], 1.0) / avg_w)
            if int(unit // c["price"] // 100) * 100 > 0:
                ok += 1
        if ok >= min(3, len(top)):
            return t
    return 1


def allocate(cands: list, budget: float, target: int = TARGET_HOLDINGS,
             max_names: int = MAX_HOLDINGS) -> tuple[list, float]:
    """Tier順に、予算の範囲で割り当てる。"""
    avg_w = sum(TIER_WEIGHT.values()) / len(TIER_WEIGHT)
    cash = budget
    plan = []
    for c in cands:
        if len(plan) >= max_names:
            break
        unit = budget / target * (TIER_WEIGHT.get(c["tier"], 1.0) / avg_w)
        shares = int(unit // c["price"] // 100) * 100
        if shares <= 0:
            continue
        cost = shares * c["price"]
        if cost > cash:
            # 予算が残り少なければ、買える範囲まで減らす
            shares = int(cash // c["price"] // 100) * 100
            if shares <= 0:
                continue
            cost = shares * c["price"]
        cash -= cost
        plan.append({**c, "shares": shares, "cost": cost,
                     "budget": unit})

    # 端数が余ったら、上位の銘柄から買い増して埋める。
    # 1銘柄が予算の2倍を超えないようにして、集中しすぎを防ぐ。
    if plan and cash > 0:
        cap = budget / target * 2.0
        for _ in range(50):
            bought = False
            for p_ in plan:
                if p_["cost"] + p_["price"] * 100 > cap:
                    continue
                if p_["price"] * 100 > cash:
                    continue
                p_["shares"] += 100
                p_["cost"] += p_["price"] * 100
                cash -= p_["price"] * 100
                bought = True
            if not bought:
                break
    return plan, cash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amount", type=float, required=True,
                    help="投資できる金額（円）")
    ap.add_argument("--target", type=int, default=TARGET_HOLDINGS,
                    help="目標保有銘柄数（既定15）")
    ap.add_argument("--max-names", type=int, default=MAX_HOLDINGS,
                    help="同時に持つ最大銘柄数（既定20）")
    args = ap.parse_args()

    if args.amount <= 0:
        sys.exit("金額を正の数で指定してください。")

    data = load_results()
    cands, stat = pick(data)
    target = fit_target(cands, args.amount, args.target)
    plan, left = allocate(cands, args.amount, target, args.max_names)

    gen = data.get("generated_at") or data.get("date") or "（日付不明）"
    th = data.get("thresholds", {})

    out = []
    out.append(f"# 配分案 — {yen(args.amount)}")
    out.append("")
    out.append(f"判定日：{gen}")
    if th:
        out.append(f"設定：利回り {th.get('min_yield', '?')}％以上 ／ "
                   f"分位 {data.get('yield_sample_n', '?')}か月")
    out.append("")

    # ── 候補の絞り込み過程 ──
    out.append("## 候補の絞り込み")
    out.append("")
    out.append("| 段階 | 件数 |")
    out.append("|---|---|")
    out.append(f"| 対象銘柄 | {stat['全体']} |")
    out.append(f"| スクリーニング通過 | {stat['スクリーニング通過']} |")
    out.append(f"| うち BUYシグナル | {stat['BUY']} |")
    if stat["緊急撤退で除外"]:
        out.append(f"| 緊急撤退で除外 | −{stat['緊急撤退で除外']} |")
    out.append(f"| **買える候補** | **{len(cands)}** |")
    out.append("")

    if not plan:
        out.append("## 買える銘柄がありません")
        out.append("")
        if not cands:
            out.append("条件を満たす銘柄がゼロでした。**見送ってください。**")
            out.append("")
            out.append("相場が悪いからではなく、"
                       "**いま割安と判定される銘柄がないだけ**です。")
            out.append("翌営業日にまた判定されます。")
        else:
            cheapest = min(c["price"] for c in cands) * 100
            out.append(f"候補は{len(cands)}件ありますが、"
                       f"**金額が足りません。**")
            out.append("")
            out.append(f"いちばん安い候補でも、100株で {yen(cheapest)} 必要です。")
            out.append("")
            out.append(f"**{yen(cheapest)} 以上を用意するか、"
                       "単元未満株（S株・ミニ株）が使える証券会社を"
                       "ご検討ください。**")
        print("\n".join(out))
        return 0

    # ── 配分案 ──
    used = args.amount - left
    out.append("## 買う銘柄")
    out.append("")
    out.append("| Tier | コード | 銘柄 | 利回り | 分位 | 株価 | 株数 | 金額 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for p in plan:
        out.append(f"| {p['tier']} | {p['code']} | {p['name'][:14]} | "
                   f"{p['yield']:.2f}％ | Q{p['pct']:.0f} | "
                   f"{p['price']:,.0f} | {p['shares']:,}株 | "
                   f"{p['cost']:,.0f}円 |")
    out.append("")
    out.append(f"**合計 {yen(used)}（{used / args.amount * 100:.0f}％）"
               f" ／ 残り {yen(left)}**")
    out.append("")

    # ── 業種の内訳 ──
    sec = {}
    for p in plan:
        sec[p["sector"] or "（不明）"] = sec.get(p["sector"] or "（不明）", 0) \
            + p["cost"]
    if sec:
        out.append("## 業種の内訳")
        out.append("")
        out.append("| 業種 | 金額 | 比率 |")
        out.append("|---|---|---|")
        for s_, v in sorted(sec.items(), key=lambda x: -x[1]):
            out.append(f"| {s_} | {v:,.0f}円 | {v / used * 100:.0f}％ |")
        out.append("")
        top = max(sec.values()) / used * 100
        if top >= 40:
            out.append(f"**1業種に{top:.0f}％が集中しています。**")
            out.append("")
            out.append("検証では業種の上限を設けても成績は変わりませんでしたが、"
                       "偏りが気になるなら、")
            out.append("**資産全体のなかでこの戦略の比率を下げる**ほうが"
                       "筋が通ります。")
            out.append("")

    # ── 使い切れなかった場合 ──
    if left > args.amount * 0.05:
        out.append("## 残った資金について")
        out.append("")
        if len(cands) <= len(plan):
            out.append(f"**条件を満たす銘柄が{len(cands)}件しかありませんでした。**")
            out.append("")
            out.append("相場が悪いからではなく、"
                       "**いま割安と判定される銘柄が少ない**だけです。")
            out.append("")
            out.append("翌営業日以降、新しい候補が出たときに回してください。")
        else:
            out.append("上限（20銘柄）に達したため、残りました。")
        out.append("")
        out.append("> 検証では、**まとまった資金は早めに入れたほうが"
                   "成績が良い**と出ています")
        out.append("> （初日一括 +5.3pt ／ 2年かけて分割 −0.6pt）。")
        out.append("> **意図的に温存する理由はありません。**")
        out.append("")

    # ── 注意 ──
    out.append("---")
    out.append("")
    out.append("**発注前に確認してください**")
    out.append("")
    out.append("- 株価は判定時点のものです。寄り付きで変わります")
    out.append("- 決算発表の直前・直後は、数字が古い可能性があります")
    out.append("- すでに保有している銘柄が含まれていないか")
    out.append("")
    out.append("*過去の検証にもとづく機械的な配分であり、"
               "将来の結果を保証するものではありません。*")

    text = "\n".join(out)
    print(text)

    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        with open(sm, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

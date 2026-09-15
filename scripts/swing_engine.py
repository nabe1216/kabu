#!/usr/bin/env python3
"""レンジ（ボックス）売買を、仮想で並走記録する。

目的
    バックテストでは「レンジ売買は効かない」と出た。
    ただしその検証は 2019-2026 の上昇相場のデータによるもので、
    横ばい相場では結論が変わる可能性が残っている。
    そこで、これから先の実データを記録して確かめる。

やること
    毎営業日、results.json を読み
      ・レンジの中にいる銘柄を判定する
      ・下限に近づいたら仮想で買う
      ・上限に近づいたら仮想で売る
    売買と評価額を記録し、本番トラックと並べて比較できるようにする。

判定の条件（バックテストと同じ）
    ・過去60営業日の高値と安値でレンジの上下を決める
    ・幅が 2〜20％ に収まっていればレンジとみなす
    ・下限 +1％以内で買い、上限 −2％以内で売る
    ・レンジを割っても損切りはしない（そのまま持つ）

本番トラック（portfolio_engine.py）には一切影響しない。
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = DATA / "results.json"
STATE = DATA / "swing_state.json"
HISTORY = DATA / "swing_history.json"

JST = ZoneInfo("Asia/Tokyo")

# ── 設定 ──
INITIAL_CASH = 10_000_000      # 元本（本番トラックと同額にして比べやすくする）
MAX_HOLDINGS = 20              # 同時に持つ上限
TARGET_HOLDINGS = 15           # 1銘柄あたりの予算の逆算に使う

BOX_WIDTH_MIN = 0.02           # レンジとみなす値幅の下限（2％）
BOX_WIDTH_MAX = 0.20           # 同 上限（20％）
BUY_AT = 0.01                  # 下限から何％以内で買うか
SELL_AT = 0.02                 # 上限から何％以内で売るか

TAX_RATE = 0.20315             # 譲渡益への課税
SLIP = 0.001                   # 約定のずれ（片道0.1％）


def load(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] {path.name} を読めません: {e}")
        return default


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def box_of(stock: dict) -> tuple[float, float] | None:
    """その銘柄のレンジの上下を返す。レンジでなければ None。

    results.json の box（generate.py が出しているボックス判定）を使う。
    幅が条件から外れていれば対象外。
    """
    b = stock.get("box") or {}
    hi, lo = b.get("upper"), b.get("lower")
    if not hi or not lo or lo <= 0 or hi <= lo:
        return None
    width = (hi - lo) / lo
    if not (BOX_WIDTH_MIN <= width <= BOX_WIDTH_MAX):
        return None
    return float(hi), float(lo)


def main() -> int:
    res = load(RESULTS, None)
    if not res:
        sys.exit("results.json がありません。先に generate.py を実行してください。")

    stocks = {s["code"]: s for s in res.get("stocks", []) if s.get("code")}
    today = datetime.now(JST).date().isoformat()

    st = load(STATE, {
        "started_at": today, "cash": INITIAL_CASH,
        "initial_cash": INITIAL_CASH, "holdings": [],
        "realized_pl": 0.0, "tax_paid": 0.0, "loss_pool": 0.0,
        "trades_count": 0,
    })
    hi = load(HISTORY, {"daily_snapshots": [], "trades": []})

    # 同じ日に二重で回さない
    if hi["daily_snapshots"] and hi["daily_snapshots"][-1].get("date") == today:
        print(f"{today} はすでに記録済みです。何もしません。")
        return 0

    held = {h["code"]: h for h in st["holdings"]}
    log = []

    # ── 売り：上限に届いたら降りる ──
    for code in list(held):
        s = stocks.get(code)
        if not s or not s.get("price"):
            continue
        box = box_of(s)
        if not box:
            continue
        upper, _ = box
        price = float(s["price"])
        if price < upper * (1 - SELL_AT):
            continue

        h = held[code]
        eff = price * (1 - SLIP)
        gross = (eff - h["buy_price"]) * h["qty"]
        tax = 0.0
        if gross > 0:
            taxable = max(0.0, gross - st["loss_pool"])
            st["loss_pool"] = max(0.0, st["loss_pool"] - gross)
            tax = taxable * TAX_RATE
        else:
            st["loss_pool"] += -gross
        st["cash"] += h["qty"] * eff - tax
        st["realized_pl"] += gross
        st["tax_paid"] += tax
        st["trades_count"] += 1

        days = (date.fromisoformat(today)
                - date.fromisoformat(h["bought_at"])).days
        hi["trades"].append({
            "date": today, "action": "SELL", "code": code,
            "name": s.get("name", code), "qty": h["qty"],
            "price": round(eff, 1), "amount": round(h["qty"] * eff),
            "realized_pl": round(gross), "tax": round(tax),
            "holding_days": days, "reason": "上限に到達",
        })
        log.append(f"  売り {code} {s.get('name','')} "
                   f"{h['qty']}株 @{eff:,.0f} 損益{gross:+,.0f}")
        del held[code]

    # ── 買い：下限に近づいたら拾う ──
    cands = []
    for code, s in stocks.items():
        if code in held or not s.get("price"):
            continue
        if s.get("emergency_exit"):
            continue
        box = box_of(s)
        if not box:
            continue
        upper, lower = box
        price = float(s["price"])
        near = (price - lower) / lower
        if 0 <= near <= BUY_AT or price < lower:
            cands.append((near, code, s, price))
    cands.sort()

    total_now = st["cash"] + sum(
        (stocks.get(c, {}).get("price") or h["buy_price"]) * h["qty"]
        for c, h in held.items())
    unit = total_now / TARGET_HOLDINGS

    for _, code, s, price in cands:
        if len(held) >= MAX_HOLDINGS:
            break
        qty = int(unit // price // 100) * 100
        if qty <= 0:
            continue
        eff = price * (1 + SLIP)
        cost = qty * eff
        if cost > st["cash"]:
            qty = int(st["cash"] // eff // 100) * 100
            if qty <= 0:
                continue
            cost = qty * eff
        st["cash"] -= cost
        st["trades_count"] += 1
        held[code] = {"code": code, "name": s.get("name", code),
                      "qty": qty, "buy_price": round(eff, 1),
                      "bought_at": today, "tier": s.get("tier"),
                      "sector": s.get("sector", "")}
        hi["trades"].append({
            "date": today, "action": "BUY", "code": code,
            "name": s.get("name", code), "qty": qty,
            "price": round(eff, 1), "amount": round(cost),
            "reason": "下限に接近",
        })
        log.append(f"  買い {code} {s.get('name','')} {qty}株 @{eff:,.0f}")

    # ── 評価 ──
    hold_list = []
    hold_val = 0.0
    for code, h in held.items():
        px = stocks.get(code, {}).get("price") or h["buy_price"]
        val = px * h["qty"]
        hold_val += val
        pl = val - h["buy_price"] * h["qty"]
        hold_list.append({**h, "price": px, "value": round(val),
                          "unrealized_pl": round(pl),
                          "unrealized_pl_pct": round(
                              pl / (h["buy_price"] * h["qty"]) * 100, 2)})
    st["holdings"] = sorted(hold_list, key=lambda x: -x["value"])
    st["total_value"] = round(st["cash"] + hold_val)
    st["updated_at"] = today

    hi["daily_snapshots"].append({
        "date": today, "total_value": st["total_value"],
        "cash": round(st["cash"]), "holdings": len(held),
        "realized_pl": round(st["realized_pl"]),
        "tax_paid": round(st["tax_paid"]),
    })
    hi["daily_snapshots"] = hi["daily_snapshots"][-1500:]
    hi["trades"] = hi["trades"][-500:]

    save(STATE, st)
    save(HISTORY, hi)

    ret = st["total_value"] - st["initial_cash"]
    print(f"=== レンジ売買の記録 {today} ===")
    for l in log:
        print(l)
    if not log:
        print("  売買なし")
    print(f"\n  評価総額 {st['total_value']:,}円"
          f"（{ret:+,}円 / {ret / st['initial_cash'] * 100:+.2f}％）")
    print(f"  保有 {len(held)}銘柄 ／ 現金 {st['cash']:,.0f}円")
    print(f"  確定損益 {st['realized_pl']:+,.0f}円 ／ "
          f"支払った税金 {st['tax_paid']:,.0f}円 ／ 売買 {st['trades_count']}回")
    print("\n  ※ これは仮想の記録です。本番トラックには影響しません。")
    print("    バックテストでは効かないと出ましたが、"
          "相場が変わったときに確かめるための材料です。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

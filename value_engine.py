#!/usr/bin/env python3
"""低PBR順のルールを、仮想で並走記録する。

目的
    過去7年の検証では、「利回り4％以上の中で、PBRの安い側20％を、
    Tier順・PBRの低い順に買う」形が、いまの本番（利回りの高い順）を
    上回った。ただしその優位の多くは2023年のPBR改革のあとに生まれている。
    これから先の実データで、本当に上回り続けるかを確かめる。

ルール（検証の pbr_low_tier と同じ）
    買う  … スクリーニング8条件を通り、利回りが足切り以上、緊急撤退でなく、
            PBRが全銘柄の安い側20％に入る銘柄。
            Tier順（S→A→B）、同じTierなら PBR の低い順。
    予算  … 総資産 ÷ 15 × （Tierの重み ÷ 1.5）。100株単位。最大20銘柄。
    売る  … 減配・業績急変（緊急撤退）のときだけ。
    配当  … 権利月に予想配当の半分を受け取る（税引後）。

本番（portfolio_engine.py）には一切影響しない。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# リポジトリ直下に置いても、scripts/ に置いても動くように、data フォルダを探す
_here = Path(__file__).resolve().parent
DATA = _here / "data" if (_here / "data").exists() else _here.parent / "data"
RESULTS = DATA / "results.json"
STATE = DATA / "value_state.json"
HISTORY = DATA / "value_history.json"
JST = ZoneInfo("Asia/Tokyo")

INITIAL_CASH = 10_000_000
MAX_HOLDINGS = 20
TARGET_HOLDINGS = 15
TIER_WEIGHT = {"S": 2.0, "A": 1.5, "B": 1.0}
PBR_TOP = 80          # 全銘柄の中で PBR が安い側 20％（順位が80％以上）
TAX_RATE = 0.20315
SLIP = 0.001


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


def pbr_rank(stocks: list[dict]) -> dict[str, float]:
    """全銘柄の中での PBR の順位（％）。安いほど大きい値。"""
    vals = [(s["code"], float(s["pbr"])) for s in stocks
            if s.get("pbr") not in (None, "") and float(s["pbr"]) > 0]
    vals.sort(key=lambda x: -x[1])           # 高い順に並べ、後ろほど安い
    n = len(vals)
    return {c: (i + 1) / n * 100 for i, (c, _) in enumerate(vals)} if n else {}


def main() -> int:
    res = load(RESULTS, None)
    if not res:
        sys.exit("results.json がありません。先に generate.py を実行してください。")
    stocks_list = [s for s in res.get("stocks", []) if s.get("code")]
    stocks = {s["code"]: s for s in stocks_list}
    min_yield = float((res.get("thresholds") or {}).get("min_yield") or 4.0)
    today = datetime.now(JST).date().isoformat()
    month = today[:7]

    st = load(STATE, {"started_at": today, "initial_cash": INITIAL_CASH,
                      "cash": INITIAL_CASH, "holdings": [], "realized_pl": 0.0,
                      "tax_paid": 0.0, "loss_pool": 0.0, "trades_count": 0,
                      "dividend_received": 0.0, "div_paid_keys": []})
    hi = load(HISTORY, {"daily_snapshots": [], "trades": [], "dividends": []})
    if hi["daily_snapshots"] and hi["daily_snapshots"][-1].get("date") == today:
        print(f"{today} はすでに記録済みです。何もしません。")
        return 0

    held = {h["code"]: h for h in st["holdings"]}
    paid = set(st.get("div_paid_keys", []))
    log = []

    # ── 配当：権利月に予想配当の半分 ──
    mnum = int(today[5:7])
    for code, h in held.items():
        s = stocks.get(code, {})
        key = f"{code}:{month}"
        if key in paid:
            continue
        if mnum in (s.get("fiscal_month"), s.get("interim_month")) and s.get("forecast_dps"):
            amt = h["qty"] * float(s["forecast_dps"]) / 2 * (1 - TAX_RATE)
            st["cash"] += amt
            st["dividend_received"] = st.get("dividend_received", 0.0) + amt
            paid.add(key)
            hi["dividends"].append({"date": today, "code": code, "name": h.get("name", code),
                                    "shares": h["qty"], "amount": round(amt)})

    # ── 売り：減配・業績急変のときだけ ──
    for code in list(held):
        s = stocks.get(code)
        if not s or not s.get("emergency_exit") or not s.get("price"):
            continue
        h = held[code]
        eff = float(s["price"]) * (1 - SLIP)
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
        hi["trades"].append({"date": today, "action": "SELL", "code": code,
                             "name": h.get("name", code), "qty": h["qty"],
                             "price": round(eff, 1), "amount": round(h["qty"] * eff),
                             "realized_pl": round(gross), "tax": round(tax),
                             "reason": "減配・業績急変"})
        log.append(f"  売り {code} {h.get('name','')} {h['qty']}株（緊急撤退）")
        del held[code]

    # ── 買い：8条件・利回り・PBRの安い側20％。Tier順、PBRの低い順 ──
    rank = pbr_rank(stocks_list)
    order = {"S": 0, "A": 1, "B": 2}
    cands = [s for s in stocks_list
             if s["code"] not in held and s.get("screening_pass")
             and not s.get("emergency_exit") and s.get("price")
             and float(s.get("current_yield") or 0) >= min_yield
             and rank.get(s["code"], 0) >= PBR_TOP]
    cands.sort(key=lambda s: (order.get(s.get("tier") or "B", 9), float(s.get("pbr") or 99)))

    total = st["cash"] + sum(float(stocks.get(c, {}).get("price") or h["buy_price"]) * h["qty"]
                             for c, h in held.items())
    for s in cands:
        if len(held) >= MAX_HOLDINGS:
            break
        price = float(s["price"])
        unit = total / TARGET_HOLDINGS * (TIER_WEIGHT.get(s.get("tier") or "B", 1.0) / 1.5)
        qty = int(unit // price // 100) * 100
        if qty <= 0:
            continue
        eff = price * (1 + SLIP)
        if qty * eff > st["cash"]:
            qty = int(st["cash"] // eff // 100) * 100
            if qty <= 0:
                continue
        st["cash"] -= qty * eff
        st["trades_count"] += 1
        held[s["code"]] = {"code": s["code"], "name": s.get("name", s["code"]),
                           "qty": qty, "buy_price": round(eff, 1), "bought_at": today,
                           "tier": s.get("tier"), "sector": s.get("sector", "")}
        hi["trades"].append({"date": today, "action": "BUY", "code": s["code"],
                             "name": s.get("name", s["code"]), "qty": qty,
                             "price": round(eff, 1), "amount": round(qty * eff),
                             "reason": f"PBR {float(s.get('pbr') or 0):.2f}"})
        log.append(f"  買い {s['code']} {s.get('name','')} {qty}株 PBR {float(s.get('pbr') or 0):.2f}")

    # ── 評価 ──
    hold_list, hold_val = [], 0.0
    for code, h in held.items():
        px = float(stocks.get(code, {}).get("price") or h["buy_price"])
        val = px * h["qty"]
        hold_val += val
        pl = val - h["buy_price"] * h["qty"]
        hold_list.append({**h, "price": px, "value": round(val), "unrealized_pl": round(pl),
                          "unrealized_pl_pct": round(pl / (h["buy_price"] * h["qty"]) * 100, 2)})
    st["holdings"] = sorted(hold_list, key=lambda x: -x["value"])
    st["total_value"] = round(st["cash"] + hold_val)
    st["updated_at"] = today
    st["div_paid_keys"] = sorted(paid)[-400:]
    hi["daily_snapshots"].append({"date": today, "total_value": st["total_value"],
                                  "cash": round(st["cash"]), "holdings": len(held)})
    hi["daily_snapshots"] = hi["daily_snapshots"][-1500:]
    hi["trades"] = hi["trades"][-500:]
    hi["dividends"] = hi["dividends"][-500:]
    save(STATE, st)
    save(HISTORY, hi)

    ret = st["total_value"] - st["initial_cash"]
    print(f"=== 低PBR順の記録 {today} ===")
    print("\n".join(log) if log else "  売買なし")
    print(f"\n  評価総額 {st['total_value']:,}円（{ret:+,}円 / {ret / st['initial_cash'] * 100:+.2f}％）")
    print(f"  保有 {len(held)}銘柄 ／ 現金 {st['cash']:,.0f}円 ／ 候補 {len(cands)}件")
    print("  ※ 仮想の記録です。本番には影響しません。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

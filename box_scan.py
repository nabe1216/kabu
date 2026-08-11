#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ボックス銘柄スキャン（J-Quants V2 対応）

generate.py と同じ流儀で書いてあります。
  認証   : J_QUANTS_API_KEY を x-api-key ヘッダーに付けるだけ（トークン交換は不要）
  一覧   : /v2/equities/master     … Mkt / ScaleCat でプライムの大型〜中型株に絞る
  株価   : /v2/equities/bars/daily … 短縮カラム名を V1 相当に正規化して使う

やること:
  1. ユニバースを取得
  2. 銘柄ごとに日次株価を取得
  3. box_detect.py の判定にかけて 0〜100 でスコア化
  4. data/box.json と data/box.csv に書き出し、実行ログに一覧を表示

使い方:
  python box_scan.py                  直近120営業日で判定
  python box_scan.py --window 240     1年で判定
  python box_scan.py --limit 30       まず30銘柄だけで試す
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

import pandas as pd
import requests

try:
    from box_detect import BoxConfig, detect_box
except ImportError:
    sys.exit("box_detect.py が見つかりません。box_scan.py と同じ場所に置いてください。")

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("box-scan")

JQUANTS_BASE = "https://api.jquants.com"
API_SLEEP_SEC = 0.55          # Standard は 120 req/min。余裕を持たせる
API_TIMEOUT_SEC = 30
API_MAX_RETRIES = 3

MARKET_CODE_PRIME = "0111"
SCALE_TARGETS = {"TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"}

OUTPUT_DIR = Path("data")


# ────────────────────────────────── APIクライアント
class JQuantsClient:
    """J-Quants V2 クライアント。generate.py と同じ認証方式。"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("J_QUANTS_API_KEY is empty")
        self.api_key = api_key
        self.session = requests.Session()

    def get(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        url = f"{JQUANTS_BASE}{path}"
        rows_all: list[dict[str, Any]] = []
        pagination_key: str | None = None

        while True:
            req = dict(params or {})
            if pagination_key:
                req["pagination_key"] = pagination_key

            for attempt in range(API_MAX_RETRIES):
                try:
                    resp = self.session.get(
                        url, params=req,
                        headers={"x-api-key": self.api_key},
                        timeout=API_TIMEOUT_SEC)
                    if resp.status_code == 429:
                        wait = 2 ** (attempt + 1)
                        log.warning("レート制限。%d秒待機します", wait)
                        time.sleep(wait)
                        continue
                    if resp.status_code in (401, 403):
                        sys.exit(f"認証に失敗しました（{resp.status_code}）: {resp.text[:200]}\n"
                                 "J_QUANTS_API_KEY の値をご確認ください。")
                    resp.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == API_MAX_RETRIES - 1:
                        raise
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"リトライ上限に達しました: {url}")

            payload = resp.json()
            rows = payload.get("data")
            if rows is None:
                for k, v in payload.items():
                    if k != "pagination_key" and isinstance(v, list):
                        rows = v
                        break
            if rows:
                rows_all.extend(rows)

            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                break
            time.sleep(API_SLEEP_SEC)

        return rows_all

    @staticmethod
    def _normalize(q: dict[str, Any]) -> dict[str, Any]:
        """V2の短縮カラム名を V1 相当に揃える。"""
        mapping = {"O": "Open", "H": "High", "L": "Low", "C": "Close",
                   "AdjO": "AdjustmentOpen", "AdjH": "AdjustmentHigh",
                   "AdjL": "AdjustmentLow", "AdjC": "AdjustmentClose"}
        out = dict(q)
        for short, long_name in mapping.items():
            if short in out and long_name not in out:
                out[long_name] = out[short]
        return out

    def master(self) -> list[dict[str, Any]]:
        return self.get("/v2/equities/master", {})

    def bars(self, code: str, frm: str, to: str) -> list[dict[str, Any]]:
        rows = self.get("/v2/equities/bars/daily",
                        {"code": code, "from": frm, "to": to})
        return [self._normalize(r) for r in rows]


# ────────────────────────────────── 補助
def normalize_code(code: str) -> str:
    code = str(code).strip()
    return code[:4] if len(code) == 5 and code.endswith("0") else code


def to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_universe(client: JQuantsClient) -> list[dict[str, str]]:
    log.info("銘柄一覧を取得中…")
    info = client.master()
    log.info("取得件数: %d", len(info))

    uni = []
    for row in info:
        if row.get("Mkt", "") != MARKET_CODE_PRIME:
            continue
        if (row.get("ScaleCat", "") or "") not in SCALE_TARGETS:
            continue
        code = normalize_code(row.get("Code", ""))
        if not code:
            continue
        uni.append({
            "code": code,
            "name": row.get("CoName", "") or row.get("CoNameEn", ""),
            "sector": row.get("S33Nm", ""),
        })
    log.info("対象ユニバース: %d銘柄", len(uni))
    return uni


# ────────────────────────────────── メイン
def main() -> int:
    api_key = os.environ.get("J_QUANTS_API_KEY")
    if not api_key:
        log.error("J_QUANTS_API_KEY 環境変数が設定されていません")
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=120, help="判定期間（営業日）")
    ap.add_argument("--min-score", type=float, default=70.0)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="試し実行用。先頭N銘柄だけ処理")
    ap.add_argument("--include-broken", action="store_true",
                    help="レンジを抜けた銘柄も候補に含める（既定は除外）")
    ap.add_argument("--include-down", action="store_true",
                    help="下降チャネルも候補に含める（既定は除外）")
    args = ap.parse_args()

    client = JQuantsClient(api_key)
    uni = build_universe(client)
    if args.limit:
        uni = uni[:args.limit]
        log.info("試し実行のため %d銘柄に絞ります", len(uni))

    today = date.today()
    to_date = today.strftime("%Y-%m-%d")
    # 休場日を見込んで多めに取る
    frm_date = (today - timedelta(days=int(args.window * 1.7) + 30)).strftime("%Y-%m-%d")

    cfg = BoxConfig(window=args.window)
    rows, failures = [], 0

    log.info("株価を取得して判定中（%d銘柄、数分かかります）…", len(uni))
    for i, u in enumerate(uni, 1):
        time.sleep(API_SLEEP_SEC)
        try:
            q = client.bars(u["code"], frm_date, to_date)
        except Exception as e:
            log.warning("取得失敗 %s: %s", u["code"], e)
            failures += 1
            continue
        if not q:
            continue

        df = pd.DataFrame(q)
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        close = df.get("AdjustmentClose")
        if close is None:
            close = df.get("Close")
        if close is None:
            continue
        close = pd.to_numeric(close, errors="coerce").dropna()
        if len(close) < args.window * 0.6:
            continue

        high = pd.to_numeric(df.get("AdjustmentHigh", df.get("High")), errors="coerce")
        low = pd.to_numeric(df.get("AdjustmentLow", df.get("Low")), errors="coerce")

        r = detect_box(close, high, low, cfg)
        rows.append({
            "code": u["code"], "name": u["name"], "sector": u["sector"],
            "score": r["score"], "type": r["type"], "status": r["status"],
            "position": r["position"], "lower": r["lower"], "upper": r["upper"],
            "width_pct": r["width_pct"], "slope_pct": r["slope_annual_pct"],
            "adf_t": r["adf_t"], "crosses": r["crosses"],
            "last": round(float(close.iloc[-1]), 1),
        })

        if i % 50 == 0:
            log.info("進捗 %d/%d（該当 %d件）", i, len(uni), len(rows))

    if not rows:
        print(f"\n判定できる銘柄がありませんでした。（失敗 {failures}件）")
        return 0

    allf = pd.DataFrame(rows)

    # ── 全体の分布を必ず出す。しきい値はこれを見て決める ──
    print(f"\n■ スコアの分布（判定した {len(allf)}銘柄 / 判定期間 {args.window}営業日）\n")
    bins = [(90, 101), (80, 90), (70, 80), (60, 70), (0, 60)]
    for lo, hi in bins:
        n = int(((allf["score"] >= lo) & (allf["score"] < hi)).sum())
        bar = "█" * int(n / max(len(allf), 1) * 40)
        print(f"  {lo:>3}点以上 {n:>4}件 ({n/len(allf)*100:>4.1f}%) {bar}")
    print("\n  種別の内訳:")
    for t, n in allf["type"].value_counts().items():
        print(f"    {t:<12}{n:>4}件")
    print("\n  状態の内訳:")
    for t, n in allf["status"].value_counts().items():
        print(f"    {t:<12}{n:>4}件")

    # ── 買い候補として妥当なものだけに絞る ──
    out = allf[allf["score"] >= args.min_score].copy()
    n_before = len(out)

    excluded = []
    if not args.include_broken:
        # レンジを抜けた銘柄はボックス売買の前提が崩れている
        broken = out["status"].isin(["上抜け", "下抜け"]) | (out["position"] < -5) | (out["position"] > 105)
        excluded.append(("レンジを抜けている", int(broken.sum())))
        out = out[~broken]
    if not args.include_down:
        # 右肩下がりのレンジは、下限で買っても次の下限がさらに低い
        down = out["type"] == "下降チャネル"
        excluded.append(("下降チャネル", int(down.sum())))
        out = out[~down]

    print(f"\n■ 絞り込み（{args.min_score}点以上 {n_before}件 から）")
    for label, n in excluded:
        print(f"    − {label}: {n}件を除外")
    print(f"    → 候補 {len(out)}件")

    if out.empty:
        print("\n条件を満たす銘柄がありませんでした。min_score を下げてお試しください。")
        return 0

    # 並べ替え: 点数が高く、レンジの下寄り（買い場）にあるものを上に。
    # 下限ちょうどではなく少し上（15%付近）を最良として、離れるほど減点する。
    out["rank"] = out["score"] - (out["position"] - 15).abs() * 0.30
    out = out.sort_values("rank", ascending=False).head(args.top).drop(columns="rank")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "box.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "box.json").open("w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
            "window": args.window, "min_score": args.min_score,
            "universe_count": len(uni), "evaluated_count": len(allf),
            "failure_count": failures,
            "items": out.to_dict("records"),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n■ 買い候補 {len(out)}件\n")
    print(f"{'コード':<7}{'銘柄':<20}{'点':>5}{'種別':>13}{'状態':>10}"
          f"{'位置':>7}{'現在値':>9}{'下限':>9}{'上限':>9}{'幅':>7}")
    print("-" * 100)
    for _, x in out.iterrows():
        print(f"{x['code']:<7}{str(x['name'])[:18]:<20}{x['score']:>5.0f}"
              f"{x['type']:>13}{x['status']:>10}{x['position']:>6.0f}%"
              f"{x['last']:>9.1f}{x['lower']:>9.1f}{x['upper']:>9.1f}{x['width_pct']:>6.1f}%")
    print(f"\n書き出しました: data/box.csv, data/box.json（失敗 {failures}件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

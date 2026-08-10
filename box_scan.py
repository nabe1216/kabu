#!/usr/bin/env python3
"""
ボックス銘柄スキャン（GitHub Actions から実行する用）

やること:
  1. J-Quants から直近の日次株価をまとめて取得
  2. box_detect.py の判定にかける
  3. 結果を data/box.json と data/box.csv に書き出し、実行ログに一覧を表示

必要なもの:
  box_detect.py が同じ場所にあること
  環境変数に J-Quants のリフレッシュトークンが入っていること
  （GitHub の Secrets に登録済みのものをそのまま使えます）

使い方:
  python box_scan.py                    直近120営業日で判定
  python box_scan.py --window 240       1年で判定
  python box_scan.py --min-score 75     しきい値を上げる
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta

import pandas as pd
import requests

try:
    from box_detect import BoxConfig, detect_box
except ImportError:
    sys.exit("box_detect.py が見つかりません。同じフォルダに置いてください。")

JQ = "https://api.jquants.com/v1"

# Secrets の名前は環境によって違うので、よくある候補を順に探す
TOKEN_KEYS = ["J_QUANTS_API_KEY", "JQUANTS_REFRESH_TOKEN", "JQUANTS_TOKEN",
              "JQ_REFRESH_TOKEN", "REFRESH_TOKEN", "JQUANTS_MAIL_PASSWORD"]


def get_secret() -> str:
    for k in TOKEN_KEYS:
        v = os.environ.get(k)
        if v and v.strip():
            print(f"認証情報を {k} から読み込みました", file=sys.stderr)
            return v.strip()
    sys.exit("J-Quants の認証情報が環境変数に見つかりません。\n"
             f"次のいずれかの名前で設定してください: {', '.join(TOKEN_KEYS)}")


def jq_auth(secret: str) -> str:
    """認証情報の形式を自動で判別してIDトークンを取得する。

    J-Quantsのリフレッシュトークンは有効期限が1週間程度と短いため、
    シークレットにはメールアドレスとパスワードが入っていることが多い。
    次の順に試す。

      1. JSON形式        {"mailaddress": "...", "password": "..."}
      2. 区切り形式      メールアドレス:パスワード（改行区切りも可）
      3. リフレッシュトークンそのもの
    """
    mail = pw = None

    # 1) JSON
    if secret.lstrip().startswith("{"):
        try:
            j = json.loads(secret)
            mail = j.get("mailaddress") or j.get("mail") or j.get("email")
            pw = j.get("password") or j.get("pass")
        except json.JSONDecodeError:
            pass

    # 2) メールアドレスとパスワードの組
    if not mail and "@" in secret:
        for sep in ("\n", ",", ":", "\t", " "):
            if sep in secret:
                a, _, b = secret.partition(sep)
                if "@" in a and b.strip():
                    mail, pw = a.strip(), b.strip()
                    break

    if mail and pw:
        print("メールアドレスとパスワードで認証します", file=sys.stderr)
        r = requests.post(f"{JQ}/token/auth_user",
                          json={"mailaddress": mail, "password": pw}, timeout=30)
        if r.status_code != 200:
            sys.exit(f"ログインに失敗しました（{r.status_code}）: {r.text[:200]}")
        refresh = r.json().get("refreshToken")
        if not refresh:
            sys.exit("refreshToken を取得できませんでした。")
    else:
        print("リフレッシュトークンとして扱います", file=sys.stderr)
        refresh = secret

    r = requests.post(f"{JQ}/token/auth_refresh",
                      params={"refreshtoken": refresh}, timeout=30)
    if r.status_code != 200:
        sys.exit(
            f"IDトークンの取得に失敗しました（{r.status_code}）: {r.text[:200]}\n"
            "シークレットの中身をご確認ください。\n"
            "  ・リフレッシュトークンの場合、有効期限は1週間程度です\n"
            "  ・メールアドレスとパスワードなら、改行かコンマで区切って登録してください")
    return r.json()["idToken"]


def jq_get(path: str, token: str, **params) -> list:
    out, key = [], None
    while True:
        p = dict(params)
        if key:
            p["pagination_key"] = key
        r = requests.get(f"{JQ}{path}", headers={"Authorization": f"Bearer {token}"},
                         params=p, timeout=60)
        if r.status_code != 200:
            return out
        j = r.json()
        body = next((v for k, v in j.items()
                     if isinstance(v, list) and k != "pagination_key"), [])
        out.extend(body)
        key = j.get("pagination_key")
        if not key:
            return out


def fetch_universe(token: str) -> pd.DataFrame:
    """プライム・スタンダードの銘柄一覧"""
    df = pd.DataFrame(jq_get("/listed/info", token))
    if df.empty:
        sys.exit("銘柄一覧を取得できませんでした。")
    df = df[df["MarketCode"].isin(["0111", "0112"])]
    cols = ["Code", "CompanyName"]
    return df[[c for c in cols if c in df.columns]]


def fetch_prices(token: str, days: int) -> pd.DataFrame:
    """日付ごとに全銘柄の終値を取得する。
       銘柄ごとに引くより呼び出し回数が少なく済む。"""
    end = date.today()
    start = end - timedelta(days=int(days * 1.6) + 20)   # 休場日ぶんを見込む
    frames, d, n = [], start, 0
    while d <= end:
        if d.weekday() < 5:
            rows = jq_get("/prices/daily_quotes", token, date=d.isoformat())
            if rows:
                df = pd.DataFrame(rows)
                keep = [c for c in ["Date", "Code", "AdjustmentClose",
                                    "AdjustmentHigh", "AdjustmentLow"] if c in df.columns]
                frames.append(df[keep])
                n += 1
                if n % 20 == 0:
                    print(f"  ... {d} まで取得（{n}営業日）", file=sys.stderr)
        d += timedelta(days=1)
    if not frames:
        sys.exit("株価を取得できませんでした。")
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"])
    for c in ["AdjustmentClose", "AdjustmentHigh", "AdjustmentLow"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["AdjustmentClose"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=120, help="判定期間（営業日）")
    ap.add_argument("--min-score", type=float, default=70.0)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    token = jq_auth(get_secret())

    print("銘柄一覧を取得中…", file=sys.stderr)
    uni = fetch_universe(token)
    names = dict(zip(uni["Code"], uni.get("CompanyName", uni["Code"])))

    print(f"株価を取得中（{args.window}営業日ぶん、数分かかります）…", file=sys.stderr)
    px = fetch_prices(token, args.window)
    px = px[px["Code"].isin(set(uni["Code"]))]

    cfg = BoxConfig(window=args.window)
    print("判定中…", file=sys.stderr)

    rows = []
    for code, g in px.sort_values("Date").groupby("Code"):
        if len(g) < args.window * 0.6:
            continue
        r = detect_box(g["AdjustmentClose"],
                       g.get("AdjustmentHigh"), g.get("AdjustmentLow"), cfg)
        if r.get("score", 0) < args.min_score:
            continue
        rows.append({
            "code": code, "name": names.get(code, ""),
            "score": r["score"], "type": r["type"], "status": r["status"],
            "position": r["position"], "lower": r["lower"], "upper": r["upper"],
            "width_pct": r["width_pct"], "slope_pct": r["slope_annual_pct"],
            "adf_t": r["adf_t"], "crosses": r["crosses"],
            "last": round(float(g["AdjustmentClose"].iloc[-1]), 1),
        })

    if not rows:
        print(f"\n{args.min_score}点以上の銘柄はありませんでした。"
              f"--min-score を下げて試してください。")
        return

    out = pd.DataFrame(rows)
    # 買い場に近い順：点数が高く、レンジ下限に近いものを上に
    out["rank"] = out["score"] - out["position"] * 0.25
    out = out.sort_values("rank", ascending=False).head(args.top).drop(columns="rank")

    os.makedirs(args.outdir, exist_ok=True)
    out.to_csv(f"{args.outdir}/box.csv", index=False, encoding="utf-8-sig")
    with open(f"{args.outdir}/box.json", "w", encoding="utf-8") as f:
        json.dump({"asOf": date.today().isoformat(), "window": args.window,
                   "minScore": args.min_score, "items": out.to_dict("records")},
                  f, ensure_ascii=False, indent=2)

    print(f"\n■ ボックス候補 {len(out)}件（判定期間 {args.window}営業日）\n")
    print(f"{'コード':<7}{'銘柄':<18}{'点':>5}{'種別':>12}{'状態':>9}"
          f"{'位置':>7}{'現在値':>9}{'下限':>9}{'上限':>9}{'幅':>7}")
    print("-" * 96)
    for _, x in out.iterrows():
        print(f"{x['code']:<7}{str(x['name'])[:16]:<18}{x['score']:>5.0f}"
              f"{x['type']:>12}{x['status']:>9}{x['position']:>6.0f}%"
              f"{x['last']:>9.1f}{x['lower']:>9.1f}{x['upper']:>9.1f}{x['width_pct']:>6.1f}%")
    print(f"\n書き出しました: {args.outdir}/box.csv, {args.outdir}/box.json")


if __name__ == "__main__":
    main()

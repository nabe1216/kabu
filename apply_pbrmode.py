#!/usr/bin/env python3
"""買いの判定と並べ方を切り替える（低PBR順に対応した版）。

    pbr      … スクリーニングを通り、利回りが足切り以上で、
                PBR が全銘柄の安い側20％に入る銘柄を BUY。
                同じ Tier の中では PBR の低い順に買う。（新しいルール）
    yield    … スクリーニングを通り、利回りが足切り以上なら BUY。
                同じ Tier の中では利回りの高い順。（ひとつ前のルール）
    quantile … 利回りが過去3年の Q75 以上で BUY。（最初のルール）

なぜ変えるか
    16〜32条件の総当たりで、低PBR順が利回り順を88〜100％の条件で上回った。
    2023年のPBR改革より前の期間でも上回っていた。
    閾値（10/20/30％）・約定のずれ（30bps）を変えても崩れなかった。
    外部要因への依存の合計は +0.80pt で、目安（+1pt以内）に収まった。
    ただし円安・金利・原油への依存は合わせて約1.5pt増えている。

書き換えるファイル
    scripts/generate.py       … 判定（PBRの順位をつけ、安い側20％以外は BUY にしない）
    scripts/portfolio_engine.py … 同じ Tier の中の並べ方（buy_sort を使う）

目印が見つからない場合や文法エラーが出た場合は、何も保存せずに止まる。
"""
import ast
import os
import re
import sys
from pathlib import Path

GEN = Path("scripts/generate.py")
PORT = Path("scripts/portfolio_engine.py")

POST_BLOCK = '''    # ── 低PBR順（BUY_MODE == 'pbr'）──
    # 全銘柄の中で PBR が安い側 PBR_TOP_PCT% に入る銘柄だけを BUY に残し、
    # 同じ Tier の中では PBR の低い順に買えるよう buy_sort を付ける。
    _pbrs = sorted([(x['code'], x['pbr']) for x in stocks
                    if x.get('pbr') is not None and x['pbr'] > 0], key=lambda t: -t[1])
    _n = len(_pbrs)
    _rank = {c: (i + 1) / _n * 100 for i, (c, _) in enumerate(_pbrs)} if _n else {}
    for x in stocks:
        x['pbr_rank'] = round(_rank.get(x['code'], 0.0), 1)
        if BUY_MODE == 'pbr':
            x['buy_sort'] = x['pbr'] if x.get('pbr') else 99.0
            if x.get('signal') == 'BUY' and _rank.get(x['code'], 0.0) < PBR_TOP_PCT:
                x['signal'] = 'NEUTRAL'
        else:
            x['buy_sort'] = -(x.get('current_yield') or 0)

'''


def main() -> int:
    mode = os.environ.get("MODE", "pbr")
    if mode not in ("pbr", "yield", "quantile"):
        sys.exit(f"MODE は pbr / yield / quantile です（指定: {mode}）")
    for f in (GEN, PORT):
        if not f.exists():
            sys.exit(f"{f} がありません。")
    g = g0 = GEN.read_text(encoding="utf-8")
    pe = pe0 = PORT.read_text(encoding="utf-8")
    log = []

    # 1. BUY_MODE（ひとつ前の patch-buymode で入っているはず）
    m = re.search(r"^BUY_MODE\s*=\s*'(\w+)'", g, re.M)
    if not m:
        sys.exit("BUY_MODE が見つかりません。先に patch-buymode を実行してください。")
    if m.group(1) != mode:
        g = re.sub(r"^BUY_MODE\s*=\s*'\w+'", f"BUY_MODE = '{mode}'", g, count=1, flags=re.M)
        log.append(f"買いの判定 … {m.group(1)} → {mode}")

    # 2. PBR の閾値
    if "PBR_TOP_PCT" not in g:
        m2 = re.search(r"^BUY_MODE\s*=.*$", g, re.M)
        g = g.replace(m2.group(0), m2.group(0) + "\n"
                      "# 低PBR順のとき、全銘柄の中で PBR が安い側何％までを買うか。\n"
                      "# 80 なら「安い側20％」。検証で 10/20/30％ いずれも崩れなかったので、\n"
                      "# 最初に決めた 20％ を使う。\n"
                      "PBR_TOP_PCT = 80", 1)
        log.append("PBR の閾値（安い側20％）を追加")

    # 3. 判定：pbr のときも、まず利回りで BUY を出す（あとで PBR で絞る）
    if "if BUY_MODE in ('yield', 'pbr'):" not in g:
        if "    if BUY_MODE == 'yield':" not in g:
            sys.exit("determine_signal の新しいルールの分岐が見つかりません。中止します。")
        g = g.replace("    if BUY_MODE == 'yield':", "    if BUY_MODE in ('yield', 'pbr'):", 1)
        log.append("判定の関数を低PBR順にも対応")

    # 4. 全銘柄を処理したあとで、PBR の順位をつけて絞る
    if "_rank = {c: (i + 1) / _n * 100" not in g:
        anchor = None
        for cand in ("    _save_stmts_cache()\n", "    # 統計\n"):
            if g.count(cand) == 1:
                anchor = cand
                break
        if not anchor:
            sys.exit("PBR の順位を差し込む場所が見つかりません。中止します。")
        g = g.replace(anchor, POST_BLOCK + anchor, 1)
        log.append("全銘柄の PBR の順位をつけ、安い側20％以外は BUY にしない処理を追加")

    # 5. portfolio_engine の並べ方を buy_sort に
    old = "-(s.get('current_yield') or 0),"
    new = "s.get('buy_sort', -(s.get('current_yield') or 0)),"
    if new not in pe:
        if pe.count(old) != 1:
            sys.exit(f"portfolio_engine.py の並べ替えの目印が {pe.count(old)} 箇所です。中止します。")
        pe = pe.replace(old, new, 1)
        log.append("portfolio_engine の並べ方を buy_sort に変更（低PBR順なら PBR の低い順）")

    for src, name in ((g, "generate.py"), (pe, "portfolio_engine.py")):
        try:
            ast.parse(src)
        except SyntaxError as e:
            sys.exit(f"{name} の文法が壊れます。中止します: {e}")
    if g != g0:
        GEN.write_text(g, encoding="utf-8")
    if pe != pe0:
        PORT.write_text(pe, encoding="utf-8")

    out = ["## 買いの判定の変更", ""]
    out += [f"- {x}" for x in log] if log else [f"すでに {mode} です。変更はありません。"]
    out += ["",
            "**pbr** … 8条件・利回り4%以上・PBRが安い側20%。Tier順、PBRの低い順（新しいルール）",
            "**yield** … 8条件・利回り4%以上。Tier順、利回りの高い順（ひとつ前）",
            "**quantile** … 利回りが過去3年のQ75以上（最初のルール）", "",
            "今夜のシグナル生成から反映されます。戻すときは MODE に yield を入れて再実行してください。"]
    for x in log:
        print(x)
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        with open(sm, "a", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    else:
        print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

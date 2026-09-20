#!/usr/bin/env python3
"""対象にする規模区分を選べるようにする。

なぜ必要か
    プライム全銘柄（1,555社）に広げたところ、データの取得に118分かかり、
    ワークフローの上限90分に収まらなかった。
    株価は9年分を1銘柄ずつ取るため、社数を減らすしか手がない。

    時価総額で絞ろうとしたが、銘柄一覧には時価総額が含まれておらず、
    各銘柄を取得したあとでしか分からない。つまり事前には絞れない。
    使えるのは TOPIX の規模区分（ScaleCat）だけ。

段階
    core   … Core30 / Large70 / Mid400            約489社・40分
    small1 … 上記 + TOPIX Small 1                 社数は回してみないと分からない
    prime  … プライム全銘柄                        約1,555社・118分（枠外）

目印が見つからない場合や文法エラーが出た場合は、何も保存せずに止まる。
"""
import ast
import os
import re
import sys
from pathlib import Path

TARGET = Path("scripts/generate.py")
VALID = ("core", "small1", "prime")


def main() -> int:
    scope = os.environ.get("SCOPE", "small1")
    if scope not in VALID:
        sys.exit(f"scope は {' / '.join(VALID)} のいずれかです（指定: {scope}）")
    if not TARGET.exists():
        sys.exit(f"{TARGET} がありません。")

    s = before = TARGET.read_text(encoding="utf-8")
    log = []

    m = re.search(r"^UNIVERSE_SCOPE\s*=\s*'(\w+)'", s, re.M)
    if not m:
        sys.exit("UNIVERSE_SCOPE の行が見つかりません。"
                 "先に patch-universe を実行してください。")
    cur = m.group(1)
    if cur != scope:
        s = re.sub(r"^UNIVERSE_SCOPE\s*=\s*'\w+'",
                   f"UNIVERSE_SCOPE = '{scope}'", s, count=1, flags=re.M)
        log.append(f"対象の範囲 … {cur} → {scope}")

    # 絞り込みの条件を、段階に応じたものに置き換える
    old = re.search(
        r"^(\s*)# core のときだけ規模区分で絞る\n"
        r"\s*if UNIVERSE_SCOPE == 'core' and scale_cat not in SCALE_TARGETS:",
        s, re.M)
    if old:
        indent = old.group(1)
        new = (f"{indent}# 段階に応じて規模区分で絞る。\n"
               f"{indent}#   core   … Core30 / Large70 / Mid400\n"
               f"{indent}#   small1 … 上記 + TOPIX Small 1\n"
               f"{indent}#   prime  … 絞らない（社数が多く時間内に収まらない）\n"
               f"{indent}_targets = set(SCALE_TARGETS)\n"
               f"{indent}if UNIVERSE_SCOPE == 'small1':\n"
               f"{indent}    _targets |= {{'TOPIX Small 1'}}\n"
               f"{indent}if UNIVERSE_SCOPE != 'prime' and scale_cat not in _targets:")
        s = s.replace(old.group(0), new, 1)
        log.append("規模区分の条件を段階式に変更")
    elif "_targets = set(SCALE_TARGETS)" not in s:
        print("規模区分で絞っている箇所が見つかりません。候補を表示します：\n")
        for i, l in enumerate(s.split("\n"), 1):
            if "SCALE_TARGETS" in l or "scale_cat" in l:
                print(f"  {i:>5}: {l.rstrip()}")
        sys.exit("中止しました。")

    # 何社になったかを分かりやすく出す
    a2 = "        log.info('Universe filter: prime_skip=%d, scale_skip=%d, kept=%d',"
    if a2 in s and "対象の範囲:" not in s:
        s = s.replace(a2,
                      "        log.info('対象の範囲: %s', UNIVERSE_SCOPE)\n" + a2, 1)
        log.append("ログに対象の範囲を出すよう追加")

    if s == before:
        print("変更はありませんでした。")
        return 0

    ast.parse(s)
    TARGET.write_text(s, encoding="utf-8")
    for x in log:
        print(x)

    est = {"core": "約489社・40分", "small1": "回してみないと分からない",
           "prime": "約1,555社・118分（枠外）"}
    out = ["## generate.py の変更", ""]
    out += [f"- {x}" for x in log]
    out += ["", f"対象を **{scope}**（{est[scope]}）にしました。", "",
            "### 次にやること", "",
            "1. 「Generate Dividend Signals」を実行",
            "2. ログの `Universe size:` を見る", "",
            "**900社以下** … 70分前後。おそらく収まります",
            "**1,100社以上** … 85分超。枠に収まらない見込み", "",
            "収まらなければ、このワークフローを `core` で戻してください。"]
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        with open(sm, "a", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

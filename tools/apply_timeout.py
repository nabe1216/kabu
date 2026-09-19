#!/usr/bin/env python3
"""daily.yml の実行時間の上限を変える。

なぜ必要か
    対象をプライム全銘柄（約1,557社）に広げたため、
    初回は財務データも全銘柄ぶん取得することになり、
    90分では足りずタイムアウトする。

    2回目以降はキャッシュが効いて株価だけの取得になるので、
    50〜60分に収まる見込み。つまり上限を延ばすのは初回のためのもの。

使い方
    MINUTES に分数を入れて実行する。
    初回を通したら 90 に戻しておくと、
    万一の暴走で時間を使い切るのを防げる。

目印が見つからない場合は何も保存せずに止まる。
"""
import os
import re
import sys
from pathlib import Path

TARGET = Path(".github/workflows/daily.yml")


def main() -> int:
    minutes = int(os.environ.get("MINUTES", "180"))
    if not (5 <= minutes <= 350):
        sys.exit(f"分数は 5〜350 で指定してください（指定: {minutes}）")
    if not TARGET.exists():
        sys.exit(f"{TARGET} がありません。")

    s = before = TARGET.read_text(encoding="utf-8")

    m = re.search(r"^(\s*)timeout-minutes:\s*(\d+)", s, re.M)
    if m:
        cur = m.group(2)
        if str(cur) == str(minutes):
            print(f"すでに {minutes} 分です。何もしません。")
            return 0
        s = re.sub(r"^(\s*)timeout-minutes:\s*\d+",
                   lambda x: f"{x.group(1)}timeout-minutes: {minutes}",
                   s, count=1, flags=re.M)
        msg = f"実行時間の上限 … {cur}分 → {minutes}分"
    else:
        # timeout-minutes が無い場合は runs-on の次に足す
        m2 = re.search(r"^(\s*)runs-on:.*$", s, re.M)
        if not m2:
            print("runs-on の行が見つかりません。ファイルの中身を表示します：\n")
            for i, l in enumerate(s.split("\n")[:60], 1):
                print(f"  {i:>3}: {l}")
            sys.exit("中止しました。")
        indent = m2.group(1)
        s = s.replace(m2.group(0),
                      f"{m2.group(0)}\n{indent}timeout-minutes: {minutes}", 1)
        msg = f"実行時間の上限を {minutes}分 に設定（新規）"

    if s == before:
        print("変更はありませんでした。")
        return 0

    # YAML として壊れていないか確かめる
    try:
        import yaml
        yaml.safe_load(s)
    except ImportError:
        pass
    except Exception as e:
        sys.exit(f"YAML が壊れます。中止します: {e}")

    TARGET.write_text(s, encoding="utf-8")
    print(msg)

    out = ["## daily.yml の変更", "", f"- {msg}", ""]
    if minutes > 90:
        out += ["初回は財務データを全銘柄ぶん取得するため、90分では足りません。",
                "この上限で1回だけ完走させてください。", "",
                "2回目以降はキャッシュが効いて50〜60分に収まる見込みです。",
                "**完走を確認したら、90 に戻しておくことをお勧めします。**", ""]
    out += ["次に「Generate Dividend Signals」を実行してください。"]
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        with open(sm, "a", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""generate.py に「時間切れ前に自分で止まる」仕組みを入れる。

なぜ必要か
    対象を1,557社に広げたところ、90分の上限で強制終了するようになった。
    強制終了だと、そのあとのコミットのステップまで到達しないため、
    作りかけのキャッシュがリポジトリに残らない。
    次回もゼロからやり直しになり、いつまでも完走できない。

やること
    起動からの経過時間を測り、決めた分数を超えたら
    財務データの取得をやめてキャッシュから返す。
    処理は最後まで走りきるので、コミットのステップも実行され、
    作りかけのキャッシュがリポジトリに残る。

    これを2〜3回繰り返せば、全銘柄ぶんのキャッシュが揃う。
    揃ったあとは株価だけの取得になり、時間内に収まる。

目印が見つからない場合や文法エラーが出た場合は、何も保存せずに止まる。
"""
import ast
import os
import re
import sys
from pathlib import Path

TARGET = Path("scripts/generate.py")

BUDGET_BLOCK = '''# 財務データの取得に使ってよい時間（分）。
# ワークフローの上限（90分）より短くしておく。
# 超えたら取得をやめてキャッシュから返し、処理は最後まで走らせる。
# そうしないと強制終了になり、コミットまで到達せず
# 作りかけのキャッシュが残らない。
STMTS_TIME_BUDGET_MIN = {budget}
_START_TIME = time.monotonic()
_budget_hit = False


def _out_of_time():
    """使ってよい時間を超えたか。超えた瞬間に一度だけ知らせる。"""
    global _budget_hit
    if STMTS_TIME_BUDGET_MIN <= 0:
        return False
    over = (time.monotonic() - _START_TIME) / 60 >= STMTS_TIME_BUDGET_MIN
    if over and not _budget_hit:
        _budget_hit = True
        log.warning('財務データの取得を%d分で打ち切ります。'
                    'ここまでの分を保存し、残りは次回に回します。',
                    STMTS_TIME_BUDGET_MIN)
    return over


'''

GET_OLD = """            got = self.get('/v2/fins/summary', params)
            c['data'][code] = got"""
GET_NEW = """            if _out_of_time():
                return []          # 時間切れ。次回に回す
            got = self.get('/v2/fins/summary', params)
            c['data'][code] = got"""


def main() -> int:
    budget = int(os.environ.get("BUDGET", "70"))
    if not TARGET.exists():
        sys.exit(f"{TARGET} がありません。")
    s = before = TARGET.read_text(encoding="utf-8")
    log = []

    if "STMTS_TIME_BUDGET_MIN" in s:
        m = re.search(r"^STMTS_TIME_BUDGET_MIN\s*=\s*(\d+)", s, re.M)
        cur = m.group(1) if m else "?"
        if str(cur) == str(budget):
            print(f"すでに {budget} 分です。何もしません。")
            return 0
        s = re.sub(r"^STMTS_TIME_BUDGET_MIN\s*=\s*\d+",
                   f"STMTS_TIME_BUDGET_MIN = {budget}", s, count=1, flags=re.M)
        log.append(f"取得に使う時間 … {cur}分 → {budget}分")
    else:
        if "STMTS_CACHE_DAYS" not in s:
            sys.exit("先に patch-cache を実行してください。")

        # time を読み込む
        if not re.search(r"^import time$", s, re.M):
            m = re.search(r"^import sys$", s, re.M) or re.search(r"^import os$", s, re.M)
            if m:
                s = s.replace(m.group(0), m.group(0) + "\ntime_imported = True\nimport time", 1)
                s = s.replace("time_imported = True\n", "")
                log.append("time を読み込むように追加")
            else:
                s = "import time\n" + s
                log.append("time を先頭で読み込むように追加")

        # 時間の判定を足す（キャッシュの定義の直前）
        a1 = "# 財務データを何日もたせるか。"
        if s.count(a1) != 1:
            sys.exit(f"目印1が {s.count(a1)} 箇所です。中止します。")
        s = s.replace(a1, BUDGET_BLOCK.format(budget=budget) + a1, 1)
        log.append("時間切れの判定を追加")

        # 取得の直前で判定する
        if s.count(GET_OLD) != 1:
            sys.exit(f"目印2が {s.count(GET_OLD)} 箇所です。中止します。")
        s = s.replace(GET_OLD, GET_NEW, 1)
        log.append("時間を超えたら取得をやめるように変更")

    if s == before:
        print("変更はありませんでした。")
        return 0

    ast.parse(s)
    TARGET.write_text(s, encoding="utf-8")
    for x in log:
        print(x)

    out = ["## generate.py の変更", ""]
    out += [f"- {x}" for x in log]
    out += ["", f"財務データの取得を **{budget}分** で打ち切ります。", "",
            "残りの処理は最後まで走るので、コミットのステップも実行され、",
            "作りかけのキャッシュがリポジトリに残ります。", "",
            "**2〜3回実行すれば、全銘柄ぶんが揃います。**", "",
            "揃ったあとは株価だけの取得になり、50〜60分に収まる見込みです。", "",
            "### 次にやること", "",
            "1. `daily.yml` のコミット対象に `data/stmts_cache.json` を足す",
            "2. 「Generate Dividend Signals」を実行",
            "3. `data/stmts_cache.json` が増えているか確認",
            "4. 揃うまで2〜3回くり返す"]
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        with open(sm, "a", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""generate.py に、財務データのキャッシュを組み込む。

なぜ必要か
    対象をプライム全銘柄（約1,557社）に広げたところ、
    1時間31分でタイムアウトした。
    株価と財務で1銘柄あたり2回APIを呼んでいるため、
    財務を毎日取らなければ、ほぼ半分の時間で済む。

やること
    JQuantsClient.get_statements にキャッシュを被せる。
      ・data/stmts_cache.json に保存する
      ・指定した日数より新しければ、APIを呼ばずにそれを使う
      ・古くなっていたら取り直して保存する

注意
    決算発表の反映が、最大で指定日数ぶん遅れる。
    減配や業績急変の検知も、その分だけ遅れる。

目印が見つからない場合や文法エラーが出た場合は、何も保存せずに止まる。
"""
import ast
import os
import re
import sys
from pathlib import Path

TARGET = Path("scripts/generate.py")

CACHE_BLOCK = '''# 財務データを何日もたせるか。0 ならキャッシュを使わない。
# 株価と財務で1銘柄あたり2回APIを呼ぶため、財務を毎日取らなければ
# 実行時間がほぼ半分になる。決算の反映はその日数ぶん遅れる。
STMTS_CACHE_DAYS = {days}
_STMTS_CACHE_PATH = DATA_DIR / 'stmts_cache.json'
_stmts_cache = None


def _load_stmts_cache():
    """財務データのキャッシュを読む。壊れていれば空で始める。"""
    global _stmts_cache
    if _stmts_cache is not None:
        return _stmts_cache
    _stmts_cache = {{'saved_at': '', 'data': {{}}}}
    if STMTS_CACHE_DAYS > 0 and _STMTS_CACHE_PATH.exists():
        try:
            with _STMTS_CACHE_PATH.open(encoding='utf-8') as f:
                c = json.load(f)
            saved = parse_date(c.get('saved_at'))
            if saved is not None:
                age = (date.today() - saved).days
                if age < STMTS_CACHE_DAYS:
                    _stmts_cache = c
                    log.info('財務データのキャッシュを使います（%d日前・%d銘柄）',
                             age, len(c.get('data', {{}})))
                else:
                    log.info('財務データのキャッシュが%d日前なので取り直します', age)
        except Exception as e:
            log.warning('財務データのキャッシュを読めません: %s', e)
    return _stmts_cache


def _save_stmts_cache():
    """取得した財務データを保存する。"""
    if STMTS_CACHE_DAYS <= 0 or _stmts_cache is None:
        return
    if not _stmts_cache.get('data'):
        return
    try:
        _stmts_cache['saved_at'] = date.today().isoformat()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _STMTS_CACHE_PATH.open('w', encoding='utf-8') as f:
            json.dump(_stmts_cache, f, ensure_ascii=False)
        log.info('財務データのキャッシュを保存しました（%d銘柄）',
                 len(_stmts_cache['data']))
    except Exception as e:
        log.warning('財務データのキャッシュを保存できません: %s', e)


'''

GET_OLD = "        return self.get('/v2/fins/summary', params)"
GET_NEW = """        if STMTS_CACHE_DAYS > 0:
            c = _load_stmts_cache()
            hit = c['data'].get(code)
            if hit is not None:
                return hit
            got = self.get('/v2/fins/summary', params)
            c['data'][code] = got
            return got
        return self.get('/v2/fins/summary', params)"""


def main() -> int:
    days = int(os.environ.get("DAYS", "7"))
    if not TARGET.exists():
        sys.exit(f"{TARGET} がありません。")
    s = before = TARGET.read_text(encoding="utf-8")
    log = []

    # すでに入っていれば、日数だけ変える
    if "STMTS_CACHE_DAYS" in s:
        m = re.search(r"^STMTS_CACHE_DAYS\s*=\s*(\d+)", s, re.M)
        cur = m.group(1) if m else "?"
        if str(cur) == str(days):
            print(f"すでに {days} 日です。何もしません。")
            return 0
        s = re.sub(r"^STMTS_CACHE_DAYS\s*=\s*\d+",
                   f"STMTS_CACHE_DAYS = {days}", s, count=1, flags=re.M)
        log.append(f"財務データの保持日数 … {cur} → {days}")
    else:
        # 1. キャッシュの読み書きを足す
        a1 = "def safe_float(value: Any) -> float | None:"
        if s.count(a1) != 1:
            sys.exit(f"目印1（safe_float）が {s.count(a1)} 箇所です。中止します。")
        s = s.replace(a1, CACHE_BLOCK.format(days=days) + a1, 1)
        log.append("キャッシュの読み書きを追加")

        # 2. get_statements にキャッシュを被せる
        if s.count(GET_OLD) != 1:
            sys.exit(f"目印2（fins/summary）が {s.count(GET_OLD)} 箇所です。中止します。")
        s = s.replace(GET_OLD, GET_NEW, 1)
        log.append("get_statements にキャッシュを適用")

        # 3. 全銘柄を回し終えたところで保存する
        a3 = None
        for cand in ("    # 統計",
                     "    bucket_counts: dict[str, int] = defaultdict(int)"):
            if s.count(cand) >= 1:
                a3 = cand
                break
        if not a3:
            print("保存を差し込む場所が見つかりません。候補を表示します：\n")
            for i, l in enumerate(s.split("\n"), 1):
                if "bucket_counts" in l or "統計" in l:
                    print(f"  {i:>5}: {l.rstrip()}")
            sys.exit("中止しました。")
        s = s.replace(a3, "    _save_stmts_cache()\n\n" + a3, 1)
        log.append("実行の最後にキャッシュを保存")

    if s == before:
        print("変更はありませんでした。")
        return 0

    ast.parse(s)
    TARGET.write_text(s, encoding="utf-8")
    for x in log:
        print(x)

    out = ["## generate.py の変更", ""]
    out += [f"- {x}" for x in log]
    out.append("")
    if days > 0:
        out += [f"財務データを **{days}日** もたせます。", "",
                "初回は全銘柄ぶん取得するので、これまでどおりの時間がかかります。",
                f"2回目以降、{days}日のあいだは株価だけの取得になり、",
                "実行時間がほぼ半分になるはずです。", "",
                f"**注意** … 決算発表の反映が最大{days}日遅れます。",
                "減配や業績急変の検知も同様です。", ""]
    else:
        out += ["キャッシュを使いません（毎日取得します）。", ""]
    out += ["次に「Generate Dividend Signals」を実行してください。", "",
            "戻すときは、このワークフローを 0 で再実行してください。"]
    sm = os.environ.get("GITHUB_STEP_SUMMARY")
    if sm:
        with open(sm, "a", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

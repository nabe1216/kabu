#!/usr/bin/env python3
"""
有価証券報告書の【配当政策】から「減配しない」約束を読み取る（2026年11月13日に追加）。

- 対象：data/results.json の銘柄（毎日の判定の範囲・約970社）と、検証運用の持ち株
- 各社の最新の有価証券報告書（書類種別 120）を EDINET API v2 で探し、【配当政策】の文を取り出す
  1. XBRL を CSV にしたもの（type=5）の「DividendPolicy」を含む要素
  2. なければ本文（type=1 の inline XBRL）の「DividendPolicy」を含むテキストブロック
  3. それもなければ本文の見出し「配当政策」のあとの文
- 文の中に「累進配当」「減配しない」「維持または増配」などの言葉があるかで判定し、data/dividend_policy.json に出す
- 2回目からは、書類が変わっていない会社は読み直さない（毎月動かしても、取り直すのは新しく出た有価証券報告書だけ）

出典：EDINET（金融庁）。公共データ利用規約（第1.0版）に従い、結果のファイルに出典と、DIVIDEND HEIST が
判定のために抜き出し・加工したことを書く。API の利用規約に合わせ、1秒に1回より速くは呼ばない。

使い方：EDINET_API_KEY=… python scripts/edinet_policy.py [--days 460] [--max-docs 0] [--codes 7203,8058]
        python scripts/edinet_policy.py --selftest   （作り物のデータで読み取りと判定を確かめる。キー不要）
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'data' / 'results.json'
STATE = ROOT / 'data' / 'portfolio_state.json'
OUT = ROOT / 'data' / 'dividend_policy.json'
CACHE = ROOT / 'edinet_cache' / 'lists'          # 書類一覧の控え（Actions のキャッシュに置く。リポジトリには入れない）
API = 'https://api.edinet-fsa.go.jp/api/v2'
PARSER_VERSION = 1                                 # 読み取り・判定の決まりを変えたら上げる（全社を読み直す）
WAIT_SEC = 1.1                                     # 呼び出しの間隔
JST = timezone(timedelta(hours=9))
SOURCE_NOTE = ('出典：EDINET（金融庁）に提出された有価証券報告書の【配当政策】。'
               'DIVIDEND HEIST が判定のために文を抜き出し・加工した（公共データ利用規約 第1.0版）。'
               '判定は言葉による機械的なもので、会社の約束の全部を表すものではない')

# ---------------------------------------------------------------------------
# 判定（言葉の決まり）
# ---------------------------------------------------------------------------
_OR = r'(?:または|もしくは|又は|若しくは|ないし|あるいは|・|、)'
POSITIVE = [
    ('累進配当', r'累進配当'),
    ('累進的な配当', r'累進的(?:な|に)?(?:配当|増配)'),
    ('配当を累進的に', r'配当(?:金)?(?:を|の|は)?(?:段階的・?)?累進的'),
    ('累進的に行う', r'累進的に(?:行|実施|増加|増や)'),
    ('減配しない', r'減配(?:は|を)?(?:原則として|原則)?(?:行わ|実施せ|実施し|せ|し)(?:ず|ない)'),
    ('減額しない', r'減額(?:は|を)?(?:原則として|原則)?(?:行わ|実施せ|実施し|せ|し)(?:ず|ない)'),
    ('維持または増配', r'(?:維持|据え置き|据置き|据置)' + _OR + r'(?:増配|増額)'),
    ('増配または維持', r'(?:増配|増額)' + _OR + r'(?:維持|据え置き|据置き|据置)'),
    ('下げることなく', r'(?:下げる|引き下げる|減らす)ことなく'),
    ('前期を下回らない', r'(?:前期|前年度|前事業年度|前年)[^。]{0,20}下回らない'),
]
FLAGS = [
    ('期限', r'(?:中期経営計画|中計|計画期間|期間中|期間において|年度まで|月期まで|年度から|月期から|年度において)'),
    ('条件', r'(?:限り|場合|CET1|範囲|前提|条件)'),
    ('目指す・努める', r'(?:目指|めざ|努め|努力)'),
    ('原則', r'原則'),
    ('今期だけ', r'(?:当期|今期|当事業年度|20\d\d年\d{1,2}月期)の(?:1株当たり)?(?:年間)?配当'),
]


def norm(text: str) -> str:
    """全角・半角をそろえ、空白と改行を消す（判定は空白なしの文で行う）。"""
    t = unicodedata.normalize('NFKC', text or '')
    return re.sub(r'\s+', '', t)


def _sentence(t: str, a: int, b: int) -> str:
    s = t.rfind('。', 0, a) + 1
    e = t.find('。', b)
    return t[s:(e + 1 if e >= 0 else len(t))]


def classify(text: str) -> dict:
    """配当政策の文から「減配しない」約束を探す。見つけた言葉・前後の文・ただし書きを返す。"""
    t = norm(text)
    hits = []
    for label, pat in POSITIVE:
        for m in re.finditer(pat, t):
            win = t[max(0, m.start() - 25): m.end() + 15]
            # 「配当性向を累進的に引き上げる」は、1株の配当を下げない約束ではないので数えない
            if '累進' in m.group(0) and '配当性向' in win and re.search(r'引き?上げ', win):
                continue
            hits.append((m.start(), m.end(), label, m.group(0)))
    if not hits:
        return {'progressive': False, 'phrase': None, 'snippet': None, 'flags': [], 'text_len': len(t)}
    hits.sort()
    a, b, label, phrase = hits[0]
    sent = _sentence(t, a, b)
    flags = [name for name, pat in FLAGS if re.search(pat, sent)]
    if '今期だけ' in flags and re.search(r'(?:毎期|毎年|継続|方針)', sent):
        flags.remove('今期だけ')
    snip = sent if len(sent) <= 120 else t[max(0, a - 50): b + 50]
    return {'progressive': True, 'phrase': label, 'snippet': snip, 'flags': flags,
            'labels': sorted({h[2] for h in hits}), 'text_len': len(t)}


# ---------------------------------------------------------------------------
# 書類から【配当政策】の文を取り出す
# ---------------------------------------------------------------------------
def strip_html(s: str) -> str:
    s = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', s)
    s = re.sub(r'(?i)<br\s*/?>|</p>|</div>|</tr>|</li>', '\n', s)
    s = re.sub(r'<[^>]+>', ' ', s)
    return html.unescape(s)


def _decode(raw: bytes) -> str:
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16')
    for enc in ('utf-8-sig', 'cp932', 'utf-16'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def from_csv_zip(data: bytes) -> str | None:
    """XBRL を CSV にしたもの（type=5）から、要素名に DividendPolicy を含む行の値を取り出す。"""
    best = None
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if not name.lower().endswith('.csv'):
                continue
            txt = _decode(z.read(name))
            first = txt.split('\n', 1)[0]
            delim = '\t' if first.count('\t') >= first.count(',') else ','
            vi = None                               # 「値」の列（見出しの行があればそこから、なければ最後の列）
            for row in csv.reader(io.StringIO(txt), delimiter=delim):
                if not row:
                    continue
                if vi is None and '値' in [c.strip() for c in row]:
                    vi = [c.strip() for c in row].index('値')
                    continue
                if not any('dividendpolicy' in (c or '').lower() for c in row[:2]):
                    continue
                cells = [c for c in row if c is not None]
                val = cells[vi] if (vi is not None and vi < len(cells)) else next((c for c in reversed(cells) if c.strip()), '')
                val = strip_html(val)
                if best is None or len(val) > len(best):
                    best = val
    return best


def _ix_blocks(doc: str, key: str = 'dividendpolicy') -> list[str]:
    """inline XBRL の ix:nonNumeric のうち、name に key を含むものの中身（入れ子も含めて）を返す。"""
    out = []
    low = doc.lower()
    for m in re.finditer(r'<ix:nonnumeric\b[^>]*>', low):
        tag = low[m.start():m.end()]
        nm = re.search(r'name="([^"]+)"', tag)
        if not nm or key not in nm.group(1):
            continue
        depth, pos = 1, m.end()
        tok = re.compile(r'<ix:nonnumeric\b[^>]*>|</ix:nonnumeric\s*>')
        while depth and pos < len(low):
            t = tok.search(low, pos)
            if not t:
                pos = len(low)
                break
            depth += -1 if t.group(0).startswith('</') else 1
            pos = t.end()
        out.append(doc[m.end():pos])
    return out


def from_xbrl_zip(data: bytes) -> tuple[str | None, str]:
    """本文（type=1）の inline XBRL から取り出す。なければ見出し「配当政策」のあとの文。"""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if 'publicdoc' in n.lower() and n.lower().endswith(('.htm', '.html', '.xhtml'))]
        texts = {n: _decode(z.read(n)) for n in names}
    for n, doc in texts.items():
        blocks = _ix_blocks(doc)
        if blocks:
            return strip_html(max(blocks, key=len)), 'ixbrl'
    for n, doc in sorted(texts.items()):
        plain = norm(strip_html(doc))
        i = plain.find('配当政策】')
        if i < 0:
            i = plain.find('配当政策')
        if i >= 0:
            seg = plain[i:i + 2000]
            j = re.search(r'[4４]【|【コーポレート・ガバナンス', seg[10:])
            return (seg[:j.start() + 10] if j else seg), 'heading'
    return None, 'none'


# ---------------------------------------------------------------------------
# EDINET API
# ---------------------------------------------------------------------------
class KeyError_(RuntimeError):
    pass


_last_call = [0.0]


def http_get(path: str, params: dict, key: str, binary: bool = False):
    q = dict(params)
    q['Subscription-Key'] = key
    url = f'{API}/{path}?{urllib.parse.urlencode(q)}'
    for attempt in range(5):
        wait = WAIT_SEC - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'dividend-heist/1.0'}), timeout=90) as r:
                body = r.read()
                ctype = r.headers.get('Content-Type', '')
            if binary and 'json' in ctype.lower():
                info = json.loads(body.decode('utf-8', errors='replace'))
                st = str(info.get('StatusCode') or info.get('metadata', {}).get('status') or '')
                if st in ('401', '403'):
                    raise KeyError_(info)
                return None                       # 書類がない・取り下げなど
            return body if binary else json.loads(body.decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise KeyError_(f'HTTP {e.code}')
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(10 * (attempt + 1) ** 2)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'EDINET に5回つながらなかった：{path}')


def day_list(d: date, key: str, today: date) -> list[dict]:
    """その日に出た書類の一覧。7日より前の日は控えを使う（あとから変わらないので）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'{d.isoformat()}.json'
    if f.exists() and (today - d).days > 7:
        try:
            return json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            pass
    js = http_get('documents.json', {'date': d.isoformat(), 'type': 2}, key)
    meta = (js or {}).get('metadata', {})
    if str(meta.get('status', '200')) in ('401', '403'):
        raise KeyError_(meta)
    rows = [{k: r.get(k) for k in ('docID', 'secCode', 'filerName', 'docTypeCode', 'ordinanceCode', 'formCode',
                                   'submitDateTime', 'periodEnd', 'csvFlag', 'xbrlFlag', 'withdrawalStatus')}
            for r in (js or {}).get('results') or [] if r.get('docTypeCode') in ('120',) and r.get('secCode')]
    f.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    return rows


def target_codes() -> dict[str, str]:
    codes = {}
    try:
        for s in json.loads(RESULTS.read_text(encoding='utf-8')).get('stocks', []):
            codes[str(s['code'])[:4].upper()] = s.get('name', '')
    except Exception as e:
        print(f'results.json を読めない（{e}）')
    try:
        for h in json.loads(STATE.read_text(encoding='utf-8')).get('holdings', []):
            codes.setdefault(str(h['code'])[:4].upper(), h.get('name', ''))
    except Exception:
        pass
    return codes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=460, help='何日さかのぼって有価証券報告書を探すか')
    ap.add_argument('--max-docs', type=int, default=0, help='読む書類の上限（0＝上限なし。試しに動かすとき用）')
    ap.add_argument('--codes', default='', help='この銘柄だけ（カンマ区切り）')
    ap.add_argument('--max-minutes', type=float, default=100.0,
                    help='書類を読むのをこの分数で打ち切って保存する（次の実行で続きから読む）')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    key = os.environ.get('EDINET_API_KEY', '').strip()
    if not key:
        print('EDINET の API キー（GitHub の Secrets の EDINET_API_KEY）が登録されていません。何もせずに終わります。')
        return 0

    today = datetime.now(JST).date()
    codes = target_codes()
    if a.codes:
        want = {c.strip().upper()[:4] for c in a.codes.split(',') if c.strip()}
        codes = {c: n for c, n in codes.items() if c in want} or {c: '' for c in want}
    print(f'対象 {len(codes)}社。{a.days}日さかのぼって、最新の有価証券報告書を探します')

    latest: dict[str, dict] = {}
    try:
        for i in range(a.days + 1):
            d = today - timedelta(days=i)
            for r in day_list(d, key, today):
                c = str(r['secCode'])[:4].upper()
                if c not in codes or str(r.get('withdrawalStatus') or '0') != '0':
                    continue
                if c not in latest or (r.get('submitDateTime') or '') > (latest[c].get('submitDateTime') or ''):
                    latest[c] = r
            if i and i % 60 == 0:
                print(f'  {i}日分を見た（見つかった有価証券報告書 {len(latest)}社）')
    except KeyError_ as e:
        print(f'EDINET が API キーを受け付けませんでした（{e}）。GitHub の Secrets の EDINET_API_KEY を確かめてください。')
        return 2

    prev = {}
    if OUT.exists():
        try:
            pj = json.loads(OUT.read_text(encoding='utf-8'))
            if pj.get('parser_version') == PARSER_VERSION:
                prev = pj.get('companies', {})
        except Exception:
            pass

    companies, n_read, methods = {}, 0, {}
    t0, stopped = time.time(), False

    def save(final: bool) -> dict:
        prog = {c: v for c, v in companies.items() if v.get('progressive')}
        out = {
            'source': SOURCE_NOTE,
            'checked_at': today.isoformat(),
            'parser_version': PARSER_VERSION,
            'days': a.days,
            'complete': final and not stopped,
            'counts': {'targets': len(codes), 'found': len(latest), 'read_now': n_read,
                       'progressive': len(prog), 'methods': methods},
            'companies': {**({k: v for k, v in prev.items() if k not in companies} if not final or stopped else {}),
                          **companies},
        }
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
        return prog

    for c in sorted(codes):
        r = latest.get(c)
        if not r:
            continue
        p = prev.get(c)
        if p and p.get('doc_id') == r['docID']:
            companies[c] = p
            methods[p.get('method', '?')] = methods.get(p.get('method', '?'), 0) + 1
            continue
        if (a.max_docs and n_read >= a.max_docs) or (time.time() - t0) / 60 > a.max_minutes:
            stopped = True
            continue
        text, method = None, 'none'
        try:
            if str(r.get('csvFlag')) == '1':
                z = http_get(f'documents/{r["docID"]}', {'type': 5}, key, binary=True)
                if z:
                    text = from_csv_zip(z)
                    method = 'csv' if text else 'none'
            if not text or len(norm(text)) < 30:      # CSV に文がない・短すぎる（切れている）ときは本文から
                z = http_get(f'documents/{r["docID"]}', {'type': 1}, key, binary=True)
                if z:
                    t2, m2 = from_xbrl_zip(z)
                    if t2 and len(norm(t2)) >= len(norm(text or '')):
                        text, method = t2, m2
        except KeyError_ as e:
            print(f'EDINET が API キーを受け付けませんでした（{e}）。')
            return 2
        except (zipfile.BadZipFile, RuntimeError) as e:
            print(f'  {c} {r.get("filerName")}：読めなかった（{e}）')
        n_read += 1
        res = classify(text or '')
        companies[c] = {
            'name': codes.get(c) or r.get('filerName'), 'doc_id': r['docID'],
            'submitted': (r.get('submitDateTime') or '')[:10], 'period_end': r.get('periodEnd'),
            'method': method, **res,
        }
        methods[method] = methods.get(method, 0) + 1
        if n_read % 50 == 0:
            print(f'  {n_read}社を読んだ（{(time.time() - t0) / 60:.0f}分）')
            save(False)

    prog = save(True)
    if stopped:
        print(f'\n時間（{a.max_minutes:.0f}分）か上限で読むのを打ち切った。もう一度動かすと続きから読む')
    print(f'\n対象 {len(codes)}社 ／ 有価証券報告書が見つかった {len(latest)}社 ／ 今回読んだ {n_read}社')
    print(f'文の取り出し方：{methods}（none＝配当政策の文が見つからなかった）')
    print(f'「減配しない」と書いている会社：{len(prog)}社')
    for c, v in sorted(prog.items())[:40]:
        print(f'  {c} {v.get("name", "")[:12]}：{v.get("phrase")}' + (f'（{"・".join(v["flags"])}）' if v.get('flags') else ''))
    return 0


# ---------------------------------------------------------------------------
# 確かめ（作り物のデータ。キー不要）
# ---------------------------------------------------------------------------
def selftest() -> int:
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(('一致 ' if cond else 'NG   ') + name + (f'　{detail}' if detail else ''))

    cases = [
        ('武田', '毎年の1株当たり年間配当金を増額または維持する累進配当の方針を採用しています。', True, []),
        ('しずおか', '2027年度までに「配当性向50％以上」へ累進的に引上げます。', False, []),
        ('トヨタ', '配当金につきましては、安定的・継続的に増配を行うよう努めてまいります。', False, []),
        ('関西電力', '連結配当性向25~35％を目安に、配当の維持または増配に努めます。', True, ['目指す・努める']),
        ('ソニーFG', '1株当たり年間配当額の減額は原則行わず、安定的な配当の成長を目指す方針です。', True, ['原則']),
        ('キヤノン', '2026年12月期の1株当たり配当金については、配当金額を下げることなく、配当性向40％を目途に', True, ['今期だけ']),
        ('東京海上', '配当を株主還元の基本と位置づけ、利益成長に応じて持続的に高める方針としており', False, []),
        ('JFE', '配当性向30％程度とし、80円/株を下限とします。', False, []),
        ('三井物産', '中期経営計画2029の期間（2027年3月期～2029年3月期）において、配当維持または増配を行う累進配当を継続します。', True, ['期限']),
        ('三菱UFJ（有報）', '1株当たり配当金の安定的・持続的な増加を基本方針とし', False, []),
        ('双日', '当期純利益による株主資本の積み上げが、株主還元による株主資本の減少幅を上回る限りにおいて、累進的に増配となる配当方針', True, ['条件']),
        ('全角空白', '当社は、累 進 配 当を基本方針としております。', True, []),
        ('三井住友FG', '配当は持続的な利益成長を勘案し累進的に行うものとし、本中期経営計画では毎期の増配を原則とします。', True, ['期限']),
        ('配当性向を累進的に', '配当性向を段階的に累進的に引き上げていく方針です。', False, []),
        ('減配の実績の説明', '当期は業績悪化により減配となりました。', False, []),
    ]
    for nm, txt, exp, fl in cases:
        r = classify(txt)
        check(f'判定：{nm}', r['progressive'] == exp and all(f in r['flags'] for f in fl),
              f'{r["phrase"]}・{r["flags"]}')

    # CSV（UTF-16・タブ区切り）
    rows = [['要素ID', '項目名', 'コンテキストID', '相対年度', '連結・個別', '期間・時点', 'ユニットID', '単位', '値'],
            ['jpcrp_cor:DividendPolicyTextBlock', '配当政策 [テキストブロック]', 'FilingDateInstant', '提出日時点', 'その他', '時点', '', '',
             '<p>当社は、株主還元を経営の重要課題と位置づけ、<b>累進配当</b>を基本方針としております。</p>'],
            ['jpcrp_cor:NetSales', '売上高', 'CurrentYearDuration', '当期', '連結', '期間', 'JPY', '円', '1000']]
    buf = io.StringIO()
    csv.writer(buf, delimiter='\t').writerows(rows)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, 'w') as z:
        z.writestr('XBRL_TO_CSV/jpcrp030000-asr-001_E00000-000_2026-03-31_01_2026-06-20.csv', buf.getvalue().encode('utf-16'))
    t = from_csv_zip(zb.getvalue())
    check('CSV から取り出す', t and '累進配当' in t and '<' not in t, (t or '')[:40])
    rows2 = [['要素ID', '項目名', '値'], ['jpcrp_cor:DividendPolicyTextBlock', '配当政策', '累進配当を継続。']]
    buf = io.StringIO()
    csv.writer(buf, delimiter='\t').writerows(rows2)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, 'w') as z:
        z.writestr('XBRL_TO_CSV/b.csv', buf.getvalue().encode('utf-16'))
    t = from_csv_zip(zb.getvalue())
    check('CSV から取り出す（要素名より短い文）', t == '累進配当を継続。', t or '')
    rows3 = [['jpcrp_cor:DividendPolicyTextBlock', '配当政策', 'x', '', '', '', '', '', '維持または増配を基本とする。']]
    buf = io.StringIO()
    csv.writer(buf, delimiter='\t').writerows(rows3)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, 'w') as z:
        z.writestr('XBRL_TO_CSV/c.csv', buf.getvalue().encode('utf-16'))
    t = from_csv_zip(zb.getvalue())
    check('CSV から取り出す（見出しの行なし）', t == '維持または増配を基本とする。', t or '')

    # inline XBRL（入れ子あり）
    doc = ('<html><body><h2>3【配当政策】</h2><ix:nonNumeric name="jpcrp_cor:DividendPolicyTextBlock" contextRef="x">'
           '<p>当社は、<ix:nonNumeric name="jpcrp_cor:Other" contextRef="x">安定配当</ix:nonNumeric>を基本とし、'
           '減配は原則として行わない方針です。</p></ix:nonNumeric><p>4【株価の推移】</p></body></html>')
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, 'w') as z:
        z.writestr('XBRL/PublicDoc/0104010_honbun_jpcrp030000-asr-001_ixbrl.htm', doc.encode('utf-8'))
    t, m = from_xbrl_zip(zb.getvalue())
    check('inline XBRL から取り出す（入れ子）', m == 'ixbrl' and t and '減配は原則として行わない' in norm(t) and '株価の推移' not in t, f'{m}・{norm(t or "")[:40]}')
    r = classify(t or '')
    check('取り出した文の判定', r['progressive'] and '原則' in r['flags'], f'{r["phrase"]}・{r["flags"]}')

    # 見出しだけ（テキストブロックの印がない）
    doc2 = '<html><body><p>３【配当政策】</p><p>当社は、配当の維持または増配を基本方針としております。</p><p>４【コーポレート・ガバナンスの状況等】</p></body></html>'
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, 'w') as z:
        z.writestr('XBRL/PublicDoc/0104010_honbun_x_ixbrl.htm', doc2.encode('utf-8'))
    t, m = from_xbrl_zip(zb.getvalue())
    check('見出しから取り出す', m == 'heading' and t and '維持または増配' in t and 'ガバナンス' not in t, f'{m}・{(t or "")[:40]}')
    print('\n全部一致' if ok else '\n一致しない項目がある')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

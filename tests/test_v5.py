"""engine v5.1 のテスト（接続・復旧・配当・切り替え）と、固定記録 v1・D1（配当予想の修正）・財務キャッシュ・点検。架空の銘柄だけを使う。
python3 tests/test_v5.py で実行。"""
import importlib
import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))


class Sim:
    """generate.py の運用の部分（judge_lag → update_portfolio → mark_processed）を、そのまま再現する。"""

    def __init__(self, codes, state_seed=None):
        self.dir = Path(tempfile.mkdtemp())
        import portfolio_engine as pe
        import judge_lag as jl
        importlib.reload(pe)
        importlib.reload(jl)
        self.pe, self.jl = pe, jl
        pe.STATE_PATH = self.dir / 'portfolio_state.json'
        pe.HISTORY_PATH = self.dir / 'portfolio_history.json'
        pe.TOPUP = pe.REBALANCE = pe.HARVEST = False      # 月ごとの作業は、このテストでは止める（株数を変えないため）
        self.inputs = self.dir / 'paper_inputs'
        self.codes = codes
        if state_seed:
            st = pe.init_state()
            st.update(state_seed)
            pe.save_json(pe.STATE_PATH, st)

    def stocks(self, day, judge):
        out = []
        for c in self.codes:
            j = judge.get(c, {})
            out.append({
                'code': c, 'name': f'架空{c}', 'tier': 'B', 'current_yield': 4.0, 'forecast_dps': 100.0,
                'fiscal_month': 12, 'interim_month': 6, 'price': 1000.0,
                'price_history': [{'d': day, 'o': 1000.0, 'c': 1000.0}],
                'signal': 'BUY' if j.get('buy') else 'NEUTRAL', 'screening_pass': bool(j.get('buy')),
                'emergency_exit': bool(j.get('exit')), 'emergency_hold': False,
            })
        return out

    def run(self, day, judge, fail_engine=False, crash_before_mark=False, at=None):
        """day の夜の実行。judge はその日の判定（翌実行日に約定する）。戻り値は、この実行で増えた売買。
        at は実行した時刻（日本時間の ISO 文字列）。省くと day の 20:00"""
        pe, jl = self.pe, self.jl
        from datetime import datetime as _dt
        jl._now = (lambda a=at or f'{day}T20:00:00+09:00': _dt.fromisoformat(a))
        hist = pe.load_json(pe.HISTORY_PATH, pe.init_history())
        n0 = len(hist['trades'])
        stocks = self.stocks(day, judge)
        paper, info = jl.lagged_stocks(stocks, self.inputs)
        called = False
        if paper is not None:
            pe.TODAY_OVERRIDE = date.fromisoformat(info['price_date'])
            orig = pe.process_dividends
            if fail_engine:
                def boom(*a, **k):
                    raise RuntimeError('わざと失敗')
                pe.process_dividends = boom
            try:
                pe.update_portfolio(paper, None)
                called = True
                if not crash_before_mark:
                    jl.mark_processed(self.inputs, info)
            except RuntimeError:
                pass
            finally:
                pe.process_dividends = orig
        hist = pe.load_json(pe.HISTORY_PATH, pe.init_history())
        new = hist['trades'][n0:]
        return {'info': info, 'called': called, 'buy': [t['code'] for t in new if t['action'] == 'BUY'],
                'sell': [t['code'] for t in new if t['action'] == 'SELL'], 'n': len(new)}

    def state(self):
        return self.pe.load_json(self.pe.STATE_PATH, self.pe.init_state())

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def test_connection_and_recovery():
    s = Sim(['1001', '1002', '1003'])
    r1 = s.run('2026-12-01', {'1001': {'buy': True}})
    check('接続1：初日は前の判定がないので売買しない', r1['n'] == 0 and r1['info']['decided_from'] is None, str(r1['info']))
    r2 = s.run('2026-12-02', {'1002': {'buy': True}})
    check('接続2：翌営業日に、前日の判定（1001を買う）だけを約定', r2['buy'] == ['1001'] and r2['info']['decided_from'] == '2026-12-01',
          f"buy={r2['buy']} from={r2['info']['decided_from']}")
    r2b = s.run('2026-12-02', {'1002': {'buy': True}, '1001': {'exit': True}})
    check('接続3：同じ日の再実行は運用の更新を呼ばない（売買0件）', (not r2b['called']) and r2b['n'] == 0 and 'skip' in r2b['info'], str(r2b['info']))
    st = s.state()
    check('接続3b：再実行のあとも持ち株は1001の1社', [h['code'] for h in st['holdings']] == ['1001'])
    # 再実行で判定が変わった（例：データを直して動かし直した）場合は、新しい判定が保存され、翌営業日に使われる
    # 復旧1：判定の保存後に、運用の更新が失敗 → 再実行で一度だけ約定
    r3 = s.run('2026-12-03', {'1003': {'buy': True}}, fail_engine=True)
    check('復旧1a：失敗した回は売買が残らない', r3['n'] == 0 and not r3['called'])
    check('復旧1b：処理済みの日付は進まない', s.jl.last_processed(s.inputs) == '2026-12-02', s.jl.last_processed(s.inputs))
    r3b = s.run('2026-12-03', {'1003': {'buy': True}})
    check('復旧1c：再実行で前日（12/2・再実行で直した方）の判定を一度だけ約定（1002を買い、1001を売る）',
          r3b['buy'] == ['1002'] and r3b['sell'] == ['1001'] and r3b['info']['decided_from'] == '2026-12-02',
          f"buy={r3b['buy']} sell={r3b['sell']} from={r3b['info']['decided_from']}")
    r3c = s.run('2026-12-03', {'1003': {'buy': True}})
    check('復旧1d：もう一度動かしても二重にならない', r3c['n'] == 0 and not r3c['called'])
    # 復旧2：運用の更新は成功、処理済みを書く前に止まった → 再実行で二重にしない
    r4 = s.run('2026-12-04', {}, crash_before_mark=True)
    check('復旧2a：12/3の判定の1003を買う', r4['buy'] == ['1003'])
    r4b = s.run('2026-12-04', {})
    check('復旧2b：再実行は engine の確認で売買0件、処理済みが進む', r4b['n'] == 0 and s.jl.last_processed(s.inputs) == '2026-12-04',
          f"n={r4b['n']} processed={s.jl.last_processed(s.inputs)}")
    st = s.state()
    check('復旧2c：持ち株は1002・1003の2社', sorted(h['code'] for h in st['holdings']) == ['1002', '1003'],
          str(sorted(h['code'] for h in st['holdings'])))
    hist = s.pe.load_json(s.pe.HISTORY_PATH, s.pe.init_history())
    check('全体：買い3件・売り1件だけ（二重なし）', sum(1 for t in hist['trades'] if t['action'] == 'BUY') == 3
          and sum(1 for t in hist['trades'] if t['action'] == 'SELL') == 1)
    s.close()


def test_dividends():
    # 2026年12月：権利付き最終日 12/28（月）、権利落ち日 12/29（火）
    s = Sim(['A001', 'B001', 'C001', 'D001'])
    check('配当0：2026年12月の権利付き最終日は12月28日', s.pe._cum_date(2026, 12) == date(2026, 12, 28))
    s.run('2026-12-21', {'B001': {'buy': True}, 'D001': {'buy': True}})
    s.run('2026-12-22', {})                                   # B・D を買う
    s.run('2026-12-23', {})
    s.run('2026-12-24', {})
    s.run('2026-12-25', {'A001': {'buy': True}, 'B001': {'exit': True}})
    r28 = s.run('2026-12-28', {'C001': {'buy': True}, 'D001': {'exit': True}})   # A を買い、B を売る
    check('配当1：12/28 に A を買い B を売る', r28['buy'] == ['A001'] and r28['sell'] == ['B001'], f"{r28['buy']} {r28['sell']}")
    st = s.state()
    check('配当1b：権利付き最終日の時点では、まだ権利はつけない', not st.get('div_entitled'))
    r29 = s.run('2026-12-29', {})                              # 権利をつけてから、C を買い D を売る
    check('配当1c：12/29 に C を買い D を売る', r29['buy'] == ['C001'] and r29['sell'] == ['D001'], f"{r29['buy']} {r29['sell']}")
    ents = {e['code']: e for e in s.state().get('div_entitled', [])}
    check('配当2：権利付き最終日に買った A に権利がつく', 'A001' in ents and ents['A001']['shares'] == 400, str(ents.get('A001')))
    check('配当3：権利付き最終日に売った B にはつかない', 'B001' not in ents)
    check('配当4：権利落ち日に買った C にはつかない', 'C001' not in ents)
    check('配当5：権利落ち日に売った D には、前日の株数でつく', 'D001' in ents and ents['D001']['shares'] == 400, str(ents.get('D001')))
    check('配当6：入金日は3か月後の月末（2027-03-31）以後', all(e['pay_from'] == '2027-03-31' for e in ents.values()))
    s.run('2026-12-30', {})
    cash_before = s.state()['cash']
    s.run('2027-03-30', {})
    check('配当7：3/30 はまだ入金しない', s.state()['cash'] == cash_before)
    s.run('2027-03-31', {})
    st = s.state()
    paid = [e for e in st['div_entitled'] if e.get('paid')]
    gross = 2 * round(50 * 400)
    check('配当8：3/31 に A・D の2件を入金（税引き前 40,000円）', len(paid) == 2 and st.get('dividend_gross') == gross,
          f"paid={len(paid)} gross={st.get('dividend_gross')}")
    s.close()


def test_resume_across_month():
    s = Sim(['E001'])
    s.run('2026-12-24', {'E001': {'buy': True}})
    s.run('2026-12-25', {})                                   # E を買う
    s.run('2026-12-28', {})                                   # 権利付き最終日に実行して、そのあと止まる
    r = s.run('2027-01-05', {})                               # 年が明けてから再開
    ents = {e['code']: e for e in s.state().get('div_entitled', [])}
    check('配当9：権利付き最終日のあと止まり、翌月に再開しても、12月の権利をつける',
          'E001' in ents and ents['E001']['record'] == '2026-12' and ents['E001']['shares'] == 400, str(ents))
    r2 = s.run('2027-01-06', {})
    check('配当10：二重につけない', len(s.state().get('div_entitled', [])) == 1)
    s.close()


def test_switch_from_legacy():
    # これまでの last_updated は実行した日（夜中に終わると翌日の日付）。株価の日付と同じでも、初日の売買を飛ばさない
    s = Sim(['F001'], state_seed={'last_updated': '2026-12-01'})
    s.run('2026-11-30', {'F001': {'buy': True}})              # 切り替え前日の判定（この日は売買しない）
    r = s.run('2026-12-01', {})
    st = s.state()
    check('切替1：これまでの last_updated が株価の日付と同じでも、初日に約定する', r['buy'] == ['F001'],
          f"buy={r['buy']} legacy={st.get('legacy_last_updated')}")
    check('切替2：日付の基準が株価の日付に切り替わる', st.get('date_basis') == 'price_date' and st.get('last_updated') == '2026-12-01')
    s.close()


def test_calendar():
    import market_calendar as mc
    exp = {(2026, 3): '2026-03-27', (2026, 9): '2026-09-28', (2026, 12): '2026-12-28', (2027, 4): '2027-04-27',
           (2027, 12): '2027-12-28', (2028, 12): '2028-12-27'}
    got = {k: mc.cum_date(*k).isoformat() for k in exp}
    check('暦：権利付き最終日（祝日・年末を数える）', got == exp, str({k: v for k, v in got.items() if exp[k] != v}))
    check('暦：12/31・1/2・1/3・祝日は休業日', not any(mc.is_business_day(date.fromisoformat(d)) for d in
                                               ['2026-12-31', '2027-01-01', '2027-01-04'][:2] + ['2026-11-23', '2027-01-11']))



def _fake(code, day, buy=False, exit_=False, fiscal=12, interim=6):
    return {'code': code, 'name': f'架空{code}', 'tier': 'B', 'current_yield': 4.0, 'forecast_dps': 100.0,
            'fiscal_month': fiscal, 'interim_month': interim, 'price': 1000.0, 'pbr': 0.8, 'safety_y1': 1.0,
            'price_history': [{'d': day, 'o': 1000.0, 'c': 1000.0}],
            'signal': 'BUY' if buy else 'NEUTRAL', 'signal_pre_rank': 'BUY' if buy else 'NEUTRAL',
            'screening_pass': bool(buy), 'emergency_exit': bool(exit_), 'emergency_hold': False}


def test_frozen_dividends():
    """固定記録 v1（frozen_v1_run → frozen_v1_engine）の配当を、本番 v5.1 と同じ場面で確かめる（形 YIELD を見る）。"""
    import frozen_v1_engine as fe
    import frozen_v1_run as fr
    importlib.reload(fe)
    importlib.reload(fr)
    d = Path(tempfile.mkdtemp())
    fr.OUT = d / 'v1'
    fr.check_frozen = lambda: (True, '')            # 指紋は別のテスト（test_frozen_fingerprint）で見る
    fe.TOPUP = fe.REBALANCE = fe.HARVEST = False
    codes = ['A001', 'B001', 'C001', 'D001']

    def run(day, judge, at=None):
        from datetime import datetime as _dt
        fr._now = (lambda a=at or f'{day}T20:00:00+09:00': _dt.fromisoformat(a))
        st = [_fake(c, day, buy=judge.get(c) == 'buy', exit_=judge.get(c) == 'exit') for c in codes]
        return fr.update(st, None, None)

    check('固定0：2026年12月の権利付き最終日は12月28日（これまでの近似は29日）', fe._cum_date(2026, 12) == date(2026, 12, 28))
    run('2026-12-21', {'B001': 'buy', 'D001': 'buy'})
    run('2026-12-22', {})
    run('2026-12-23', {})
    run('2026-12-24', {})
    run('2026-12-25', {'A001': 'buy', 'B001': 'exit'})
    r28 = run('2026-12-28', {'C001': 'buy', 'D001': 'exit'})
    tr = r28['variants']['YIELD']['trades']
    check('固定1：12/28 に A を買い B を売る', sorted((t['action'], t['code']) for t in tr) == [('BUY', 'A001'), ('SELL', 'B001')], str(tr))
    st = json.loads((fr.OUT / 'variant_yield_state.json').read_text(encoding='utf-8'))
    check('固定1b：権利付き最終日の時点では、まだ権利はつけない', not st.get('div_entitled'))
    run('2026-12-29', {})
    st = json.loads((fr.OUT / 'variant_yield_state.json').read_text(encoding='utf-8'))
    ents = {e['code']: e for e in st.get('div_entitled', [])}
    check('固定2：権利付き最終日に買った A に権利がつく', 'A001' in ents, str(list(ents)))
    check('固定3：権利付き最終日に売った B にはつかない', 'B001' not in ents)
    check('固定4：権利落ち日に買った C にはつかない', 'C001' not in ents)
    check('固定5：権利落ち日に売った D には、前日の株数でつく', 'D001' in ents and ents['D001']['shares'] == ents['A001']['shares'],
          str(ents.get('D001')))
    check('固定6：入金日は3か月後の月末（2027-03-31）以後', all(e['pay_from'] == '2027-03-31' for e in ents.values()))
    run('2027-03-30', {})
    st = json.loads((fr.OUT / 'variant_yield_state.json').read_text(encoding='utf-8'))
    check('固定7：3/30 はまだ入金しない', not any(e.get('paid') for e in st['div_entitled']))
    run('2027-03-31', {})
    st = json.loads((fr.OUT / 'variant_yield_state.json').read_text(encoding='utf-8'))
    check('固定8：3/31 に A・D の2件を入金', sum(1 for e in st['div_entitled'] if e.get('paid')) == 2)
    shutil.rmtree(d, ignore_errors=True)


def test_frozen_fingerprint():
    import frozen_v1_run as fr
    importlib.reload(fr)
    ok, why = fr.check_frozen()
    check('固定9：固定し直した指紋が、いまのファイルと一致する（12/1 に止まらない）', ok, why)


def _st(doc, per, disc, fy_end, f=None, nx=None, nxt_fy=None, t='15:00'):
    return {'DiscNo': '','DocType': doc, 'CurPerType': per, 'DiscDate': disc, 'DiscTime': t, 'CurFYEn': fy_end,
            'CurPerEn': fy_end if per == 'FY' else disc, 'NxtFYEn': nxt_fy or '',
            'FDivAnn': '' if f is None else str(f), 'NxFDivAnn': '' if nx is None else str(nx), 'DivAnn': ''}


def test_d1_revision():
    import generate as g
    q1 = _st('1QFinancialStatements_Consolidated_JP', '1Q', '2026-08-03', '2027-03-31', f=98)
    q1['CurPerEn'] = '2026-06-30'
    fy = _st('FYFinancialStatements_Consolidated_JP', 'FY', '2026-05-15', '2026-03-31', nx=90, nxt_fy='2027-03-31')
    rev = _st('DividendForecastRevision', 'FY', '2026-10-29', '2027-03-31', f=112)
    v, d = g.get_forecast_dps_with_date([fy, q1, rev])
    check('D1-1：短信より後に出た同じ期の配当予想の修正を使う', v == 112.0 and d == date(2026, 10, 29), f'{v} {d}')
    check('D1-1b：分割を直す版（v3）も同じ', g.forecast_dps_v3([fy, q1, rev], []) == 112.0)
    rev_old = _st('DividendForecastRevision', 'FY', '2026-07-20', '2027-03-31', f=95)
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev_old])
    check('D1-2：修正が短信より前なら、短信の予想のまま', v == 98.0, str(v))
    rev_other = _st('DividendForecastRevision', 'FY', '2026-10-29', '2026-03-31', f=80)
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev_other])
    check('D1-3：違う期（前の期）の修正は使わない', v == 98.0, str(v))
    rev_2q = _st('EarnForecastRevision', '2Q', '2026-10-29', '2027-03-31', f=120)
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev_2q])
    check('D1-4：2Q 型の業績修正の行も、同じ期の年間配当の予想として使う（型で分けない）', v == 120.0, str(v))
    rev_2q_old = _st('EarnForecastRevision', '2Q', '2026-07-28', '2027-03-31', f=60)
    rev_2q_old['CurPerEn'] = '2026-09-30'
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev_2q_old])
    check('D1-4b：期末が新しい修正の行を1回目の探索で拾わない（後に出た1Qの短信の98円を使う）', v == 98.0, str(v))
    rev_zero = _st('DividendForecastRevision', 'FY', '2026-10-29', '2027-03-31', f=0)
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev_zero])
    check('D1-9：修正の0円（無配の予想）は有効な値として使う', v == 0.0, str(v))
    check('D1-9b：分割を直す版（v3）も0円', g.forecast_dps_v3([fy, q1, rev_zero], []) == 0.0, str(g.forecast_dps_v3([fy, q1, rev_zero], [])))
    rev_st = _st('DividendForecastRevision', 'FY', '2026-10-29', '2027-03-31', f=130)
    rev_st['CurFYSt'] = '2026-10-01'
    q1['CurFYSt'] = '2026-04-01'
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev_st])
    check('D1-10：期の終わりが同じでも始まりが違う（決算期の変更など）修正は使わない', v == 98.0, str(v))
    del q1['CurFYSt']
    a_ = [fy, q1, rev, rev_old, rev_other]
    v1, _ = g.get_forecast_dps_with_date(a_)
    v2, _ = g.get_forecast_dps_with_date(list(reversed(a_)))
    check('D1-11：入力の並びを逆にしても同じ値', v1 == v2 == 112.0, f'{v1} {v2}')
    t1 = _st('DividendForecastRevision', 'FY', '2026-10-29', '2027-03-31', f=112)
    t2 = _st('EarnForecastRevision', 'FY', '2026-10-29', '2027-03-31', f=110)
    v1, _ = g.get_forecast_dps_with_date([fy, q1, t1, t2])
    v2, _ = g.get_forecast_dps_with_date([fy, q1, t2, t1])
    check('D1-12：まったく同じ時刻に値が2つあれば小さい方（並びに左右されない）', v1 == v2 == 110.0, f'{v1} {v2}')
    check('D1-12b：同じ時刻の違う値は「競合」として記録する', any(c['values'] == [110.0, 112.0] for c in g._FORECAST_CONFLICTS),
          str(g._FORECAST_CONFLICTS))
    q1z = _st('1QFinancialStatements_Consolidated_JP', '1Q', '2026-08-05', '2027-03-31', f=0)
    q1z['CurPerEn'] = '2026-06-30'
    fy54 = _st('FYFinancialStatements_Consolidated_JP', 'FY', '2026-05-08', '2026-03-31', nx=54, nxt_fy='2027-03-31')
    v, _ = g.get_forecast_dps_with_date([fy54, q1z])
    check('D1-13：最新の短信の年間予想が0円なら0円（前の54円にさかのぼらない。カカクコム型）', v == 0.0, str(v))
    check('D1-13b：分割を直す版（v3）も0円', g.forecast_dps_v3([fy54, q1z], []) == 0.0, str(g.forecast_dps_v3([fy54, q1z], [])))
    q1b = _st('1QFinancialStatements_Consolidated_JP', '1Q', '2026-08-05', '2027-03-31', f=None)
    q1b['CurPerEn'] = '2026-06-30'
    v, _ = g.get_forecast_dps_with_date([fy54, q1b])
    check('D1-14：空欄（未定）なら、これまでどおり前の予想（54円）にさかのぼる', v == 54.0, str(v))
    # 実績の出ている期より前の予想まではさかのぼらない（シャープ・日産型：2022年・2024年の予想を使っていた）
    old3q = _st('3QFinancialStatements_Consolidated_JP', '3Q', '2022-02-08', '2022-03-31', f=40)
    old3q['CurPerEn'] = '2021-12-31'
    fy_now = _st('FYFinancialStatements_Consolidated_JP', 'FY', '2026-05-12', '2026-03-31', nxt_fy='2027-03-31')
    fy_now['DivAnn'] = '0.0'
    q1_now = _st('1QFinancialStatements_Consolidated_JP', '1Q', '2026-08-05', '2027-03-31', f=None)
    q1_now['CurPerEn'] = '2026-06-30'
    v, d_ = g.get_forecast_dps_with_date([old3q, fy_now, q1_now])
    check('D1-15：今の期の予想がすべて空欄なら、何年も前の予想ではなく直近の実績（0円）', v == 0.0 and d_ == date(2026, 5, 12), f'{v} {d_}')
    fy_prev = _st('FYFinancialStatements_Consolidated_JP', 'FY', '2026-05-14', '2026-03-31', nxt_fy='2027-03-31')
    fy_prev['DivAnn'] = '137.0'
    q3_prev = _st('3QFinancialStatements_Consolidated_JP', '3Q', '2026-02-02', '2026-03-31', f=116)
    q3_prev['CurPerEn'] = '2025-12-31'
    v, _ = g.get_forecast_dps_with_date([q3_prev, fy_prev, q1_now])
    check('D1-16：来期の予想が未定なら、前の期の予想（116円）ではなく、その期の実績（137円）', v == 137.0, str(v))
    # 2026年11月24日（GPT のレビュー）：本決算の短信で来期の予想が空欄でも、そのあとに出た来期の修正は使う
    fy_bl = _st('FYFinancialStatements_Consolidated_JP', 'FY', '2026-05-14', '2026-03-31', nxt_fy='2027-03-31')
    fy_bl['DivAnn'] = '50.0'
    rv60 = _st('DividendForecastRevision', 'FY', '2026-06-20', '2027-03-31', f=60)
    rv0 = _st('DividendForecastRevision', 'FY', '2026-06-20', '2027-03-31', f=0)
    v, d_ = g.get_forecast_dps_with_date([fy_bl, rv60])
    check('D1-17：来期の予想が空欄の短信のあとに出た来期の修正（60円）を使う', v == 60.0 and d_ == date(2026, 6, 20), f'{v} {d_}')
    check('D1-17b：分割を直す版（v3）も60円', g.forecast_dps_v3([fy_bl, rv60], []) == 60.0, str(g.forecast_dps_v3([fy_bl, rv60], [])))
    v, _ = g.get_forecast_dps_with_date([fy_bl, rv0])
    check('D1-18：その修正が0円なら0円（実績の50円に戻らない）', v == 0.0, str(v))
    check('D1-18b：分割を直す版（v3）も0円', g.forecast_dps_v3([fy_bl, rv0], []) == 0.0, str(g.forecast_dps_v3([fy_bl, rv0], [])))
    v, _ = g.get_forecast_dps_with_date([fy_bl])
    check('D1-18c：修正がなければ、これまでどおり実績（50円）', v == 50.0, str(v))
    v, _ = g.get_forecast_dps_with_date([fy, rev])
    check('D1-5：本決算の来期予想のあとに出た修正も使う', v == 112.0, str(v))
    rev_blank = _st('DividendForecastRevision', 'FY', '2026-10-29', '2027-03-31', f=None)
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev_blank])
    check('D1-6：予想が空（未定）の修正は使わない', v == 98.0, str(v))
    rev_earn = _st('EarnForecastRevision', 'FY', '2026-10-29', '2027-03-31', f=105, t='13:00')
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev, rev_earn])
    check('D1-7：修正が2つあれば、いちばん新しく出た方（時刻まで見る）', v == 112.0, str(v))
    rev_earn['DiscTime'] = '16:00:00'
    rev['DiscTime'] = '15:30'
    v, _ = g.get_forecast_dps_with_date([fy, q1, rev, rev_earn])
    check('D1-7b：同じ日なら時刻の遅い方（15:30 と 16:00:00 を比べられる）', v == 105.0, str(v))
    # 分割の前に出た「分割後の基準」の修正（KeePer 型）：分割を直す版で値がずれない
    fy2 = _st('FYFinancialStatements_NonConsolidated_JP', 'FY', '2026-08-14', '2026-06-30', nx=100, nxt_fy='2027-06-30')
    rv2 = _st('DividendForecastRevision', 'FY', '2026-09-15', '2027-06-30', f=50)
    ev = [(date(2026, 10, 29), 0.5)]
    check('D1-8：分割の前に出た分割後の基準の修正は、v3 で二重に半分にしない', g.forecast_dps_v3([fy2, rv2], ev) == 50.0,
          str(g.forecast_dps_v3([fy2, rv2], ev)))


def test_stmts_cache_saved_at():
    import generate as g
    importlib.reload(g)
    d = Path(tempfile.mkdtemp())
    g._STMTS_CACHE_PATH = d / 'stmts_cache.json.gz'
    g._STMTS_CACHE_LEGACY = d / 'stmts_cache.json'
    g.OUTPUT_DIR = d
    import gzip
    from datetime import timedelta
    old = (date.today() - timedelta(days=3)).isoformat()
    with gzip.open(g._STMTS_CACHE_PATH, 'wt', encoding='utf-8') as f:
        json.dump({'saved_at': old, 'schema': g._STMTS_SCHEMA, 'data': {'1301': [{'DiscDate': '2026-01-01'}]}}, f)
    g._stmts_cache = None
    g._stmts_loaded_saved_at = ''
    g._load_stmts_cache()
    g._save_stmts_cache()
    c = json.load(gzip.open(g._STMTS_CACHE_PATH, 'rt', encoding='utf-8'))
    check('キャッシュ1：読んだだけの日は、取得日を今日で上書きしない', c['saved_at'] == old, c['saved_at'])
    exp = (date.today() - timedelta(days=g.STMTS_CACHE_DAYS)).isoformat()
    with gzip.open(g._STMTS_CACHE_PATH, 'wt', encoding='utf-8') as f:
        json.dump({'saved_at': exp, 'schema': g._STMTS_SCHEMA, 'data': {'1301': [{'DiscDate': '2026-01-01'}]}}, f)
    g._stmts_cache = None
    g._stmts_loaded_saved_at = ''
    c2 = g._load_stmts_cache()
    check('キャッシュ2：期限が来たら空から取り直す', c2['data'] == {}, str(len(c2['data'])))
    c2['data']['1301'] = [{'DiscDate': '2026-11-20'}]
    g._save_stmts_cache()
    c = json.load(gzip.open(g._STMTS_CACHE_PATH, 'rt', encoding='utf-8'))
    check('キャッシュ3：取り直した日は、取得日が今日になる', c['saved_at'] == date.today().isoformat(), c['saved_at'])
    # 取り直しを月曜に始めて途中で止め、別の日に続けて終えた：取得日は始めた日（先に取った分を若返らせない）
    started = (date.today() - timedelta(days=g.STMTS_CACHE_DAYS + 1)).isoformat()
    with gzip.open(g._STMTS_CACHE_PATH, 'wt', encoding='utf-8') as f:
        json.dump({'saved_at': '', 'schema': g._STMTS_SCHEMA, 'started_at': started, 'data': {'1301': [{'DiscDate': '2026-01-01'}]}}, f)
    g._stmts_cache = None
    g._stmts_loaded_saved_at = ''
    c6 = g._load_stmts_cache()
    c6['data']['1302'] = [{'DiscDate': '2026-11-20'}]
    g._save_stmts_cache()
    c = json.load(gzip.open(g._STMTS_CACHE_PATH, 'rt', encoding='utf-8'))
    check('キャッシュ6：途中で止めて続けた取り直しの取得日は、始めた日', c['saved_at'] == started, c['saved_at'])
    g._stmts_cache = None
    g._stmts_loaded_saved_at = ''
    c7 = g._load_stmts_cache()
    check('キャッシュ7：始めた日から期限が過ぎていれば、次の回で取り直す', c7['data'] == {}, str(len(c7['data'])))
    # いまの本番のキャッシュ（版なし・取得日は新しく見えるが中身は古い）は、直したあとの最初の実行で捨てる
    with gzip.open(g._STMTS_CACHE_PATH, 'wt', encoding='utf-8') as f:
        json.dump({'saved_at': date.today().isoformat(), 'data': {'1301': [{'DiscDate': '2026-09-18'}]}}, f)
    g._stmts_cache = None
    g._stmts_loaded_saved_at = ''
    c4 = g._load_stmts_cache()
    check('キャッシュ4：版のない（直す前の）キャッシュは、取得日が今日でも捨てて取り直す', c4['data'] == {}, str(len(c4['data'])))
    with gzip.open(g._STMTS_CACHE_PATH, 'wt', encoding='utf-8') as f:
        json.dump({'saved_at': '', 'data': {'1301': [{'DiscDate': '2026-09-18'}]}}, f)
    g._stmts_cache = None
    g._stmts_loaded_saved_at = ''
    c5 = g._load_stmts_cache()
    check('キャッシュ5：版のない作りかけのキャッシュも引き継がない', c5['data'] == {}, str(len(c5['data'])))
    shutil.rmtree(d, ignore_errors=True)


def test_health_check_price_basis():
    import health_check as hc
    importlib.reload(hc)
    d = Path(tempfile.mkdtemp())
    (d / 'results.json').write_text(json.dumps({'stocks': [], 'latest_price_date': '2026-12-01'}), encoding='utf-8')
    (d / 'portfolio_state.json').write_text(json.dumps({'last_updated': '2026-12-01', 'date_basis': 'price_date',
                                                        'holdings': [], 'cash': 1, 'total_value': 1}), encoding='utf-8')
    hc.RESULTS, hc.STATE = d / 'results.json', d / 'portfolio_state.json'
    out = hc.run(today=date(2026, 12, 2))
    codes = [w['code'] for w in out.get('warnings', out.get('warns', []))]
    check('点検：株価の日付で更新していれば、実行が翌日でも「更新されていない」と出さない', 'pf_not_updated' not in codes, str(codes))
    shutil.rmtree(d, ignore_errors=True)


def test_lookahead_guard():
    """判定を保存した時刻が、約定する日の始値（9:00）より後なら、その判定では約定しない。"""
    s = Sim(['G001', 'G002'])
    s.run('2026-12-01', {})
    r = s.run('2026-12-02', {'G001': {'buy': True}}, at='2026-12-03T10:30:00+09:00')   # 12/2 の判定を 12/3 の取引中に作った
    r3 = s.run('2026-12-03', {})
    check('先読み1：約定の日の始値より後に保存した判定では、売買しない', r3['n'] == 0 and 'late' in r3['info'], str(r3['info']))
    s2 = Sim(['H001'])
    s2.run('2026-12-01', {})
    s2.run('2026-12-02', {'H001': {'buy': True}})                                  # 12/2 の夜に正しく保存
    s2.run('2026-12-02', {}, at='2026-12-03T10:30:00+09:00')                       # 12/3 の取引中に、12/2 の分を動かし直した
    r4 = s2.run('2026-12-03', {})
    check('先読み2：翌営業日の取引が始まったあとは、前日の判定を上書きしない（夜に保存した判定で約定）',
          r4['buy'] == ['H001'], f"buy={r4['buy']} info={r4['info']}")
    s3 = Sim(['E001'])
    s3.run('2026-12-01', {'E001': {'buy': True}})                                  # 12/1 の判定（約定予定 12/2）
    r5 = s3.run('2026-12-03', {})                                                   # 12/2 の実行が抜けた
    check('失効1：約定予定日（12/2）を過ぎた判定は、12/3 の寄付きに持ち越さない', r5['n'] == 0 and 'expired' in r5['info'],
          f"n={r5['n']} info={r5['info']}")
    s4 = Sim(['E002'])
    s4.run('2026-11-20', {'E002': {'buy': True}}, at='2026-11-24T00:13:00+09:00')  # 休日をはさむ（11/23 は祝日）
    r6 = s4.run('2026-11-24', {})
    check('失効2：休日をはさんでも、次の営業日（11/24）の判定は使う', r6['buy'] == ['E002'] and 'expired' not in r6['info'],
          f"buy={r6['buy']} info={r6['info']}")
    s3.close(); s4.close()
    s.close(); s2.close()


def test_frozen_lookahead():
    import frozen_v1_engine as fe
    import frozen_v1_run as fr
    importlib.reload(fe)
    importlib.reload(fr)
    d = Path(tempfile.mkdtemp())
    fr.OUT = d / 'v1'
    fr.check_frozen = lambda: (True, '')
    fe.TOPUP = fe.REBALANCE = fe.HARVEST = False
    from datetime import datetime as _dt

    def run(day, judge, at=None):
        fr._now = (lambda a=at or f'{day}T20:00:00+09:00': _dt.fromisoformat(a))
        return fr.update([_fake('J001', day, buy=judge.get('J001') == 'buy')], None, None)

    r1 = run('2026-12-01', {'J001': 'buy'}, at='2026-12-02T10:00:00+09:00')      # 12/1 の判定を 12/2 の取引中に作った
    r2 = run('2026-12-02', {})
    tr = r2['variants']['YIELD']['trades']
    check('固定10：始値より後に保存した判定では約定しない（評価だけ進む）', not tr and r2.get('late_judgment') and r2['decided_from'] is None,
          f"trades={tr} late={r2.get('late_judgment')}")
    r3 = run('2026-12-03', {'J001': 'buy'})
    r4 = run('2026-12-07', {})                                                      # 12/4 の実行が抜けた（12/5・6 は土日）
    tr4 = r4['variants']['YIELD']['trades']
    check('固定11：約定予定日（12/4）を過ぎた判定は失効（12/7 に持ち越さない）', not tr4 and r4.get('expired_judgment') and r4['decided_from'] is None,
          f"trades={tr4} expired={r4.get('expired_judgment')}")
    shutil.rmtree(d, ignore_errors=True)



def test_stale_price_rows():
    import generate as g
    ph = lambda d: [{'d': '2026-11-18', 'c': 1}, {'d': d, 'c': 1}]
    st = [{'code': 'A001', 'price_history': ph('2026-11-20')}, {'code': 'B001', 'price_history': ph('2026-11-20')},
          {'code': 'C001', 'price_history': ph('2026-11-17')}]
    un = []
    keep, latest, stale = g.drop_stale_price_rows(st, un)
    check('行なし1：最後の行が全体の最新日より古い会社は外す（古い終値で判定しない）',
          [x['code'] for x in keep] == ['A001', 'B001'] and stale == ['C001'] and latest == '2026-11-20', f'{stale} {latest}')
    check('行なし2：外した理由を unprocessed に残す（理由不明ではない）',
          len(un) == 1 and un[0]['code'] == 'C001' and '株価の行がない' in un[0]['reason'] and '2026-11-17' in un[0]['reason'], str(un))
    un2 = []
    keep2, _, stale2 = g.drop_stale_price_rows(st[:2], un2)
    check('行なし3：全社の最後の行がそろっていれば何も外さない', len(keep2) == 2 and not stale2 and not un2)


if __name__ == '__main__':
    test_calendar()
    test_connection_and_recovery()
    test_dividends()
    test_resume_across_month()
    test_switch_from_legacy()
    test_frozen_dividends()
    test_frozen_fingerprint()
    test_d1_revision()
    test_stmts_cache_saved_at()
    test_health_check_price_basis()
    test_lookahead_guard()
    test_frozen_lookahead()
    test_stale_price_rows()
    ok = sum(1 for r in RESULTS if r[1])
    print(f'\n{ok}/{len(RESULTS)} PASS')
    json.dump([{'name': n, 'ok': o, 'detail': d} for n, o, d in RESULTS],
              open(ROOT / 'tests' / 'result_v5.json', 'w'), ensure_ascii=False, indent=1)
    sys.exit(0 if ok == len(RESULTS) else 1)

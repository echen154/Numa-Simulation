import asyncio, json, os, sys, tempfile, time, numpy as np, pytest
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import simmodel as M, router as R, ratelimit as RL, strategy as ST, announcements as AN, etf as ETF
from engine import Engine
from library import Library
ROOT = os.environ.get('HURRICANE_RESEARCH', os.path.join(HERE, '..', '..', 'hurricane-season-research'))
LIB = os.path.join(HERE, '..', 'data', 'lib_train.npz')


# ---------------- valuation ----------------
@pytest.mark.skipif(not os.path.exists(ROOT), reason='research data not present')
def test_settlement_formula_on_untouched_seasons():
    """replay seasons through the engine; after the round-3 final report fair == settlement (within 0.15)"""
    lib = Library(LIB)
    for s in (3, 47, 88):
        e = Engine(lib); e.exclude_season = s
        for l in open(os.path.join(ROOT, 'history', f'season-{s:03d}', 'feed.jsonl')):
            e.on_message(json.loads(l))
        f, S, info = e.fair()
        se = json.load(open(os.path.join(ROOT, 'history', f'season-{s:03d}', 'settlement.json')))
        assert info['phase'] == 'final' and info['complete']
        assert np.abs(f - np.array([se[k] for k in M.SYMS])).max() < 0.15


def test_insurance_layers():
    z = np.zeros((1, 5))
    for L, h, r in [(300, 75, 90), (400, 75, 90), (850, 75 - 0.06 * 450, 90), (1300, 21, 90), (1700, 21, 90 - 0.035 * 400)]:
        out = M.settle(z, z, 0, L)[0]
        assert abs(out[8] - h) < 1e-9 and abs(out[9] - r) < 1e-9


# ---------------- engine robustness ----------------
def test_engine_ignores_malformed_and_requires_complete_data():
    e = Engine(Library(LIB))
    assert e.on_message({'type': 'storm', 'round': 1, 'age': 'x'}) in (None, 'storm')
    e.on_message({'type': 'storm', 'round': 1, 'age': 100, 'x': 400, 'y': 500, 'vx': 0, 'vy': -1, 'radius': 80, 'intensity': 1.0, 'seq': 5})
    e.on_message({'type': 'storm', 'round': 1, 'age': 102, 'x': float('nan'), 'y': 500, 'radius': 80, 'intensity': 1.0, 'seq': 6})
    assert not e.complete()              # joined mid-storm: obs at age 0 missing -> never trade
    f, S, info = e.fair()
    assert info['complete'] is False and ST.beta_for(info) == 0.0
    e.on_message({'type': 'damage_report', 'round': 1, 'stage': 'final', 'damage_pct': {'PORT': 'bad'}})
    assert 1 not in e.final_done


# ---------------- strategy ----------------
def test_policy_no_trade_without_quotes_or_when_incomplete():
    p = ST.Policy(); fair = np.full(12, 50.0)
    assert p.targets(fair, {'phase': 'final', 'complete': True}, {}, {}, {}) == []
    mids = {s: 40.0 for s in ST.SYMS}; hss = {s: 0.05 for s in ST.SYMS}
    assert p.targets(fair, {'phase': 'final', 'complete': False}, mids, hss, {}) == []
    o = p.targets(fair, {'phase': 'final', 'complete': True}, mids, hss, {s: 0 for s in ST.SYMS})
    assert o and all(side == 'buy' for _, side, *_ in o)
    assert all(q <= M.LIMITS[s] * 0.34 + 1 for s, _, q, *_ in o)


# ---------------- rate limiter ----------------
def test_shared_rate_limiter_across_instances():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'rl.json')
        a = RL.SharedRateLimiter(p, team_share=20, per_key=10, n_keys=3); b = RL.SharedRateLimiter(p, team_share=20, per_key=10, n_keys=3)
        got = sum(1 for i in range(40) if (a if i % 2 else b).acquire(1) is not None)
        assert got == int(20 * 0.85)                  # normal priority keeps 15% for urgent
        assert a.acquire(1, 'urgent') is not None


# ---------------- router (fake exchange) ----------------
class FakeApi:
    def __init__(self, timeout_next=False):
        self.calls = []; self.open = []; self.hist = []; self.timeout_next = timeout_next; self.pos = {'PORT': 150}
    async def req(self, method, path, key_idx=0, body=None):
        self.calls.append((method, path, body))
        if method == 'GET' and path == '/positions':
            return 200, {'positions': [{'symbol': k, 'quantity': v} for k, v in self.pos.items()]}
        if method == 'GET' and path == '/orders/open':
            return 200, {'orders': self.open}
        if method == 'GET' and path == '/orders':
            return 200, {'orders': self.hist}
        if method == 'POST' and path == '/orders':
            o = {'order_id': f'id{len(self.hist)}', **body, 'filled_quantity': 0, 'status': 'pending'}
            self.hist.insert(0, o); self.open.append(o)
            if self.timeout_next:
                self.timeout_next = False; raise asyncio.TimeoutError()
            return 200, {'order_id': o['order_id'], 'status': 'pending'}
        if method == 'DELETE':
            return 200, {'status': 'cancelled'}
        return 404, {}


def mk_router(api, dry=False):
    d = tempfile.mkdtemp()
    return R.OrderRouter(api, RL.SharedRateLimiter(os.path.join(d, 'rl.json'), 100, 60, 3), M.LIMITS, d, lambda *a, **k: None, dry_run=dry, live_ok=not dry)


def test_router_capacity_counts_team_position_and_teammate_orders():
    api = FakeApi(); r = mk_router(api); r.set_competition('c1')
    api.open.append({'order_id': 'teammate1', 'symbol': 'PORT', 'side': 'buy', 'quantity': 30, 'filled_quantity': 0, 'price': 70})
    asyncio.run(r.sync())
    assert r.room('PORT', 'buy') == 200 - 150 - 30
    oid = asyncio.run(r.place('PORT', 'buy', 100, 70.0))
    assert oid and api.calls[-1][2]['quantity'] == 20            # clipped to remaining room


def test_router_never_cancels_foreign_orders_and_never_cancel_all():
    api = FakeApi(); r = mk_router(api); r.set_competition('c1')
    api.open.append({'order_id': 'teammate1', 'symbol': 'HAVEN', 'side': 'sell', 'quantity': 5, 'filled_quantity': 0, 'price': 40})
    asyncio.run(r.sync()); asyncio.run(r.cancel_own())
    assert not any(c[0] == 'DELETE' for c in api.calls)
    assert not asyncio.run(r.cancel('teammate1'))
    assert not any('/orders/all' in c[1] for c in api.calls)


def test_router_timeout_is_recovered_by_price_signature():
    api = FakeApi(timeout_next=True); r = mk_router(api); r.set_competition('c1')
    res = asyncio.run(r.place('GRID', 'sell', 10, 38.5))
    assert res is None and len(r.uncertain) == 1
    assert r.room('GRID', 'sell') == 400 - 10                    # capacity stays reserved while uncertain
    asyncio.run(r.sync())
    assert len(r.uncertain) == 0 and any(o['sym'] == 'GRID' for o in r.owned.values())


def test_dry_run_sends_nothing():
    api = FakeApi(); r = mk_router(api, dry=True); r.set_competition('c1')
    asyncio.run(r.place('PORT', 'buy', 10, 70.0))
    assert not any(c[0] in ('POST', 'DELETE') for c in api.calls)


# ---------------- announcements ----------------
@pytest.mark.parametrize('h,kind,rule', [
    ('HavenBridge raises its retention to $500M', 'exact', 'haven_attachment'),
    ('Meridian Re now attaches above $1,500 million', 'exact', 'rein_attachment'),
    ('Archipelago Ferries declares a special dividend of $2.00 per share', 'exact', 'per_share_cash'),
    ('GridLock to harden substations', 'directional', 'directional_keyword'),
    ('Officials hold a press conference', 'unknown', None)])
def test_announcement_classifier(h, kind, rule):
    c = AN.classify({'headline': h, 'symbols': []})
    assert c['kind'] == kind and c['rule'] == rule


# ---------------- ETF ----------------
def test_etf_arb_requires_executable_profit():
    lv = lambda p, q=100: [{'price': p, 'quantity': q}]
    comp = dict(PORT=70, ISLE=50, AQUA=27, GRID=36, FERRY=18, HAVEN=40); basket = 70 + 50 + 54 + 36 + 36 + 40
    B = {s: (lv(p - .05), lv(p + .05)) for s, p in comp.items()}
    B['ARCH'] = (lv(basket - 0.2, 20), lv(basket + 0.2, 20))
    assert ETF.best_arb('ARCH', B) is None                       # inside costs -> no trade
    B['ARCH'] = (lv(basket + 2.0, 20), lv(basket + 2.4, 20))
    a = ETF.best_arb('ARCH', B); assert a and a['direction'] == 'sell_etf' and a['profit'] > 0


# ---------------- sole-trader mode (competition day: one process, all three keys) ----------------
def test_sole_trader_adopts_orphans_and_kill_uses_cancel_all():
    api = FakeApi(); d = tempfile.mkdtemp()
    r = R.OrderRouter(api, RL.SharedRateLimiter(os.path.join(d, 'rl.json'), 170, 57, 3), M.LIMITS, d, lambda *a, **k: None,
                      dry_run=False, live_ok=True, sole=True)
    r.set_competition('c1')
    api.open.append({'order_id': 'orphan1', 'symbol': 'HAVEN', 'side': 'sell', 'quantity': 5, 'filled_quantity': 0, 'price': 40})
    asyncio.run(r.sync())
    assert 'orphan1' in r.owned and r.owned['orphan1']['tag'] == 'adopted'
    asyncio.run(r.kill_all())
    assert any(c[0] == 'DELETE' and c[1] == '/orders/all' for c in api.calls)


def test_shared_mode_still_never_cancels_all():
    api = FakeApi(); r = mk_router(api); r.set_competition('c1')
    asyncio.run(r.kill_all())
    assert not any('/orders/all' in c[1] for c in api.calls)


def test_redundant_feeds_deduplicate_by_seq():
    import marketdata as MD
    got = []
    md = MD.MarketData('https://x', ['k1', 'k2', 'k3'], M.SYMS, lambda m: got.append(m['seq']), lambda *a, **k: None)
    msg = {'type': 'news', 'message': {'type': 'storm', 'round': 1, 'age': 0, 'seq': 7}}
    for _ in range(3):                                   # the same message arrives on three connections
        md.handle_frame(msg)
    md.handle_frame({'type': 'news', 'message': {'type': 'storm', 'round': 1, 'age': 2, 'seq': 8}})
    assert got == [7, 8] and md.dupes == 2
    md.handle_frame({'type': 'quote', 'symbol': 'PORT', 'bid': 70.0, 'ask': 70.2})
    assert md.books['PORT'].valid() and abs(md.books['PORT'].mid - 70.1) < 1e-9
    md.handle_frame({'type': 'quote', 'symbol': 'PORT', 'bid': None, 'ask': 70.2})
    assert not md.books['PORT'].valid()                  # one-sided book -> never traded

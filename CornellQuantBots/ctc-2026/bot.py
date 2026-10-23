"""Hurricane Season bot v2 (CTC 2026, Systematic Equities).

DRY-RUN BY DEFAULT. Live order submission requires ALL of:  --live  and  environment CTC_ALLOW_LIVE=1.

  python bot.py --mode testnet                      # dry run against the testnet (logs intended orders)
  CTC_ALLOW_LIVE=1 python bot.py --mode live --live # competition day

Keys: one per line in ~/.ctc/keys.txt (override with --keys-file or CTC_KEYS="k1,k2,k3"). Never in source or logs.
Run directory (logs, state, control.json, rate-limit file): ./run (override with CTC_RUN_DIR).
"""
import argparse, asyncio, json, os, sys, time, traceback
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import simmodel as M
from library import Library
from engine import Engine
from strategy import Policy, SYMS
from router import Api, OrderRouter
from ratelimit import SharedRateLimiter
from marketdata import MarketData
from announcements import AnnouncementHandler
import etf as ETF
import control as CTL

BASE_URL = 'https://cornelltradingcompetition.org'
RUN = os.environ.get('CTC_RUN_DIR', 'run')
PARAM_DEFAULTS = dict(HAVEN_ATT=M.HAVEN_ATT, HAVEN_EXH=M.HAVEN_EXH, REIN_ATT=M.REIN_ATT)


def load_keys(path):
    env = os.environ.get('CTC_KEYS')
    if env:
        return [k.strip() for k in env.split(',') if k.strip()]
    path = os.path.expanduser(path)
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith('#')]


class Bot:
    def __init__(self, a, keys):
        os.makedirs(RUN, exist_ok=True)
        self.a = a; self.logf = open(os.path.join(RUN, a.log), 'a')
        self.live = bool(a.live and os.environ.get('CTC_ALLOW_LIVE') == '1')
        lib_path = os.path.join(HERE, 'data', 'lib_all.npz' if os.path.exists(os.path.join(HERE, 'data', 'lib_all.npz')) else 'lib_train.npz')
        self.eng = Engine(Library(lib_path)); self.eng.use_cone = bool(a.cone)
        self.api = Api(BASE_URL if a.url is None else a.url, keys)
        self.lim = SharedRateLimiter(os.path.join(RUN, 'ratelimit.json'), team_share=a.team_share, per_key=57 if a.sole_trader else 55, n_keys=len(keys))
        self.router = OrderRouter(self.api, self.lim, M.LIMITS, os.path.join(RUN, 'state'), self.log, dry_run=not self.live, live_ok=self.live, sole=a.sole_trader)
        self.policy = Policy(size_scale=a.size)
        self.router.on_fill = self.on_fill
        self.md = MarketData(self.api.base.replace('/api/v1', ''), keys, SYMS, self.on_news, self.log)
        self.ann = AnnouncementHandler(self.log, self.alert, auto_exact=not a.no_auto_announce)
        self.symbols = set(a.symbols.split(',')) if a.symbols else set(SYMS)
        self.fair = None; self.scen = None; self.info = {}; self.fair_ts = 0.0
        self.state = None; self.comp = None; self.ftime = None
        self.urgent = asyncio.Event(); self.recompute = asyncio.Event()
        self.ctl_seq = -1; self.killed = False; self.etf_on = False; self.ctl_pause = {}
        self.arb_inv = {s: 0 for s in SYMS}          # inventory held by the ETF-arb module (kept out of the core targets)
        self.date2season = {}
        if a.mode == 'testnet':
            try:
                man = json.load(open(os.path.join(a.research, 'history', 'manifest.json')))
                self.date2season = {st['open'][:10]: s['season'] for s in man['seasons'] for st in s['storms']}
            except Exception as e:
                self.log('no_manifest', err=repr(e))
        self.log('start', live=self.live, mode=a.mode, lib=lib_path, symbols=sorted(self.symbols), team_share=a.team_share, sole=a.sole_trader, n_keys=len(keys))
        print(f"bot v2 | {len(keys)} keys | {'SOLE TRADER' if a.sole_trader else 'shared account'} | mode={a.mode} | {'LIVE ORDERS' if self.live else 'DRY RUN (no orders sent)'} | run dir {RUN}", flush=True)

    # ---------------- utils ----------------
    def log(self, kind, **kw):
        self.logf.write(json.dumps({'t': round(time.time(), 3), 'kind': kind, **kw}, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
        self.logf.flush()

    def alert(self, text):
        print('\a\n' + '!' * 70 + f'\n{time.strftime("%H:%M:%S")} {text}\n' + '!' * 70, flush=True); self.log('ALERT', text=text)

    def on_fill(self, sym, side, q, price, tag):
        self.log('fill', sym=sym, side=side, qty=q, price=price, tag=tag)
        if tag.startswith('etf'):
            self.arb_inv[sym] += q if side == 'buy' else -q
        elif price:
            self.policy.guard.on_fill(sym, side, q, price)

    def on_news(self, m):
        ev = self.eng.on_message(m)
        if ev == 'final':
            self.log('final_report', round=m.get('round'), dmg=m.get('damage_pct')); self.recompute.set(); self.urgent.set()
        elif ev == 'prelim':
            self.log('prelim_report', round=m.get('round'), dmg=m.get('damage_pct')); self.recompute.set()
        elif ev == 'announcement':
            # stale orders on affected symbols become dangerous: cancel ours first
            syms = m.get('symbols') or SYMS
            asyncio.create_task(self._cancel_syms(syms))
            self.ann.apply(m, self.router, self.policy); self.recompute.set(); self.urgent.set()
        elif ev == 'news':
            self.log('news', msg=m)
        elif m.get('type') == 'storm_formed':
            self.log('storm_formed', round=m.get('round'), storm=m.get('storm'))

    async def _cancel_syms(self, syms):
        for s in syms:
            await self.router.cancel_own(s, priority='urgent')

    def core_pos(self):
        """position the core strategy manages: team position minus ETF-arb inventory"""
        return {s: self.router.team_pos.get(s, 0) - self.arb_inv.get(s, 0) for s in SYMS}

    # ---------------- loops ----------------
    async def clock_loop(self):
        while True:
            try:
                st, d = await self.api.req('GET', '/competitions/time')
                if st == 200:
                    cid = d.get('competition_id')
                    if cid != self.comp:
                        if self.comp is not None:
                            self.eng.reset(); self.md.reset_seq(); self.arb_inv = {s: 0 for s in SYMS}
                            self.log('new_competition', comp=cid)
                        self.comp = cid; self.router.set_competition(cid)
                    self.state = d.get('state'); self.ftime = d.get('fictional_time')
                    if self.a.mode == 'testnet' and self.ftime:
                        self.eng.exclude_season = self.date2season.get(self.ftime[:10])
            except Exception as e:
                self.log('clock_err', err=repr(e))
            await asyncio.sleep(2)

    async def fair_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                f, S, info = await loop.run_in_executor(None, self.eng.fair)
                if np.all(np.isfinite(f)):
                    self.fair, self.scen, self.info, self.fair_ts = f, S, info, time.time()
                    mids = {s: self.md.books[s].mid for s in SYMS}
                    self.log('fair', info=info, fair=np.round(f, 3), mid=[None if mids[s] is None else round(mids[s], 3) for s in SYMS])
                else:
                    self.log('fair_nan', info=info)
            except Exception as e:
                self.log('fair_err', err=repr(e), tb=traceback.format_exc()[-600:])
            try:
                await asyncio.wait_for(self.recompute.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            self.recompute.clear()

    async def sync_loop(self):
        while True:
            try:
                await self.router.sync()
                self.policy.guard.update({s: self.md.books[s].mid for s in SYMS})
            except Exception as e:
                self.log('sync_err', err=repr(e))
            await asyncio.sleep(1.0)

    def tradable(self):
        return (self.state == 'running' and not self.killed and not self.router.paused and self.fair is not None
                and time.time() - self.fair_ts < 4 and self.md.connected and time.time() - self.router.last_sync < 5)

    async def trade_loop(self):
        while True:
            try:
                await asyncio.wait_for(self.urgent.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            urgent = self.urgent.is_set(); self.urgent.clear()
            if urgent:
                for _ in range(20):                      # wait (max 1 s) for the fair value that includes the event
                    if time.time() - self.fair_ts < 0.3:
                        break
                    await asyncio.sleep(0.05)
            try:
                if self.tradable():
                    await self.trade_once(urgent)
            except Exception as e:
                self.log('trade_err', err=repr(e), tb=traceback.format_exc()[-800:])

    async def trade_once(self, urgent):
        now = time.time(); mids = {}; hss = {}
        for s in SYMS:
            if s not in self.symbols:
                continue
            b = self.md.books[s]
            if not b.valid(now):
                b = await self.md.refresh_book(self.api, s)       # stale/empty quote -> REST book
                if b is None or not b.valid(time.time()):
                    continue                                       # empty or one-sided book: never trade blind
            mids[s] = b.mid; hss[s] = (b.ask - b.bid) / 2
        pos = self.core_pos()
        paused = lambda s: self.router.sym_paused.get(s, 0) > now or self.ctl_pause.get(s, 0) > now
        orders = self.policy.targets(self.fair, self.info, mids, hss, pos, paused, urgent=urgent and self.info.get('phase') == 'final')
        if urgent and self.info.get('phase') != 'final':
            urgent = False
        tasks = []
        for s, side, qty, px, prio, val in orders:
            own = self.router.own_open(s)
            same = [oid for oid, o in own.items() if o['side'] == side and abs((o['price'] or 0) - px) <= max(0.3 * hss[s], 0.01) and now - o['ts'] < 8]
            if same:
                continue
            tasks.append(self._replace(s, own, side, qty, px, prio))
        # symbols without a target: cancel our stale working orders there
        targeted = {o[0] for o in orders}
        for s in SYMS:
            if s not in targeted:
                for oid, o in self.router.own_open(s).items():
                    if o.get('tag') in ('core', 'adopted') and now - o['ts'] > 3:
                        tasks.append(self.router.cancel(oid))
        if tasks:
            await asyncio.gather(*tasks)

    async def _replace(self, s, own, side, qty, px, prio):
        for oid, o in own.items():
            if o.get('tag') in ('core', 'adopted'):
                await self.router.cancel(oid, prio)
        await self.router.place(s, side, qty, px, tag='core', priority=prio)

    async def etf_loop(self):
        while True:
            await asyncio.sleep(2.0)
            if not (self.etf_on and self.tradable()):
                continue
            try:
                books = {}
                for s in ['ARCH', 'PORT', 'ISLE', 'AQUA', 'GRID', 'FERRY', 'HAVEN', 'REBLD', 'RECON', 'MAT', 'GEN']:
                    b = await self.md.refresh_book(self.api, s)
                    if b is not None and b.levels and b.levels[0] and b.levels[1]:
                        books[s] = b.levels
                for e in ('ARCH', 'REBLD'):
                    arb = ETF.best_arb(e, books, room=self.router.room)
                    if arb is None:
                        continue
                    self.log('etf_arb_seen', **{k: v for k, v in arb.items() if k != 'legs'}, legs=arb['legs'])
                    await asyncio.gather(*[self.router.place(s, side, q, px, tag='etf', priority='normal') for s, side, q, px in arb['legs']])
                    await asyncio.sleep(1.5)
                    for s, _, _, _ in arb['legs']:
                        for oid, o in self.router.own_open(s).items():
                            if o.get('tag') == 'etf':
                                await self.router.cancel(oid)
            except Exception as e:
                self.log('etf_err', err=repr(e), tb=traceback.format_exc()[-600:])

    async def control_loop(self):
        while True:
            await asyncio.sleep(1.0)
            c = CTL.load()
            if c.get('seq', 0) == self.ctl_seq:
                continue
            self.ctl_seq = c.get('seq', 0); self.log('control', cfg=c)
            self.router.paused = bool(c['paused'])
            if c['paused']:
                await self.router.cancel_own(priority='urgent')
            if c['kill'] and not self.killed:
                self.killed = True; self.router.paused = True
                await self.router.kill_all(); self.alert('KILL: resting orders cancelled, trading stopped')
            self.ctl_pause = {s: float(u) for s, u in c['sym_pause'].items()}
            for s, u in self.ctl_pause.items():
                if u > time.time():
                    await self.router.cancel_own(s, priority='urgent')
            self.policy.size_scale = float(c['size_scale'])
            self.policy.fair_override = {s: tuple(v) for s, v in c['fair_override'].items()}
            for k, v in PARAM_DEFAULTS.items():
                setattr(M, k, float(c['params'].get(k, v)))
            M.ADJ_ADD[:] = 0; M.ADJ_MUL[:] = 1
            for s, (kind, v) in c['adjust'].items():
                j = SYMS.index(s)
                if kind == 'add': M.ADJ_ADD[j] += v
                else: M.ADJ_MUL[j] *= v
            self.etf_on = bool(c['etf']) and self.a.enable_etf
            self.recompute.set()

    async def status_loop(self):
        while True:
            await asyncio.sleep(10)
            if self.fair is None:
                continue
            parts = []
            for j, s in enumerate(SYMS):
                m = self.md.books[s].mid
                parts.append(f"{s}:{self.router.team_pos.get(s, 0)}@{'-' if m is None else f'{m:.2f}'}/f{self.fair[j]:.2f}")
            print(time.strftime('%H:%M:%S'), self.ftime, self.state, self.info, 'rate', self.lim.usage(), '|', ' '.join(parts), flush=True)

    async def run(self):
        await self.api.start()
        try:
            await asyncio.gather(self.md.run(self.a.ws_conns), self.clock_loop(), self.fair_loop(), self.sync_loop(), self.trade_loop(),
                                 self.etf_loop(), self.control_loop(), self.status_loop())
        finally:
            await self.api.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['testnet', 'live'], required=True)
    ap.add_argument('--live', action='store_true', help='actually send orders (also needs CTC_ALLOW_LIVE=1)')
    ap.add_argument('--keys-file', default='~/.ctc/keys.txt')
    ap.add_argument('--url', default=None)
    ap.add_argument('--log', default='bot.jsonl')
    ap.add_argument('--sole-trader', action='store_true', help='this process is the ONLY trader on the team account (competition day)')
    ap.add_argument('--team-share', type=int, default=None, help='order actions/min this bot may use (team limit 180; default 170 sole, 100 shared)')
    ap.add_argument('--ws-conns', type=int, default=None, help='parallel market-data connections (default: one per key)')
    ap.add_argument('--size', type=float, default=1.0, help='scale all target sizes')
    ap.add_argument('--symbols', default='', help='comma list; default all 12 (use to split symbols with a teammate bot)')
    ap.add_argument('--enable-etf', action='store_true', help='allow the ETF arb module (still off until control.py etf on)')
    ap.add_argument('--cone', action='store_true', help='use the official-cone-adjusted forecaster (mixed evidence, off by default)')
    ap.add_argument('--no-auto-announce', action='store_true', help='never auto-price announcements; always pause + alert')
    ap.add_argument('--research', default=os.path.join(HERE, '..', 'hurricane-season-research'))
    a = ap.parse_args()
    if a.team_share is None:
        a.team_share = 170 if a.sole_trader else 100
    keys = load_keys(a.keys_file)
    if not keys:
        sys.exit('no API keys found')
    asyncio.run(Bot(a, keys).run())


if __name__ == '__main__':
    main()

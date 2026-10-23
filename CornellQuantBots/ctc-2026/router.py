"""Centralized order router: the ONLY component that sends order actions.

Verified API behaviour (testnet, read-only probes, 2026-10-22):
  * GET /positions        -> TEAM positions (identical for all three keys; changed while only a teammate traded).
  * GET /orders           -> TEAM order history (identical for all keys), newest first, last 100, no owner field;
                             cleared when a new competition starts.
  * GET /orders/open      -> TEAM-wide resting orders, no owner field (confirmed: our key returned an order placed
                             by a teammate's bot on 2026-10-22).
  * DELETE /orders/{id}   -> works for our own ids; we never try ids we did not create.
  * DELETE /orders/all    -> used only in sole-trader mode (competition day: one process owns all three keys), for the
                             kill switch. In shared mode it is never used.
Rules enforced here:
  * We own exactly the order ids returned by our own POSTs (persisted to disk per competition).
  * Capacity check before every order: |team position + all resting team orders on that side + our uncertain
    orders + this order| <= limit (the exchange counts resting orders against the limit).
  * A POST that times out is 'uncertain': its capacity stays reserved and the router looks for it in the
    team history (same symbol/side/qty/price, created after we sent it). If it is not found within 20 s it
    is treated as not placed.
  * Every action passes the shared rate limiter. Dry-run never sends anything.
"""
import asyncio, json, os, time, uuid
import aiohttp

UA = {'User-Agent': 'ctc-hurricane-v2/2.0'}
SIG = 0.000037   # 37 millionths: marks our limit prices (prices may have up to 6 decimals)


def has_sig(price):
    return price is not None and abs((round(float(price) * 1e6) % 100) - 37) < 0.5


class Api:
    def __init__(self, base, keys, timeout=4.0):
        self.base = base.rstrip('/') + '/api/v1'; self.keys = keys; self.timeout = timeout; self.session = None

    async def start(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def req(self, method, path, key_idx=0, body=None):
        h = {**UA, 'X-API-Key': self.keys[key_idx % len(self.keys)], 'Content-Type': 'application/json'}
        async with self.session.request(method, self.base + path, headers=h, json=body,
                                        timeout=aiohttp.ClientTimeout(total=self.timeout)) as r:
            txt = await r.text()
            try:
                return r.status, json.loads(txt)
            except Exception:
                return r.status, {'raw': txt[:300]}


class OrderRouter:
    def __init__(self, api, limiter, limits, state_dir, log, dry_run=True, live_ok=False, sole=False):
        self.sole = sole   # True: this process is the only trader on the team account (all open orders are ours)
        self.api = api; self.lim = limiter; self.limits = limits; self.log = log
        self.dry = dry_run or not live_ok
        self.state_dir = state_dir; os.makedirs(state_dir, exist_ok=True)
        self.comp = None
        self.owned = {}        # oid -> dict(sym, side, qty, price, filled, status, ts, tag)
        self.uncertain = {}    # local id -> dict(sym, side, qty, price, t_sent)
        self.team_pos = {}     # sym -> qty (team)
        self.team_open = []    # resting orders on the team account (all owners)
        self.own_fills = {}    # sym -> signed filled qty from our own orders (this competition)
        self.paused = False; self.sym_paused = {}
        self.last_sync = 0.0
        self.on_fill = None   # callback(sym, side, qty, price, tag)

    # ---------- persistence ----------
    def _path(self):
        return os.path.join(self.state_dir, f'router_{self.comp}.json')

    def set_competition(self, comp):
        if comp == self.comp:
            return
        self.comp = comp; self.owned = {}; self.uncertain = {}; self.own_fills = {}
        try:
            d = json.load(open(self._path()))
            self.owned = d.get('owned', {}); self.own_fills = d.get('own_fills', {})
            self.log('router_resume', comp=comp, owned=len(self.owned))
        except Exception:
            pass

    def save(self):
        if self.comp is None:
            return
        tmp = self._path() + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'owned': self.owned, 'own_fills': self.own_fills}, f)
        os.replace(tmp, self._path())

    # ---------- capacity ----------
    def resting_same_side(self, sym, side):
        return sum(max(0, o['quantity'] - o.get('filled_quantity', 0)) for o in self.team_open
                   if o['symbol'] == sym and o['side'] == side)

    def uncertain_same_side(self, sym, side):
        return sum(u['qty'] for u in self.uncertain.values() if u['sym'] == sym and u['side'] == side)

    def room(self, sym, side):
        pos = self.team_pos.get(sym, 0); lim = self.limits[sym]
        rest = self.resting_same_side(sym, side) + self.uncertain_same_side(sym, side)
        return int(lim - pos - rest) if side == 'buy' else int(lim + pos - rest)

    def own_open(self, sym=None):
        return {k: v for k, v in self.owned.items() if v['status'] in ('pending', 'partial', 'open') and (sym is None or v['sym'] == sym)}

    # ---------- actions ----------
    async def place(self, sym, side, qty, price, tag='', priority='normal', order_type='limit'):
        if self.paused or self.sym_paused.get(sym, 0) > time.time():
            return None
        qty = int(min(qty, max(self.room(sym, side), 0)))
        if qty < 1:
            return None
        if order_type == 'limit' and (price is None or not (price > 0)):
            return None
        k = self.lim.acquire(1, priority)
        if k is None:
            self.log('rate_limited', sym=sym, side=side, qty=qty); return None
        body = {'symbol': sym, 'side': side, 'order_type': order_type, 'quantity': qty}
        if order_type == 'limit':
            # price signature: the 5th-6th decimals carry our marker so our orders are identifiable in team history
            body['price'] = round(round(float(price), 4) + SIG, 6)
        if self.dry:
            oid = 'dry-' + uuid.uuid4().hex[:12]
            self.owned[oid] = dict(sym=sym, side=side, qty=qty, price=body.get('price'), filled=0, status='pending', ts=time.time(), tag=tag)
            self.log('dry_order', oid=oid, tag=tag, **body); return oid
        lid = uuid.uuid4().hex; self.uncertain[lid] = dict(sym=sym, side=side, qty=qty, price=body.get('price'), t_sent=time.time())
        try:
            st, d = await self.api.req('POST', '/orders', k, body)
        except Exception as e:
            self.log('order_uncertain', err=repr(e), tag=tag, **body); return None   # stays uncertain
        if st == 200 and d.get('order_id'):
            self.uncertain.pop(lid, None)
            oid = d['order_id']; status = d.get('status', 'pending')
            self.owned[oid] = dict(sym=sym, side=side, qty=qty, price=body.get('price'), filled=qty if status == 'filled' else 0,
                                   status=status, ts=time.time(), tag=tag)
            if status == 'filled':
                self._add_fill(sym, side, qty, body.get('price'), tag)
            self.log('order', oid=oid, status=status, tag=tag, **body); self.save()
            return oid
        self.uncertain.pop(lid, None)
        self.log('order_rejected', st=st, resp=d, tag=tag, **body)
        return None

    async def cancel(self, oid, priority='normal'):
        o = self.owned.get(oid)
        if o is None or o['status'] not in ('pending', 'partial', 'open'):
            return False
        k = self.lim.acquire(1, priority)
        if k is None:
            return False
        if self.dry:
            o['status'] = 'cancelled'; self.log('dry_cancel', oid=oid); return True
        try:
            st, d = await self.api.req('DELETE', f'/orders/{oid}', k)
        except Exception as e:
            self.log('cancel_err', oid=oid, err=repr(e)); return False
        self.log('cancel', oid=oid, st=st, resp=d)
        if st == 200:
            o['status'] = d.get('status', 'cancelled')
        self.save(); return st == 200

    async def cancel_own(self, sym=None, priority='normal'):
        for oid in list(self.own_open(sym)):
            await self.cancel(oid, priority)

    async def kill_all(self):
        """emergency: sole trader -> one DELETE /orders/all (1 action), then per-id cleanup; otherwise own ids only"""
        if self.sole and not self.dry:
            k = self.lim.acquire(1, 'urgent')
            try:
                st, d = await self.api.req('DELETE', '/orders/all', 0 if k is None else k)
                self.log('cancel_all', st=st, resp=d)
            except Exception as e:
                self.log('cancel_all_err', err=repr(e))
        await self.cancel_own(priority='urgent')

    async def etf(self, sym, action, qty, priority='normal'):
        """create/redeem; one action each. Disabled unless explicitly enabled in config (semantics unverified)."""
        k = self.lim.acquire(1, priority)
        if k is None:
            return None
        if self.dry:
            self.log('dry_etf', sym=sym, action=action, qty=qty); return 'dry'
        try:
            st, d = await self.api.req('POST', f'/etf/{sym}/{action}', k, {'quantity': int(qty)})
        except Exception as e:
            self.log('etf_uncertain', sym=sym, action=action, err=repr(e)); return None
        self.log('etf', sym=sym, action=action, qty=qty, st=st, resp=d)
        return d if st == 200 else None

    def _add_fill(self, sym, side, q, price=None, tag=''):
        self.own_fills[sym] = self.own_fills.get(sym, 0) + (q if side == 'buy' else -q)
        if self.on_fill:
            try:
                self.on_fill(sym, side, q, price, tag)
            except Exception:
                pass

    # ---------- reconciliation ----------
    async def sync(self):
        """positions + resting orders (team) + our order statuses. Never modifies teammates' orders."""
        st, d = await self.api.req('GET', '/positions')
        if st == 200:
            self.team_pos = {p['symbol']: int(p['quantity']) for p in d.get('positions', [])}
        st, d = await self.api.req('GET', '/orders/open')
        open_ids = set()
        if st == 200:
            self.team_open = d.get('orders', [])
            for o in self.team_open:
                open_ids.add(o['order_id'])
                if self.sole and o['order_id'] not in self.owned:
                    # sole trader: an unknown resting order can only be ours (e.g. left by a crashed run) -> adopt it
                    self.owned[o['order_id']] = dict(sym=o['symbol'], side=o['side'], qty=int(o['quantity']), price=o.get('price'),
                                                     filled=int(o.get('filled_quantity', 0)), status=o.get('status', 'pending'),
                                                     ts=0.0, tag='adopted')
                    self.log('order_adopted', oid=o['order_id'], sym=o['symbol'])
                mine = self.owned.get(o['order_id'])
                if mine:
                    nf = int(o.get('filled_quantity', 0))
                    if nf > mine['filled']:
                        self._add_fill(mine['sym'], mine['side'], nf - mine['filled'], mine['price'], mine.get('tag', '')); mine['filled'] = nf
                    mine['status'] = o.get('status', mine['status'])
        # owned orders that left the open list, and uncertain POSTs: look them up in the team history
        gone = [oid for oid, o in self.owned.items() if o['status'] in ('pending', 'partial', 'open') and oid not in open_ids
                and not oid.startswith('dry-')]
        if (gone or self.uncertain) and st == 200:
            hs, hist = await self.api.req('GET', '/orders')
            if hs == 200:
                H = {o['order_id']: o for o in hist.get('orders', [])}
                for oid in gone:
                    mine = self.owned[oid]; h = H.get(oid)
                    if h is None:
                        if time.time() - mine['ts'] > 30:
                            mine['status'] = 'unknown'; self.log('order_lost', oid=oid)
                        continue
                    nf = int(h.get('filled_quantity', 0))
                    if nf > mine['filled']:
                        self._add_fill(mine['sym'], mine['side'], nf - mine['filled'], mine['price'], mine.get('tag', '')); mine['filled'] = nf
                    mine['status'] = h.get('status', 'done')
                for lid, u in list(self.uncertain.items()):
                    cand = [o for o in hist.get('orders', []) if o['order_id'] not in self.owned and o['symbol'] == u['sym']
                            and (u['price'] is None or has_sig(o.get('price')))
                            and o['side'] == u['side'] and int(o['quantity']) == u['qty']
                            and (u['price'] is None or abs(float(o.get('price') or 0) - u['price']) < 1e-6)]
                    if cand:
                        o = cand[0]
                        self.owned[o['order_id']] = dict(sym=u['sym'], side=u['side'], qty=u['qty'], price=u['price'],
                                                         filled=0, status='pending', ts=u['t_sent'], tag='recovered')
                        nf = int(o.get('filled_quantity', 0))
                        if nf:
                            self._add_fill(u['sym'], u['side'], nf, u['price'], 'recovered'); self.owned[o['order_id']]['filled'] = nf
                        self.owned[o['order_id']]['status'] = o.get('status', 'pending')
                        del self.uncertain[lid]; self.log('order_recovered', oid=o['order_id'])
                    elif time.time() - u['t_sent'] > 20:
                        del self.uncertain[lid]; self.log('uncertain_dropped', **u)
        self.last_sync = time.time(); self.save()

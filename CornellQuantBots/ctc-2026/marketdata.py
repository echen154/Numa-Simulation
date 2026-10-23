"""Market data: redundant WebSocket feeds (one connection per key, up to 25 allowed per team) with shared
sequence de-duplication, reconnect, staleness tracking, and REST order-book fallback.

Redundancy matters: a single dropped connection means missed storm observations, and the engine then refuses
to trade the rest of that round. With 2-3 parallel connections, a message is lost only if every feed misses it.
"""
import asyncio, json, time
import websockets

STALE_S = 3.0


class Book:
    __slots__ = ('bid', 'ask', 'bsz', 'asz', 'ts', 'levels')

    def __init__(self):
        self.bid = self.ask = None; self.bsz = self.asz = 0; self.ts = 0.0; self.levels = None

    def valid(self, now=None):
        now = now or time.time()
        return (self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > self.bid
                and now - self.ts < STALE_S)

    @property
    def mid(self):
        return (self.bid + self.ask) / 2 if self.bid is not None and self.ask is not None else None


class MarketData:
    def __init__(self, base, keys, symbols, on_news, log):
        self.ws_url = base.rstrip('/').replace('https://', 'wss://').replace('http://', 'ws://') + '/ws/v1/market-data'
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.symbols = symbols; self.on_news = on_news; self.log = log
        self.books = {s: Book() for s in symbols}
        self.conn = {}            # connection index -> connected?
        self.last_frame = 0.0; self.reconnects = 0
        self.seen_seq = set(); self.dupes = 0

    @property
    def connected(self):
        return any(self.conn.values())

    def reset_seq(self):
        self.seen_seq = set()

    def handle_frame(self, d):
        """process one decoded frame (shared by all connections). Returns True if it was a new news message."""
        t = d.get('type')
        if t == 'quote':
            b = self.books.get(d.get('symbol'))
            if b is not None:
                b.bid = d.get('bid'); b.ask = d.get('ask'); b.bsz = d.get('bid_size') or 0; b.asz = d.get('ask_size') or 0
                b.ts = time.time()
            return False
        if t != 'news':
            return False
        m = d.get('message')
        if not isinstance(m, dict):
            return False
        seq = m.get('seq')
        if seq is not None and m.get('type') != 'snapshot':
            if seq in self.seen_seq:
                self.dupes += 1
                return False
            self.seen_seq.add(seq)
        try:
            self.on_news(m)
        except Exception as e:
            self.log('news_handler_err', err=repr(e), msg=str(m)[:300])
        return True

    async def _one(self, i):
        backoff = 1.0
        sub = {'action': 'subscribe', 'symbols': self.symbols, 'channels': ['news', 'quotes']}
        key = self.keys[i % len(self.keys)]
        while True:
            try:
                async with websockets.connect(self.ws_url, additional_headers={'X-API-Key': key, 'User-Agent': 'ctc-hurricane-v2/2.0'},
                                              max_size=2 ** 22, ping_interval=20) as ws:
                    await ws.send(json.dumps(sub))   # exactly one subscription message, within 10 s
                    self.conn[i] = True; backoff = 1.0; self.log('ws_connected', conn=i, n=self.reconnects)
                    async for frame in ws:
                        self.last_frame = time.time()
                        try:
                            d = json.loads(frame)
                        except Exception:
                            self.log('ws_bad_frame', conn=i, frame=str(frame)[:200]); continue
                        self.handle_frame(d)
            except Exception as e:
                self.conn[i] = False; self.reconnects += 1
                self.log('ws_err', conn=i, err=repr(e)); await asyncio.sleep(backoff); backoff = min(backoff * 2, 15)

    async def run(self, n_conn=None):
        n = n_conn or len(self.keys)
        await asyncio.gather(*[self._one(i) for i in range(n)])

    async def refresh_book(self, api, sym):
        """REST fallback when the quote is stale/empty: full depth (also used by the ETF module)."""
        try:
            st, d = await api.req('GET', f'/orderbook/{sym}')
        except Exception:
            return None
        if st != 200:
            return None
        b = self.books[sym]; bids = d.get('bids') or []; asks = d.get('asks') or []
        b.levels = (bids, asks)
        b.bid = bids[0]['price'] if bids else None; b.ask = asks[0]['price'] if asks else None
        b.bsz = bids[0]['quantity'] if bids else 0; b.asz = asks[0]['quantity'] if asks else 0
        b.ts = time.time()
        return b

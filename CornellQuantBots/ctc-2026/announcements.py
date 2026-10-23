"""Announcement handler.

The research data contains no announcements, so NOTHING here is calibrated on real examples. The parser is
deterministic and only acts automatically when the headline states a number that maps onto a known model
parameter. Everything else pauses the affected symbols and asks a human.

Classes of result:
  exact       : a number + a known parameter (insurance attachment/limit, per-share cash, % change of value,
                share of losses covered) -> model adjusted, trade the 45 s house lag with full confidence
  directional : sign is clear but size is not ("hardens", "evacuates", "cancels contract") -> pause + alert,
                human may set an override (control.py)
  unknown     : pause + alert
Every decision is logged with the rule that fired.
"""
import re, time
import simmodel as M

NAMES = {'PORT': ['cascadia pacific', 'port', 'terminals'], 'ISLE': ['isle resorts', 'resort'], 'AQUA': ['aquavida', 'fisher'],
         'GRID': ['gridlock', 'grid', 'utility', 'utilities', 'substation'], 'FERRY': ['ferr'], 'RECON': ['reconex'],
         'MAT': ['materials holdings', 'materials'], 'GEN': ['backup power', 'generator'], 'HAVEN': ['havenbridge', 'haven'],
         'REIN': ['meridian', 'reinsur']}
SYM_IDX = {s: i for i, s in enumerate(M.SYMS[:10])}
MONEY = r'\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(m|mn|million|b|bn|billion)?'
PCT = r'([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent)'


def _money(txt):
    m = re.search(MONEY, txt, re.I)
    if not m:
        return None
    v = float(m.group(1).replace(',', '')); u = (m.group(2) or '').lower()
    return v * 1000 if u.startswith('b') else v


def symbols_in(text, given):
    found = [s for s in given or [] if s in SYM_IDX]
    t = text.lower()
    for s, keys in NAMES.items():
        if s not in found and any(k in t for k in keys):
            found.append(s)
    return found


def classify(msg):
    """-> dict(kind, symbols, action, detail, rule)"""
    h = (msg.get('headline') or '').strip(); t = h.lower()
    syms = symbols_in(h, msg.get('symbols'))
    res = dict(kind='unknown', symbols=syms, action=None, detail=h, rule=None)
    money = _money(h); pct = re.search(PCT, t)
    # 1) insurance layer changes (most valuable: nonlinear payoffs)
    if 'HAVEN' in syms and money is not None and re.search(r'retention|attach|deductible|starts? (?:paying|at)', t):
        res.update(kind='exact', action=('param', 'HAVEN_ATT', money), rule='haven_attachment'); return res
    if ('HAVEN' in syms or 'REIN' in syms) and money is not None and re.search(r'limit|cap|exhaust|up to|tops? out', t):
        if 'REIN' in syms and 'HAVEN' not in syms:
            res.update(kind='exact', action=('param', 'REIN_ATT', money), rule='rein_attachment'); return res
        res.update(kind='exact', action=('param', 'HAVEN_EXH', money), rule='haven_limit'); return res
    if 'REIN' in syms and money is not None and re.search(r'attach|retention|above', t):
        res.update(kind='exact', action=('param', 'REIN_ATT', money), rule='rein_attachment'); return res
    # 2) per-share cash (special dividend, buyback at price, fine per share)
    m = re.search(r'\$\s?([0-9]+(?:\.[0-9]+)?)\s*(?:per|a|/)\s*share', t)
    if m and len(syms) == 1:
        v = float(m.group(1)); sign = -1 if re.search(r'fine|penalt|charge|levy|tax', t) else 1
        res.update(kind='exact', action=('add', syms[0], sign * v), rule='per_share_cash'); return res
    # 3) explicit % change of company value / revenue
    if pct and len(syms) == 1 and re.search(r'value|worth|revenue|earnings|profit|assets', t):
        v = float(pct.group(1)) / 100; sign = -1 if re.search(r'cut|fall|drop|lose|loss|reduc|down|lower|decline', t) else 1
        if re.search(r'rise|increase|up|higher|boost|gain|raise', t) or sign < 0:
            res.update(kind='exact', action=('mul', syms[0], 1 + sign * v), rule='pct_value_change'); return res
    # 4) directional only
    if syms and re.search(r'harden|reinforc|evacuat|shut|close|suspend|halt|downgrade|upgrade|contract|subsid|aid|bailout|bankrupt|default|merger|acqui', t):
        res.update(kind='directional', rule='directional_keyword'); return res
    return res


class AnnouncementHandler:
    def __init__(self, log, alert, auto_exact=True, pause_s=120, window_s=45):
        self.log = log; self.alert = alert; self.auto_exact = auto_exact; self.pause_s = pause_s; self.window_s = window_s
        self.history = []

    def apply(self, msg, router, policy):
        c = classify(msg); now = time.time(); self.history.append((now, msg, c))
        self.log('announcement', headline=msg.get('headline'), symbols=msg.get('symbols'), cls=c)
        syms = c['symbols'] or [s for s in (msg.get('symbols') or [])]
        if c['kind'] == 'exact' and self.auto_exact:
            a = c['action']
            if a[0] == 'param':
                setattr(M, a[1], float(a[2]))
            elif a[0] == 'add':
                M.ADJ_ADD[SYM_IDX[a[1]]] += a[2]
            elif a[0] == 'mul':
                M.ADJ_MUL[SYM_IDX[a[1]]] *= a[2]
            # trade the house's ~45 s repricing lag at full beta for the affected symbols (fair recomputed next tick)
            policy.announce_until = {s: now + self.window_s for s in syms + ['ARCH', 'REBLD']}
            self.alert(f"AUTO-PRICED announcement ({c['rule']}): {msg.get('headline')} -> {a}. Uncalibrated rule: check it! "
                       f"Undo with: python control.py reset-adjustments")
            return c
        for s in syms or M.SYMS:
            router.sym_paused[s] = now + self.pause_s
        self.alert(f"ANNOUNCEMENT NOT PRICED ({c['kind']}): {msg.get('headline')} symbols={syms}. Paused {self.pause_s}s. "
                   f"Use control.py to set a fair override, an adjustment, or unpause.")
        return c

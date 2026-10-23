"""Human control for a running bot (edits control.json in the run directory; the bot reads it every second).

  python control.py status
  python control.py pause                 # no new orders (positions kept, own resting orders cancelled)
  python control.py resume
  python control.py kill                  # cancel resting orders and stop trading
  python control.py unkill                # clear the kill flag (needed before restarting a killed bot)
  python control.py pause-sym HAVEN 120   # pause one symbol for N seconds
  python control.py unpause-sym HAVEN
  python control.py size 0.5              # scale all target sizes
  python control.py fair HAVEN 41.5 60    # force our fair value for 60 s (e.g. after an announcement you priced by hand)
  python control.py adjust PORT add -5    # model adjustment: settle value +/- dollars per share (or: mul 0.9)
  python control.py param HAVEN_ATT 500   # insurance layer parameter (HAVEN_ATT, HAVEN_EXH, REIN_ATT) in $M
  python control.py reset-adjustments
  python control.py etf on|off
"""
import json, os, sys, time

PATH = os.path.join(os.environ.get('CTC_RUN_DIR', 'run'), 'control.json')
DEFAULT = dict(paused=False, kill=False, size_scale=1.0, sym_pause={}, fair_override={}, adjust={}, params={}, etf=False, seq=0)


def load():
    try:
        return {**DEFAULT, **json.load(open(PATH))}
    except Exception:
        return dict(DEFAULT)


def save(c):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    c['seq'] = c.get('seq', 0) + 1; c['updated'] = time.time()
    tmp = PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(c, f, indent=1)
    os.replace(tmp, PATH)


def main(a):
    c = load(); cmd = a[0] if a else 'status'
    if cmd == 'status':
        print(json.dumps(c, indent=1)); return
    if cmd == 'pause': c['paused'] = True
    elif cmd == 'resume': c['paused'] = False
    elif cmd == 'kill': c['kill'] = True
    elif cmd == 'unkill': c['kill'] = False
    elif cmd == 'pause-sym': c['sym_pause'][a[1]] = time.time() + float(a[2] if len(a) > 2 else 3600)
    elif cmd == 'unpause-sym': c['sym_pause'].pop(a[1], None)
    elif cmd == 'size': c['size_scale'] = float(a[1])
    elif cmd == 'fair': c['fair_override'][a[1]] = [float(a[2]), time.time() + float(a[3] if len(a) > 3 else 60)]
    elif cmd == 'adjust': c['adjust'][a[1]] = [a[2], float(a[3])]
    elif cmd == 'param': c['params'][a[1]] = float(a[2])
    elif cmd == 'reset-adjustments': c['adjust'] = {}; c['params'] = {}; c['fair_override'] = {}; c['reset_adj'] = time.time()
    elif cmd == 'etf': c['etf'] = (a[1] == 'on')
    else:
        print(__doc__); return
    save(c); print('ok', cmd, a[1:])


if __name__ == '__main__':
    main(sys.argv[1:])

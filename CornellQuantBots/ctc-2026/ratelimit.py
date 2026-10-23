"""Order-action rate limiting shared by every process of ours on this machine.

Exchange limits: 60 order actions/min per key, 180/min per team (all keys, all agents).
Teammates' bots consume the same team budget, so we only allow ourselves `team_share` per minute.
The state lives in a small JSON file protected by an exclusive file lock (fcntl, works on macOS),
so several of our processes (main bot, ETF module, manual tools) cannot exceed the budget together.
"""
import fcntl, json, os, time

WINDOW = 60.0


class SharedRateLimiter:
    def __init__(self, path, team_share=100, per_key=55, n_keys=3):
        self.path = path; self.team_share = team_share; self.per_key = per_key; self.n_keys = n_keys
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump({'events': []}, f)

    def _locked(self, fn):
        with open(self.path, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                try:
                    st = json.load(f)
                except json.JSONDecodeError:
                    st = {'events': []}
                now = time.time()
                st['events'] = [e for e in st['events'] if now - e[0] < WINDOW]
                res = fn(st, now)
                f.seek(0); f.truncate(); json.dump(st, f)
                return res
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def acquire(self, n=1, priority='normal'):
        """Reserve n actions. Returns key index to use, or None if over budget.
        'normal' requests leave 15% of the budget free for 'urgent' ones (final report, kill switch)."""
        def fn(st, now):
            team_used = len(st['events'])
            cap = self.team_share if priority == 'urgent' else int(self.team_share * 0.85)
            if team_used + n > cap:
                return None
            used = [0] * self.n_keys
            for e in st['events']:
                if 0 <= e[1] < self.n_keys:
                    used[e[1]] += 1
            k = min(range(self.n_keys), key=lambda i: used[i])
            if used[k] + n > self.per_key:
                return None
            st['events'] += [[now, k]] * n
            return k
        return self._locked(fn)

    def usage(self):
        return self._locked(lambda st, now: len(st['events']))

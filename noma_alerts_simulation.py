from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

STEP_MIN = 5                 # the patch samples every 5 minutes
SPD = 24 * 60 // STEP_MIN    # 288 samples in a day

# Rule A, as described in the prompt. I am guessing at the exact cutoffs, which
# is the first thing I would confirm with the team.
POP_THRESHOLDS = {"hr": 100.0, "rr": 24.0, "temp": 38.0}

# Rule B. A deviation has to be unusual for this patient (3 robust SDs) AND big
# enough that a nurse would care (the floors). Three SDs of a very steady
# patient can be 4 bpm, which is nothing.
# TODO: these floors are my guess. A clinician should set them.
DEVIATION_SD = 3.0
FLOORS = {"hr": 12.0, "rr": 4.0, "temp": 0.5}

# How long before the same problem is allowed to alert again.
REFRACTORY_A = 30     # a threshold crossing is a moment
REFRACTORY_B = 240    # a sustained state should not re-page every hour


@dataclass
class Patient:
    hr: np.ndarray
    rr: np.ndarray
    temp: np.ndarray
    onset: int | None      # index where the decline starts, None if they stay well
    ramp: int              # samples from onset to "a nurse would see this anyway"

    @property
    def apparent(self):
        # The moment the patch stops being worth anything, because the
        # deterioration is now obvious. Warning time is measured back from here.
        return None if self.onset is None else self.onset + self.ramp


def simulate_cohort(n_patients=40, days=7, n_events=8, ramp_hours=6, seed=7):
    rng = np.random.default_rng(seed)
    T = days * SPD
    ramp = int(ramp_hours * 60 / STEP_MIN)
    sick = set(rng.choice(n_patients, n_events, replace=False).tolist())
    cohort = []

    for p in range(n_patients):
        # resting values, then sensor and physiological noise on top
        hr = rng.normal(76, 9) + rng.normal(0, 3.0, T)
        rr = rng.normal(16, 2.5) + rng.normal(0, 1.1, T)
        temp = rng.normal(36.8, .25) + rng.normal(0, .10, T)

        # walking, meals, anxiety, a patch that has come loose
        for _ in range(rng.integers(4, 7) * days):
            i = rng.integers(0, T - 12)
            dur = rng.integers(3, 8)
            hr[i:i + dur] += rng.normal(26, 6)
            rr[i:i + dur] += rng.normal(5, 1.5)
            if rng.random() < .15:
                temp[i:i + dur] += rng.normal(.7, .2)

        onset = None
        if p in sick:
            onset = int(rng.integers(int(1.5 * SPD), int((days - 1.2) * SPD)))
            slope = np.clip((np.arange(T) - onset) / ramp, 0, 1)
            hr += 26 * slope
            rr += 8 * slope
            temp += .9 * slope

        cohort.append(Patient(hr, rr, temp, onset, ramp))
    return cohort


def collapse(indices, refractory_min):
    gap = max(1, refractory_min // STEP_MIN)
    alerts = []
    for i in indices:
        if not alerts or i - alerts[-1] >= gap:
            alerts.append(int(i))
    return alerts


def baseline(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, 1.4826 * mad


def held_for(flag, window_min):
    w = max(1, window_min // STEP_MIN)
    cum = np.concatenate(([0], np.cumsum(flag.astype(int))))
    out = np.zeros(flag.size, dtype=bool)
    out[w - 1:] = (cum[w:] - cum[:-w]) == w     # answer lands at the window's end
    return out


def rule_a(p):
    fire = ((p.hr > POP_THRESHOLDS["hr"])
            | (p.rr > POP_THRESHOLDS["rr"])
            | (p.temp > POP_THRESHOLDS["temp"]))
    return collapse(np.flatnonzero(fire), REFRACTORY_A)


def rule_b(p, persist_min=45, n_vitals=1, tiers=False):
    flags = []
    for name, series in (("hr", p.hr), ("rr", p.rr), ("temp", p.temp)):
        med, sd = baseline(series[:SPD])                  # first 24h of wear
        cutoff = max(DEVIATION_SD * sd, FLOORS[name])
        flags.append(held_for(series - med > cutoff, persist_min))

    n_flagged = np.sum(flags, axis=0)
    alerts = collapse(np.flatnonzero(n_flagged >= n_vitals), REFRACTORY_B)
    if not tiers:
        return alerts
    # how the alert reaches a human: two or more vitals wakes someone up,
    # one vital can wait for the shift digest
    return [(i, "page" if n_flagged[i] >= 2 else "digest") for i in alerts]


def evaluate(cohort, alerts_by_patient, grace_hours=2.0, **config):
    grace = int(grace_hours * 60 / STEP_MIN)
    total = true_alerts = caught = 0
    monitored_days = 0.0
    leads = []

    for patient, alerts in zip(cohort, alerts_by_patient):
        if patient.onset is None:
            horizon = patient.hr.size
            total += len(alerts)
        else:
            horizon = patient.apparent + grace
            kept = [i for i in alerts if i <= horizon]
            hits = [i for i in kept if i >= patient.onset]
            total += len(kept)
            true_alerts += len(hits)
            if hits:
                caught += 1
                leads.append((patient.apparent - min(hits)) * STEP_MIN / 60)
        monitored_days += horizon * STEP_MIN / (60 * 24)

    n_events = sum(p.onset is not None for p in cohort)
    return {
        "alerts_per_patient_day": total / monitored_days,
        "share_in_event_window": true_alerts / total if total else 0.0,
        "events_caught": caught / n_events,
        "median_lead_h": float(np.median(leads)) if leads else 0.0,
        "total_alerts": total,
        "monitored_days": round(monitored_days, 1),
        "config": config,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--patients", type=int, default=40)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--events", type=int, default=8)
    ap.add_argument("--persist", type=int, default=45)
    ap.add_argument("--vitals", type=int, default=1)
    args = ap.parse_args()

    cohort = simulate_cohort(args.patients, args.days, args.events, seed=args.seed)

    current = evaluate(cohort, [rule_a(p) for p in cohort])
    proposed = evaluate(cohort, [rule_b(p, args.persist, args.vitals) for p in cohort],
                        persist_min=args.persist, n_vitals=args.vitals)

    # Sweeping both knobs rather than reporting the setting that flatters me.
    # The interesting part is the shape, not any single row.
    sweep = []
    for persist in (15, 30, 45, 60, 90, 120):
        for n_vitals in (1, 2):
            sweep.append(evaluate(cohort, [rule_b(p, persist, n_vitals) for p in cohort],
                                  persist_min=persist, n_vitals=n_vitals))

    print(f"{'rule':<32}{'alerts/pt-day':>14}{'meaningful':>12}{'events':>9}{'warning':>10}")
    for label, s in (("current, population cutoffs", current),
                     (f"proposed, baseline + {args.persist} min", proposed)):
        print(f"{label:<32}{s['alerts_per_patient_day']:>14.2f}{s['share_in_event_window']:>11.0%}"
              f"{s['events_caught']:>9.0%}{s['median_lead_h']:>9.1f}h")

    print("\nsweep")
    for s in sweep:
        c = s["config"]
        print(f"  {c['persist_min']:>3} min, {c['n_vitals']} vital(s): "
              f"{s['alerts_per_patient_day']:>6.2f}/pt-day, "
              f"{s['events_caught']:>4.0%} of events, {s['median_lead_h']:>4.1f}h warning")

    json.dump({"current": current, "proposed": proposed, "sweep": sweep,
               "meta": {"synthetic": True, "patients": args.patients, "days": args.days,
                        "events": args.events, "step_min": STEP_MIN, "ramp_hours": 6,
                        "seed": args.seed, "proposed_persist_min": args.persist,
                        "proposed_n_vitals": args.vitals}},
              open("noma_summary.json", "w"), indent=2)

    # one patient for the figure: the first who actually gets sick
    ex = next(i for i, p in enumerate(cohort) if p.onset is not None)
    json.dump({"hr": cohort[ex].hr.tolist(), "onset": cohort[ex].onset,
               "current": rule_a(cohort[ex]),
               "proposed": rule_b(cohort[ex], args.persist, args.vitals)},
              open("noma_example.json", "w"))


if __name__ == "__main__":
    main()

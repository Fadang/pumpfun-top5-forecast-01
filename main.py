"""pumpfun-top5-forecast-01 agent v7 — Monte Carlo trajectory forecaster
(v6 architecture) with LONG-HORIZON corrections.

v7 changes vs v6 (all horizon-aware; short-horizon behaviour ~unchanged):
  1. ADAPTIVE SOFT CAP: v6 mean-reverted simulated caps toward a hard-coded
     3.9, calibrated on the 50-min practice market (slots 3.0-3.6). The final
     market's slots are ~4.5-5.0, so v6's MC equilibrium sat ~0.3-0.5 below
     the real slot level and its max-selected size centre ran ~+0.35 too
     high in falling-slot worlds (synthetic eval_long: size -23.36 in the
     cap*=3.8 world vs -16.00 in cap*=4.3). v7 estimates the soft cap from
     the input history itself (median place-1 cap over the last 12 marks,
     floored at v6's 3.9 so short-horizon behaviour is unchanged).
  2. SLOT-TREND EXTRAPOLATION: per-place slot levels + a damped linear
     trend of place-1 caps (last 12 marks), extrapolated to each read with
     a saturating factor 12/(12+h) and clamped to [-0.3, +0.9]. It lifts
     the phantom centre and centres a new BROAD size component.
  3. SIZE WIDTH GROWS WITH HORIZON: prior-mixture sd 0.30 ->
     0.30 + 0.015*sqrt(mean_h); plus a broad Normal component centred on
     the extrapolated slot level (sd 0.45) with weight min(0.35, mean_h/150).
     At 7-13 h the true read caps spread ~0.8 log10 across reads; a tight
     distribution centred 0.3 off scores catastrophically.
  4. ORDER AT VERY LONG HORIZON: named_cap gains a branch for mean_h > 12
     (0.20, was 0.40): at 7-13 h most top-5 places are newcomers, so named
     mass mostly steals from the unnamed share (synthetic worlds: v6 order
     -24.4 vs -23.03 for naming nothing). The MC order weight also rises
     toward 0.85 there (the stale persistence prior is the worst picker at
     long horizon).
Deterministic: fixed seed, no wall clock, ties by mint ascending.
Stdlib only, Python 3.13.
"""
import bisect
import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
NSIMS_SHORT = 400   # horizons <= 12 steps
NSIMS_LONG = 250    # horizons > 12 steps (final run: 42-78 steps)
MAX_CANDS = 80
KAPPA = 0.25        # per-step pull back toward the adaptive soft cap
W_MC_ORD_MAX = 0.7  # max MC weight for ORDER probs
W_MC_ORD_LONG = 0.85  # MC order weight at very long horizon (prior is stale)
W_MC_SIZE = 0.6  # MC weight for SIZE distributions
SIZE_NAMED = 0.95   # share of size mass named; rest spreads over unnamed bins
MAX_NAMED_PER_PLACE = 12
DRIFT_RATE_MAX = 0.06
DRIFT_TOTAL_MAX = 0.60
# v7 long-horizon knobs
NAMED_CAP_LONG = 0.20   # named order mass cap when mean_h > 12 steps
SIZE_SD_H = 0.015       # prior-size sd grows by this * sqrt(mean_h)
BROAD_SD = 0.45         # sd of the broad size component (extrapolated slot)
BROAD_W_MAX = 0.35      # max weight of the broad size component
TREND_CLAMP_LO = -0.3   # extrapolated slot shift, lower clamp
TREND_CLAMP_HI = 0.9    # extrapolated slot shift, upper clamp
SLOT_WINDOW = 12        # marks used for slot levels / trend
# v4 prior knobs (identical to main_v4.py)
TEMP = 0.18
UNNAMED_PRIOR = 0.45
SIZE_SD = 0.3
SLOPE_CAP = 0.30
POOL_BONUS = 0.08
COMPLETE_BONUS = 0.04
CURVE_PENALTY = -0.05
RANK_KERNEL_SD = 2.5
SLOPE_WINDOW = 4


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def log_cap(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def erfc(x):
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


def norm_cdf(x, mean, sd):
    return 0.5 * erfc(-(x - mean) / (sd * math.sqrt(2.0)))


def bucket(cap):
    # fine buckets: dynamics differ sharply by level (see diag_v6.py):
    #   <1.6 floor mass (sits at ~1.45, churns out), 1.6-2.5 danger zone
    #   (mean -0.21, sd 0.48), 2.5-3.0 pumpers (+0.24/step),
    #   3.0-3.5 steady growth (+0.05/step, never dies in-sample),
    #   3.5+ top coins (same +0.05 drift, mean-reverted above the adaptive soft cap).
    # NOTE: boundary is 1.6, not 2.0: a coin at 1.98 is a young pumper,
    # not a floor coin (the floor mass sits at exactly ~1.45).
    if cap < 1.6:
        return 0
    if cap < 2.5:
        return 1
    if cap < 3.0:
        return 2
    if cap < 3.5:
        return 3
    return 4


def size_bin(x, edges):
    if x < edges[0]:
        return 0
    if x >= edges[-1]:
        return 81
    return bisect.bisect_right(edges, x)


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task.get("output", "forecast.json")

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                marks.append(json.loads(line))
    marks = [m for m in marks if m.get("ok") and m.get("rows")]
    if not marks:
        raise SystemExit("the history holds no ok mark")
    last = marks[-1]
    last_at = parse_time(last["at"])
    hist_until = parse_time(task["history_until"]) if task.get("history_until") else last_at + 600
    reads = task.get("reads") or [hist_until + 600 * (i + 1) for i in range(7)]
    horizons = sorted({max(0, int(round((parse_time(r) - hist_until) / 600.0))) for r in reads})
    if not horizons:
        horizons = [1]
    max_h = horizons[-1]
    h_set = set(horizons)
    mean_h = sum(horizons) / len(horizons)
    named_cap = (0.75 if mean_h <= 1.5 else (0.55 if mean_h <= 3.5
                 else (0.40 if mean_h <= 12 else NAMED_CAP_LONG)))
    # ---- v7: slot levels + trend from the input history itself ----
    # slot_med[k]: median place-k log-cap over the last SLOT_WINDOW marks
    # (robust level of each top slot). slope1: least-squares slope of the
    # place-1 slot over the same window (per step). The extrapolated slot
    # level at horizon h is slot_med[k] + trend_shift(h), with a saturating
    # extrapolation (shift -> slope*12 as h -> inf) clamped to [-0.3, +0.9].
    # The MC's soft cap is this level, floored at v6's 3.9 (so short-horizon
    # behaviour is unchanged: on the practice market the slots sit below
    # 3.9; on the final run the market is 11 h old and its slots sit far
    # above it, where the hard-coded 3.9 actively dragged the MC down).
    # NOTE: every v7 correction below is gated on mean_h > 12 steps. The
    # real-data splits only reach 10-40 min, where v6's MC is already well
    # calibrated; the corrections exist for the 7-13 h final horizon only.
    win = marks[-SLOT_WINDOW:]
    slot_med = {}
    for k in range(1, places + 1):
        vals = sorted(log_cap(m["rows"][k - 1]) for m in win if len(m.get("rows") or []) >= k)
        slot_med[k] = vals[len(vals) // 2] if vals else log_cap(marks[-1]["rows"][k - 1])
    xs = list(range(len(win)))
    ys = [log_cap(m["rows"][0]) for m in win if m.get("rows")]
    n_xy = len(ys)
    if n_xy >= 2:
        mx = sum(xs[:n_xy]) / n_xy
        my = sum(ys) / n_xy
        den = sum((x - mx) ** 2 for x in xs[:n_xy])
        slope1 = (sum((x - mx) * (y - my) for x, y in zip(xs[:n_xy], ys)) / den
                  if den else 0.0)
    else:
        slope1 = 0.0
    slope1 = max(-0.10, min(0.10, slope1))
    soft_cap = max(3.9, slot_med[1])
    long_regime = mean_h > 12

    def trend_shift(h):
        if not long_regime:
            return 0.0
        return max(TREND_CLAMP_LO,
                   min(TREND_CLAMP_HI, slope1 * h * 12.0 / (12.0 + h)))

    def slot_level(k, h):
        return slot_med[k] + trend_shift(h)

    size_sd_eff = SIZE_SD + (SIZE_SD_H * math.sqrt(mean_h) if long_regime else 0.0)
    broad_w = min(BROAD_W_MAX, mean_h / 150.0) if long_regime else 0.0
    # calibration thickness: number of usable consecutive-mark transitions.
    # The MC's per-coin order picks are only trustworthy once its return /
    # death / graduation rates are estimated from several transitions; its
    # aggregate size histogram is more robust, so it keeps a fixed weight.
    n_trans = sum(1 for t in range(len(marks) - 1)
                  if marks[t].get("rows") and marks[t + 1].get("rows"))
    w_ord = W_MC_ORD_MAX * min(1.0, n_trans / 6.0)
    if mean_h > 12:
        # very long horizon: the persistence prior's picks are stale, trust
        # the MC's survivors more (its weight rises toward W_MC_ORD_LONG).
        w_ord = min(W_MC_ORD_LONG, w_ord + 0.15)
    # young-market drift: a fresh market's top-5 slots rise fast as it fills
    # in (~+0.05 log10/step in the practice market's first hour), decaying
    # to zero by age 2 h.  It lifts the phantom centre and the size samples
    # (newcomers arrive at the FUTURE top level), never the ranking.
    # The TOTAL drift is capped (DRIFT_TOTAL_MAX): the warm-up lasts ~2 h,
    # so the drift must not keep growing linearly out to 7-13 h horizons.
    # If t0 is after the last mark the task data is inconsistent -> no drift.
    t0 = parse_time(task["t0"]) if task.get("t0") else parse_time(marks[0]["at"])
    market_age_min = (last_at - t0) / 60.0
    if market_age_min < 0:
        young_drift = 0.0
    else:
        young_drift = 0.05 * max(0.0, 1.0 - market_age_min / 120.0)

    # ---- calibrate dynamics from the input history itself ----
    traj = {}
    for i, m in enumerate(marks):
        for r in m["rows"]:
            traj.setdefault(r["mint"], []).append(
                (i, log_cap(r), r["venue"], bool(r.get("complete")), r["created_at"]))
    returns = {}       # (venue, bucket) -> [log-returns]
    all_ret = {"pool": [], "curve": [], "all": []}
    death = {}         # (venue, bucket) -> [gone, total]
    grad_n = grad_d = 0
    for t in range(len(marks) - 1):
        a = {r["mint"]: r for r in marks[t]["rows"]}
        b = {r["mint"]: r for r in marks[t + 1]["rows"]}
        for mint, ra in a.items():
            key = (ra["venue"], bucket(log_cap(ra)))
            d = death.setdefault(key, [0, 0])
            d[1] += 1
            if mint in b:
                rb = b[mint]
                r = log_cap(rb) - log_cap(ra)
                returns.setdefault(key, []).append(r)
                all_ret.setdefault(ra["venue"], []).append(r)
                all_ret["all"].append(r)
                if ra["venue"] == "curve":
                    grad_d += 1
                    if rb["venue"] == "pool":
                        grad_n += 1
            else:
                d[0] += 1
    for v in ("pool", "curve"):
        if not all_ret.get(v):
            all_ret[v] = [0.05] if v == "pool" else [-0.02]
    if not all_ret["all"]:
        all_ret["all"] = [0.0]

    def ret_pool(venue, cap):
        p = returns.get((venue, bucket(cap)))
        if p and len(p) >= 3:
            return p
        p = all_ret.get(venue)
        if p:
            return p
        return all_ret["all"]

    def p_death(venue, cap):
        gone, tot = death.get((venue, bucket(cap)), [0, 0])
        # light smoothing only: the raw hazard matters (2.5-3 coins die 1/3
        # of the time per step; heavy smoothing made simulated rockets
        # nearly immortal and the MC overconfident).
        p = (gone + 0.5) / (tot + 5.0)
        return min(0.60, max(0.005, p))

    p_grad = (grad_n + 1.0) / (grad_d + 50.0) if grad_d else 0.014

    # newcomer anchor: mean cap of newcomer top-5 coins in the last 3 marks
    seen = set()
    newcomer_caps = []
    for m in marks[-3:]:
        for r in m["rows"][:places]:
            if r["mint"] not in seen:
                newcomer_caps.append(log_cap(r))
        seen |= {r["mint"] for r in m["rows"]}
    if len(newcomer_caps) < 5:
        newcomer_caps = [log_cap(r) for r in marks[-1]["rows"][:places]]
    newcomer_mean = sum(newcomer_caps) / len(newcomer_caps)
    pool_ret = all_ret.get("pool") or [0.05]
    drift_rate = min(DRIFT_RATE_MAX, max(0.0, sum(pool_ret) / len(pool_ret)))
    last_top5 = sorted((log_cap(r) for r in marks[-1]["rows"][:places]), reverse=True)
    last_mid = last_top5[len(last_top5) // 2]

    def phantom_count(h):
        # enough phantoms to fill the top 5 at long horizons: in reality
        # 4-5 of the top 5 at 30-40 min are newcomers, so the sim needs
        # ~5-8 phantoms per read, not 2-3 (with only 2-3, places 4-5 fall
        # to crashed named coins and the size histogram piles up < 2.00).
        return min(10, 4 + int(0.7 * h))

    # chart floor: a coin below the last mark's smallest charted cap has
    # dropped out of the top-200 (dead for our purposes).
    floor_cap = min(log_cap(r) for r in last["rows"])

    # ---- candidates: top MAX_CANDS by last log-cap ----
    last_rows = sorted(last["rows"], key=lambda r: r["rank"])
    cands = [{"mint": r["mint"], "cap": log_cap(r), "venue": r["venue"]}
             for r in last_rows[:MAX_CANDS]]

    # ---- Monte Carlo ----
    rng = random.Random(SEED)
    nsims = NSIMS_SHORT if max_h <= 12 else NSIMS_LONG
    counts = {k: {} for k in range(1, places + 1)}
    size_samples = {k: [] for k in range(1, places + 1)}
    n_reads = len(horizons)

    for _ in range(nsims):
        paths = []
        for c in cands:
            cap = c["cap"]
            venue = c["venue"]
            alive = True
            snap = {}
            if 0 in h_set:
                snap[0] = cap
            nxt = sorted(h for h in h_set if h > 0)
            ni = 0
            for step in range(1, max_h + 1):
                if alive:
                    if rng.random() < p_death(venue, cap):
                        alive = False
                    else:
                        pool = ret_pool(venue, cap)
                        cap = cap + pool[rng.randrange(len(pool))]
                        if cap > soft_cap:
                            cap -= KAPPA * (cap - soft_cap)
                        if cap < floor_cap:
                            alive = False
                        elif venue == "curve" and rng.random() < p_grad:
                            venue = "pool"
                while ni < len(nxt) and nxt[ni] == step:
                    snap[nxt[ni]] = cap if alive else None
                    ni += 1
                if ni >= len(nxt):
                    break
            paths.append((c["mint"], snap))
        for h in horizons:
            scored = []
            for mint, snap in paths:
                cap = snap.get(h)
                if cap is not None:
                    scored.append((cap, mint))
            scored.sort(key=lambda x: (-x[0], x[1] or ""))
            # Phantom centre: the SIMULATED place-1 cap at this read, minus
            # a gap (newcomers that take top-5 places arrive just below the
            # top of the chart: pooled empirical gaps 0.0-0.35, tail to
            # ~0.95), plus the young-market drift (newcomers arrive at the
            # FUTURE top level).  Centring on the simulated median was
            # wrong: the median coin crashes, so phantoms sat ~1.0 too low
            # and the size histogram piled up below 2.00 on young markets.
            if scored:
                top1 = scored[0][0]
                centre = top1 + min(DRIFT_TOTAL_MAX, young_drift * h) + trend_shift(h)
            else:
                centre = (max(last_mid, newcomer_mean)
                          + min(DRIFT_TOTAL_MAX, young_drift * h + drift_rate * h)
                          + trend_shift(h))
            for _ in range(phantom_count(h)):
                gap = abs(rng.gauss(0.35, 0.35))
                scored.append((centre - gap + rng.gauss(0.0, 0.15), None))
            scored.sort(key=lambda x: (-x[0], x[1] or ""))
            for k, (cap, mint) in enumerate(scored[:places], 1):
                # Size samples record the SLOT's cap, not just the coin's:
                # in a young market the top-5 slots rise ~+0.13/step (place-1
                # slot, including replacement by hotter newcomers), faster
                # than any single incumbent coin's +0.05/step drift.  The
                # young-market drift is added to NAMED coins' size samples
                # (phantoms already carry it via their centre, capped the
                # same way).  Ranking is unaffected (uniform within a read).
                size_samples[k].append(cap + min(DRIFT_TOTAL_MAX, young_drift * h) if mint is not None else cap)
                if mint is not None:
                    counts[k][mint] = counts[k].get(mint, 0) + 1

    total = float(nsims * n_reads)
    mc_prob = {k: {} for k in range(1, places + 1)}
    for k in range(1, places + 1):
        for mint, cnt in counts[k].items():
            mc_prob[k][mint] = cnt / total
    mc_size = {k: {} for k in range(1, places + 1)}
    for k in range(1, places + 1):
        cnt = {}
        for x in size_samples[k]:
            b = size_bin(x, edges)
            cnt[b] = cnt.get(b, 0) + 1
        n = len(size_samples[k])
        if n:
            for b, c in cnt.items():
                mc_size[k][b] = c / n

    # ---- v4 persistence prior ----
    prior_cands = []
    for mint, pts in traj.items():
        pts.sort()
        li, lcap, venue, complete, created = pts[-1]
        recent = pts[-SLOPE_WINDOW:]
        n = len(recent)
        if n >= 2:
            xs = [p[0] for p in recent]
            ys = [p[1] for p in recent]
            mx = sum(xs) / n
            my = sum(ys) / n
            den = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
        else:
            slope = 0.0
        slope = max(-0.15, min(0.15, slope))
        mu = lcap + max(-SLOPE_CAP, min(SLOPE_CAP, slope * 0))  # HORIZON_STEPS=0
        if venue == "pool":
            mu += POOL_BONUS
        elif complete:
            mu += COMPLETE_BONUS
        else:
            mu += CURVE_PENALTY
        age_min = (last_at - parse_time(created)) / 60.0
        if age_min <= 15 and slope > 0:
            mu += 0.05
        prior_cands.append({"mint": mint, "mu": mu, "last_cap": lcap})
    prior_cands.sort(key=lambda c: (-c["mu"], c["mint"]))
    for i, c in enumerate(prior_cands):
        c["pred_rank"] = i + 1
    top_prior = prior_cands[:30]
    mu_max = top_prior[0]["mu"]
    prior_prob = {k: {} for k in range(1, places + 1)}
    for k in range(1, places + 1):
        scored = []
        for c in top_prior:
            w = math.exp((c["mu"] - mu_max) / TEMP)
            w *= math.exp(-((c["pred_rank"] - k) ** 2) / (2 * RANK_KERNEL_SD ** 2))
            scored.append((w, c))
        tot = sum(w for w, _ in scored)
        for w, c in sorted(scored, reverse=True)[:MAX_NAMED_PER_PLACE]:
            p = (1.0 - UNNAMED_PRIOR) * w / tot
            if p >= 0.0005:
                prior_prob[k][c["mint"]] = p
    prior_size = {k: {} for k in range(1, places + 1)}
    for k in range(1, places + 1):
        pk = prior_prob[k]
        if not pk:
            continue
        tot_p = sum(pk.values())
        mix = []
        for mint, p in pk.items():
            c = next(c for c in top_prior if c["mint"] == mint)
            # size mixture means carry the same young-market slot drift
            # (capped like everywhere else); order uses the undrifted mu.
            mix.append((p / tot_p, c["mu"] + min(DRIFT_TOTAL_MAX, young_drift * mean_h)))
        for b in range(82):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == 81 else edges[b]
            p = sum(w * (norm_cdf(hi, mu, size_sd_eff) - norm_cdf(lo, mu, size_sd_eff))
                    for w, mu in mix)
            if p >= 0.000001:
                prior_size[k][b] = p

    # ---- blend + output (separate weights: order follows calibration
    # thickness, size keeps a fixed MC weight) ----
    probabilities = []
    for k in range(1, places + 1):
        mints = set(mc_prob[k]) | set(prior_prob[k])
        blended = {}
        for mint in mints:
            p = w_ord * mc_prob[k].get(mint, 0.0) + (1 - w_ord) * prior_prob[k].get(mint, 0.0)
            if p >= 0.0005:
                blended[mint] = p
        ranked = sorted(blended.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_NAMED_PER_PLACE]
        rows = []
        # per-coin cap 0.45, as in v4: no single coin may take more than 45%
        # of a place (it leaves more unnamed mass, which is what scores when
        # the place is taken by a newcomer, and it caps the damage when the
        # confident pick is wrong).
        ranked = [(m, min(p, 0.45)) for m, p in ranked]
        s = sum(p for _, p in ranked)
        if s > named_cap:
            f = named_cap / s
            ranked = [(m, p * f) for m, p in ranked]
        for mint, p in ranked:
            if p >= 0.0005:
                rows.append({"mint": mint, "place": k, "p": round(p, 6)})
        s = sum(r["p"] for r in rows)
        if s > named_cap:
            f = named_cap / s
            for r in rows:
                r["p"] = round(r["p"] * f, 6)
        probabilities.extend(rows)

    # ---- v7 size: re-centre the MC histogram on the extrapolated slot level
    # (the MC's max-selected slot caps run ~+0.3 too high when true slots
    # fall) and mix in a broad Normal centred on that level. ----
    def bin_centre(b):
        if b == 0:
            return 1.975
        if b == 81:
            return 6.025
        return (edges[b - 1] + edges[b]) / 2.0

    mc_size_c = {k: {} for k in range(1, places + 1)}
    for k in range(1, places + 1):
        tot = sum(mc_size[k].values())
        if tot <= 0:
            continue
        mc_mean = sum(p * bin_centre(b) for b, p in mc_size[k].items()) / tot
        # re-centring is a long-regime correction only (see note above).
        shift = 0
        if long_regime:
            shift = int(round((slot_level(k, mean_h) - mc_mean) / 0.05))
            shift = max(-10, min(10, shift))
        for b, p in mc_size[k].items():
            nb = max(0, min(81, b + shift))
            mc_size_c[k][nb] = mc_size_c[k].get(nb, 0.0) + p

    sizes = []
    for k in range(1, places + 1):
        centre_k = slot_level(k, mean_h)
        # iterate all 82 bins: the broad component has mass everywhere
        # (tiny in the far tails); restricting to mc/prior bins would drop
        # part of its weight into the unnamed remainder.
        bins = range(82)
        rows = []
        for b in sorted(bins):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == 81 else edges[b]
            p_broad = norm_cdf(hi, centre_k, BROAD_SD) - norm_cdf(lo, centre_k, BROAD_SD)
            p = ((1.0 - broad_w) * (W_MC_SIZE * mc_size_c[k].get(b, 0.0)
                 + (1 - W_MC_SIZE) * prior_size[k].get(b, 0.0))
                 + broad_w * p_broad) * SIZE_NAMED
            if p >= 0.000001:
                rows.append({"place": k, "bin": b, "p": round(p, 6)})
        ssum = sum(r["p"] for r in rows)
        if ssum > 1.0:
            f = 1.0 / ssum
            for r in rows:
                r["p"] = math.floor(r["p"] * f * 1000000) / 1000000
        sizes.extend(rows)

    agg = {}
    for k in range(1, places + 1):
        for mint, cnt in counts[k].items():
            agg[mint] = agg.get(mint, 0.0) + cnt * (places + 1 - k)
    top5 = [m for m, _ in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))[:places]]
    if len(top5) < places:
        for r in last_rows:
            if r["mint"] not in top5:
                top5.append(r["mint"])
            if len(top5) == places:
                break
    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()

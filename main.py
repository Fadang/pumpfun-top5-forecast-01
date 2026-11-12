"""pumpfun-top5-forecast-01 agent (Python 3.13, stdlib only).

Strategy: persistence + momentum, calibrated on the kit's practice history.
Practice facts that drive it: the top 5 is pool (graduated) coins only after the
first ~30 min; rank persistence rises over time (top-5 overlap 1/5 early, 4/5 late);
new graduates can enter the top 5 within ~5 min of creation; curve coins fade.
So each coin gets a predicted future log-cap mu = last log-cap + damped recent
slope (per-10-min log-cap trend, shrunk) + pool/graduation bonus. Place
probabilities come from a softmax over mu with a rank-proximity kernel, leaving
~50% of every place unnamed (newcomers born after the cut take ~2 of the top 5
in practice). Sizes are a normal mixture over the place's likely coins, sd 0.55
(wider than the house baseline's 0.35: the answer reads are 7-13 h out).
Deterministic: no wall clock, no randomness; ties break by mint ascending.
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR/forecast.json.
"""
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tuned knobs (v1) ---
TEMP = 0.18
UNNAMED = 0.45       # share of each place left unnamed (newcomers / movers)
SIZE_SD = 0.3
SIZE_NAMED = 1.0
HORIZON_STEPS = 0
SLOPE_CAP = 0.30     # max |slope contribution| in log10 units
POOL_BONUS = 0.08
COMPLETE_BONUS = 0.04
CURVE_PENALTY = -0.05
RANK_KERNEL_SD = 2.5
MAX_NAMED_PER_PLACE = 12
SLOPE_WINDOW = 4     # points used for the per-10-min slope fit (99 = whole history)
SIZE_SHIFT = 0.0     # shift added to the size mixture mean (negative = decay)
SIZE_USE_MU = True   # size mixture mean: mu (True) or raw last log-cap (False)


def erfc(x):
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


def norm_cdf(x, mean, sd):
    return 0.5 * erfc(-(x - mean) / (sd * math.sqrt(2.0)))


def log_cap(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def parse_time(s):
    # ISO 8601 UTC -> epoch seconds (stdlib only, no dateutil)
    from datetime import datetime, timezone
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task["places"])
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

    # Per-coin trajectory: list of (mark_index, log_cap, venue, complete, created_at)
    traj = {}
    for i, m in enumerate(marks):
        for r in m["rows"]:
            traj.setdefault(r["mint"], []).append(
                (i, log_cap(r), r["venue"], bool(r.get("complete")), r["created_at"]))

    cands = []
    for mint, pts in traj.items():
        pts.sort()
        li, lcap, venue, complete, created = pts[-1]
        # slope per 10 min from least squares over last SLOPE_WINDOW points
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
        mu = lcap + max(-SLOPE_CAP, min(SLOPE_CAP, slope * HORIZON_STEPS))
        if venue == "pool":
            mu += POOL_BONUS
        elif complete:
            mu += COMPLETE_BONUS
        else:
            mu += CURVE_PENALTY
        age_min = (last_at - parse_time(created)) / 60.0
        if age_min <= 15 and slope > 0:
            mu += 0.05
        cands.append({"mint": mint, "mu": mu, "last_cap": lcap, "venue": venue,
                      "age": age_min, "n": len(pts)})
    # predicted order: mu desc, mint asc for ties
    cands.sort(key=lambda c: (-c["mu"], c["mint"]))
    for i, c in enumerate(cands):
        c["pred_rank"] = i + 1
    top = cands[:30]
    mu_max = top[0]["mu"]

    probabilities = []
    for k in range(1, places + 1):
        scored = []
        for c in top:
            w = math.exp((c["mu"] - mu_max) / TEMP)
            w *= math.exp(-((c["pred_rank"] - k) ** 2) / (2 * RANK_KERNEL_SD ** 2))
            scored.append((w, c))
        tot = sum(w for w, _ in scored)
        named_mass = 1.0 - UNNAMED
        named = []
        for w, c in sorted(scored, reverse=True)[:MAX_NAMED_PER_PLACE]:
            p = named_mass * w / tot
            if p > 0.45:
                p = 0.45
            if p >= 0.0005:
                named.append({"mint": c["mint"], "place": k, "p": round(p, 6)})
        # renormalise after cap/rounding so sum stays <= named_mass
        s = sum(r["p"] for r in named)
        if s > named_mass:
            f = named_mass / s
            for r in named:
                r["p"] = round(r["p"] * f, 6)
        probabilities.extend(named)

    # sizes: per place, mixture of Normals over the place's likely coins
    sizes = []
    for k in range(1, places + 1):
        named_k = [r for r in probabilities if r["place"] == k]
        if not named_k:
            continue
        tot_p = sum(r["p"] for r in named_k)
        mix = []
        for r in named_k:
            c = next(c for c in top if c["mint"] == r["mint"])
            mix.append((r["p"] / tot_p,
                        (c["mu"] if SIZE_USE_MU else c["last_cap"]) + SIZE_SHIFT))
        for b in range(82):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == 81 else edges[b]
            p = 0.0
            for w, mu in mix:
                p += w * (norm_cdf(hi, mu, SIZE_SD) - norm_cdf(lo, mu, SIZE_SD))
            p *= SIZE_NAMED
            if p >= 0.000001:
                sizes.append({"place": k, "bin": b, "p": round(p, 6)})
        # keep per-place size sum <= 1 (rounding can push it over when SIZE_NAMED=1)
        place_sizes = [s for s in sizes if s["place"] == k]
        ssum = sum(s["p"] for s in place_sizes)
        if ssum > 1.0:
            f = 1.0 / ssum
            for s in place_sizes:
                # floor (not round) so the rescaled sum cannot exceed 1.0
                s["p"] = math.floor(s["p"] * f * 1000000) / 1000000
    top5 = [c["mint"] for c in cands[:places]]
    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exact binomial intervals for closed-loop success counts, generated from artifacts.

Project rule: prose never carries a typed number. A success rate of 0/61 has an upper bound that a
reader will want, and that bound must come from a script reading the artifact, not from arithmetic
done in a chat window. This emits, for every closed-loop ``result.json`` given, the exact
(Clopper-Pearson) interval on the success count together with the count it was computed from.

Why exact rather than bootstrap: with three seeds, percentile bootstrap intervals are not certified
-- Agarwal et al. (2021) report they under-cover below ten runs -- and with zero successes a
bootstrap over episodes is degenerate, since every resample also contains zero successes. The exact
binomial interval has neither problem: it is a statement about a single binomial proportion and is
valid at any n, including n with zero events.

Clopper-Pearson is implemented directly from the binomial CDF (``math.comb``), so there is no SciPy
dependency and the arithmetic is exact rational until the final float comparison. For the k = 0 case
the closed form ``1 - alpha**(1/n)`` is used and cross-checked against the search, because that case
is the one this project actually needs and a silent disagreement there would be worth knowing about.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Exact terms, summed low to high."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(1.0, total)


def clopper_pearson(k: int, n: int, confidence: float = 0.95, *, two_sided: bool = True
                    ) -> tuple[float, float]:
    """Exact binomial interval by bisection on the binomial CDF.

    Lower limit solves P(X >= k | p) = alpha; upper limit solves P(X <= k | p) = alpha.
    """
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, {n}]")
    alpha = (1.0 - confidence) / (2.0 if two_sided else 1.0)

    if k == 0:
        lo = 0.0
    else:
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2.0
            # P(X >= k) = 1 - P(X <= k-1)
            if 1.0 - binom_cdf(k - 1, n, mid) > alpha:
                b = mid
            else:
                a = mid
        lo = a

    if k == n:
        hi = 1.0
    else:
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2.0
            if binom_cdf(k, n, mid) > alpha:
                a = mid
            else:
                b = mid
        hi = b
    return lo, hi


def summarize(k: int, n: int) -> dict[str, Any]:
    lo95, hi95 = clopper_pearson(k, n, 0.95, two_sided=True)
    _, hi95_one = clopper_pearson(k, n, 0.95, two_sided=False)
    out: dict[str, Any] = {
        "successes": k,
        "episodes": n,
        "point_estimate": k / n if n else None,
        "exact_95_two_sided": [lo95, hi95],
        "exact_95_upper_one_sided": hi95_one,
    }
    if k == 0:
        # Closed forms for the zero-event case, as an independent check on the search.
        closed_two = 1.0 - (0.025 ** (1.0 / n))
        closed_one = 1.0 - (0.05 ** (1.0 / n))
        out["closed_form_check"] = {
            "two_sided_upper": closed_two,
            "one_sided_upper": closed_one,
            "agrees_with_search": (abs(closed_two - hi95) < 1e-9
                                   and abs(closed_one - hi95_one) < 1e-9),
        }
        out["rule_of_three_approx"] = 3.0 / n
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, nargs="*",
                    help="closed-loop result.json files; each must carry aggregate.success_rate "
                         "and aggregate.episodes_evaluated")
    ap.add_argument("--count", nargs=2, type=int, metavar=("K", "N"),
                    help="compute for a bare count instead of reading artifacts")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report: dict[str, Any] = {"gate": "success_interval_v1", "entries": []}

    if args.count:
        k, n = args.count
        report["entries"].append({"source": "explicit_count", **summarize(k, n)})

    for path in args.results:
        data = json.loads(path.read_text())
        agg = data.get("aggregate")
        if agg is None:
            print(f"[skip] {path}: no aggregate block", flush=True)
            continue
        n = int(agg["episodes_evaluated"])
        rate = agg.get("success_rate")
        if rate is None:
            print(f"[skip] {path}: no success_rate", flush=True)
            continue
        k = int(round(float(rate) * n))
        if abs(float(rate) * n - k) > 1e-6:
            raise SystemExit(
                f"{path}: success_rate {rate} x {n} episodes is not an integer count; refusing to "
                "guess the numerator."
            )
        report["entries"].append({
            "source": str(path),
            "run_id": data.get("run_id"),
            "checkpoint_update": data.get("checkpoint_update"),
            **summarize(k, n),
        })

    for e in report["entries"]:
        lo, hi = e["exact_95_two_sided"]
        print(f"{e.get('run_id') or e['source']}: {e['successes']}/{e['episodes']} "
              f"= {e['point_estimate']:.4f}; exact 95% CI [{lo:.4f}, {hi:.4f}]; "
              f"one-sided 95% upper {e['exact_95_upper_one_sided']:.4f}", flush=True)
        if "closed_form_check" in e:
            c = e["closed_form_check"]
            print(f"    zero-event closed form: two-sided {c['two_sided_upper']:.6f}, "
                  f"one-sided {c['one_sided_upper']:.6f}, agrees={c['agrees_with_search']}; "
                  f"rule of three {e['rule_of_three_approx']:.6f}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[done] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

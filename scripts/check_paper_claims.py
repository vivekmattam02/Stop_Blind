#!/usr/bin/env python3
"""Check that numbers written in the manuscripts match the artifacts and code they come from.

Prose in this project is supposed to carry no typed numbers, but a LaTeX macro can go stale and a
hand-written sentence can slip in. This walks both manuscripts, finds each load-bearing figure, and
compares it against the artifact or source file it is supposed to have come from.

Searches ALL section files of a manuscript rather than a guessed one: an earlier version of this
check looked for a value in the wrong file and reported a mismatch that did not exist, which is a
worse failure than not checking at all.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tex_corpus(*globs: str) -> str:
    """All LaTeX text for a manuscript, concatenated."""
    out = []
    for g in globs:
        for p in sorted(ROOT.glob(g)):
            out.append(p.read_text(encoding="utf-8"))
    return "\n".join(out)


def has_number(corpus: str, value: int | float, *, decimals: int | None = None) -> bool:
    """True if the value appears, tolerating LaTeX thousands separators and $ math."""
    if decimals is not None:
        return f"{value:.{decimals}f}" in corpus
    plain = str(int(value))
    latex = f"{int(value):,}".replace(",", "{,}")
    comma = f"{int(value):,}"
    return any(v in corpus for v in (plain, latex, comma))


# StopBlind release build: the paper-2 checks in the working copy of this script cover an
# unpublished campaign and are omitted here. The paper-1 checks below are unchanged.
def main() -> int:
    rows: list[tuple[bool, str, str, str]] = []

    def chk(ok: bool, name: str, paper: str, source: str) -> None:
        rows.append((bool(ok), name, paper, source))

    # ---------------- Paper 1 ----------------
    p1 = tex_corpus("paper/sections/*.tex", "paper/main.tex")
    figs = json.loads(
        (ROOT / "artifacts/derived/action_prior_values.json").read_text())["figures"]
    prior = next(f for f in figs if "inflection_coefficient" in f)
    chk(has_number(p1, prior["inflection_coefficient"], decimals=2), "P1 inflection coefficient",
        "2.55", f"{prior['inflection_coefficient']:.2f}")
    # The manuscript must state the ratio the treated arm ACTUALLY applied, computed on its own
    # 3,072-decision training set. The split-level ratio (37.4) was never applied to the loss; this
    # check previously enforced that number and so held a wrong claim in place. See CR-047.
    bal = json.loads(
        (ROOT / "artifacts/spark/native_c0_action_balance_v1/manifest.json").read_text())
    w = bal["action_balance_contract"]["weights"]["weights"]
    applied = w["STOP"] / w["MOVE_FORWARD"]
    chk(has_number(p1, applied, decimals=1), "P1 applied STOP/FORWARD weight",
        f"{applied:.1f}", f"{applied:.4f} from action_balance_contract.weights")

    # Every generated macro the manuscript uses must exist in the generated values file.
    gen_dir = ROOT / "paper/generated"
    macro_files = list(gen_dir.glob("*.tex")) if gen_dir.is_dir() else []
    defined = set()
    for f in macro_files:
        defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read_text()))
    used = set(re.findall(r"\\(Pr[A-Z]\w+|C0[A-Z]\w+)\{?\}?", p1))
    missing = sorted(u for u in used if u not in defined)
    chk(not missing, "P1 macros all defined",
        f"{len(used)} macros used", f"{len(defined)} defined; missing: {missing[:4] or 'none'}")

    width = max(len(r[1]) for r in rows)
    for ok, name, paper, source in rows:
        print(("  OK   " if ok else "  FAIL ") + f"{name:<{width}}  paper: {paper:<28} source: {source}")
    passed = sum(r[0] for r in rows)
    print()
    print(("PASS " if passed == len(rows) else "*** MISMATCH *** ") + f"({passed}/{len(rows)})")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

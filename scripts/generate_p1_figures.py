#!/usr/bin/env python3
"""Generate the StopBlind manuscript's figures from verified artifacts only.

Two panels the paper argues in prose but never showed:

``p1_blind_metric``  the selection signal improving while the behaviour that decides the task
                     vanishes -- cross-entropy falling monotonically while predicted STOP goes to
                     zero and stays there. This is the paper's claim in one figure.
``p1_trajectories``  what the selected policy then does in closed loop: distance to the goal over
                     the full 500-step cap, per episode, showing approach without termination.

Each figure writes a provenance sidecar naming every artifact and field it was built from.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPARK = Path("artifacts/spark")
C_CE, C_STOP, C_GOAL = "#0072B2", "#D55E00", "#009E73"
# Treatment identity, used identically in every panel it appears in. Both slots clear the
# CVD-separation and 3:1 contrast checks against white; line style repeats the distinction.
C_UNWEIGHTED, C_WEIGHTED = "#0072B2", "#D55E00"

plt.rcParams.update({
    # STIX matches the Times text of the NeurIPS style, so figure labels read as part of the page.
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    # TrueType (Type 42) glyphs, not Type 3: camera-ready checks flag Type 3 fonts in a PDF.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _stop_count(cal: dict[str, Any]) -> int:
    return sum(1 for d in cal["per_decision"]
               if d["action_decision"]["predicted_action"] == "STOP")


def fig_blind_metric(out: Path) -> dict[str, Any]:
    """Selection CE against predicted STOP, over training, for every C0 run that reported.

    Colour encodes the TREATMENT and means the same thing in both panels, so a reader who learns
    "blue is unweighted" on the top panel keeps that on the bottom. Line style repeats the same
    distinction, so the figure survives greyscale printing and colour-vision deficiency.

    The CE panel is clipped to the band where the runs actually separate. Every run starts at
    5.49 nats at update 0 -- one segment that would otherwise occupy four fifths of the panel and
    compress the entire result into a flat smear. The off-scale start is annotated, not hidden.
    """
    # Explicit, not a glob. The seed-20260825 pair does not follow the seed_replication naming
    # (`native_c0_development_v1` and `native_c0_action_balance_v1`), so a glob on that prefix
    # silently drops a third of the study and yields a two-seed figure for a three-seed paper.
    C0_RUNS = (
        "native_c0_development_v1",                                 # unweighted, seed 20260825
        "native_c0_seed_replication_unweighted_s20260826_v1",
        "native_c0_seed_replication_unweighted_s20260827_v1",
        "native_c0_action_balance_v1",                              # weighted, seed 20260825
        "native_c0_seed_replication_inverse_frequency_s20260826_v1",
        "native_c0_seed_replication_inverse_frequency_s20260827_v1",
    )
    runs = [str(SPARK / d / "result.json") for d in C0_RUNS]
    missing = [r for r in runs if not Path(r).is_file()]
    if missing:
        raise SystemExit(f"selection-surface runs missing: {missing}")
    fig, (ax, axs) = plt.subplots(2, 1, figsize=(2.67, 2.50), sharex=True,
                                  gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.16})
    used: list[str] = []
    ce0: list[float] = []
    best = {"ce": None, "run": None}
    stop_dead: list[int] = []
    for path in runs:
        res = json.loads(Path(path).read_text())
        cals = res.get("calibrations") or []
        if not cals:
            continue
        used.append(path)
        name = Path(path).parent.name
        weighted = "inverse_frequency" in name or "action_balance" in name
        u = [c["update_step"] for c in cals]
        ce = [c["summary"]["mean_restricted_action_ce"] for c in cals]
        st = [_stop_count(c) for c in cals]
        ce0.append(ce[0])
        colour = C_WEIGHTED if weighted else C_UNWEIGHTED
        style = dict(lw=1.2, color=colour, ls=(0, (3.2, 1.6)) if weighted else "-",
                     solid_capstyle="round")
        ax.plot(u, ce, **style)
        axs.plot(u, st, **style)
        if not weighted:
            if best["ce"] is None or min(ce) < best["ce"]:
                best["ce"] = min(ce)
            nz = [uu for uu, s in zip(u, st) if s > 0]
            stop_dead.append(min([uu for uu, s in zip(u, st) if s == 0], default=u[-1])
                             if not nz[1:] else max(nz))

    for a in (ax, axs):
        a.grid(axis="y", color="#E9E9E9", lw=0.5)
        a.set_axisbelow(True)
        a.spines["right"].set_visible(False)

    ax.set_ylim(0.60, 1.75)
    # Without explicit ticks matplotlib starts labelling at 1.00, leaving the 0.7--0.8 band
    # unlabelled -- the band containing the selection minimum the whole paper is about.
    ax.set_yticks([0.75, 1.00, 1.25, 1.50, 1.75])
    ax.set_ylabel("selection CE (nats)", fontsize=6.5)
    if ce0:
        ax.annotate(f"all runs start at {max(ce0):.2f} nats, above this panel",
                    xy=(0.025, 0.015), xycoords="axes fraction", ha="left", va="bottom",
                    fontsize=6, color="#666666",
                    bbox=dict(fc="white", ec="none", pad=1.2))
    ax.plot([], [], color=C_UNWEIGHTED, lw=1.2, label="unweighted (3 seeds)")
    ax.plot([], [], color=C_WEIGHTED, lw=1.2, ls=(0, (3.2, 1.6)),
            label="weighted (3 seeds)")
    ax.legend(loc="upper right", handlelength=2.2, frameon=True, framealpha=0.92,
              edgecolor="none", facecolor="white", borderpad=0.35).set_zorder(6)

    axs.set_ylim(-2.0, 37)
    axs.set_ylabel("STOP predicted\n(of 256)", fontsize=6.5)
    axs.set_xlabel("optimizer update", fontsize=6.5)
    axs.axhline(0, color="#BBBBBB", lw=0.6, zorder=1)
    # No in-panel annotation for the STOP collapse: at this column width it overruns the axes,
    # and the caption already states it ("falls from 28 to zero in all three unweighted runs and
    # never returns"). A figure should not repeat its own caption to say the same thing worse.
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return {"figure": out.name, "sources": used,
            "ce_axis_clipped_to": [0.60, 1.75], "ce_at_update_0": max(ce0) if ce0 else None,
            "fields": ["calibrations[].update_step",
                       "calibrations[].summary.mean_restricted_action_ce",
                       "calibrations[].per_decision[].action_decision.predicted_action"]}

def fig_trajectories(out: Path, run: Path, goal_radius_m: float = 3.0,
                    clip_m: float = 30.0) -> dict[str, Any]:
    """Distance to goal over the step cap, per episode, for one closed-loop run.

    Clipped to ``clip_m``. A single episode wanders to 78 m; drawn to full scale it pushes the
    other sixty into the bottom fifth of the panel and the goal radius onto the axis line. The
    excursion is annotated with its true peak rather than silently cropped.
    """
    eps = sorted(glob.glob(str(run / "episodes" / "*.json")))
    fig, ax = plt.subplots(figsize=(2.67, 1.82))
    entered = 0
    n = 0
    above: list[float] = []
    for f in eps:
        d = json.loads(Path(f).read_text())
        dist = [s["distance_to_goal"] for s in d["steps"] if s.get("distance_to_goal") is not None]
        if not dist:
            continue
        n += 1
        near = min(dist) <= goal_radius_m
        entered += bool(near)
        if max(dist) > clip_m:
            above.append(max(dist))
        ax.plot(range(len(dist)), dist, lw=0.6 if near else 0.5, alpha=0.85 if near else 0.30,
                color=C_GOAL if near else "#9A9A9A", zorder=3 if near else 1)
    ax.set_ylim(0, clip_m)
    ax.grid(axis="y", color="#E9E9E9", lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["right"].set_visible(False)
    ax.axhline(goal_radius_m, color=C_STOP, lw=0.9, ls=":", zorder=4)
    ax.annotate(f"{goal_radius_m:.0f} m goal radius", (0.988, goal_radius_m + 0.55),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=6, color=C_STOP, zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.0))
    if above:
        ax.annotate(f"{len(above)} episode leaves the panel, peaking at {max(above):.0f} m"
                    if len(above) == 1 else
                    f"{len(above)} episodes leave the panel, peaking at {max(above):.0f} m",
                    xy=(0.985, 0.965), xycoords="axes fraction", ha="right", va="top",
                    fontsize=6, color="#666666",
                    bbox=dict(fc="white", ec="none", pad=1.0))
    ax.plot([], [], color=C_GOAL, lw=1.2, label=f"enters the radius ({entered})")
    ax.plot([], [], color="#9A9A9A", lw=1.2, label=f"does not ({n - entered})")
    ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.4,
              bbox_to_anchor=(0.0, 0.93), frameon=True, framealpha=0.92,
              edgecolor="none", facecolor="white", borderpad=0.35).set_zorder(6)
    ax.set_xlabel("step (cap 500)", fontsize=6.5)
    ax.set_ylabel("distance to goal (m)", fontsize=6.5)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return {"figure": out.name, "sources": [str(run)], "episodes": n,
            "episodes_entering_radius": entered, "goal_radius_m": goal_radius_m,
            "y_axis_clipped_to_m": clip_m,
            "episodes_exceeding_clip": len(above),
            "max_distance_m": round(max(above), 1) if above else None,
            "fields": ["steps[].distance_to_goal"]}

def fig_stop_moments(out: Path, run: Path, index_path: Path, object_root: Path,
                    want: int = 4) -> dict[str, Any]:
    """The agent's own view at every moment the expert says STOP, and what the model said instead.

    This is the paper's claim as an image rather than a number: at the goal, with the target in
    view, the selected checkpoint emits MOVE_FORWARD.

    The checkpoint shown is the one the paper's own selection rule picks -- the lowest selection
    cross-entropy on the frozen surface -- not the last one trained. Every STOP-target decision
    that checkpoint faced is shown, so the panel is a complete enumeration rather than a sample;
    there is nothing to curate and nothing omitted. ``want`` caps the panel count if a future
    surface carries more of them.
    """
    import torch
    res = json.loads((run / "result.json").read_text())
    cals = res.get("calibrations") or []
    if not cals:
        raise SystemExit("run reports no calibrations")
    sel = min(cals, key=lambda c: c["summary"]["mean_restricted_action_ce"])
    pool = [d for d in sel["per_decision"] if d["target_action"] == "STOP"]
    total_on_surface = len(pool)
    if not pool:
        raise SystemExit("no STOP-target decisions on the selection surface")

    by_key = {e["cache_key"]: e["relative_path"]
              for e in json.loads(index_path.read_text())["entries"]}
    pool = [d for d in pool if d["cache_key"] in by_key][:want]
    if not pool:
        raise SystemExit(
            f"none of the STOP-target cache keys resolve in {index_path}; the run and the cache "
            "index do not correspond")
    fig, axes = plt.subplots(1, len(pool), figsize=(1.687 * len(pool), 1.95))
    axes = [axes] if len(pool) == 1 else list(axes)
    used = [str(run / "result.json")]
    for ax, d in zip(axes, pool):
        rel = by_key.get(d["cache_key"])
        obj = torch.load(object_root / rel, map_location="cpu", weights_only=False)
        used.append(str(object_root / rel))
        ax.imshow(obj["rgb_frames"][-1].numpy())    # the frame the decision is made on
        pred = d["action_decision"]["predicted_action"]
        scores = d["action_decision"]["scores"]
        p_stop = scores["STOP"]["restricted_probability"]
        p_pred = scores[pred]["restricted_probability"]
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"expert: STOP\nmodel: {pred.replace('_',' ').lower()} ({p_pred:.2f})",
                     fontsize=6.5, pad=3)
        ax.set_xlabel(f"P(STOP) = {p_stop:.3f}", fontsize=6.5, labelpad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(0.8)
    shown = len(pool)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return {"figure": out.name, "sources": used, "decisions_shown": shown,
            "selected_update_step": sel["update_step"],
            "selected_by": "lowest summary.mean_restricted_action_ce across calibrations",
            "selection_ce": round(sel["summary"]["mean_restricted_action_ce"], 4),
            "stop_target_decisions_on_surface": total_on_surface,
            "is_complete_enumeration": shown == total_on_surface,
            "predicted": [d["action_decision"]["predicted_action"] for d in pool],
            "p_stop": [d["action_decision"]["scores"]["STOP"]["restricted_probability"]
                       for d in pool]}
def fig_metric_decomposition(out: Path, run: Path) -> dict[str, Any]:
    """What the selection metric is made of, split by the action the expert asked for.

    The selection number is a mean over decisions, so each expert action contributes its own mean
    cross-entropy weighted by how often that action is asked for. Those four contributions sum to
    the reported selection CE exactly -- the figure asserts an identity, not an estimate, and the
    provenance records the residual so a reader can check it.

    This is the paper's argument as arithmetic. STOP carries by far the worst per-decision CE and
    is never predicted, yet it is rare enough that it moves only a small fraction of the number
    used to choose the checkpoint.
    """
    import statistics
    res = json.loads((run / "result.json").read_text())
    cals = res.get("calibrations") or []
    if not cals:
        raise SystemExit("run reports no calibrations")
    sel = min(cals, key=lambda c: c["summary"]["mean_restricted_action_ce"])
    dec = sel["per_decision"]
    n_all = len(dec)
    reported = sel["summary"]["mean_restricted_action_ce"]

    order = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
    rows = []
    for act in order:
        ces = [d["restricted_action_ce"] for d in dec if d["target_action"] == act]
        if not ces:
            continue
        mean_ce = statistics.mean(ces)
        rows.append({"action": act, "n": len(ces), "mean_ce": mean_ce,
                     "contribution": len(ces) / n_all * mean_ce})
    total = sum(r["contribution"] for r in rows)
    stop = next((r for r in rows if r["action"] == "STOP"), None)
    n_pred_stop = sum(1 for d in dec
                      if d["action_decision"]["predicted_action"] == "STOP")

    y = list(range(len(rows)))[::-1]                       # first row at the top
    hi = "#D55E00"
    mute = "#B4B4B4"
    colours = [hi if r["action"] == "STOP" else mute for r in rows]
    labels = [r["action"].replace("_", " ").lower() for r in rows]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 1.95), sharey=True,
                                   gridspec_kw={"width_ratios": [1.0, 1.5], "wspace": 0.12,
                                                "top": 0.80, "bottom": 0.24})
    axl.barh(y, [r["n"] for r in rows], color=colours, height=0.62)
    axr.barh(y, [r["contribution"] for r in rows], color=colours, height=0.62)
    for a in (axl, axr):
        a.set_yticks(y)
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0)
        a.grid(axis="x", color="#ECECEC", lw=0.5)
        a.set_axisbelow(True)
    axl.set_yticklabels(labels, fontsize=7)
    axl.set_xlim(0, max(r["n"] for r in rows) * 1.30)
    axr.set_xlim(0, max(r["contribution"] for r in rows) * 1.42)
    renderer = fig.canvas.get_renderer()
    for yy, r in zip(y, rows):
        axl.text(r["n"] + n_all * 0.018, yy, f"{r['n']}", va="center", fontsize=6.5,
                 color="#333333")
        value = axr.text(r["contribution"] + total * 0.022, yy,
                         f"{r['contribution']:.3f}", va="center", fontsize=6.5, color="#333333")
        # The per-decision mean follows its value at a fixed gap, so every row reads the same way.
        end = axr.transData.inverted().transform(value.get_window_extent(renderer))[1][0]
        axr.text(end + total * 0.012, yy, f"({r['mean_ce']:.2f} each)", va="center",
                 fontsize=6, color="#8A8A8A")
    axl.set_xlabel(f"decisions the expert\nasked for (of {n_all})", fontsize=6.5,
                   linespacing=1.4, labelpad=4)
    axr.set_xlabel("contribution to the\nselection CE (nats)", fontsize=6.5,
                   linespacing=1.4, labelpad=4)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return {"figure": out.name, "sources": [str(run / "result.json")],
            "selected_update_step": sel["update_step"],
            "selection_ce_reported": round(reported, 4),
            "selection_ce_reconstructed": round(total, 4),
            "reconstruction_residual": round(abs(total - reported), 8),
            "decision_count": n_all,
            "stop_predicted_count": n_pred_stop,
            "per_action": [{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in r.items()} for r in rows],
            "fields": ["calibrations[].per_decision[].target_action",
                       "calibrations[].per_decision[].restricted_action_ce",
                       "calibrations[].summary.mean_restricted_action_ce"]}


def fig_selection_schematic(out: Path, run: Path, closed_loop: Path,
                            arm: str = "unweighted_s20260826") -> dict[str, Any]:
    """What checkpoint selection measures, against what deployment does.

    Two block diagrams on one scale. Selection is open loop: every input comes from the expert's
    recorded episode. Deployment is closed loop: the policy's action goes to the simulator, which
    returns the next view. Every number is read from the artifacts at draw time and checked
    against the statement it supports, so the drawing cannot drift from the runs.
    """
    from matplotlib.patches import FancyArrowPatch, Rectangle

    res = json.loads((run / "result.json").read_text())
    cals = res["calibrations"]
    sel = min(cals, key=lambda c: c["summary"]["mean_restricted_action_ce"])
    n_dec = len(sel["per_decision"])
    stop_idx = [i for i, d in enumerate(sel["per_decision"]) if d["target_action"] == "STOP"]
    n_stop = len(stop_idx)
    updates = [c["update_step"] for c in cals]

    cl = json.loads((closed_loop / "result.json").read_text())["checkpoint_summaries"][arm]
    hist = cl["action_histogram"]
    n_actions = sum(hist.values())
    n_eps = cl["episode_count"]
    cap = 500
    # The drawing states these outright, so they are checked rather than assumed.
    emitted = sorted(a for a, k in hist.items() if k)
    assert emitted == ["MOVE_FORWARD"], f"{arm} emits {emitted}"
    assert cl["stop_rate"] == 0, f"{arm} stop rate {cl['stop_rate']}"
    assert cl["step_cap_rate"] == 1, f"{arm} step-cap rate {cl['step_cap_rate']}"
    assert n_actions == n_eps * cap, f"{n_actions} actions in {n_eps} capped episodes"
    side = 16
    assert n_dec == side * side, f"{n_dec} decisions do not fill a {side}x{side} grid"

    ink, mute, line, fill, hi = "#1A1A1A", "#5A5A5A", "#333333", "#EBEBEB", C_STOP
    W, Hgt = 5.5, 2.02
    fig = plt.figure(figsize=(W, Hgt))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, Hgt); ax.axis("off")
    fs, fs_small = 8, 7
    lead = fs / 72 * 1.3           # line spacing in inches

    def lines_at(x, yc, rows, ha="left"):
        top = yc + lead * (len(rows) - 1) / 2
        for k, (txt, colour) in enumerate(rows):
            ax.text(x, top - k * lead, txt, ha=ha, va="center", fontsize=fs, color=colour,
                    zorder=3)

    def box(x0, y0, w, h, rows, fc="white"):
        ax.add_patch(Rectangle((x0, y0), w, h, fc=fc, ec=line, lw=0.6, zorder=2))
        lines_at(x0 + w / 2, y0 + h / 2, rows, ha="center")

    def arrow(points):
        for (xa, ya), (xb, yb) in zip(points[:-2], points[1:-1]):
            ax.plot([xa, xb], [ya, yb], color=line, lw=0.6, solid_capstyle="butt", zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=6,
                                     lw=0.6, color=line, shrinkA=0, shrinkB=0, zorder=1))

    def heading(y, title, note):
        tx = ax.text(0.02, y, title, ha="left", va="top", fontsize=8.5, weight="bold",
                     color=ink)
        fig.canvas.draw()
        x1 = ax.transData.inverted().transform(
            tx.get_window_extent(fig.canvas.get_renderer()))[1][0]
        ax.text(x1 + 0.06, y, note, ha="left", va="top", fontsize=8.5, color=mute)

    # Shared columns: input, policy, output, the row's fourth element, then its result.
    xi, wi = 0.02, 0.88
    xp, wp = 1.07, 0.94
    xo, wo = 2.18, 1.04
    xf, wf = 3.39, 0.56
    xr = 4.15                    # results column: text only, no arrow into it
    bh = 0.42

    # ---------------- selection: open loop ----------------
    heading(Hgt - 0.02, "Selection", "(teacher-forced)")
    ya = 1.30
    yc = ya + bh / 2
    box(xi, ya, wi, bh, [("expert's frames", ink), ("and instruction", ink)])
    box(xp, ya, wp, bh, [("policy", ink), ("(each checkpoint)", mute)], fc=fill)
    # Selection scores the probability given to the expert's action (cross-entropy), not
    # whether the top-scoring action matches it.
    box(xo, ya, wo, bh, [("cross-entropy of the", ink), ("expert's action", ink)])
    arrow([(xi + wi, yc), (xp, yc)])
    arrow([(xp + wp, yc), (xo, yc)])
    arrow([(xo + wo, yc), (xf, yc)])
    # The held-out decisions, one cell each in their recorded order; STOP targets in colour.
    cell = wf / side
    gy = yc - wf / 2
    for i in range(n_dec):
        r, c_ = divmod(i, side)
        ax.add_patch(Rectangle((xf + c_ * cell + cell * 0.12, gy + (side - 1 - r) * cell
                                + cell * 0.12), cell * 0.76, cell * 0.76,
                               fc=hi if i in stop_idx else "#C9C9C9", ec="none", zorder=2))
    ax.text(xf + wf / 2, gy - 0.04, f"{n_stop} of {n_dec} ask for STOP", ha="center", va="top",
            fontsize=fs_small, color=hi)
    lines_at(xr, yc, [(f"Mean over the {n_dec}", ink),
                      ("decisions is lowest", ink),
                      (f"at update {sel['update_step']}, which", ink),
                      ("is selected", ink)])

    # ---------------- deployment: closed loop ----------------
    heading(1.00, "Deployment", "(closed loop)")
    yb = 0.32
    yc = yb + bh / 2
    box(xi, yb, wi, bh, [("its own frames", ink), ("and instruction", ink)])
    box(xp, yb, wp, bh, [("policy", ink), (f"(update {sel['update_step']})", mute)], fc=fill)
    box(xo, yb, wo, bh, [("action", ink)])
    box(xf, yb, wf, bh, [("simulator", ink)])
    arrow([(xi + wi, yc), (xp, yc)])
    arrow([(xp + wp, yc), (xo, yc)])
    arrow([(xo + wo, yc), (xf, yc)])
    # The loop: the simulator's next view returns to the policy's input.
    yl = yb - 0.14
    arrow([(xf + wf / 2, yb), (xf + wf / 2, yl), (xi + wi / 2, yl), (xi + wi / 2, yb)])
    ax.text((xi + wi / 2 + xf + wf / 2) / 2, yl - 0.03, "next frames", ha="center", va="top",
            fontsize=fs_small, color=mute)
    lines_at(xr, yc, [(f"All {n_actions:,} actions in", ink),
                      (f"{n_eps} episodes move forward;", ink),
                      ("no STOP, so every episode", ink),
                      (f"ends at the {cap}-step cap", ink)])

    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out, format="pdf")
    plt.close(fig)
    return {"figure": out.name,
            "sources": [str(run / "result.json"), str(closed_loop / "result.json")],
            "selected_update_step": sel["update_step"],
            "selection_ce": round(sel["summary"]["mean_restricted_action_ce"], 4),
            "selection_decisions": n_dec, "stop_targets_on_surface": n_stop,
            "closed_loop_arm": arm, "closed_loop_episodes": n_eps,
            "closed_loop_actions": n_actions, "closed_loop_action_histogram": hist,
            "closed_loop_stop_rate": cl["stop_rate"], "closed_loop_step_cap_rate": cl["step_cap_rate"],
            "fields": ["calibrations[].summary.mean_restricted_action_ce",
                       "calibrations[].per_decision[].target_action",
                       "checkpoint_summaries[arm].action_histogram",
                       "checkpoint_summaries[arm].stop_rate"]}


def fig_closed_loop_behavior(out: Path, analysis: Path, closed_loop: Path) -> dict[str, Any]:
    """Constant-action episode counts per seed, and the weighted-minus-unweighted contrasts.

    Redrawn in the same typeface and palette as the rest of the paper's figures. The earlier
    version of this panel was produced by a separate drawing library in a sans-serif corporate
    style, which read as a slide rather than a figure and did not match the four beside it.
    """
    an = json.loads(analysis.read_text())
    beh = an["checkpoint_behavior"]
    boot = an["treatment_mean_scene_cluster_bootstrap"]["metrics"]
    seeds = ["20260825", "20260826", "20260827"]
    unw = [beh[f"unweighted_s{s}"]["constant_action_episode_count"] for s in seeds]
    wei = [beh[f"inverse_frequency_s{s}"]["constant_action_episode_count"] for s in seeds]
    n_eps = beh[f"unweighted_s{seeds[0]}"]["episode_count"]

    rows = []
    for key, label in (("navigation_error", "final error"), ("minimum_distance", "closest approach")):
        m = boot[key]
        rows.append((label, m["point_delta"], m["lower_95"], m["upper_95"]))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.5, 2.05),
                                   gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.42})

    x = list(range(len(seeds))); w = 0.36
    axl.bar([i - w / 2 for i in x], unw, w, color=C_UNWEIGHTED, label="unweighted")
    axl.bar([i + w / 2 for i in x], wei, w, color=C_WEIGHTED, label="weighted")
    for i, (a, b) in enumerate(zip(unw, wei)):
        axl.text(i - w / 2, a + 1.2, str(a), ha="center", fontsize=6)
        axl.text(i + w / 2, b + 1.2, str(b), ha="center", fontsize=6)
    axl.set_xticks(x); axl.set_xticklabels([str(s) for s in seeds], fontsize=6.5)
    axl.set_ylim(0, n_eps * 1.18)
    axl.set_xlabel("seed", fontsize=6.5)
    axl.set_ylabel(f"episodes repeating one\naction (of {n_eps})", fontsize=6.5, linespacing=1.35)
    axl.grid(axis="y", color="#ECECEC", lw=0.5); axl.set_axisbelow(True)
    axl.spines["right"].set_visible(False)
    axl.legend(fontsize=6, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.24), handlelength=1.4, columnspacing=1.2)

    y = list(range(len(rows)))[::-1]
    axr.axvline(0, color="#BBBBBB", lw=0.7, zorder=1)
    for yy, (label, pt, lo, hi) in zip(y, rows):
        # A paired difference belongs to neither treatment, so it is drawn in neutral ink.
        colour = "#333333"
        axr.plot([lo, hi], [yy, yy], color=colour, lw=1.1, solid_capstyle="round", zorder=3)
        axr.plot([pt], [yy], marker="o", ms=4.0, color=colour, zorder=4)
        # U+2212, the typographic minus, as on the axis ticks.
        axr.text(pt, yy + 0.30, f"{pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]".replace("-", "\u2212"),
                 ha="center", fontsize=5.8, color="#333333", zorder=5,
                 bbox={"fc": "white", "ec": "none", "pad": 0.6})
    # Two-line labels keep "closest approach" clear of the left panel's bars.
    axr.set_yticks(y); axr.set_yticklabels([r[0].replace(" ", "\n") for r in rows], fontsize=6.5)
    axr.set_ylim(-0.6, len(rows) - 0.25)
    axr.set_xlabel("weighted minus unweighted (m)", fontsize=6.5)
    axr.grid(axis="x", color="#ECECEC", lw=0.5); axr.set_axisbelow(True)
    axr.spines["right"].set_visible(False); axr.spines["left"].set_visible(False)
    axr.tick_params(axis="y", length=0)

    fig.savefig(out, format="pdf")
    plt.close(fig)
    return {"figure": out.name, "sources": [str(analysis)],
            "constant_action_unweighted": unw, "constant_action_weighted": wei,
            "episode_count": n_eps,
            "contrasts": [{"metric": r[0], "point_delta": round(r[1], 4),
                           "lower_95": round(r[2], 4), "upper_95": round(r[3], 4)} for r in rows],
            "fields": ["checkpoint_behavior[arm].constant_action_episode_count",
                       "treatment_mean_scene_cluster_bootstrap.metrics[*].point_delta/lower_95/upper_95"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument("--c0-run", type=Path,
                    default=SPARK / "native_c0_seed_replication_unweighted_s20260826_v1")
    ap.add_argument("--cache-index", type=Path,
                    default=SPARK / "native_c0_development_cache_v1" / "cache_index.json")
    ap.add_argument("--object-root", type=Path,
                    default=Path("checkpoints/native_c0_h4_content_cache_v1"))
    ap.add_argument("--closed-loop-analysis", type=Path,
                    default=Path("artifacts/context/native_c0_closed_loop_analysis_v1/analysis.json"))
    ap.add_argument("--c0-closed-loop", type=Path,
                    default=SPARK / "native_c0_closed_loop_v1")
    ap.add_argument("--closed-loop", type=Path,
                    default=SPARK / "native_chunk_closed_loop_134k_v1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    prov = [fig_blind_metric(args.out / "p1_blind_metric.pdf")]
    if (args.c0_run / "result.json").is_file():
        prov.append(fig_metric_decomposition(
            args.out / "p1_metric_decomposition.pdf", args.c0_run))
    if args.c0_run.is_dir() and args.cache_index.is_file() and args.object_root.is_dir():
        prov.append(fig_stop_moments(args.out / "p1_stop_moments.pdf", args.c0_run,
                                     args.cache_index, args.object_root))
    if (args.c0_run / "result.json").is_file() and (args.c0_closed_loop / "result.json").is_file():
        prov.append(fig_selection_schematic(
            args.out / "p1_selection_schematic.pdf", args.c0_run, args.c0_closed_loop))
    if args.closed_loop_analysis.is_file():
        prov.append(fig_closed_loop_behavior(
            args.out / "native_c0_closed_loop_behavior.pdf",
            args.closed_loop_analysis, args.c0_closed_loop))
    if (args.closed_loop / "episodes").is_dir():
        prov.append(fig_trajectories(args.out / "p1_trajectories.pdf", args.closed_loop))
    (args.out / "provenance_p1.json").write_text(
        json.dumps({"gate": "p1_figures_v1", "figures": prov}, indent=2))
    for p in prov:
        print(f"[fig] {p['figure']:<26} from {len(p['sources'])} artifact source(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

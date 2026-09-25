"""PT: final performance vs number of preferences, human vs synthetic labels.

Reads the per-seed eval_summary.json written by run_grid_ms_count.sh:

    <runs_root>/<prefix>-N<N>/seed_<s>/eval_summary.json                  human
    <runs_root>/<prefix>-N<N>-<scripted_tag>/seed_<s>/eval_summary.json   synthetic

Both arms train on the SAME pairs per (N, seed); only the label source differs.
Optionally draws the ground-truth-reward IQL ceiling as a horizontal band.

Metric: last10_eval_reward (mean of the final 10 evaluations, each 10 episodes),
aggregated across seeds as mean +/- s.e.

Usage (compute node, pt env):
    python -m scripts.experiment_grid.plots.plot_pt_count_axis
    python -m scripts.experiment_grid.plots.plot_pt_count_axis --budgets 10 50 100 250 750 1000
    python -m scripts.experiment_grid.plots.plot_pt_count_axis --no_baseline --err sd
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms

RUNS_ROOT = Path("/scratch/marzii/PT/lunarlander/grid_mixture_ms")
BASELINE_DIR = RUNS_ROOT / "lunarlander-truereward-iql"
OUT_DEFAULT = Path("/scratch/marzii/PT/figures/pt_count_axis_human_vs_synthetic.png")

HUMAN_C = "#1f77b4"
SYNTH_C = "#d62728"
# two-sided t_{0.975} by degrees of freedom, for 95% CIs from few seeds
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
       15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
       27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def spread(sd, n, kind):
    """sd, standard error, or the half-width of a 95% CI over n seeds.

    sd and n may be scalars or arrays (one entry per budget).
    """
    sd, n = np.asarray(sd, dtype=float), np.asarray(n, dtype=float)
    if kind == "sd":
        return sd
    se = sd / np.sqrt(n)
    if kind != "ci95":
        return se
    t = np.vectorize(lambda k: T95.get(int(k) - 1, 1.96))(n)
    return se * t

BASE_C = "#555555"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budgets", type=int, nargs="+",
                   default=[10, 50, 100, 250, 750, 1000])
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--runs_root", type=Path, default=RUNS_ROOT)
    p.add_argument("--prefix", default="lunarlander-human-count-v2")
    p.add_argument("--scripted_tag", default="scriptedlabels",
                   help="suffix of the synthetic-label run dirs")
    p.add_argument("--metric", default="last10_eval_reward",
                   help="key inside eval_summary.json (metric_source=eval_summary only)")
    p.add_argument("--metric_source", choices=["final_eval", "eval_summary"],
                   default="final_eval",
                   help="final_eval: mean_return from final_eval_<ep>ep.json, i.e. 50 "
                        "deterministic episodes of the SAVED final policy. "
                        "eval_summary: last10_eval_reward, the mean of the last 10 "
                        "mid-training evaluations (10 episodes each).")
    p.add_argument("--episodes", type=int, default=50,
                   help="which final_eval_<episodes>ep.json to read")
    p.add_argument("--human_tag", default=None,
                   help="suffix of the human run dirs; defaults to 'policy' for "
                        "final_eval (those runs saved their policies) and '' otherwise")
    p.add_argument("--err", choices=["se", "sd", "ci95"], default="se",
                   help="error bars: standard error, standard deviation, or "
                        "95% CI over seeds (t-based)")
    p.add_argument("--xscale", choices=["linear", "log"], default="linear",
                   help="linear starts the label axis at 0; log spreads the small budgets")
    p.add_argument("--baseline_dir", type=Path, default=BASELINE_DIR,
                   help="ground-truth-reward IQL runs; drawn as a horizontal band")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--title", default="PT on LunarLander: human vs synthetic preferences")
    p.add_argument("--aaai", action="store_true",
                   help="AAAI-27 single-column export: 3.3in wide, 9pt, no title, "
                        "Type-42 fonts, tight bbox, writes .pdf and 300-dpi .png")
    p.add_argument("--xlabel", default=None)
    p.add_argument("--ylabel", default=None)
    p.add_argument("--xticks", type=int, nargs="+", default=None,
                   help="labelled x ticks (default: every budget)")
    p.add_argument("--legend_loc", default=None)
    p.add_argument("--legend_bbox", type=float, nargs=2, default=None)
    p.add_argument("--dump_json", type=Path, default=None,
                   help="also write the plotted series, baseline and axis settings as JSON "
                        "(used to build the combined multi-panel figure)")
    return p.parse_args()


def read_arm(runs_root: Path, cond: str, seeds: int, metric: str,
             source: str = "final_eval", episodes: int = 50):
    """Per-seed metric values for one condition dir.

    final_eval   -> mean_return from final_eval_<episodes>ep.json (50 deterministic
                    episodes of the saved final policy)
    eval_summary -> <metric> from eval_summary.json (mid-training evaluations)
    """
    vals = []
    d = runs_root / cond
    for s in range(seeds):
        if source == "final_eval":
            f = d / f"seed_{s}" / f"final_eval_{episodes}ep.json"
            key = "mean_return"
        else:
            f = d / f"seed_{s}" / "eval_summary.json"
            key = metric
        if f.exists():
            try:
                vals.append(float(json.load(open(f))[key]))
            except (KeyError, json.JSONDecodeError):
                print(f"  ! unreadable: {f}")
    return np.asarray(vals, dtype=float)


def series(runs_root, prefix, budgets, seeds, metric, tag=None,
           source="final_eval", episodes=50):
    """(x, mean, err_raw, n) over the budgets that have data."""
    xs, means, sds, ns = [], [], [], []
    for n in budgets:
        cond = f"{prefix}-N{n}" + (f"-{tag}" if tag else "")
        v = read_arm(runs_root, cond, seeds, metric, source, episodes)
        if len(v) == 0:
            print(f"  (skip, no runs) {cond}")
            continue
        xs.append(n)
        means.append(v.mean())
        sds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        ns.append(len(v))
        print(f"  {cond:<58} n={len(v):<3} mean={v.mean():7.1f} sd={sds[-1]:6.1f}")
    return np.array(xs), np.array(means), np.array(sds), np.array(ns)


def main():
    a = parse_args()

    human_tag = a.human_tag
    if human_tag is None:
        human_tag = "policy" if a.metric_source == "final_eval" else ""

    print(f"metric source: {a.metric_source}"
          + (f" ({a.episodes} deterministic episodes)" if a.metric_source == "final_eval"
             else f" ({a.metric})"))
    print("human arm:")
    xh, mh, sh, nh = series(a.runs_root, a.prefix, a.budgets, a.seeds, a.metric,
                            tag=human_tag, source=a.metric_source, episodes=a.episodes)
    print("synthetic arm:")
    xs_, ms, ss, ns = series(a.runs_root, a.prefix, a.budgets, a.seeds, a.metric,
                             tag=a.scripted_tag, source=a.metric_source,
                             episodes=a.episodes)

    def err(sd, n):
        return spread(sd, n, a.err)

    if a.aaai:
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                             "axes.labelsize": 9, "xtick.labelsize": 9,
                             "ytick.labelsize": 9, "legend.fontsize": 9})
    lw, cap, msz = (1.2, 2, 4) if a.aaai else (2, 3, 6)
    dump = {"series": [], "baseline": None}
    fig, ax = plt.subplots(figsize=(3.3, 2.4) if a.aaai else (6.4, 4.2))

    if not a.no_baseline and a.baseline_dir.exists():
        b = read_arm(a.baseline_dir.parent, a.baseline_dir.name, a.seeds, a.metric,
                     a.metric_source, a.episodes)
        if len(b):
            bm, bsd = b.mean(), (b.std(ddof=1) if len(b) > 1 else 0.0)
            be = spread(bsd, len(b), a.err)
            ax.axhline(bm, color=BASE_C, ls="--", lw=1.0 if a.aaai else 1.4, zorder=1)
            ax.axhspan(bm - be, bm + be, color=BASE_C, alpha=0.15, zorder=0)
            dump["baseline"] = dict(mean=float(bm), err=float(be), n=int(len(b)), color=BASE_C)
            # ax.text(0.02, 0.97,
            #         f"ground-truth reward IQL  ({bm:.0f}, n={len(b)})",
            #         transform=ax.transAxes, color=BASE_C, fontsize=8, va="top")
            # print(f"baseline: n={len(b)} mean={bm:.1f} sd={bsd:.1f}")

    if len(xh):
        ax.errorbar(xh, mh, yerr=err(sh, nh), marker="o", color=HUMAN_C, lw=lw, ms=msz,
                    capsize=cap, label=(f"human preferences, {nh.max()} seeds" if a.aaai
                                        else f"human preferences (n={nh.max()} seeds)"))
        dump["series"].append(dict(x=xh.tolist(), mean=mh.tolist(), err=err(sh, nh).tolist(),
                                   n=nh.tolist(), color=HUMAN_C, marker="o", ls="-"))
    if len(xs_):
        ax.errorbar(xs_, ms, yerr=err(ss, ns), marker="s", color=SYNTH_C, lw=lw, ms=msz,
                    ls="--", capsize=cap, label=(f"synthetic preferences, {ns.max()} seeds" if a.aaai
                                                 else f"synthetic preferences (n={ns.max()} seeds)"))
        dump["series"].append(dict(x=xs_.tolist(), mean=ms.tolist(), err=err(ss, ns).tolist(),
                                   n=ns.tolist(), color=SYNTH_C, marker="s", ls="--"))

    if a.xscale == "log":
        ax.set_xscale("log")
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks(a.budgets)
    else:
        ax.set_xlim(0, max(a.budgets) * 1.04)   # linear axis anchored at 0
        ax.set_xticks(a.budgets)
    if a.xticks:
        ax.set_xticks(a.xticks)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(a.xlabel or "number of preference labels (N)")
    ylab = (f"final return\n({a.episodes} deterministic eval episodes)"
            if a.metric_source == "final_eval"
            else "final return  (mean of last 10 mid-training evals)")
    ax.set_ylabel(a.ylabel or ylab)
    if not a.aaai:
        ax.set_title(a.title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc=a.legend_loc or "lower left", frameon=not a.aaai,
              bbox_to_anchor=a.legend_bbox,
              handlelength=2.2 if a.aaai else None, borderaxespad=0.1 if a.aaai else 0.5)
    fig.tight_layout()

    if a.dump_json:
        for s, lab in zip(dump["series"], ax.get_legend_handles_labels()[1]):
            s["label"] = lab
        dump.update(xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), xscale=ax.get_xscale(),
                    xticks=[float(t) for t in ax.get_xticks()],
                    xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
        a.dump_json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dump, open(a.dump_json, "w"), indent=1)
        print(f"wrote {a.dump_json}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.aaai:
        # tight bbox plus a small uneven margin (left, bottom, right, top; inches)
        # so every text box sits fully inside the page
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        bbox = matplotlib.transforms.Bbox.from_extents(bb.x0 - 0.12, bb.y0 - 0.10,
                                                       bb.x1 + 0.04, bb.y1 + 0.04)
        for ext in (".pdf", ".png"):
            o = a.out.with_suffix(ext)
            fig.savefig(o, dpi=300, bbox_inches=bbox)
            print(f"\nwrote {o}")
    else:
        fig.savefig(a.out, dpi=200)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

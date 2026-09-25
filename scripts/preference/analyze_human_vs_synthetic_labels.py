"""
Exact human-vs-synthetic PT label comparison.

For each N and seed:
1. Load the human preference pairs + labels from the human label folder.
2. Reconstruct the synthetic (scripted-teacher) label for those exact same pairs
   from the true dataset rewards, the same way PT does at load time
   (JaxPref/reward_transform.py:298-310, label_type=1).
3. Compare labels one-by-one.
4. Report:
      Human:    %A, %B, %tie, %skip
      Synthetic:%A, %B, %tie, %skip
      Agreement: exact label equality %

No near-tie threshold, no return-gap analysis.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


# Verified PT encoding
A = 0
B = 1
TIE = -1

# Change this only if your code uses a specific skip value.
# SKIP = -2
VALID = (A, B, TIE)

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[10, 50, 100, 250, 750, 1000],
    )
    p.add_argument("--seeds", type=int, default=30)

    # Human label folders, e.g.
    # root/prefix-N100-s0/
    p.add_argument("--human_root", type=Path,
               default=Path("/scratch/marzii/PT/human_label/human"),
    )
    
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("/scratch/marzii/PT/lunarlander/seed_0/render/mixture-v2/"
                     "lunarlander-mixture-v2.hdf5"),
    )
    
    p.add_argument("--human_prefix", default="lunarlander-human-count-v2")

    p.add_argument("--out_csv", type=Path,
                default=Path("/scratch/marzii/PT/analysis/human_vs_synthetic/pt_human_synthetic_agreement.csv"),
    )

    # Change these if the stored label filenames differ.
    p.add_argument("--human_label_file", default="label_human")

    p.add_argument("--query_len", type=int, default=100)

    return p.parse_args()


def load_pickle(path):
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f))


def load_condition(folder, n, qlen, label_filename):
    """
    Returns:
        starts_a
        starts_b
        labels
    """
    starts_a = load_pickle(folder / f"indices_num{n}_q{qlen}")
    starts_b = load_pickle(folder / f"indices_2_num{n}_q{qlen}")
    labels = load_pickle(folder / label_filename)

    if not (len(starts_a) == len(starts_b) == len(labels)):
        raise ValueError(
            f"Length mismatch in {folder}: "
            f"A={len(starts_a)}, B={len(starts_b)}, labels={len(labels)}"
        )

    return (
        starts_a.astype(np.int64),
        starts_b.astype(np.int64),
        labels.astype(np.int64),
    )


def percentages(labels):
    n = len(labels)

    if n == 0:
        return {
            "A": np.nan,
            "B": np.nan,
            "tie": np.nan,
            "skip": np.nan,
        }

    # this: 100 * float(np.mean(~np.isin(labels, VALID))) --> calculates the percentage of labels that are not in VALID (A, B, TIE). It counts how many labels are invalid and divides by the total number of labels to get a percentage of invalid labels.
    return {
        "A": 100 * np.mean(labels == A),
        "B": 100 * np.mean(labels == B),
        "tie": 100 * np.mean(labels == TIE),
        "skip": 100 * float(np.mean(~np.isin(labels, VALID))),
    }
    
def load_rewards(dataset_path):
    """True per-step environment rewards for the offline dataset."""
    import h5py
    with h5py.File(dataset_path, "r") as f:
        return f["rewards"][:].astype(np.float64)


def synthetic_labels(rewards, starts_a, starts_b, qlen):
    """Reconstruct the scripted teacher for the SAME pairs.

    A pair is two start indices into the dataset; segment X is
    rewards[start : start + qlen]. The teacher compares the summed TRUE reward
    of the two segments and prefers the larger; a tie is produced only when the
    two sums are EXACTLY equal (reward_transform.py:298-310, label_type=1).
    """
    csum = np.concatenate([[0.0], np.cumsum(rewards, dtype=np.float64)])
    ret_a = csum[starts_a + qlen] - csum[starts_a]
    ret_b = csum[starts_b + qlen] - csum[starts_b]
    lab = np.where(ret_a > ret_b, A, B).astype(np.int64)
    lab[ret_a == ret_b] = TIE
    return lab, ret_a, ret_b


def main():
    args = parse_args()
    rewards = load_rewards(args.dataset)
    print(f"dataset: {args.dataset}  ({len(rewards)} transitions)")
    rows = []

    for n in args.budgets:

        print(f"\n=== N={n} ===")

        for seed in range(args.seeds):

            human_folder = (
                args.human_root
                / f"{args.human_prefix}-N{n}-s{seed}"
            )
            
            if not human_folder.exists():
                print(f"Missing human folder: {human_folder}")
                continue

            h_a, h_b, h_labels = load_condition(
                human_folder,
                n,
                args.query_len,
                args.human_label_file,
            )

            if (h_a.max() + args.query_len > len(rewards)
                    or h_b.max() + args.query_len > len(rewards)):
                raise ValueError(
                    f"N={n}, seed={seed}: a segment runs past the end of "
                    f"{args.dataset} -- wrong dataset for these pairs?"
                )

            s_labels, ret_a, ret_b = synthetic_labels(
                rewards, h_a, h_b, args.query_len
            )
            
            # ---- Verify exact pair alignment ----

            if len(h_labels) != len(s_labels):
                raise ValueError(
                    f"N={n}, seed={seed}: {len(h_labels)} human labels vs "
                    f"{len(s_labels)} synthetic."
                )


            # ---- Exact label comparison ----

            matches = h_labels == s_labels

            h_pct = percentages(h_labels)
            s_pct = percentages(s_labels)

            row = {
                "N": n,
                "seed": seed,
                "pairs": len(h_labels),

                "human_A_pct": h_pct["A"],
                "human_B_pct": h_pct["B"],
                "human_tie_pct": h_pct["tie"],
                "human_skip_pct": h_pct["skip"],

                "synthetic_A_pct": s_pct["A"],
                "synthetic_B_pct": s_pct["B"],
                "synthetic_tie_pct": s_pct["tie"],
                "synthetic_skip_pct": s_pct["skip"],

                "n_match": int(matches.sum()),
                "n_mismatch": int((~matches).sum()),
                "agreement_pct": 100 * matches.mean(),
            }

            rows.append(row)

        # ---- Aggregate: mean over the per-seed values (seeds stay independent) ----

        seed_rows = [r for r in rows if r["N"] == n]

        if seed_rows:
            def mean_of(key):
                return float(np.mean([r[key] for r in seed_rows]))

            agreements = np.array([r["agreement_pct"] for r in seed_rows], dtype=float)
            # ddof=1: sample sd across the seed-level agreement values
            agree_sd = float(np.std(agreements, ddof=1)) if len(agreements) > 1 else 0.0

            print(f"Seeds: {len(seed_rows)}   pairs/seed: {seed_rows[0]['pairs']}")

            print(
                f"Human: "
                f"A={mean_of('human_A_pct'):.1f}%  "
                f"B={mean_of('human_B_pct'):.1f}%  "
                f"Tie={mean_of('human_tie_pct'):.1f}%  "
                f"Skip={mean_of('human_skip_pct'):.1f}%"
            )

            print(
                f"Synthetic: "
                f"A={mean_of('synthetic_A_pct'):.1f}%  "
                f"B={mean_of('synthetic_B_pct'):.1f}%  "
                f"Tie={mean_of('synthetic_tie_pct'):.1f}%  "
                f"Skip={mean_of('synthetic_skip_pct'):.1f}%"
            )

            print(
                f"Agreement: "
                f"{agreements.mean():.1f}% +/- {agree_sd:.1f}% "
                f"(mean +/- sd over {len(agreements)} seeds)"
            )

    # ---- Save per-seed CSV ----

    if rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
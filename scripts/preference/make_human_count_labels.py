"""Materialise a human-count label subset: top-N of a per-seed permutation of the 1000-pair
human preference pool, in PT's human_label/<env>/ format.

Reproduces exactly how the existing lunarlander-human-count-N{5,10,100,250,750,1000} sets were
built -- verified against N=10 seeds 0,1,2:

    idx = np.random.default_rng(seed).permutation(len(pool))[:N]      # permutation ORDER, not sorted

so subsets nest across N for a fixed seed (N10 subset of N50 subset of N100 ...).

Writes EXACTLY three files per env folder (indices_num{N}_q{L}, indices_2_num{N}_q{L},
label_human). Nothing else may live there: JaxPref does
`sorted(os.listdir(base_path))` and unpacks into exactly three names, so a stray file
either raises or silently loads the wrong array. See memory pt-human-label-layout.

    python -m scripts.preference.make_human_count_labels --n 50 --seeds 30
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np

POOL = Path("/scratch/marzii/PT/lunarlander/human_pref_clean/labels_n1000_clean.pkl")
OUT  = Path("/scratch/marzii/PT/human_label/human")
WEB_TO_PT = {-1: 0, 0: -1, 1: 1}          # same remap as labels_web_to_pt_format.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--prefix", default="lunarlander-human-count")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    w = pickle.load(open(POOL, "rb"))
    labels, pa, pb = w["labels"], np.asarray(w["pair_starts_a"]), np.asarray(w["pair_starts_b"])
    tsec = np.asarray(w["time_sec"], float)
    keep = [i for i, l in enumerate(labels) if l is not None]
    assert len(keep) == len(labels), f"pool has {len(labels)-len(keep)} skips; sampling rule assumes none"
    L = w["metadata"]["query_len"]
    P = len(keep)
    print(f"pool: {P} labelled pairs, query_len={L}")

    for s in range(a.seeds):
        idx = np.random.default_rng(s).permutation(P)[: a.n]
        d = OUT / f"{a.prefix}-N{a.n}-s{s}"
        sel_a = pa[idx].astype(np.int32)
        sel_b = pb[idx].astype(np.int32)
        sel_l = [WEB_TO_PT[int(labels[i])] for i in idx]
        secs  = float(tsec[idx].sum())
        if a.dry_run:
            print(f"  {d.name}: n={len(idx)} first5={list(idx[:5])} label_time={secs:.1f}s")
            continue
        d.mkdir(parents=True, exist_ok=True)
        existing = sorted(p.name for p in d.iterdir())
        if existing:
            raise SystemExit(f"REFUSING to write: {d} already contains {existing}")
        pickle.dump(sel_a, open(d / f"indices_num{a.n}_q{L}", "wb"))
        pickle.dump(sel_b, open(d / f"indices_2_num{a.n}_q{L}", "wb"))
        pickle.dump(sel_l, open(d / "label_human", "wb"))
        assert len(list(d.iterdir())) == 3, f"{d} must hold exactly 3 files"
        meta = OUT / "_grid_metadata" / f"{a.prefix}-N{a.n}-s{s}.label_alignment.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"actual_data_seed": -1, "label_alignment": 1.0,
                   "source": "human_pref_clean_n1000", "human_label_time_sec": round(secs, 2)},
                  open(meta, "w"))
        print(f"  wrote {d.name}: n={len(idx)} label_time={secs:.1f}s")


if __name__ == "__main__":
    main()

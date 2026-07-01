# PT Worklog

Running log of changes. Newest at top. Keep entries to 1–2 sentences.

| Date | What | Why / Result |
|---|---|---|
| 2026-06-30 | Re-ran all noise sweeps (random + flip × N=50/1000, 11 levels × 30 seeds) under `ex*` names. | Regenerate the noise axis under the exact-% convention without overwriting old results. |
| 2026-06-30 | Switched noise injection from per-label coin (`rng.random < p`) to exact-% selection (`rng.choice`); added `deterministic_flip` mode. | Coin made the corrupted *count* wobble at small N=50 (20% → ~14–26%); exact gives clean 20%. Flip = inverted labels (anti-signal) vs random = uniform replace (info destruction). |
| 2026-06-30 | Consolidated grid scripts into one gen→submit→run trio per axis; moved pipeline runs to `oracle_vs_human_vs_pebble/`, plots to `plots/`, smoke tests to `smoke_test/`. | Too many overlapping copies; now one clear pair per axis (noise/count). |

# Oracle vs Human vs PEBBLE (A/B/C comparison)

Full PT pipeline (reward model → IQL) run per **labeler × dataset**, a few seeds each.
This is the A/B/C study: how PT does with oracle vs real-human vs PEBBLE-generated labels.

- **A = oracle**, **B = human**, **C = pebble**
- Dataset suffix: *(none)* = `medium-v2`, `_mixture` = mixture, `_mr` = `medium-replay`

| | medium-v2 | mixture | medium-replay |
|---|---|---|---|
| oracle | `run_pt_pipeline_oracle.sh` | `..._oracle_mixture.sh` | `..._oracle_mr.sh` |
| human | `run_pt_pipeline_human.sh` | `..._human_mixture.sh` | `..._human_mr.sh` |
| pebble | `run_pt_pipeline_pebble_labels.sh` | `..._pebble_mixture.sh` | `..._pebble_mr.sh` |

Extras: `run_pt_pipeline_{human,pebble}_extra_seeds.sh` add more seeds to existing runs;
`run_pt_reward_oracle100.sh` trains the reward model only (oracle, 100 labels).

Each script `cd`s to the repo root, so run from anywhere:
`sbatch scripts/oracle_vs_human_vs_pebble/<file>.sh`

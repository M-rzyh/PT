# Human-Facing Difficulty Axes for LunarLander — PT vs GAIL (design / prioritization)

Scope: **human supervisors only** (the agent/automated-teacher arms are out of scope here).
Goal: frame *how* each axis is implemented for the two human pipelines, so we can pick which to
run first and show the supervisor example clips before any labeling.

## The two human pipelines (recap)

| | What the human does | How difficulty is injected |
|---|---|---|
| **PT (preferences)** | Watches **pairs of pre-recorded clips**, picks the better one. No control. | **Indirect** — the compared trajectories are *generated* under the modified dynamics, rendered to video, then labeled. |
| **GAIL (demonstrations)** | **Plays live** (teleoperates), producing demos. | **Direct** — the dynamics/controls are modified *while the human plays*. |

This is exactly the PT-indirect / GAIL-direct split in the notes, and it drives everything below.

## Three kinds of difficulty (the key distinction for prioritization)

1. **Perception** — degrades how well the human can *see/comprehend*. Frame-blanking, region-masking, **speed-up**. Natural for **both** arms (PT watches, GAIL watches-while-playing).
2. **Control interface** — degrades how well the human can *act*. **Sticky actions, action delay**. Native to **GAIL** (there is a control loop to corrupt). For **PT there is no control loop**, so these can only be baked into trajectory *generation* → they become a **data-distribution** change, not a perceptual difficulty. ⇐ important asymmetry.
3. **Task / dynamics** — makes the underlying task easier/harder. **Gravity, wind, slow-down**. Both arms, via the physics; also changes the eval task.

---

## Axis 1 — Sticky actions

**Mechanism:** with probability `p`, the env repeats the *previous* action instead of the one just issued (ALE-style sticky actions, Machado et al. 2018). A thin env **wrapper**.

- **GAIL (direct):** wrap the `human_demo` env → the human's keypress is overridden by the previous action w.p. `p`. Feels like a sticky/unresponsive controller; the human over/under-corrects → degraded demos.
- **PT (indirect):** roll the mixture policies through the sticky wrapper → trajectories with laggy control → render → the human labels clean-speed video.
- **Honest caveat:** for PT this is a *distribution* change (the clips just show worse flying); the human's *judging* is not perceptually harder. The GAIL story is much stronger than the PT story here.
- **What the human sees:** GAIL = a live lander that ignores ~`p` of your inputs; PT = a clip of a lander that responds a beat late and overshoots.
- **Levels:** `p ∈ {0, 0.25, 0.5, 0.75}`.

## Axis 2 — Action delay

**Mechanism:** buffer actions in a FIFO queue and apply each one `k` steps later (latency). A thin **wrapper**. At 50 FPS, `k` steps = `k × 20 ms`.

- **GAIL (direct):** the human's input takes effect `k` steps later — high-latency control, hard to stabilize.
- **PT (indirect):** generate the compared trajectories under delayed-action dynamics.
- **Same caveat as sticky:** native to GAIL; for PT it's a distribution change, not a perception difficulty.
- **What the human sees:** GAIL = press-now-happens-later lag; PT = a clip that behaves as if the pilot reacts late.
- **Levels:** `k ∈ {0, 3, 6, 12}` steps ≈ `0 / 60 / 120 / 240 ms`.

## Axis 3 — Simulation speed (harder = faster, easier = slower)

**MEASURED, not assumed** (`scripts/preference/gen_speed_examples.sh`, clips in
`$SCRATCH/PT/lunarlander/speed_demo/`). Source clip: 600×400, **20 fps, 100 frames, 5.000 s**.

### The baseline is not 1×

Each rendered frame is **one physics step** (LunarLander `FPS = 50`, dt = 20 ms). So 100 frames
= **2.0 s of real lander time**, shown over **5.0 s**. The clips we label today already play at
**0.4× real time** (2.5× slow motion). Define **real time ≡ 50 fps playback**.

⚠️ **The three arms run at three different speeds.** Real time ≡ 50 steps/s (`FPS = 50`).
`requested` = what the code asks for; `achieved` = measured `sum(ep_length)/sum(duration_sec)`
from `timing.csv`. They differ because `human_demo.py:454` sleeps `1.0/fps` **after** doing the
render + JPEG encode, so each step costs `overhead + 1/fps` rather than `1/fps`.

| Arm / session | requested | achieved | vs real time | source of the fps |
|---|---|---|---|---|
| PT label clips | 20 fps | 20 fps | **0.40×** | `rollout_with_video.py --fps` (default 20) + hardcoded `build_rendered_mixture.py:219` |
| GAIL clean `session_2` | `--fps 25` (0.50×) | 22.6 steps/s | **0.45×** | `start_demo_server.sh:18` (by elimination + mtime) |
| GAIL `blank50` (b=10) | `--fps 50` (1.00×) | 37.9 steps/s | **0.76×** | `~/.bash_history:1053` — **exact command, proven** |

⇒ **The b=10 GAIL blanking comparison confounds blanking with a 2× requested (1.68× achieved)
speed-up**: the blanked demos were played near real time, the clean baseline at half speed.
PT's 0.40× happens to sit close to the *clean* GAIL rate; `blank50` is the outlier.

Fix before any speed study: (a) make fps an explicit option on **both** renderers, (b) write a
`session_meta.json` with requested + achieved fps, (c) pick one anchor for both arms.

### Mechanism, and where it actually caps

| Direction | Mechanism | Cap |
|---|---|---|
| **Slower** | **retime**: keep all 100 frames, lower the fps | **None.** Verified 1 fps → a 100 s clip. Only tedium bounds it. |
| **Faster** | **retime** up to 60 fps | **1.2× real time.** Past 60 fps a display shows nothing more. |
| **Faster still** | **subsample**: keep every Kth frame, play at 50 fps | Unbounded, but **lossy** — the human never sees the dropped physics steps. |

So the honest statement is *not* "the sim has no speed cap so anything goes". Lossless speed-up
is capped at **1.2× real time** by the 60 Hz display. **Every speed beyond that destroys
information** — which makes fast-forward a *perception* difficulty of the same family as frame
blanking (it drops frames), not a pure "watch it faster" knob. Nice for the story; state it.

### Verified clip ladder

| Clip | fps | frames | duration | vs real time | mechanism |
|---|---|---|---|---|---|
| `slowest_0.02x_1fps` | 1 | 100 | 100.0 s | 0.02× | retime |
| `slow_0.10x_5fps` | 5 | 100 | 20.0 s | 0.10× | retime |
| `slow_0.20x_10fps` | 10 | 100 | 10.0 s | 0.20× | retime |
| `base_0.40x_20fps` | 20 | 100 | 5.0 s | **0.40× ← what we label today** | — |
| `faster_0.60x_30fps` | 30 | 100 | 3.33 s | 0.60× | retime |
| `realtime_1.00x_50fps` | 50 | 100 | 2.0 s | 1.00× | retime |
| `maxlossless_1.20x_60fps` | 60 | 100 | 1.67 s | 1.20× | **last lossless speed** |
| `drop2_2.0x_realtime` | 50 | 50 | 1.0 s | 2.0× | subsample K=2 |
| `drop3_3.0x_realtime` | 50 | 34 | 0.68 s | 3.0× | subsample K=3 |
| `drop5_5.0x_realtime` | 50 | 20 | 0.40 s | 5.0× | subsample K=5 |

### My proposed levels (watch the clips before committing)

**PT (judging a pair).** Anchor on today's 0.4× so the existing labels stay the baseline:

| Level | Playback | Why |
|---|---|---|
| **Easier** | 10 fps (0.2×), 10 s | twice as long to watch; a gentle "does slower help?" control |
| **Base** | 20 fps (0.4×), 5 s | unchanged — reuses existing labels |
| **Harder** | 50 fps (1.0×), 2 s | real time; still every physics step; the honest first hard step |
| **Hardest** | drop3 (3.0×), 0.68 s | 2/3 of frames gone; two 0.68 s clips to compare is brutal |

Pair-judging doubles the load: at 3× you watch two clips in 1.4 s combined. I expect the coarse
"which one landed?" judgment to survive 3×, and fine control-quality judgments to die at 1.0×.

**GAIL (playing live).** Human visual reaction ≈ 250 ms is fixed, and the lander needs corrections
every few hundred ms, so the control budget — not vision — binds:

| Level | Wall-clock | Prediction |
|---|---|---|
| **Easier** | 0.5× | more reaction budget; should raise demo return above the +237 clean baseline |
| **Base** | 1.0× | current |
| **Harder** | 2.0× | playable but degraded; ~125 ms effective reaction budget |
| **Hardest** | 3.0× | control breakdown; ~83 ms budget ≈ below reaction time |

These are hypotheses, not results. The clips above make the PT side checkable in a minute; the
GAIL side needs a `--speed` flag on `human_demo.py` and one person playing.

## Axis 4 — Easier variations: gravity, wind, slow-down

**Gravity** — *this is easier than I first claimed.*

Both envs create the world as a Box2D `b2World`. gymnasium passes gravity in
(`b2World(gravity=(0, gravity))`); gym 0.23.1 calls bare `Box2D.b2World()` and inherits Box2D's
default `(0, -10)`. But **`world.gravity` is a writable attribute** — verified on the PT env:

```python
e = gym.make("LunarLander-v2").unwrapped
e.reset()
e.world.gravity                  # b2Vec2(0,-10)
e.world.gravity = (0.0, -5.0)    # works
```

So gravity is a **one-line post-construction patch on BOTH arms** — no gymnasium switch, no env
migration. Levels `{-10 base, -7, -5, -3}`; smaller magnitude = floatier = easier. (The constructor
kwarg is still rejected on gym 0.23.1: `__init__() got an unexpected keyword argument 'gravity'`.
Set the attribute, don't pass the kwarg.)

**Wind** — *this is harder / noisier than I first claimed, and now I can say exactly why.*

Wind exists only in gymnasium (`grep -c wind` on gym 0.23.1's `lunar_lander.py` = **0**). Its phase
comes from `wind_idx`, and three verified facts make it an uncontrolled disturbance:

1. `self.wind_idx = np.random.randint(-9999, 9999)` runs in **`__init__`**, from the **global
   NumPy RNG** — *not* `self.np_random`. Verified: pinning the global seed and varying the env seed
   (`reset(seed=0)` vs `reset(seed=7)`) gives the **same** `wind_idx`; varying the global seed
   changes it. **`env.reset(seed=...)` does not control the wind.**
2. `reset()` never re-draws it. Verified: `wind_idx` after 30 steps = −7236, and after a fresh
   `reset(seed=0)` = −7235 — it **carried over and kept incrementing**, it was not re-drawn.
   So episode *N*'s gust pattern depends on the total number of steps taken in episodes 1…*N*−1.
3. The waveform `tanh(sin(0.02k) + sin(0.01πk))` is deliberately **aperiodic**, and wind is
   switched off the moment a leg touches ground.

Together: the same seed does **not** reproduce the same wind, and episodes are **coupled** through
a counter. That is variance you cannot seed away — it inflates the error bars on every arm without
telling you anything about supervision difficulty. Hence bottom priority. (It's fixable — reseed
`wind_idx` from `self.np_random` in `reset()` — but that's patching the env, and it's a detour.)

Base LunarLander has wind **off**, so wind is really a *harder* knob; "no wind" is the easy end.

**Slow-down:** covered by Axis 3 (easier end), and it has **no lower bound**.

## Feasibility summary (measured)

| Axis | PT arm | GAIL arm |
|---|---|---|
| Sticky / delay | wrapper — portable | wrapper — portable |
| Speed | retime the rendered clips (offline) | wall-clock pacing (`--speed` flag needed) |
| **Gravity** | `env.world.gravity = (0, g)` — **works on gym 0.23.1** | constructor kwarg |
| **Wind** | **absent from gym 0.23.1** → needs gymnasium-generated trajectories | constructor kwarg, but unseeded |

## Prioritization — my recommendation (updated)

1. **Simulation speed-up** — cleanest cross-arm perception axis; clips already exist. But **first
   fix the 0.4×-vs-1.0× baseline mismatch between the arms**, and be explicit that speeds >1.2×
   real time work by dropping frames. **Do first.**
2. **Gravity (easier)** — promoted: it's a one-line attribute set on both arms, no env migration.
   Cleanest single-knob dynamics axis, and the only *easier* direction besides slow-down.
3. **Sticky actions** — strong, native GAIL axis; weaker/ambiguous for PT (distribution change,
   not perception). Run as a **GAIL-centric** study, present the PT side honestly.
4. **Action delay** — same shape as sticky; pick sticky *or* delay, not both.
5. **Wind** — demoted further: unseeded, cross-episode-coupled, PT-side absent. **Skip unless the
   supervisor specifically wants it.**

## Next step — example clips for the supervisor

Recommended: render a small **gallery of ~5–10 s example clips** (isolated scratch dir, nothing existing touched) showing exactly what the human would see per candidate:
`sticky p=0.5`, `delay k=6`, `speed 2× / 3×`, `slow 0.5×`, `gravity -5`, `wind on`.
That gives the supervisor a concrete before/after to choose from **before** any labeling. Say the word and I'll generate them.

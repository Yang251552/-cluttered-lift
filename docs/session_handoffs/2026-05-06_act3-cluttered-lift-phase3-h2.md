# Session Handoff — 2026-05-06

`cluttered-pick` (act 3) progress: phases 0, 1, 2, 3-H1, **3-H2** are done and
pushed to GitHub. README updated with the master comparison table, diagnosis
narrative, and cross-act links. Phase 4 (writeup polish) is pending.

## Repo URLs

| Act | Repo |
|---|---|
| 1 — `isaac-lab-manipulation` | https://github.com/Yang251552/isaac-lab-manipulation |
| 2 — `excavation-rl` | (private, not on GitHub) |
| 3 — **this** `cluttered-lift` | https://github.com/Yang251552/cluttered-lift |

Local working directory remains `~/Downloads/cluttered-pick/`; GitHub repo
name is `cluttered-lift`. (Renaming history is documented in the previous
handoff at `docs/session_handoffs/2026-05-05_*.md`.)

## What changed this session

### Phase 3, H2 — warm-start from act-1 `model_1499.pt`

Hypothesis: a random-init PPO can't bootstrap on the cluttered substrate
because reaching/lifting reward is too noisy under sphere contacts; a
pre-trained policy that already encodes the lift behaviour should produce
informative gradients on every successful lift.

**Verdict: supported.** Same NoCurric task that left H1 stuck at 0% lets PPO
fine-tune to 96% lift over 1500 fresh updates when initialised from act-1's
`model_1499.pt`. Same optimiser, same substrate, same seed; only the
initialisation differs.

Final five-row comparison (six-row including both H2 seeds):

| Eval | Lift @ 4 cm | Reach @ 5 cm | Reach @ 2 cm | Mean cube z (m) | Mean goal-dist (m) |
|---|---|---|---|---|---|
| Bare-table act-1 | 100.00% | 100.00% | 100.00% | 0.382 | 0.0032 |
| Granular zero-shot | 94.53% | 94.14% | 92.58% | 0.352 | 0.0374 |
| Phase 2 retrained (curric on) | 0.00% | 0.00% | 0.00% | 0.024 | 0.418 |
| Phase 3 H1 retrained (curric off) | 0.00% | 0.00% | 0.00% | 0.022 | 0.440 |
| **Phase 3 H2 warm-start, seed 0** | **96.09%** | **95.70%** | **69.53%** | **0.348** | **0.0361** |
| **Phase 3 H2 warm-start, seed 42** | **96.48%** | **96.48%** | **72.27%** | **0.346** | **0.0371** |

Side-effect worth noting: reach@2cm drops from zero-shot's 92.58% to
~70.9% averaged over both H2 seeds. Bare-table policy's settling-precision
gradient gets re-weighted toward "don't disturb spheres" robustness during
fine-tune.

See `docs/phase3-h2-warmstart.md` for the full doc, including the launch
command, rsl_rl warm-start flag confirmation, and TODO(human) research
observation paragraph.

### README + cross-act surface

- Added `## Result table` (6 rows) before the phase comparison GIFs
- Added `## Diagnosis` section that frames H1 + H2 as the two falsification
  experiments, with two TODO(human) blocks:
  - The H2 GIF placeholder (needs offscreen-render rerun on EC2)
  - The "what I'd try next" research-taste paragraph (CLAUDE.md §6
    explicitly requires this be the human's own voice)
- Added `## Sibling repos` (act 1 GitHub link, act 2 noted as private)
- Added `## Phase docs` table linking all five phase docs

Pushed in commit `89e17a5`.

## What is pending

### Phase 4 — writeup polish

Mostly mechanical now. Open items:

1. **Fill the TODO(human) blocks** (CLAUDE.md §6 — human voice required):
   - `docs/phase1-zero-shot-transfer.md` — research observation paragraph
   - `docs/phase2-train-from-scratch.md` — research observation paragraph
   - `docs/phase3-h1-no-curriculum.md` — research observation paragraph
   - `docs/phase3-h2-warmstart.md` — research observation paragraph
   - `README.md` — the "what I'd try next" paragraph in `## Diagnosis`
2. **Render H2 rollout GIF + multi-panel comparison**:
   - Restart EC2, restore offscreen render env (libGLU, freeglut3-dev,
     Xvfb, driver 580.x — see "known gotchas" §3)
   - Run `play_granular.py` on `model_2998.pt` of run
     `2026-05-06_07-53-44`
   - Convert to `results/videos/h2_warmstart_seed42.{mp4,gif}`
   - Optional: produce a phase-comparison panel (zero-shot | H1 | H2)
     and replace the H2 placeholder URL in README
3. **Optional cleanup** (carries from prior handoff):
   - Parameterise `scripts/plot_training_curves.py` (suptitle currently
     hardcoded to act-1's "1500 iters, 24.2 min")
   - Optional second-seed eval for phase 1 — phase 1 only has seed 42
     for the locked 1d substrate; phase 2 / H1 are seed-0 only because
     the 0% result wouldn't change with a second seed

After (1)+(2) the act-3 portfolio surface is complete.

## EC2 state at session end

- IP at session end: `13.63.237.226` (will change on next start/stop cycle)
- Driver: nvidia-driver-580.126.20 (passes Isaac Sim 4.5 RTX whitelist)
- libGLU + freeglut3-dev installed (offscreen video render works)
- Xvfb + x11vnc set up on port 5901 (vnc password `lift1234`)
- `train.py` and `play.py` patched on EC2 to import `granular_lift_env` (so
  the cluttered-task IDs resolve)
- New tmux launch quirk this session: the venv was not auto-sourced in fresh
  tmux sessions, and `OMNI_KIT_ACCEPT_EULA=YES` had to be exported before
  running `./isaaclab.sh`. Both fixes added to the launch wrapper inline; if
  you script future runs, prepend:
  ```
  source /home/ubuntu/isaaclab_venv/bin/activate
  export OMNI_KIT_ACCEPT_EULA=YES
  cd /home/ubuntu/IsaacLab
  ```
- New SSH key for this EC2: `~/.ssh/excavation-key.pem` (NOT
  `cmm_compute.pem`/`cmm_yang_key.pem` — those return permission denied on
  this instance)
- User instructed to stop EC2 after this handoff is committed

If the next session restarts the EC2: IP changes. Update `CLAUDE.md` §1 (or
just the handoff IP — `CLAUDE.md` defers to act-1's CLAUDE.md for that field).

## Known gotchas (carried from prior handoff)

1. **PhysX 64K material limit**: every per-body `physics_material=` on the
   sphere collection creates a separate PhysX material instance. With 1024
   envs × 64 spheres that overflows. Use the default material — i.e., do not
   set `physics_material=` on `RigidObjectCfg` for the spheres.
2. **PhysX patch buffer overflow** at `num_envs ≥ 1024` × 64 spheres prints
   `"increase PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to N"`
   warnings (saw ~16,400–16,600 typical during H2). Warnings, not errors;
   sim continues, some sphere-sphere contacts may be skipped in a fraction
   of timesteps. Acceptable for this project.
3. **Driver 535.288** (the default on Ubuntu 22.04 Deep Learning AMI):
   NVIDIA's Vulkan driver re-encodes the patch field with 8-bit truncation.
   `535.288.01` reports as `535.32.01` to Vulkan, which fails Isaac Sim's
   `≥ 535.129` whitelist. Headless training works on the rejected driver,
   but `--video` and any DISPLAY-bound rendering hang. Fix: `sudo apt
   install -y nvidia-driver-550-server` (apt usually pulls 580.x).
4. **`train.py` and `play.py` registration**: the official scripts do not
   import `granular_lift_env`, so the cluttered task IDs are unknown to gym.
   The repo's scripts at `/home/ubuntu/IsaacLab/scripts/reinforcement_learning/rsl_rl/`
   on the EC2 are patched to add `import granular_lift_env`. If you ever
   sync from upstream Isaac Lab, re-apply that patch.
5. **README image URLs**: use `https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/...`
   (no leading dash in the repo name).
6. **`plot_training_curves.py` hardcodes act-1's suptitle**. Phase 2, H1,
   and H2 curve PNGs all embed it as-is. TODO when phase 4 polish:
   parameterise with a `--title` flag.
7. **rsl_rl warm-start CLI semantics** (new this session): `--resume
   --load_run <relative_dirname> --checkpoint <model_file.pt>`. The
   `load_run` value is the run folder *name* (e.g. `2026-05-03_18-55-30`),
   relative to `<log_root>/<experiment_name>/`. The runner loads weights
   AND iter counter, so a resumed run reports `iteration 1499/2999` rather
   than `iteration 0/1500` — read the iter range as
   `start_it → start_it + max_iterations`.

## Tasks at session end

- [x] Phase 0
- [x] Phase 1
- [x] Phase 2
- [x] Phase 3 H1 (curriculum disabled, falsified)
- [x] Phase 3 H2 (warm-start, supported)
- [x] Result table + diagnosis narrative in README
- [x] Cross-act links + phase docs index in README
- [ ] Phase 4 — TODO(human) research-voice paragraphs (5 places)
- [ ] Phase 4 — H2 rollout GIF render
- [ ] Phase 4 — optional cleanups (plot script suptitle, phase 1 second seed)

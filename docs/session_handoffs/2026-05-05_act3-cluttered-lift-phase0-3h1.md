# Session Handoff — 2026-05-05

`cluttered-pick` (act 3) progress: phases 0, 1, 2, and 3-H1 are done and pushed
to GitHub. Phase 3-H2 and phase 4 are pending for the next session.

## Repo URLs

| Act | Repo |
|---|---|
| 1 — `isaac-lab-manipulation` | https://github.com/Yang251552/isaac-lab-manipulation |
| 2 — `excavation-rl` | (separate repo, not part of this session) |
| 3 — **this** `cluttered-lift` | https://github.com/Yang251552/cluttered-lift |

Note: local working directory is `~/Downloads/cluttered-pick/`, GitHub repo
name is `cluttered-lift`. Earlier in the session the local dir was named
`granular-pick/` and was renamed mid-flight; some doc internals still reference
"granular" as the substrate name (this is fine — the substrate IS a granular
proxy, the repo name is shorter).

## What is done

### Phase 0 — substrate decision (locked)

64 small spheres in an 8×8 staggered grid on the table, centred on the cube
spawn region. Cube spawn distribution (±10 cm × ±25 cm) is wider than the grid
(±14 cm × ±14 cm), which produces a deliberate ~56% inside-cluster / ~44%
outside-cluster split — this bimodality is the actual phase-1 research signal.
Do not "fix" it.

See `docs/phase0-substrate-decision.md`.

### Phase 1 — zero-shot transfer of act-1 policy

256-rollout eval × 2 seeds (separately calibrated through 4 substrate
configurations 1a / 1b / 1c / 1d before reaching the locked one).

| Eval | Lift @ 4 cm | Reach @ 5 cm | Reach @ 2 cm | Mean cube z (m) | Mean goal-dist (m) |
|---|---|---|---|---|---|
| Bare-table act-1 | 100% | 100% | 100% | 0.382 | 0.0032 |
| Granular zero-shot | 94.53% | 94.14% | 92.58% | 0.352 | 0.0374 |

Mean goal-dist degrades 12×. Success rate drops only ~7 pp because the failure
distribution is bimodal, not uniform. See `docs/phase1-zero-shot-transfer.md`.

### Phase 2 — retrain from scratch (curriculum on)

PPO with the same `LiftCubePPORunnerCfg` from act-1, no hyperparameter
changes, on the cluttered scene. Stopped early at iter 1100/1500 per
CLAUDE.md §2.1 mid-train rule.

Failure mode: at iter ~425 the `action_rate` and `joint_vel` curriculum
schedules complete their 10000-env-step ramp from −0.0001 to −0.1. The
penalty becomes an order of magnitude larger than any task reward the actor
is earning. PPO collapses the policy onto a "don't move" fixed point.

Eval: 0% / 0% / 0%. Cube basically untouched (mean z 0.024 m, mean goal-dist
0.418 m). See `docs/phase2-train-from-scratch.md`.

### Phase 3, H1 — retrain with curriculum disabled

Subclassed env `GranularFrankaCubeLiftEnvCfg_NoCurric` removes the two
curriculum schedules in `__post_init__`. Otherwise identical to phase 2.
Stopped at iter 450/1500 per CLAUDE.md §2.1 (37 min).

H1 verdict: **falsified.** Mean reward stayed flat at 4–5.5 the entire run.
Lift reward stayed below 1.0. The phase-2 cliff at iter 425 is gone (no
schedule to ramp), but no learning trajectory replaced it. Eval: 0% / 0% /
0% — same as phase 2 but reached via a different path (phase 2 the policy
briefly tried then was crushed; H1 the policy never tried).

Final four-row comparison table (this is the act-3 deliverable so far):

| Eval | Lift @ 4 cm | Reach @ 5 cm | Reach @ 2 cm | Mean cube z (m) | Mean goal-dist (m) |
|---|---|---|---|---|---|
| Bare-table act-1 | 100% | 100% | 100% | 0.382 | 0.0032 |
| Granular zero-shot | 94.53% | 94.14% | 92.58% | 0.352 | 0.0374 |
| Phase 2 retrained (curriculum on) | 0% | 0% | 0% | 0.024 | 0.418 |
| Phase 3 H1 retrained (curriculum off) | 0% | 0% | 0% | 0.022 | 0.440 |

See `docs/phase3-h1-no-curriculum.md`.

## What is pending

### Phase 3, H2 — warm-start from act-1 checkpoint

Hypothesis: the bottleneck is exploration. A random-init policy can't bootstrap
on this substrate because reaching reward is too noisy under sphere contacts.
A pre-trained policy (act-1's `model_1499.pt`) already encodes "reach, grasp,
lift" structure on bare table; if it's allowed to fine-tune under continued
PPO updates, every successful lift produces a clean gradient.

If H2 succeeds (reward climbs from ~150, lift reward stays high), the
exploration-bootstrap claim is supported.

If H2 also collapses to 0%, the implication is stronger: not even a working
warmed-up policy survives PPO updates on this substrate, pointing to something
fundamental in advantage estimation or surrogate-loss stability under noisy
contacts.

**Implementation note for next session**: rsl_rl 5.0.1 supports warm-start
via `agent_cfg.resume = True` plus `load_run` and `load_checkpoint` fields.
Verify the exact CLI flags by `grep -E "resume|load_run|load_checkpoint" /home/ubuntu/IsaacLab/scripts/reinforcement_learning/rsl_rl/cli_args.py /home/ubuntu/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` first. Draft command:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Granular-Lift-Cube-Franka-v0 \
  --headless --num_envs 1024 --logger wandb \
  --log_project_name granular-pick --seed 42 \
  --resume --load_run 2026-05-03_18-55-30 --checkpoint model_1499.pt
```

`--load_run` typically takes a relative run name under the experiment dir,
not an absolute path. Confirm on first run.

### Phase 4 — writeup

Integrate the four phase docs into a narrative section in README. Fill the
TODO research-observation paragraphs in each phase doc (these must be the
human's voice — Claude does not write them). Add cross-links to act-1 and
act-2 sibling repos. Single commit, push.

## EC2 state at session end

- IP at session end: `16.171.182.227` (will change on next start/stop cycle)
- Driver: nvidia-driver-580.126.20 (passes Isaac Sim 4.5 RTX whitelist)
- libGLU + freeglut3-dev installed (offscreen video render works)
- Xvfb + x11vnc set up on port 5901 (vnc password `lift1234`)
- `train.py` and `play.py` patched on EC2 to import `granular_lift_env` (so the
  cluttered-task IDs resolve)
- User instructed to stop EC2 after this handoff is committed

If the next session restarts the EC2: IP changes. Update `CLAUDE.md` §1
(both this repo and `isaac-lab-manipulation`).

## Known gotchas (re-encountering these wastes time)

1. **PhysX 64K material limit**: every per-body `physics_material=` cfg
   creates a separate PhysX material instance. With 1024 envs × 64 spheres
   that overflows. Use the default material — i.e., do not set
   `physics_material=` on `RigidObjectCfg` for the spheres.
2. **PhysX patch buffer overflow** at `num_envs ≥ 1024` × 64 spheres prints
   `"please increase its size to at least N"` warnings. These are warnings,
   not errors — sim continues, some sphere-sphere contacts may be skipped
   in a fraction of timesteps. Acceptable for this project.
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
   (no leading dash in the repo name — the leading-dash variant `Yang251552/-cluttered-lift`
   was an earlier wrong URL and is corrected throughout the docs).
6. **`plot_training_curves.py` hardcodes act-1's suptitle** ("PPO training,
   seed 42, 1500 iters, A10G, 24.2 min"). Phase 2 and phase 3 curve PNGs
   embed it as-is. TODO when phase 4 polish: parameterise the script with
   a `--title` flag.

## Cleanup the next session can do

- Parameterise `scripts/plot_training_curves.py` (suptitle).
- Fill the human-voice paragraphs in `docs/phase{1,2}-*.md` and
  `docs/phase3-h1-*.md` (each has a `TODO (human)` block).
- Optional: re-eval phase 1 with seed 42 to mirror act-1's two-seed pattern.
  Currently only phase 1 zero-shot has the seed-42 confirmation; phase 2 and
  phase 3-H1 are seed-0 only because their numbers are 0% across the board
  and a second seed wouldn't change the conclusion.

## Tasks at session end

- [x] #7 phase 1
- [x] #8 phase 2
- [x] #11 before/after GIF
- [x] #9 phase 3 H1 (H2 standalone)
- [x] #12 session handoff
- [ ] phase 3 H2 — see "What is pending" above
- [ ] #10 phase 4 writeup

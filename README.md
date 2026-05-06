# Granular Pick — A Scoped Granular-Manipulation Experiment

A small experiment on the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) +
[`rsl_rl`](https://github.com/leggedrobotics/rsl_rl) PPO stack, asking one
specific question:

> **The same PPO policy that hits 100 % on `Isaac-Lift-Cube-Franka-v0` —
> what happens when the cube is buried in a cluster of small rigid bodies?
> Where does the gap come from: observation, action, dynamics, or reward?**

This is **act 3** of a three-repo arc:

| Repo | Role |
|---|---|
| [`isaac-lab-manipulation`](../isaac-lab-manipulation) | act 1 — standard reproduction on the production stack (`Isaac-Lift-Cube-Franka-v0`, 100 % over 256 rollouts × 2 seeds) |
| [`excavation-rl`](../excavation-rl) | act 2 — self-built granular-manipulation substrate (custom Warp + hand-rolled PPO); did not converge, kept as a documented lessons-learned artefact |
| this repo | act 3 — same research interest (granular manipulation), much cheaper substrate (small-rigid-body cluster on Isaac Lab); a tractable sub-problem of act 2's question |

The deliverable is **not** "I solved granular pick". It is **a diagnosis** —
quantified data and a falsification-style discussion of where standard PPO
breaks when the contact pattern shifts from a hard table to a cluster of
freely-movable small bodies.

## Status

- [x] Phase 0 — substrate decision: multi-rigid-body proxy (8×8 grid of 64 spheres, r=2 cm) via `RigidObjectCollection` ([details](docs/phase0-substrate-decision.md))
- [x] Phase 1 — zero-shot transfer of act-1 policy: **reach@2cm drops 100% → 92.58%, mean goal-dist degrades 12×** ([details](docs/phase1-zero-shot-transfer.md))
- [x] Phase 2 — re-train PPO from scratch: **policy collapses to 0% / 0% / 0% at iter 425 when the curriculum schedule kicks in** (stopped early per CLAUDE.md §2.1 mid-train rule; [details](docs/phase2-train-from-scratch.md))
- [x] Phase 3 H1 — disable curriculum: **also collapses to 0% / 0% / 0%** ([details](docs/phase3-h1-no-curriculum.md)). Verdict: H1 falsified — the curriculum is not the actor-killer; the bottleneck is upstream (exploration bootstrap)
- [x] Phase 3 H2 — warm-start from act-1 checkpoint: **96% lift, 70% reach@2cm over 2 seeds** ([details](docs/phase3-h2-warmstart.md)). Verdict: H2 supported — a working policy survives 1500 PPO updates on this substrate; the random-init bootstrap is the bottleneck
- [ ] Phase 4 — writeup

## Result table

| Eval | Lift @ 4 cm | Reach @ 5 cm | Reach @ 2 cm | Mean cube z (m) | Mean goal-dist (m) |
|---|---|---|---|---|---|
| Bare-table act-1 (256 rollouts × 2 seeds) | 100.00% | 100.00% | 100.00% | 0.382 | 0.0032 |
| Granular zero-shot (act-1 policy, 256 × 2 seeds) | 94.53% | 94.14% | 92.58% | 0.352 | 0.0374 |
| Phase 2 retrained, curriculum on (256 × 1 seed) | 0.00% | 0.00% | 0.00% | 0.024 | 0.418 |
| Phase 3 H1 retrained, curriculum off (256 × 1 seed) | 0.00% | 0.00% | 0.00% | 0.022 | 0.440 |
| **Phase 3 H2 warm-start, seed 0** (256 rollouts) | **96.09%** | **95.70%** | **69.53%** | **0.348** | **0.0361** |
| **Phase 3 H2 warm-start, seed 42** (256 rollouts) | **96.48%** | **96.48%** | **72.27%** | **0.346** | **0.0371** |

**Phase comparison** — one rollout per phase, stacked top-to-bottom in chronological order. Future phases (H2, etc.) will be appended below using the same one-GIF-per-row layout.

**Phase 1 — zero-shot transfer of act-1 policy** (reach@2cm 92.58%, mean goal-dist degrades 12× vs bare table):

![phase 1 zero-shot](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/granular_zero_shot_seed42.gif)

**Phase 2 — retrain from scratch, curriculum on** (0% across all thresholds; policy collapses to "don't move" at iter 425 when the action_rate / joint_vel curriculum penalty completes its ramp):

![phase 2 retrained](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/granular_trained_seed42.gif)

**Phase 3 H1 — retrain from scratch, curriculum off** (0% across all thresholds; the cliff at iter 425 is gone, but no learning trajectory replaces it — H1 falsified):

![phase 3 H1 no curriculum](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/h1_nocurric_seed42.gif)

**Phase 3 H2 — warm-start from act-1 `model_1499.pt`, curriculum off** (lift 96.3%, reach@5cm 96.1%, reach@2cm 70.9% averaged over 2 seeds; warm-started PPO maintains task performance through 1500 fresh updates — H2 supported):

<!-- TODO(human): render H2 rollout GIF after restarting EC2 with offscreen render, then drop the URL here.
     Same path scheme as the others: results/videos/h2_warmstart_seed42.gif -->

## Diagnosis

Phase 1 confirms the gap is real: the same act-1 policy that hits 100% on the bare table loses 7 pp of lift success and 12× of mean goal-distance precision when 64 small spheres are placed under the cube. Phase 2 then asks the obvious follow-up — can PPO learn this substrate from scratch using the same hyperparameters that worked on the bare table? — and the answer is no: the policy collapses to "don't move" at iter 425, exactly when the `action_rate` and `joint_vel` curriculum schedules complete their −0.0001 → −0.1 ramp.

That collapse motivated two falsification experiments in phase 3.

**H1 — curriculum is the structural cause.** Removing the two schedules in `__post_init__` should let PPO learn unimpeded. *Falsified*: the curve no longer cliffs at iter 425, but mean reward also never lifts off its iter-0 baseline. Both retrains end at 0% / 0% / 0%; the curriculum was making the failure visible (the cliff), not causing it.

**H2 — exploration bootstrap is the structural cause.** A random-init policy can't find a useful gradient because reaching/lifting reward is too noisy under sphere contacts; if we initialise from a policy that already encodes the lift behaviour, PPO updates should produce informative gradients on every successful lift. *Supported*: warm-started from act-1's `model_1499.pt`, the same NoCurric task that left H1 stuck at 0% lets PPO fine-tune to 96% over 1500 fresh updates. Same optimiser, same substrate, same seed — only the initialisation differs.

**The H1/H2 contrast is the act-3 deliverable.** PPO can train on this contact-rich scene; it cannot bootstrap into it. The optimiser-level claim and the bootstrap-level claim are not the same claim, and naive PPO-from-scratch failures conflate them.

<!-- TODO(human): "what I'd try next" paragraph.
     CLAUDE.md §6: this is the part PIs read for research taste — must be the
     human's own voice. Likely candidates to discuss: (1) curriculum on the
     substrate itself (start with 0 spheres, grow density), (2) reward shaping
     on contact patterns (penalise sphere displacement during reach), (3) world
     models / model-based PPO that can plan around contact noise, (4) why a
     bare-table demo is a much cheaper "warm-start" than this whole pipeline
     and what that says about the value of teacher policies for sim-to-real
     transfer in clutter. Keep it 4–6 sentences, opinionated, no hedging. -->

## Discipline

Phase-gated, not auto-iterated. See [`CLAUDE.md`](CLAUDE.md) for why this
matters — and why the diagnostic prose in the eventual writeup must be
written by the human, not generated.

## Why "rigid-body proxy" instead of real granular media

A real continuous granular bed (sand, gravel) needs particle physics —
either PhysX PBD particles or a Warp implementation. Both exist; both blew
out act 2's budget. For act 3 the question of interest is **what does
PPO do when contact resolves through a cluster of free bodies, instead of
a single hard surface**, and that question is well-tested by 50–200
small spheres of comparable size to the manipulated cube. This is a
deliberate scope cut, documented, not a limitation glossed over.

## Sibling repos

- **Act 1** — [`isaac-lab-manipulation`](https://github.com/Yang251552/isaac-lab-manipulation): standard `Isaac-Lift-Cube-Franka-v0` reproduction. Provides the bare-table baseline (100%) and the warm-start checkpoint (`model_1499.pt`) that phase 3 H2 fine-tunes from.
- **Act 2** — `excavation-rl` (private): self-built granular-excavation substrate (custom Warp particle physics + hand-rolled PPO). Did not converge; kept as a documented engineering / lessons-learned artefact. The act-3 thesis is the deliberate counterpoint to act-2's auto-iteration mode.

## Phase docs

| Phase | Doc | One-line result |
|---|---|---|
| 0 | [`docs/phase0-substrate-decision.md`](docs/phase0-substrate-decision.md) | Substrate: 64 spheres × 2 cm × 8×8 grid via `RigidObjectCollection` |
| 1 | [`docs/phase1-zero-shot-transfer.md`](docs/phase1-zero-shot-transfer.md) | Zero-shot: lift drops 100% → 94.53%, mean goal-dist 12× |
| 2 | [`docs/phase2-train-from-scratch.md`](docs/phase2-train-from-scratch.md) | Retrain from scratch collapses at iter 425 to 0% |
| 3 H1 | [`docs/phase3-h1-no-curriculum.md`](docs/phase3-h1-no-curriculum.md) | Curriculum off → still 0%; H1 falsified |
| 3 H2 | [`docs/phase3-h2-warmstart.md`](docs/phase3-h2-warmstart.md) | Warm-start → 96%; H2 supported (exploration bootstrap is the bottleneck) |

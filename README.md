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
- [x] Phase 1 — zero-shot transfer of act-1 policy: **reach@2cm drops 100% → 91.02%, mean goal-dist degrades 13× (averaged over 2 seeds)** ([details](docs/phase1-zero-shot-transfer.md))
- [x] Phase 2 — re-train PPO from scratch: **policy collapses to 0% / 0% / 0% at iter 425 when the curriculum schedule kicks in** (stopped early per CLAUDE.md §2.1 mid-train rule; [details](docs/phase2-train-from-scratch.md))
- [x] Phase 3 H1 — disable curriculum: **also collapses to 0% / 0% / 0%** ([details](docs/phase3-h1-no-curriculum.md)). Verdict: H1 falsified — the curriculum is not the actor-killer; the bottleneck is upstream (exploration bootstrap)
- [x] Phase 3 H2 — warm-start from act-1 checkpoint: **96% lift, 70% reach@2cm over 2 seeds** ([details](docs/phase3-h2-warmstart.md)). Verdict: H2 supported — a working policy survives 1500 PPO updates on this substrate; the random-init bootstrap is the bottleneck
- [ ] Phase 4 — writeup

## Result table

| Eval | Lift @ 4 cm | Reach @ 5 cm | Reach @ 2 cm | Mean cube z (m) | Mean goal-dist (m) |
|---|---|---|---|---|---|
| Bare-table act-1 (256 rollouts × 2 seeds) | 100.00% | 100.00% | 100.00% | 0.382 | 0.0032 |
| Granular zero-shot (act-1 policy, 256 × 2 seeds) | 93.36% | 93.16% | 91.02% | 0.346 | 0.0421 |
| Phase 2 retrained, curriculum on (256 × 1 seed) | 0.00% | 0.00% | 0.00% | 0.024 | 0.418 |
| Phase 3 H1 retrained, curriculum off (256 × 1 seed) | 0.00% | 0.00% | 0.00% | 0.022 | 0.440 |
| **Phase 3 H2 warm-start, seed 0** (256 rollouts) | **96.09%** | **95.70%** | **69.53%** | **0.348** | **0.0361** |
| **Phase 3 H2 warm-start, seed 42** (256 rollouts) | **96.48%** | **96.48%** | **72.27%** | **0.346** | **0.0371** |

**Phase comparison** — one rollout per phase, stacked top-to-bottom in chronological order. Future phases (H2, etc.) will be appended below using the same one-GIF-per-row layout.

**Phase 1 — zero-shot transfer of act-1 policy** (reach@2cm 91.02%, mean goal-dist degrades 13× vs bare table; averaged over 2 seeds):

![phase 1 zero-shot](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/granular_zero_shot_seed42.gif)

**Phase 2 — retrain from scratch, curriculum on** (0% across all thresholds; policy collapses to "don't move" at iter 425 when the action_rate / joint_vel curriculum penalty completes its ramp):

![phase 2 retrained](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/granular_trained_seed42.gif)

**Phase 3 H1 — retrain from scratch, curriculum off** (0% across all thresholds; the cliff at iter 425 is gone, but no learning trajectory replaces it — H1 falsified):

![phase 3 H1 no curriculum](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/h1_nocurric_seed42.gif)

**Phase 3 H2 — warm-start from act-1 `model_1499.pt`, curriculum off** (lift 96.3%, reach@5cm 96.1%, reach@2cm 70.9% averaged over 2 seeds; warm-started PPO maintains task performance through 1500 fresh updates — H2 supported):

![phase 3 H2 warm-start](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/h2_warmstart_seed42.gif)

## Diagnosis

Phase 1 confirms the gap is real: the same act-1 policy that hits 100% on the bare table loses ~7 pp of lift success and 13× of mean goal-distance precision (averaged over 2 seeds) when 64 small spheres are placed under the cube. Phase 2 then asks the obvious follow-up — can PPO learn this substrate from scratch using the same hyperparameters that worked on the bare table? — and the answer is no: the policy collapses to "don't move" at iter 425, exactly when the `action_rate` and `joint_vel` curriculum schedules complete their −0.0001 → −0.1 ramp.

That collapse motivated two falsification experiments in phase 3.

**H1 — curriculum is the structural cause.** Removing the two schedules in `__post_init__` should let PPO learn unimpeded. *Falsified*: the curve no longer cliffs at iter 425, but mean reward also never lifts off its iter-0 baseline. Both retrains end at 0% / 0% / 0%; the curriculum was making the failure visible (the cliff), not causing it.

**H2 — exploration bootstrap is the structural cause.** A random-init policy can't find a useful gradient because reaching/lifting reward is too noisy under sphere contacts; if we initialise from a policy that already encodes the lift behaviour, PPO updates should produce informative gradients on every successful lift. *Supported*: warm-started from act-1's `model_1499.pt`, the same NoCurric task that left H1 stuck at 0% lets PPO fine-tune to 96% over 1500 fresh updates. Same optimiser, same substrate, same seed — only the initialisation differs.

**The H1/H2 contrast is the act-3 deliverable.** PPO can train on this contact-rich scene; it cannot bootstrap into it. The optimiser-level claim and the bootstrap-level claim are not the same claim, and naive PPO-from-scratch failures conflate them.

Side-by-side rollout, left to right: zero-shot act-1 policy (lift 94.5%) | phase 2 retrained, curriculum on (0%) | phase 3 H1 retrained, curriculum off (0%) | phase 3 H2 warm-started, curriculum off (lift 96%):

![phase comparison 4-panel](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/phase_comparison_4panel.gif)

## What I'd try next

If I had another budget cycle on this, the experiment I would actually run first is a substrate curriculum: start training with zero spheres on the table (i.e. the act-1 bare-table task), and grow sphere density linearly over the first ~500 iters until the substrate matches the locked phase-1 config. The reason I want to try this one specifically is that H2 already says PPO can stay on the task manifold once it is there — so a substrate that starts on bare table puts the random-init policy onto the manifold for free, and the question becomes whether the policy can keep up as the substrate gets harder around it. This also has the nice property that it does not require a separately-trained teacher checkpoint, so it is a single-run experiment that fits inside the act-3 framing. If it works, it is a stronger statement than H2 because the policy never had a privileged warm start.

The other thing I am curious about but did not do is opening the H2 evaluation per-episode and splitting the inside-cluster vs outside-cluster spawns. Phase 1 noted the bimodal split (~56% inside, ~44% outside) but never reported the split in eval — I think the H2 reach@2cm drop from 92.58% to ~70% is probably almost entirely on the inside-cluster spawns, and if that were verified it would change how I read the H2 result. The H2 fine-tuning probably did not "lose precision generally"; it probably "stayed precise on outside-cluster, gained robustness inside-cluster". That is a different and more useful story than "warm-start fine-tune trades precision for robustness", and the experiment to tell those two apart is a couple of hours of `eval_granular.py` re-runs with cluster-membership tags written out, not new training.

What I would *not* try is reward shaping on the contact pattern (e.g. penalising sphere displacement during the reach phase). I do not think that is uninteresting, but it changes the substrate definition of the problem, which makes the result hard to compare with phase 1's zero-shot transfer baseline. The whole point of act-3 was that the substrate stays fixed and the questions are about policy and optimiser; once you start re-shaping the reward you are answering a different question and the act-1 baseline stops being the right anchor.

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
| 1 | [`docs/phase1-zero-shot-transfer.md`](docs/phase1-zero-shot-transfer.md) | Zero-shot: lift drops 100% → 93.4%, mean goal-dist 13× (2 seeds avg) |
| 2 | [`docs/phase2-train-from-scratch.md`](docs/phase2-train-from-scratch.md) | Retrain from scratch collapses at iter 425 to 0% |
| 3 H1 | [`docs/phase3-h1-no-curriculum.md`](docs/phase3-h1-no-curriculum.md) | Curriculum off → still 0%; H1 falsified |
| 3 H2 | [`docs/phase3-h2-warmstart.md`](docs/phase3-h2-warmstart.md) | Warm-start → 96%; H2 supported (exploration bootstrap is the bottleneck) |

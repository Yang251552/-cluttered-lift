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
- [ ] Phase 3 H2 — warm-start from act-1 checkpoint (next session)
- [ ] Phase 4 — writeup

**Phase comparison** — one rollout per phase, stacked top-to-bottom in chronological order. Future phases (H2, etc.) will be appended below using the same one-GIF-per-row layout.

**Phase 1 — zero-shot transfer of act-1 policy** (reach@2cm 92.58%, mean goal-dist degrades 12× vs bare table):

![phase 1 zero-shot](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/granular_zero_shot_seed42.gif)

**Phase 2 — retrain from scratch, curriculum on** (0% across all thresholds; policy collapses to "don't move" at iter 425 when the action_rate / joint_vel curriculum penalty completes its ramp):

![phase 2 retrained](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/granular_trained_seed42.gif)

**Phase 3 H1 — retrain from scratch, curriculum off** (0% across all thresholds; the cliff at iter 425 is gone, but no learning trajectory replaces it — H1 falsified):

![phase 3 H1 no curriculum](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/h1_nocurric_seed42.gif)

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

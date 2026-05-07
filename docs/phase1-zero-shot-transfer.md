# Phase 1 — Zero-shot Transfer of the Act-1 Policy onto the Granular Scene

**Time spent: ≈90 min (under the 4 h budget). Budget consumed mostly on
substrate calibration, not the eval itself.**

## What was tested

Same 256-rollout eval pipeline as `isaac-lab-manipulation/scripts/eval_lift_cube.py`,
but on a granular scene that adds 64 small spheres to the standard
`Isaac-Lift-Cube-Franka-Play-v0` task. The PPO checkpoint used is the one act 1
trained on the bare-table version (`model_1499.pt`, 100% / 100% / 100% over 256
rollouts × 2 seeds on bare table).

Observation, action, and reward configs are inherited unchanged so the policy
sees the same input/output it was trained for. The only thing that changed is
the contact pattern at the cube spawn region.

## Substrate calibration: four configurations

The first three runs were calibration. The substrate parameters that produce
a research-meaningful drop in success rate are non-obvious; this section
documents how I converged on the final config.

All four calibration runs use seed 0 only (1d's second seed is added later, see "Results" section below).

| Run | Spheres | Radius | Spawn region | Spawn height | Reach @ 2 cm | Mean goal-dist |
|---|---|---|---|---|---|---|
| 1a | 64 | 1.5 cm | 30 × 50 cm wide | from sky (z=1.5 m) | 94.92% | 44.2 mm |
| 1b | 64 | 2.5 cm | 30 × 50 cm wide | from sky (z=1.5 m) | 95.70% | 35.8 mm |
| 1c | 64 | 2.0 cm | 16 × 20 cm tight | from sky (z=1.5 m) | **100.00%** | 3.0 mm |
| **1d** | **64** | **2.0 cm** | **8×8 grid, 28 × 28 cm** | **on table (z=0.025)** | **92.58%** | **37.4 mm** |

Bare-table reference (act 1, same checkpoint): 100% / 100% / 100%, mean goal-dist 3.2 mm.

## Why the first three configs underdelivered

Runs 1a–1c spawn the spheres at z = 1.5 m and let them fall onto the table
under gravity, with the reset-event scatter range applied as an offset to that
high spawn position. Two consequences:

1. The spheres are airborne for the first ~0.5 s of the episode. The robot's
   reach phase begins around the same time. By the time the cube is grasped,
   spheres have either settled away from the cube or been pushed aside by the
   robot reaching motion.
2. The lift task's success criterion is z-axis displacement of the cube (above
   4 cm) and final cube-goal distance. Spheres scattered in xy at table level
   do not occupy the lift trajectory once they have settled, so the cube goes
   straight up out of the cluster without sustained contact.

Run 1c (tight scatter from sky) is the cleanest demonstration of the problem:
even with all 64 spheres falling onto a 16 × 20 cm region centred on the cube
spawn, success rate is identical to bare table. The cube simply lifts before
the spheres are anywhere consequential.

## What 1d changes

Spheres start in an 8×8 staggered grid on the table, with their centres at
z = 2.5 cm (the same height as the cube centre). The grid spans 28 × 28 cm
centred on the cube spawn pose. A small ±5 mm jitter is applied at reset to
break determinism. Cube spawn pose-range is unchanged from the original Lift
config (±10 cm in x, ±25 cm in y).

This produces a controlled but non-trivial change to the contact pattern: the
cube is now directly inside or adjacent to the sphere cluster at episode start,
and any lift trajectory must displace at least the spheres in immediate
contact.

## Results — phase 1 final (run 1d, two seeds)

| Metric | Bare-table (act 1, 2 seeds) | Granular 1d, seed 0 | Granular 1d, seed 42 | Granular 1d, mean | Change vs bare |
|---|---|---|---|---|---|
| Lift success @ 4 cm | 100.00% | 94.53% | 92.19% | 93.36% | −6.6 pp |
| Reach success @ 5 cm | 100.00% | 94.14% | 92.19% | 93.16% | −6.8 pp |
| Reach success @ 2 cm | 100.00% | 92.58% | 89.45% | 91.02% | −9.0 pp |
| Mean cube z at episode end | 0.382 m | 0.352 m | 0.340 m | 0.346 m | −3.6 cm |
| Mean goal-dist at episode end | 3.2 mm | 37.4 mm | 46.7 mm | 42.1 mm | **×13.1** |

Two seeds agree to within ~3 pp on each success-rate metric and ~9 mm on goal-distance.
Seed-42 sits slightly worse than seed-0 on every metric, which is consistent
with the bimodal-spawn argument below — different seed sequences land
different fractions of episodes inside the sphere grid.

The success rates drop between ~7 pp at the loose lift threshold and ~9 pp at
the tight reach threshold. The mean goal-dist tells a much stronger story:
the bare-table policy hits the goal within 3 mm; on the granular scene the
average miss is 42 mm. A success-rate bucket count is the wrong reduction
here because the failures are not uniform: the policy mostly still works,
but a tail of episodes ends with the cube tens of centimetres from the goal,
and that tail is what drives the mean.

## Bimodal observation (substrate-vs-spawn geometry)

The cube's spawn region (±10 cm × ±25 cm) is wider in y than the sphere grid
(±14 cm × ±14 cm). About 56% of cube spawns fall inside the grid; the other
44% land outside the cluster on bare table. If the outside-cluster spawns
succeed at 100% (matching bare-table), the two-seed average inside-cluster
success rate is ~84% (solving 0.56·X + 0.44·1.00 = 0.9102 → X ≈ 0.84).

This bimodal split is the actual research signal phase 1 produces. It is
the structural feature that phases 2 and 3 should explain or close.

## What I take away from the bimodal split

What surprised me most about phase 1 is that the failures are not uniform across the spawn region. I had been picturing the −7 pp drop as a "the spheres make every lift slightly harder" effect, but the bimodal-spawn calculation says it is more like a 56% / 44% mixture of "this is basically a buried-cube task at ~84% success" and "this is basically the bare table at ~100% success". So the eval number 91.02% is averaging two regimes that the policy actually treats differently, and the policy itself has no observation that tells it which regime it is in — it just runs the same bare-table-trained behaviour on both. That is part of what I think is interesting about the substrate: it forces a kind of conditional behaviour without giving the policy anything to condition on, so the "smart thing to do" and the "best thing PPO can converge to" are different.

I think for phase 3 the right way to think about this is that any fix has to either give the policy a contact-state observation (which would change the act-1 transfer story — the bare-table policy would not generalise) or absorb the inside-cluster failures into training (which is what phase 2 attempts). I am not sure yet which side of that I would push first, and one thing I would honestly like to do later is split the eval log into inside-cluster and outside-cluster buckets and report them separately, since the averaged number is doing more work than it should. For now I am leaving the bimodality on purpose, because the inside / outside contrast is the one thing in phase 1 that will keep being a useful diagnostic signal for the rest of the project.

## Phase 2 setup

The granular env config is locked at run 1d's parameters (see
`scripts/granular_lift_env.py`). Phase 2 trains PPO from scratch on
`Isaac-Granular-Lift-Cube-Franka-v0` (the training-distribution variant)
using the same `LiftCubePPORunnerCfg` as act 1. No hyperparameter changes.

Risk for phase 2: 4096 envs × 64 spheres = 262 K rigid bodies in the
PhysX scene. Runs 1a–1d already produced a "patch buffer overflow" warning
at 256 envs × 64. Scaling 16× is likely to OOM or hit a contact-pair
ceiling. Default plan: start phase 2 at `--num_envs 1024`. If that
trains cleanly, raise to 2048 or 4096 with the patch budget retuned via
the sim-cfg side. If 1024 still overflows, reduce sphere count first
(half resolution: 32 spheres in a 6×6 grid at the same spacing).

## Eval logs

- [`phase1a_64x1.5cm_wide.log`](../results/logs/phase1a_64x1.5cm_wide.log) — 64 × 1.5 cm spheres, wide scatter from sky
- [`phase1b_64x2.5cm_wide.log`](../results/logs/phase1b_64x2.5cm_wide.log) — 64 × 2.5 cm spheres, wide scatter from sky
- [`phase1c_64x2.0cm_tight.log`](../results/logs/phase1c_64x2.0cm_tight.log) — 64 × 2.0 cm spheres, tight scatter from sky
- [`phase1d_64x2.0cm_grid_seed0.log`](../results/logs/phase1d_64x2.0cm_grid_seed0.log) — **locked**: 64 × 2.0 cm in 8×8 on-table grid, seed 0
- [`phase1d_64x2.0cm_grid_seed42.log`](../results/logs/phase1d_64x2.0cm_grid_seed42.log) — locked config, seed 42 (added 2026-05-07)

## Visualization

![Zero-shot policy on granular scene, 16 envs, 12 s @ 15 fps](https://raw.githubusercontent.com/Yang251552/cluttered-lift/main/results/videos/granular_zero_shot_seed42.gif)

The act-1 checkpoint, dropped onto the 1d granular scene, lifts most cubes
and pushes spheres aside. The 7-percentage-point reach@2cm drop is what
this looks like: most lifts succeed, but a subset of episodes show the cube
catching on the cluster or settling off-axis from the goal. Full mp4 at
[`results/videos/granular_zero_shot_seed42.mp4`](../results/videos/granular_zero_shot_seed42.mp4).

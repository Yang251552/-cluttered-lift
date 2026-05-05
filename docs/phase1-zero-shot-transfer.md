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

## Results — phase 1 final (run 1d)

| Metric | Bare-table (act 1) | Granular (1d) | Change |
|---|---|---|---|
| Lift success @ 4 cm | 100.00% | 94.53% | −5.5 pp |
| Reach success @ 5 cm | 100.00% | 94.14% | −5.9 pp |
| Reach success @ 2 cm | 100.00% | 92.58% | −7.4 pp |
| Mean cube z at episode end | 0.382 m | 0.352 m | −3.0 cm |
| Mean goal-dist at episode end | 3.2 mm | 37.4 mm | **×11.7** |

The success rates drop ~7 percentage points. The mean goal-dist tells a much
stronger story: the bare-table policy hits the goal within 3 mm; on the
granular scene the average miss is 37 mm. A success-rate bucket count is the
wrong reduction here because the failures are not uniform: the policy mostly
still works, but a tail of episodes ends with the cube tens of centimetres
from the goal, and that tail is what drives the mean.

## Bimodal observation (substrate-vs-spawn geometry)

The cube's spawn region (±10 cm × ±25 cm) is wider in y than the sphere grid
(±14 cm × ±14 cm). About 56% of cube spawns fall inside the grid; the other
44% land outside the cluster on bare table. If the outside-cluster spawns
succeed at 100% (matching bare-table), the inside-cluster success rate is
~85% (solving 0.56·X + 0.44·1.00 = 0.9258 → X ≈ 0.85).

This bimodal split is the actual research signal phase 1 produces. It is
the structural feature that phases 2 and 3 should explain or close.

> **TODO (human): research observation paragraph.**
>
> A few sentences in your own voice on what the bimodal split tells you,
> what you find interesting about it, and what you'd want phase 3 to test.
> Claude is not writing this part — it has to read like the engineer who
> ran the experiment, not like a narrator.

## Phase 2 setup

The granular env config is locked at run 1d's parameters (see
`scripts/granular_lift_env.py`). Phase 2 trains PPO from scratch on
`Isaac-Granular-Lift-Cube-Franka-v0` (the training-distribution variant)
using the same `LiftCubePPORunnerCfg` as act 1. No hyperparameter changes.

Risk for phase 2: 4096 envs × 64 spheres = 262 K rigid bodies in the
PhysX scene. Run 1a–1d already produced a "patch buffer overflow" warning
at 256 envs × 64. Scaling 16× is likely to OOM or hit a contact-pair
ceiling. Default plan: start phase 2 at `--num_envs 1024`. If that
trains cleanly, raise to 2048 or 4096 with the patch budget retuned via
the sim-cfg side. If 1024 still overflows, reduce sphere count first
(half resolution: 32 spheres in a 6×6 grid at the same spacing).

## Eval logs

- [`phase1a_64x1.5cm_wide.log`](../results/logs/phase1a_64x1.5cm_wide.log) — 64 × 1.5 cm spheres, wide scatter from sky
- [`phase1b_64x2.5cm_wide.log`](../results/logs/phase1b_64x2.5cm_wide.log) — 64 × 2.5 cm spheres, wide scatter from sky
- [`phase1c_64x2.0cm_tight.log`](../results/logs/phase1c_64x2.0cm_tight.log) — 64 × 2.0 cm spheres, tight scatter from sky
- [`phase1d_64x2.0cm_grid.log`](../results/logs/phase1d_64x2.0cm_grid.log) — **locked**: 64 × 2.0 cm in 8×8 on-table grid

## Visualization

![Zero-shot policy on granular scene, 16 envs, 12 s @ 15 fps](https://raw.githubusercontent.com/Yang251552/-cluttered-lift/main/results/videos/granular_zero_shot_seed42.gif)

The act-1 checkpoint, dropped onto the 1d granular scene, lifts most cubes
and pushes spheres aside. The 7-percentage-point reach@2cm drop is what
this looks like: most lifts succeed, but a subset of episodes show the cube
catching on the cluster or settling off-axis from the goal. Full mp4 at
[`results/videos/granular_zero_shot_seed42.mp4`](../results/videos/granular_zero_shot_seed42.mp4).

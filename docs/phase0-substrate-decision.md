# Phase 0 — Substrate Decision

**Time spent: ≈10 min (under the 2 h budget).**

## What was probed

On Isaac Lab v2.3.2 + isaacsim 4.5.0.0 (the act-1 install), I checked four
candidate paths for representing a granular bed:

| Path | Status |
|---|---|
| 1. PBD particle system via `omni.physx.particle` | No `*particle*` directory in the installed `isaacsim` site-packages; would require reaching into the raw Isaac Sim API and writing the wrapper from scratch. Unbounded scope. |
| 2. Isaac Lab particle wrapper | Source grep for `ParticleSystem / particle_system / PBDParticle` returns one hit (a util in `sim/utils/prims.py`, not a wrapper). No first-class API. |
| 3. Warp particles (continuous granular) | **Excluded by act-3 thesis.** That was act 2; using it again breaks the story. |
| 4. Multi-rigid-body proxy | `isaaclab.assets.RigidObjectCollection` exists and is first-class; `isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg` exists; `dexsuite_env_cfg.py` already uses MultiAsset, so there is at least one canonical example to copy from. Standard PhysX rigid bodies. |

## Decision

**Path 4 — multi-rigid-body cluster as proxy for granular media**, implemented
via `RigidObjectCollection` of 50–200 small spheres around the cube spawn
location.

## Justification

1. **Zero new infrastructure.** It uses the same PhysX rigid-body path as
   the cube, the table, and the Franka itself. No custom physics, no Warp.

2. **Performance is well-understood.** PhysX scales linearly in contact
   pairs; 64–200 small bodies on top of an environment that already runs
   4096 envs in parallel for Lift-Cube has comfortable headroom. Phase 1
   will measure exactly.

3. **It is the act-3 thesis.** Act 2 spent its budget trying to model a
   *continuous* granular medium. Act 3 deliberately scope-cuts that to "a
   *cluster* of free bodies", which keeps the research question that
   actually motivated act 2 (what does PPO do when contact resolves
   through a non-rigid pile?) and drops the part that was eating the
   engineering budget (faithful continuous-media physics).

4. **The honest writeup writes itself.** "I scoped granular media to a
   tractable proxy" is a clean line in the eventual diagnosis — it is a
   research-maturity signal, not a limitation to apologise for.

## What this rules out (deliberately)

- Studies of grain-size effects, polydispersity, friction-angle effects on
  bulk behaviour — those need real continuum physics.
- Anything that requires faithful momentum transfer through a deep static
  bed (e.g., a buried object that needs to be excavated). The bed here is
  shallow and the bodies are individually free.
- Comparison to published granular-manipulation results that use real
  particle physics — the regime is different and a comparison would
  mislead.

## Implementation hooks for phase 1

- `isaaclab.assets.RigidObjectCollection` (batch of N rigid bodies — Phase 1 confirms exact API)
- `isaaclab.sim.spawners.wrappers.MultiAssetSpawnerCfg` for spawn-time variation
- Reference example: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/dexsuite_env_cfg.py`
- Base task to fork: `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/`

## Phase 0 output: locked

- Substrate: **multi-rigid-body cluster, 50–200 small spheres**
- API: `RigidObjectCollection`
- Reset: spheres scattered uniformly within a small bounded region around
  the cube's spawn position; phase 1 will pick exact bounds based on a
  feasibility test.

Phase 1 starts with this locked.

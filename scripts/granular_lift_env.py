"""Granular-Lift-Cube-Franka environment.

Extends the official `Isaac-Lift-Cube-Franka-v0` task by spawning N small
spheres around the cube spawn region. Observations / actions / rewards
are inherited unchanged so that the act-1 PPO checkpoint is a valid
zero-shot transfer policy on this scene.

The spheres are deliberately a *proxy* for granular media — see
`docs/phase0-substrate-decision.md` for the scope justification.

Registers two task IDs on import:
    Isaac-Granular-Lift-Cube-Franka-v0       (full randomization, training)
    Isaac-Granular-Lift-Cube-Franka-Play-v0  (256-env eval, no obs noise)
"""

from __future__ import annotations

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.envs import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
    FrankaCubeLiftEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.manipulation.lift.config.franka import agents

# ----- substrate parameters (locked for phase 1) ---------------------------
N_SPHERES = 64
GRID_SIDE = 8                # 8x8 staggered grid of spheres
GRID_SPACING = 0.04          # 4 cm centre-to-centre — touching at r=2cm
SPHERE_RADIUS = 0.020        # 2.0 cm
SPHERE_MASS = 0.005          # 5 g, light enough that gripper can displace
SPHERE_FRICTION = 0.5
CLUSTER_CENTER = (0.5, 0.0)  # matches the cube spawn area centre
SPHERE_TABLE_Z = 0.025       # sphere centre z above table top
JITTER_RANGE = {             # small per-reset jitter to break determinism
    "x": (-0.005, 0.005),
    "y": (-0.005, 0.005),
    "z": (0.0, 0.0),
}
# ---------------------------------------------------------------------------


def _grid_position(idx: int) -> tuple[float, float, float]:
    """8x8 grid centred on CLUSTER_CENTER at table level."""
    row, col = divmod(idx, GRID_SIDE)
    dx = (col - (GRID_SIDE - 1) / 2.0) * GRID_SPACING
    dy = (row - (GRID_SIDE - 1) / 2.0) * GRID_SPACING
    return (CLUSTER_CENTER[0] + dx, CLUSTER_CENTER[1] + dy, SPHERE_TABLE_Z)


def _make_sphere_cfg(idx: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Sphere_{idx:03d}",
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SPHERE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # NOTE: no per-sphere physics_material — PhysX caps at 64K materials
            # globally and 1024 envs × 64 spheres would overflow. We rely on the
            # PhysX default material so all spheres share one instance.
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.65, 0.45, 0.30), roughness=0.6
            ),
        ),
        # staggered grid on table; reset adds small jitter
        init_state=RigidObjectCfg.InitialStateCfg(pos=_grid_position(idx)),
    )


def _add_spheres(cfg) -> None:
    """Attach `N_SPHERES` sphere rigid objects + their reset events to a Lift-Cube cfg."""
    for i in range(N_SPHERES):
        name = f"sphere_{i:03d}"
        setattr(cfg.scene, name, _make_sphere_cfg(i))
        setattr(
            cfg.events,
            f"reset_{name}",
            EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": JITTER_RANGE,
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(name),
                },
            ),
        )


@configclass
class GranularFrankaCubeLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """Training-distribution variant — 4096 envs, full randomization."""

    def __post_init__(self):
        super().__post_init__()
        _add_spheres(self)


@configclass
class GranularFrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg_PLAY):
    """Play / eval variant — same as the base play but with the granular cluster."""

    def __post_init__(self):
        super().__post_init__()
        _add_spheres(self)


def _disable_curriculum(cfg) -> None:
    """Remove action_rate / joint_vel curriculum schedules. Keeps the reward
    weights frozen at their base values for the entire run.

    Phase 2 observed the trained policy collapse exactly when these schedules
    finished ramping (~iter 425, matches 10000 env-steps per env). H1 tests
    whether the curriculum is the actor-killer.
    """
    for name in ("action_rate", "joint_vel"):
        if hasattr(cfg.curriculum, name):
            delattr(cfg.curriculum, name)


@configclass
class GranularFrankaCubeLiftEnvCfg_NoCurric(GranularFrankaCubeLiftEnvCfg):
    """Phase-3 H1 training variant: cluttered scene with curriculum disabled."""

    def __post_init__(self):
        super().__post_init__()
        _disable_curriculum(self)


@configclass
class GranularFrankaCubeLiftEnvCfg_NoCurric_PLAY(GranularFrankaCubeLiftEnvCfg_PLAY):
    """Phase-3 H1 eval variant."""

    def __post_init__(self):
        super().__post_init__()
        _disable_curriculum(self)


# ----- gym registration ----------------------------------------------------

_TASK_TRAIN = "Isaac-Granular-Lift-Cube-Franka-v0"
_TASK_PLAY = "Isaac-Granular-Lift-Cube-Franka-Play-v0"
_TASK_TRAIN_H1 = "Isaac-Cluttered-Lift-Cube-Franka-NoCurric-v0"
_TASK_PLAY_H1 = "Isaac-Cluttered-Lift-Cube-Franka-NoCurric-Play-v0"

if _TASK_TRAIN not in gym.registry:
    gym.register(
        id=_TASK_TRAIN,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": GranularFrankaCubeLiftEnvCfg,
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"
            ),
        },
        disable_env_checker=True,
    )

if _TASK_PLAY not in gym.registry:
    gym.register(
        id=_TASK_PLAY,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": GranularFrankaCubeLiftEnvCfg_PLAY,
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"
            ),
        },
        disable_env_checker=True,
    )

if _TASK_TRAIN_H1 not in gym.registry:
    gym.register(
        id=_TASK_TRAIN_H1,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": GranularFrankaCubeLiftEnvCfg_NoCurric,
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"
            ),
        },
        disable_env_checker=True,
    )

if _TASK_PLAY_H1 not in gym.registry:
    gym.register(
        id=_TASK_PLAY_H1,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": GranularFrankaCubeLiftEnvCfg_NoCurric_PLAY,
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"
            ),
        },
        disable_env_checker=True,
    )

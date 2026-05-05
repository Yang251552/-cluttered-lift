"""Evaluate a Lift-Cube checkpoint on the granular-pick scene.

Mirrors `isaac-lab-manipulation/scripts/eval_lift_cube.py` but registers the
new granular task before launching the env. Same metrics so the comparison
table is apples-to-apples.

Usage (on EC2):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_granular.py \\
        --checkpoint <path> [--task Isaac-Granular-Lift-Cube-Franka-Play-v0] \\
        [--num_envs 256] [--episode_steps 300]
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Granular-Lift-Cube-Franka-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episode_steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch

from isaaclab.utils.math import combine_frame_transforms
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

# Register the granular task (must run before gym.make / hydra_task_config).
import granular_lift_env  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obj = env.unwrapped.scene["object"]
    robot = env.unwrapped.scene["robot"]

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    n_envs = args_cli.num_envs
    device = env.unwrapped.device
    final_cube_z = torch.full((n_envs,), float("nan"), device=device)
    final_goal_dist = torch.full((n_envs,), float("nan"), device=device)
    recorded = torch.zeros(n_envs, dtype=torch.bool, device=device)

    with torch.inference_mode():
        for _ in range(args_cli.episode_steps):
            actions = policy(obs)
            cube_pos_w = obj.data.root_pos_w.clone()
            cmd = env.unwrapped.command_manager.get_command("object_pose")
            goal_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cmd[:, :3])
            cube_z = cube_pos_w[:, 2]
            goal_dist = torch.norm(cube_pos_w - goal_w, dim=1)

            obs, _, dones, _ = env.step(actions)

            new_done = dones.bool() & (~recorded)
            final_cube_z = torch.where(new_done, cube_z, final_cube_z)
            final_goal_dist = torch.where(new_done, goal_dist, final_goal_dist)
            recorded = recorded | new_done

            if recorded.all():
                break

    finished = recorded.sum().item()
    cz = final_cube_z[recorded]
    gd = final_goal_dist[recorded]

    lift_succ = (cz > 0.04).float().mean().item() * 100
    reach5 = (gd < 0.05).float().mean().item() * 100
    reach2 = (gd < 0.02).float().mean().item() * 100

    print("\n" + "=" * 60)
    print(f"GRANULAR EVAL  task={args_cli.task}  seed={args_cli.seed}  envs={n_envs}")
    print("=" * 60)
    print(f"Episodes finished              : {finished}/{n_envs}")
    print(f"Lift success    (cube_z > 4 cm): {lift_succ:6.2f} %")
    print(f"Reach success   (dist < 5 cm)  : {reach5:6.2f} %")
    print(f"Reach success   (dist < 2 cm)  : {reach2:6.2f} %")
    print(f"Mean cube_z   at episode end   : {cz.mean().item():.4f} m")
    print(f"Mean goal-dist at episode end  : {gd.mean().item():.4f} m")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

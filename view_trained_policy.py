#!/usr/bin/env python
"""
View a trained SAC/TQC policy playing in the FoosballEnv
with a MuJoCo viewer window.

Usage example:
    python view_trained_policy.py \
        --model-path foosball_sac_model.zip \
        --algo sac \
        --episodes 20
"""

import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer

from stable_baselines3 import SAC
from sb3_contrib import TQC

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


# -----------------------------------------------------------
# Black-player lifting (same as your sanity script)
# -----------------------------------------------------------

BLACK_ROT_ACT_IDX = [9, 11, 13, 15]


def _get_black_rot_qpos_indices(env):
    """
    Find qpos indices for the joints driven by black-rotation actuators.
    Cache them on env as _black_rot_qpos_idx.
    """
    if hasattr(env, "_black_rot_qpos_idx"):
        return env._black_rot_qpos_idx

    model = env.model
    qpos_idx_list = []

    print("[BLACK-LIFT] Resolving black rotation qpos indices...")

    for act_id in BLACK_ROT_ACT_IDX:
        if act_id >= model.nu:
            print(f"  [WARN] actuator index {act_id} >= nu={model.nu}, skipping")
            continue

        joint_id = int(model.actuator_trnid[act_id, 0])
        if joint_id < 0:
            print(
                f"  [WARN] actuator {act_id} has no joint, "
                f"trnid={model.actuator_trnid[act_id]}, skipping"
            )
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos_idx_list.append(qpos_adr)

        try:
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
        except Exception:
            act_name = f"act_{act_id}"

        try:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        except Exception:
            joint_name = f"joint_{joint_id}"

        print(
            f"  [BLACK-LIFT] actuator {act_id} ({act_name}) "
            f"-> joint {joint_id} ({joint_name}), qpos[{qpos_adr}]"
        )

    env._black_rot_qpos_idx = qpos_idx_list
    print("[BLACK-LIFT] qpos indices for black rotation joints:", qpos_idx_list)
    return qpos_idx_list


def lift_black_players(env, angle_rad=np.deg2rad(90.0)):
    """
    Rotate black players “up” so they aren't blocking the ball.
    Call this right after reset.
    """
    qpos_idx_list = _get_black_rot_qpos_indices(env)
    if not qpos_idx_list:
        print("[BLACK-LIFT] No black rotation joints found; skipping lift.")
        return

    for idx in qpos_idx_list:
        env.data.qpos[idx] = angle_rad
    mujoco.mj_forward(env.model, env.data)

    print("[BLACK-LIFT] Black players lifted by", angle_rad, "rad")


# -----------------------------------------------------------
# Episode classification helpers (same idea as your script)
# -----------------------------------------------------------

def infer_termination_type(terminated, truncated, info, timed_out: bool):
    goal_flag      = bool(info.get("goal_scored", False))
    stagnant_flag  = bool(info.get("ball_stagnant", False))
    over_time_flag = bool(info.get("over_max_time", False))

    if goal_flag:
        return "GOAL"
    if stagnant_flag:
        return "STAGNATION"
    if timed_out or truncated or over_time_flag:
        return "TIMEOUT"
    if terminated:
        return "OTHER-TERMINATED"
    return "UNKNOWN"


# -----------------------------------------------------------
# Load trained model (SAC or TQC)
# -----------------------------------------------------------

def load_model(model_path: str, algo: str):
    algo = algo.lower()
    if algo == "sac":
        model = SAC.load(model_path)
        print(f"[MODEL] Loaded SAC model from {model_path}")
    elif algo == "tqc":
        model = TQC.load(model_path)
        print(f"[MODEL] Loaded TQC model from {model_path}")
    else:
        raise ValueError("algo must be 'sac' or 'tqc'")

    return model


# -----------------------------------------------------------
# Core rollout logic with viewer + trained policy
# -----------------------------------------------------------

def run_policy_rollouts_with_viewer(
    model,
    algo_name: str = "SAC",
    n_episodes: int = 10,
    max_steps_per_ep: int = 300,
    sleep_scale: float = 1.0,
    deterministic: bool = True,
):
    """
    Run episodes where the protagonist's actions come from the trained model.
    Render with MuJoCo viewer.
    """
    # Use the full-information env, no rendering (we do raw MuJoCo viewer)
    env = FoosballEnv(
        render_mode=None,
        verbose_mode=False,
        play_until_goal=False,
    )

    model_mj = env.model
    data = env.data

    ep_stats = []

    with viewer.launch_passive(model_mj, data) as v:
        print("[VIEWER] Window opened. Running policy episodes...")

        try:
            for ep in range(n_episodes):
                obs, info = env.reset(seed=1000 + ep)
                lift_black_players(env)

                ep_return = 0.0
                steps = 0
                last_info = info
                terminated = False
                truncated = False
                timed_out = True

                print(f"\n========== POLICY EPISODE {ep + 1}/{n_episodes} ==========")

                for t in range(max_steps_per_ep):
                    # ---- use trained policy here ----
                    action, _ = model.predict(obs, deterministic=deterministic)

                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_return += float(reward)
                    steps += 1
                    last_info = info

                    if not v.is_running:
                        print("[VIEWER] Window closed by user; stopping rollouts.")
                        return
                    v.sync()

                    dt = env.model.opt.timestep * getattr(env, "frame_skip", 1)
                    time.sleep(dt * sleep_scale)

                    if terminated or truncated:
                        timed_out = False
                        break

                term_type = infer_termination_type(
                    terminated=terminated,
                    truncated=truncated,
                    info=last_info,
                    timed_out=timed_out,
                )

                ball_x = last_info.get("ball_x", None)
                ball_y = last_info.get("ball_y", None)

                print(
                    f"Episode {ep + 1:02d} finished:\n"
                    f"  steps:      {steps}\n"
                    f"  return:     {ep_return:.3f}\n"
                    f"  term_type:  {term_type}\n"
                    f"  ball_xy:    ({ball_x}, {ball_y})\n"
                    f"  terminated: {terminated}, truncated: {truncated}"
                )

                ep_stats.append((ep_return, steps, term_type))

            print("\n[VIEWER] Policy episodes done. You can close the window now.")
            time.sleep(2.0)

        finally:
            env.close()

    # Console summary
    from collections import Counter
    print("\n================== OVERALL POLICY SUMMARY ==================")
    counts = Counter(t for _, _, t in ep_stats)
    for i, (ret, steps, ttype) in enumerate(ep_stats, start=1):
        print(f"Ep {i:02d}: return={ret:8.3f}  steps={steps:3d}  type={ttype}")
    print("Termination counts:", dict(counts))


# -----------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained model .zip file (e.g., foosball_sac_model.zip)",
    )
    parser.add_argument(
        "--algo",
        choices=["sac", "tqc"],
        default="sac",
        help="Which algo was used to train the model (sac or tqc)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to view",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--sleep-scale",
        type=float,
        default=10.0,
        help=">1.0 slower, <1.0 faster, 0.0 = as fast as possible",
    )

    args = parser.parse_args()

    model = load_model(args.model_path, args.algo)
    run_policy_rollouts_with_viewer(
        model=model,
        algo_name=args.algo.upper(),
        n_episodes=args.episodes,
        max_steps_per_ep=args.max_steps,
        sleep_scale=args.sleep_scale,
        deterministic=True,
    )


if __name__ == "__main__":
    main()

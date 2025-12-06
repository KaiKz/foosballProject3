#!/usr/bin/env mjpython
"""
FoosballEnv sanity rollouts with a MuJoCo viewer.

- Lifts black players as blockers.
- Runs several random episodes, with a “kick” bias on yellow attack.
- Shows everything in the MuJoCo viewer window.
"""

import time
import numpy as np
import mujoco
from mujoco import viewer

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


# -----------------------------------------------------------
# Black-player lifting (same idea as before)
# -----------------------------------------------------------

# From your actuator mapping:
# ctrl[8]  -> b_goal_linear
# ctrl[9]  -> b_goal_rotation
# ctrl[10] -> b_def_linear
# ctrl[11] -> b_def_rotation
# ctrl[12] -> b_mid_linear
# ctrl[13] -> b_mid_rotation
# ctrl[14] -> b_attack_linear
# ctrl[15] -> b_attack_rotation
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
# Episode classification helpers
# -----------------------------------------------------------

def infer_termination_type(terminated, truncated, ep_return, info):
    """
    Heuristic label for why an episode ended.
    Adjust thresholds/flags to match what your env puts in 'info'.
    """
    goal_flag = bool(info.get("goal_scored", False))
    stagnant_flag = bool(info.get("ball_stagnant", False))
    over_time_flag = bool(info.get("over_max_time", False))

    # You showed rewards ~1000 for a successful goal; tweak if needed.
    if goal_flag or ep_return > 200.0:
        return "GOAL"
    if stagnant_flag:
        return "STAGNATION"
    if truncated or over_time_flag:
        return "TIMEOUT"
    if terminated:
        return "OTHER-TERMINATED"
    return "UNKNOWN"


# -----------------------------------------------------------
# Core rollout logic, optionally with viewer
# -----------------------------------------------------------

def run_random_rollouts_with_viewer(
    n_episodes=10,
    max_steps_per_ep=300,
    kick_steps=40,
    sleep_scale=1.0,
):
    """
    Run random episodes while streaming env.model/env.data into a MuJoCo viewer.
    """
    # Note: we don't pass render_mode here; we're using raw MuJoCo viewer.
    env = FoosballEnv(
        render_mode=None,
    )

    model = env.model
    data = env.data

    ep_stats = []

    # Open a single viewer for the whole run
    with viewer.launch_passive(model, data) as v:
        print("[VIEWER] Window opened. Running episodes...")

        try:
            for ep in range(n_episodes):
                # Reset env
                obs, info = env.reset(seed=42 + ep)
                lift_black_players(env)  # keep black guys up after each reset

                ep_return = 0.0
                steps = 0
                last_info = info
                terminated = False
                truncated = False

                print(f"\n========== EPISODE {ep + 1}/{n_episodes} ==========")

                for t in range(max_steps_per_ep):
                    # base random action
                    action = env.action_space.sample()

                    # Add a “kick” motion on yellow attack rotation (ctrl[7])
                    if t < kick_steps and action.shape[0] > 7:
                        phase = 2.0 * np.pi * (t / max(kick_steps - 1, 1))
                        kick = 0.9 * np.sin(phase)
                        action[7] = np.clip(kick, -1.0, 1.0)

                    obs, reward, terminated, truncated, info = env.step(action)

                    ep_return += float(reward)
                    steps += 1
                    last_info = info

                    # Update viewer
                    if not v.is_running:
                        print("[VIEWER] Window closed by user; stopping rollouts.")
                        return
                    v.sync()

                    # Slow down a bit so you can see the motion
                    dt = env.model.opt.timestep * getattr(env, "frame_skip", 1)
                    time.sleep(dt * sleep_scale)

                    if terminated or truncated:
                        break

                term_type = infer_termination_type(
                    terminated=terminated,
                    truncated=truncated,
                    ep_return=ep_return,
                    info=last_info,
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

            print("\n[VIEWER] Episodes done. You can close the window now.")
            time.sleep(2.0)

        finally:
            env.close()

    # Console summary
    print("\n================== OVERALL SUMMARY ==================")
    from collections import Counter
    counts = Counter(t for _, _, t in ep_stats)
    for i, (ret, steps, ttype) in enumerate(ep_stats, start=1):
        print(f"Ep {i:02d}: return={ret:8.3f}  steps={steps:3d}  type={ttype}")
    print("Termination counts:", dict(counts))


def main():
    run_random_rollouts_with_viewer(
        n_episodes=50,
        max_steps_per_ep=100,
        kick_steps=40,
        sleep_scale=1.0,  # >1.0 = slower, <1.0 = faster
    )


if __name__ == "__main__":
    main()

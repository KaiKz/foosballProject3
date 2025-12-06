import time
import numpy as np
import mujoco

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


# Actuator indices for *black* rotation actuators from your mapping:
#   ctrl[8]  -> b_goal_linear
#   ctrl[9]  -> b_goal_rotation
#   ctrl[10] -> b_def_linear
#   ctrl[11] -> b_def_rotation
#   ctrl[12] -> b_mid_linear
#   ctrl[13] -> b_mid_rotation
#   ctrl[14] -> b_attack_linear
#   ctrl[15] -> b_attack_rotation
BLACK_ROT_ACT_IDX = [9, 11, 13, 15]

def _get_black_rot_qpos_indices(env):
    """
    Find the qpos indices of the joints driven by the black-rotation actuators.
    We do this once and cache it on the env to avoid recomputing.
    """
    if hasattr(env, "_black_rot_qpos_idx"):
        return env._black_rot_qpos_idx

    model = env.model
    qpos_idx_list = []

    print("[BLACK-LIFT] resolving black rotation qpos indices...")

    for act_id in BLACK_ROT_ACT_IDX:
        if act_id >= model.nu:
            print(f"  [WARN] actuator index {act_id} >= nu={model.nu}, skipping")
            continue

        # ✅ correct indexing: (nu, 2) array, [act_id, 0] gives joint id
        joint_id = int(model.actuator_trnid[act_id, 0])
        if joint_id < 0:
            print(f"  [WARN] actuator {act_id} has no joint, trnid={model.actuator_trnid[act_id]}, skipping")
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos_idx_list.append(qpos_adr)

        # Optional: print names for sanity
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



def lift_black_players(env, angle_rad: float = np.deg2rad(90.0)):
    """
    Force all black players to be rotated 'up' by directly setting their joint qpos.
    This bypasses ctrl so env.step() cannot override it.
    """
    qpos_idx_list = _get_black_rot_qpos_indices(env)

    for qidx in qpos_idx_list:
        env.data.qpos[qidx] = angle_rad

    # Recompute derived quantities after changing qpos
    mujoco.mj_forward(env.model, env.data)


def drive_ball_toward_goal(env, direction: float, max_steps: int = 300, step_size: float = 0.2):
    """
    Teleport the ball a bit each step along +y or -y, keep black players lifted each step,
    call env.step(), render, and stop when the episode terminates.
    """

    ball_x_adr = env.ball_x_qpos_adr
    ball_y_adr = env.ball_y_qpos_adr

    ball_x0 = float(env.data.qpos[ball_x_adr])
    ball_y0 = float(env.data.qpos[ball_y_adr])
    print(f"[GOAL-TEST] Starting drive, direction={direction:+.1f}, "
          f"initial pos=({ball_x0:.4f}, {ball_y0:.4f})")

    terminated = False
    truncated = False
    info = {}

    for t in range(max_steps):
        # --- manually move ball along y ---
        env.data.qpos[ball_y_adr] += direction * step_size

        # lift black players *before* physics so collisions see the rotated pose
        lift_black_players(env)

        # propagate
        mujoco.mj_forward(env.model, env.data)

        # protagonist takes no action; we just drive the ball
        action = np.zeros(env.protagonist_action_size, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        # enforce black players up even after the step integration
        lift_black_players(env)

        # --- viewer render ---
        if env.render_mode == "human":
            env.render()
            time.sleep(0.02)

        # --- debug prints ---
        ball_pos = [float(env.data.qpos[ball_x_adr]), float(env.data.qpos[ball_y_adr])]
        if t % 10 == 0 or terminated or truncated:
            print(
                f"  t={t:3d} pos={ball_pos} "
                f"| term={terminated} trunc={truncated} "
                f"| goal_scored={info.get('goal_scored', None)}"
            )

        if terminated or truncated:
            print("\n[GOAL-TEST] Episode ended.")
            print(f"  step: {t}")
            print(f"  ball_pos: {ball_pos}")
            print(f"  info: {info}")
            break

    if not (terminated or truncated):
        print("\n[GOAL-TEST] Finished drive without termination.")
        print(f"  last ball pos: {ball_pos}")
        print(f"  info last: {info}")

    return terminated, truncated, info


def main():
    # Use render_mode="human" so a window pops up and you can see the ball & rods.
    env = FoosballEnv(
        antagonist_model=None,
        verbose_mode=True,
        render_mode="human",
    )

    # ===============================================================
    # TEST 1: drive ball toward +y (one goal)
    # ===============================================================
    obs, info = env.reset()
    mujoco.mj_forward(env.model, env.data)

    # Immediately lift black players on reset so you *see* them rotated up
    lift_black_players(env)

    print("\n==============================")
    print("  TEST 1: drive toward +y")
    print("==============================\n")

    term_pos, trunc_pos, info_pos = drive_ball_toward_goal(
        env,
        direction=+1.0,
        max_steps=300,
        step_size=0.2,
    )

    if "goal_scored" in info_pos:
        print(f"\n[RESULT] +y direction: terminated={term_pos}, goal_scored={info_pos['goal_scored']}")
    else:
        print("\n[RESULT] +y direction: terminated="
              f"{term_pos}, but 'goal_scored' not present in info")

    # ===============================================================
    # TEST 2: drive ball toward -y (other goal)
    # ===============================================================
    print("\n==============================")
    print("  TEST 2: drive toward -y")
    print("==============================\n")

    obs, info = env.reset()
    mujoco.mj_forward(env.model, env.data)
    lift_black_players(env)

    term_neg, trunc_neg, info_neg = drive_ball_toward_goal(
        env,
        direction=-1.0,
        max_steps=300,
        step_size=0.2,
    )

    if "goal_scored" in info_neg:
        print(f"\n[RESULT] -y direction: terminated={term_neg}, goal_scored={info_neg['goal_scored']}")
    else:
        print("\n[RESULT] -y direction: terminated="
              f"{term_neg}, but 'goal_scored' not present in info")

    print("\n[VIEWER] Sleeping for a few seconds before closing...")
    time.sleep(3.0)
    env.close()


if __name__ == "__main__":
    main()

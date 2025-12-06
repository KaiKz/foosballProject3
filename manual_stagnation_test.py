import numpy as np
import mujoco
import time

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


def rotate_black_players(env, angle_rad=np.pi / 2):
    """
    Hard-set all black (b_*) rod rotation joints to a given angle.
    Call this AFTER mj_reset / mj_step whenever you want the pose enforced.
    (Not necessary for stagnation, but kept here for completeness.)
    """
    joint_names = [
        "b_goal_rotation",
        "b_def_rotation",
        "b_mid_rotation",
        "b_attack_rotation",
    ]

    print("\n[BLACK ROTATE] enforcing black rod rotations...")
    for jname in joint_names:
        j_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if j_id < 0:
            print(f"[BLACK ROTATE] joint {jname} not found, skipping")
            continue

        qpos_adr = env.model.jnt_qposadr[j_id]
        before = float(env.data.qpos[qpos_adr])
        env.data.qpos[qpos_adr] = angle_rad
        after = float(env.data.qpos[qpos_adr])
        print(f"[BLACK ROTATE] {jname}: qpos[{qpos_adr}] {before:.3f} -> {after:.3f}")

    mujoco.mj_forward(env.model, env.data)


def main():
    # verbose_mode=True so you see stagnation debug prints
    env = FoosballEnv(antagonist_model=None, verbose_mode=True, render_mode=None)
    obs, info = env.reset()

    # ------------------------------------------------------------------
    # 1) Basic geom info for ball + one attack guy
    # ------------------------------------------------------------------
    ball_geom_id = getattr(
        env,
        "ball_geom_id",
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_phys"),
    )
    guy_geom_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "y_attack_guy2"
    )

    if guy_geom_id < 0:
        raise RuntimeError("Geom 'y_attack_guy2' not found in the model")

    mujoco.mj_forward(env.model, env.data)

    ball_xyz0 = env.data.geom_xpos[ball_geom_id].copy()
    guy_xyz0 = env.data.geom_xpos[guy_geom_id].copy()
    print(
        "[INIT ROW DEBUG] ball_xyz0 =", ball_xyz0,
        "guy_xyz0 =", guy_xyz0,
        "Δz =", ball_xyz0[2] - guy_xyz0[2],
    )

    # ------------------------------------------------------------------
    # 2) Put the ball NEAR the attack guy but not touching, and make it still
    # ------------------------------------------------------------------
    ball_radius = float(env.model.geom_size[ball_geom_id][0])

    guy_type = env.model.geom_type[guy_geom_id]
    guy_size = env.model.geom_size[guy_geom_id]

    if guy_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        guy_radius = float(guy_size[0])
    elif guy_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        guy_radius = float(guy_size[0])
    elif guy_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        guy_radius = float(guy_size[0])
    elif guy_type == mujoco.mjtGeom.mjGEOM_BOX:
        guy_radius = float(guy_size[1])
    else:
        guy_radius = 0.05

    margin = 0.05
    safe_offset = ball_radius + guy_radius + margin

    # Put ball a bit "behind and to the side" so it won't be hit,
    # but stays within the field.
    env.data.qpos[env.ball_x_qpos_adr] = guy_xyz0[0] - safe_offset
    env.data.qpos[env.ball_y_qpos_adr] = guy_xyz0[1] - safe_offset

    # Zero planar velocity
    env.data.qvel[env.ball_x_qvel_adr] = 0.0
    env.data.qvel[env.ball_y_qvel_adr] = 0.0

    mujoco.mj_forward(env.model, env.data)

    # IMPORTANT: sync stagnation bookkeeping to this new ball position
    ball_pos, ball_vel = env._get_ball_obs()
    env.prev_ball_y = ball_pos[1]
    env.ball_stopped_count = 0

    print(f"[AFTER MANUAL POS] ball_pos={ball_pos}, ball_vel={ball_vel}")

    # ------------------------------------------------------------------
    # 3) Optional: rotate black players once (so they don't accidentally block)
    # ------------------------------------------------------------------
    rotate_black_players(env, angle_rad=np.pi / 2)

    # ------------------------------------------------------------------
    # 4) Step with ZERO actions and watch for stagnation termination
    # ------------------------------------------------------------------
    max_steps = 400
    for t in range(max_steps):
        # protagonist does nothing
        action = np.zeros(env.protagonist_action_size, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if t % 20 == 0:
            ball_pos, ball_vel = env._get_ball_obs()
            print(
                f"t={t:3d} | pos={ball_pos} vel={ball_vel} | "
                f"ball_stagnant={info.get('ball_stagnant', False)} "
                f"out_of_play={info.get('out_of_play', False)} "
                f"over_max_time={info.get('over_max_time', False)}"
            )

        if terminated or truncated:
            print("\n=== EPISODE TERMINATED ===")
            print(f"step: {t}")
            print("ball_pos, ball_vel:", env._get_ball_obs())
            print("termination flags:", info)
            break
    else:
        print("\n[WARN] Reached max_steps without termination")

    # ------------------------------------------------------------------
    # 5) (Optional) Viewer loop if you want to see the static ball
    # ------------------------------------------------------------------
    env.render_mode = "human"
    env.render()

    while True:
        action = np.zeros(env.protagonist_action_size, dtype=np.float32)
        rotate_black_players(env, angle_rad=np.pi / 2)  # keep them rotated
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            print("[VIEWER] episode ended, resetting")
            obs, info = env.reset()
            # Re-freeze ball in a similar static position after reset
            mujoco.mj_forward(env.model, env.data)
            ball_xyz0 = env.data.geom_xpos[ball_geom_id].copy()
            guy_xyz0 = env.data.geom_xpos[guy_geom_id].copy()

            env.data.qpos[env.ball_x_qpos_adr] = guy_xyz0[0] - safe_offset
            env.data.qpos[env.ball_y_qpos_adr] = guy_xyz0[1] - safe_offset
            env.data.qvel[env.ball_x_qvel_adr] = 0.0
            env.data.qvel[env.ball_y_qvel_adr] = 0.0
            mujoco.mj_forward(env.model, env.data)

            ball_pos, ball_vel = env._get_ball_obs()
            env.prev_ball_y = ball_pos[1]
            env.ball_stopped_count = 0

            rotate_black_players(env, angle_rad=np.pi / 2)
            env.render()


if __name__ == "__main__":
    main()

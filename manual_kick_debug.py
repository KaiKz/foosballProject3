import numpy as np
import mujoco
import time

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


def rotate_black_players(env, angle_rad=np.pi / 2):
    """
    Hard-set all black (b_*) rod rotation joints to a given angle.
    Call this AFTER mj_reset / mj_step whenever you want the pose enforced.
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
        env.data.qpos[qpos_adr] = angle_rad  # 90 degrees
        after = float(env.data.qpos[qpos_adr])
        print(f"[BLACK ROTATE] {jname}: qpos[{qpos_adr}] {before:.3f} -> {after:.3f}")

    mujoco.mj_forward(env.model, env.data)


def main():
    # verbose_mode just for extra prints during debugging
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
    # 2) Compute a "safe" separation based on geom sizes (planar ball)
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

    margin = 0.02
    safe_offset = ball_radius + guy_radius + margin

    env.data.qpos[env.ball_x_qpos_adr] = guy_xyz0[0]
    env.data.qpos[env.ball_y_qpos_adr] = guy_xyz0[1] - safe_offset

    env.data.qvel[env.ball_x_qvel_adr] = 0.0
    env.data.qvel[env.ball_y_qvel_adr] = 0.0

    mujoco.mj_forward(env.model, env.data)

    # ------------------------------------------------------------------
    # 3) Resolve any *ball–player* contacts
    # ------------------------------------------------------------------
    player_geom_ids = getattr(env, "player_geom_ids", [])

    def ball_in_contact_with_player(model, data):
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if ((g1 == ball_geom_id and g2 in player_geom_ids) or
                    (g2 == ball_geom_id and g1 in player_geom_ids)):
                return True, c
        return False, None

    ball_qpos_y_index = env.ball_y_qpos_adr

    max_tries = 50
    step = 0.01
    tries = 0

    # 🔴 First time: enforce black rotation BEFORE we start resolving & stepping
    rotate_black_players(env, angle_rad=np.pi / 2)

    while True:
        touching, c = ball_in_contact_with_player(env.model, env.data)
        if not touching:
            break

        g1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2_name = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(
            f"[RESOLVE DEBUG] Still ball-player contact: {g1_name} <-> {g2_name} "
            f"dist={c.dist:.6f}; nudging ball +y"
        )

        env.data.qpos[ball_qpos_y_index] += step
        mujoco.mj_forward(env.model, env.data)

        tries += 1
        if tries >= max_tries:
            print(
                "[WARN] couldn't resolve ball-player contact by shifting; "
                "consider increasing 'margin' or safe_offset."
            )
            break

    ball_xyz0b = env.data.geom_xpos[ball_geom_id].copy()
    print(
        "[AFTER REPOSITION] ball_xyz0 =", ball_xyz0b,
        "guy_xyz0 =", guy_xyz0,
        "Δxy =", np.linalg.norm(ball_xyz0b[:2] - guy_xyz0[:2]),
    )

    touching, _ = ball_in_contact_with_player(env.model, env.data)
    if touching:
        print("[WARN] Still ball–player contact after resolution loop!")

    print("[DEBUG] Initial ball pos/vel:", env._get_ball_obs())

    # ------------------------------------------------------------------
    # 5) Slam the attack rod and print contacts every step
    # ------------------------------------------------------------------
    for t in range(200):
        action = np.zeros(env.protagonist_action_size, dtype=np.float32)
        action[6] = 1.0  # yellow attack linear

        # 💡 Re-enforce black rotation each step so actuators can't undo it
        rotate_black_players(env, angle_rad=np.pi / 2)

        ball_xyz = env.data.geom_xpos[ball_geom_id].copy()
        guy_xyz = env.data.geom_xpos[guy_geom_id].copy()
        dist_xy = np.linalg.norm(ball_xyz[:2] - guy_xyz[:2])

        print(
            f"[DIST DEBUG] t={t} "
            f"ball_xy={ball_xyz[:2]} guy_xy={guy_xyz[:2]} "
            f"dist_xy={dist_xy:.4f}"
        )

        obs, reward, terminated, truncated, info = env.step(action)

        try:
            ball_xyz = env.data.geom_xpos[ball_geom_id].copy()
            print(f"[POS DEBUG] t={t} ball_xyz = {ball_xyz}")
        except Exception as e:
            print(f"[POS DEBUG] error: {e}")

        if env.data.ncon > 0:
            print(f"[CONTACT DEBUG] step={t}, ncon={env.data.ncon}")
            for i in range(env.data.ncon):
                c = env.data.contact[i]
                g1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                g2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                print(
                    f"  con#{i}: {g1_name} <-> {g2_name} | "
                    f"dist={c.dist:.6f}"
                )

        if terminated or truncated:
            print(f"[DEBUG] Episode ended at t={t}")
            break

    # ------------------------------------------------------------------
    # 6) Viewer loop (keep enforcing rotation here too)
    # ------------------------------------------------------------------
    env.render_mode = "human"
    env.render()
    while True:
        action = np.zeros(env.protagonist_action_size, dtype=np.float32)
        action[7] = 1.0  # spin yellow attackers a bit

        # 🔁 Make sure black players stay rotated in the viewer as well
        rotate_black_players(env, angle_rad=np.pi / 2)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            obs, info = env.reset()
            rotate_black_players(env, angle_rad=np.pi / 2)
            env.render()


if __name__ == "__main__":
    main()

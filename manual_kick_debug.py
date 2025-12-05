import numpy as np
import mujoco
import time

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


def main():
    # verbose_mode just for extra prints during debugging
    env = FoosballEnv(antagonist_model=None, verbose_mode=True, render_mode=None)
    obs, info = env.reset()

    # ------------------------------------------------------------------
    # 1) Basic geom info for ball + one attack guy
    # ------------------------------------------------------------------
    # Use the cached ids from the env if they exist
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

    # Make sure geom_xpos is up to date
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
        guy_radius = float(guy_size[0])  # radius
    elif guy_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        guy_radius = float(guy_size[0])  # radius
    elif guy_type == mujoco.mjtGeom.mjGEOM_BOX:
        # Rough: treat half-length along y as the "reach" in y
        guy_radius = float(guy_size[1])
    else:
        guy_radius = 0.05  # fallback guess

    margin = 0.02  # small gap to avoid penetration
    safe_offset = ball_radius + guy_radius + margin

    # From your old notes:
    # "action[6] = +1 -> ball_y goes DOWN (smaller y), so place the ball
    # in front at smaller y (guy_y - safe_offset)."
    # Adjust here if you later flip directions.
    env.data.qpos[env.ball_x_qpos_adr] = guy_xyz0[0]           # align x
    env.data.qpos[env.ball_y_qpos_adr] = guy_xyz0[1] - safe_offset  # y in front of guy

    # Zero planar velocity for a clean test
    env.data.qvel[env.ball_x_qvel_adr] = 0.0
    env.data.qvel[env.ball_y_qvel_adr] = 0.0

    mujoco.mj_forward(env.model, env.data)

    # ------------------------------------------------------------------
    # 3) Resolve any *ball–player* contacts by nudging the ball along +y
    # ------------------------------------------------------------------
    # Use env.player_geom_ids if available (already built in __init__)
    player_geom_ids = getattr(env, "player_geom_ids", [])

    def ball_in_contact_with_player(model, data):
        """Return (True, contact) if ball_phys is colliding with any player geom."""
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if ((g1 == ball_geom_id and g2 in player_geom_ids) or
                    (g2 == ball_geom_id and g1 in player_geom_ids)):
                return True, c
        return False, None

    # We'll directly use the ball_y qpos address (planar ball)
    ball_qpos_y_index = env.ball_y_qpos_adr

    max_tries = 50
    step = 0.01  # small nudges along +y
    tries = 0

    while True:
        touching, c = ball_in_contact_with_player(env.model, env.data)
        if not touching:
            break

        g1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
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

    # ------------------------------------------------------------------
    # 4) Now we can safely check positions / distances
    # ------------------------------------------------------------------
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
        action[6] = 1.0  # max forward slide for yellow attack rod

        ball_xyz = env.data.geom_xpos[ball_geom_id].copy()
        guy_xyz = env.data.geom_xpos[guy_geom_id].copy()
        dist_xy = np.linalg.norm(ball_xyz[:2] - guy_xyz[:2])

        print(
            f"[DIST DEBUG] t={t} "
            f"ball_xy={ball_xyz[:2]} guy_xy={guy_xyz[:2]} "
            f"dist_xy={dist_xy:.4f}"
        )

        obs, reward, terminated, truncated, info = env.step(action)

        # Print ball pos
        try:
            ball_xyz = env.data.geom_xpos[ball_geom_id].copy()
            print(f"[POS DEBUG] t={t} ball_xyz = {ball_xyz}")
        except Exception as e:
            print(f"[POS DEBUG] error: {e}")

        # Print ALL contacts and names so you see exactly what’s hitting what
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
    # 6) Optional: watch things in the viewer afterwards
    # ------------------------------------------------------------------
    env.render_mode = "human"
    env.render()
    while True:
        action = np.zeros(env.protagonist_action_size, dtype=np.float32)
        action[7] = 1.0
        action[5] = 1.0
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            obs, info = env.reset()
            env.render()


if __name__ == "__main__":
    main()

import numpy as np
import mujoco
import time

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


def main():
    env = FoosballEnv(antagonist_model=None, verbose_mode=True)
    obs, info = env.reset()
    

    # ------------------------------------------------------------------
    # 1) Basic geom info for ball + one attack guy
    # ------------------------------------------------------------------
    ball_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_phys")
    guy_geom_id  = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "y_attack_guy2")
    
    ball_xyz0 = env.data.geom_xpos[ball_geom_id].copy()
    guy_xyz0  = env.data.geom_xpos[guy_geom_id].copy()
    print(
        "[INIT ROW DEBUG] ball_xyz0 =", ball_xyz0,
        "guy_xyz0 =", guy_xyz0,
        "Δz =", ball_xyz0[2] - guy_xyz0[2]
    )

    # ------------------------------------------------------------------
    # 2) Compute a "safe" separation based on geom sizes
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

    margin = 1.5  # extra gap to avoid penetration
    safe_offset = ball_radius + guy_radius + margin

    # Decide which side of the guy we want the ball on.
    # From earlier: action[6] = +1 -> ball_y goes DOWN (smaller y),
    # so place the ball in front at smaller y (guy_y - safe_offset).
    bp = env.ball_qpos_adr  # ball qpos start index

    env.data.qpos[bp + 0] = guy_xyz0[0]           # align x
    env.data.qpos[bp + 1] = guy_xyz0[1] - safe_offset  # y in front of guy
    env.data.qpos[bp + 2] = guy_xyz0[2]           # align z

    # Zero linear velocity for a clean test
    env.data.qvel[env.ball_qvel_adr : env.ball_qvel_adr + 3] = 0.0

    # ------------------------------------------------------------------
    # 3) FIRST forward pass (needed before reading contacts)
    # ------------------------------------------------------------------
    mujoco.mj_forward(env.model, env.data)

    # ------------------------------------------------------------------
    # 4) Resolve any *ball–player* contacts by nudging the ball along +y
    # ------------------------------------------------------------------
    PLAYER_GEOM_PREFIXES = ["y_attack_guy", "y_mid_guy", "y_def_guy"]

    # Collect all player geom ids
    player_geom_ids = []
    for gid in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name:  # can be None
            continue
        if any(name.startswith(prefix) for prefix in PLAYER_GEOM_PREFIXES):
            player_geom_ids.append(gid)

    def ball_in_contact_with_player(model, data):
        """Return (True, contact) if ball_phys is colliding with any player geom."""
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if ((g1 == ball_geom_id and g2 in player_geom_ids) or
                (g2 == ball_geom_id and g1 in player_geom_ids)):
                return True, c
        return False, None

    ball_qpos_y_index = bp + 1  # ball's y coordinate in qpos

    max_tries = 50
    step = 0.01  # small nudges along +y
    tries = 0

    while True:
        touching, c = ball_in_contact_with_player(env.model, env.data)
        if not touching:
            break

        # Optional: debug what we’re colliding with
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
            print("[WARN] couldn't resolve ball-player contact by shifting; "
                  "consider increasing 'margin' or safe_offset.")
            break

    # ------------------------------------------------------------------
    # 5) Now we can safely check positions / distances
    # ------------------------------------------------------------------
    ball_xyz0b = env.data.geom_xpos[ball_geom_id].copy()
    print(
        "[AFTER REPOSITION] ball_xyz0 =", ball_xyz0b,
        "guy_xyz0 =", guy_xyz0,
        "Δxy =", np.linalg.norm(ball_xyz0b[:2] - guy_xyz0[:2])
    )

    # At this point we specifically ensured no ball–player contacts.
    touching, _ = ball_in_contact_with_player(env.model, env.data)
    if touching:
        print("[WARN] Still ball–player contact after resolution loop!")

    print("[DEBUG] Initial ball pos/vel:", env._get_ball_obs())

    # ------------------------------------------------------------------
    # 6) Slam the attack rod and print contacts every step
    # ------------------------------------------------------------------
    for t in range(200):
        action = np.zeros(env.protagonist_action_size, dtype=np.float32)
        action[6] = 1.0  # max forward slide

        ball_xyz = env.data.geom_xpos[ball_geom_id].copy()
        guy_xyz  = env.data.geom_xpos[guy_geom_id].copy()
        dist_xy  = np.linalg.norm(ball_xyz[:2] - guy_xyz[:2])

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

    while True:
        action = np.zeros(env.protagonist_action_size, dtype=np.float32)
        action[6] = 1.0  # or 0.0 if you just want ball drifting
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()


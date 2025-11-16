import numpy as np
import mujoco

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv



# def test_ball_force():
#     env = FoosballEnv(antagonist_model=None, verbose_mode=False, debug_free_ball=False)
#     env.reset()

#     model, data = env.model, env.data

#     # Get ball_x / ball_y indices
#     jx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
#     jy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
#     qx = model.jnt_qposadr[jx]
#     qy = model.jnt_qposadr[jy]
#     vx = model.jnt_dofadr[jx]
#     vy = model.jnt_dofadr[jy]

#     # Put ball in a clean position and zero its velocity
#     data.qpos[qx] = 0.0
#     data.qpos[qy] = 0.0
#     data.qvel[vx] = 0.0
#     data.qvel[vy] = 0.0
#     mujoco.mj_forward(model, data)

#     print("=== BALL FORCE TEST ===")
#     for t in range(10):
#         # Apply a constant force in +y to the ball
#         data.qfrc_applied[vy] = 10.0

#         mujoco.mj_step(model, data)

#         print(
#             f"t={t:02d}  "
#             f"pos=({data.qpos[qx]:.4f}, {data.qpos[qy]:.4f})  "
#             f"vel=({data.qvel[vx]:.4f}, {data.qvel[vy]:.4f})"
#         )

#         # Clear the applied force for the next step
#         data.qfrc_applied[vy] = 0.0

#     env.close()

def test_ball_force():
    env = FoosballEnv(antagonist_model=None, verbose_mode=False, debug_free_ball=False)
    env.reset()
    model, data = env.model, env.data

    j_free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    if j_free < 0:
        raise RuntimeError("ball_free joint not found")

    qbase = model.jnt_qposadr[j_free]
    vbase = model.jnt_dofadr[j_free]

    # Clean state
    data.qpos[qbase + 0] = 0.0  # x
    data.qpos[qbase + 1] = 0.0  # y
    data.qpos[qbase + 2] = 0.08 # z
    data.qvel[vbase + 0] = 0.0  # vx
    data.qvel[vbase + 1] = 0.0  # vy
    data.qvel[vbase + 2] = 0.0  # vz
    mujoco.mj_forward(model, data)

    print("=== BALL FORCE TEST (free joint) ===")
    for t in range(10):
        # Apply force along +y (linear DOF index is vbase+1)
        data.qfrc_applied[vbase + 1] = 10.0

        mujoco.mj_step(model, data)

        x = data.qpos[qbase + 0]
        y = data.qpos[qbase + 1]
        vx = data.qvel[vbase + 0]
        vy = data.qvel[vbase + 1]

        print(f"t={t:02d} pos=({x:.4f}, {y:.4f}) vel=({vx:.4f}, {vy:.4f})")

        data.qfrc_applied[vbase + 1] = 0.0

    env.close()


def list_actuators(env):
    model = env.model
    print("=== ACTUATORS ===")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"{i:2d}: {name}")


# def place_ball_near_geom(env, geom_name="y_mid_guy3", forward_offset=3.0):
#     """
#     Put the ball in front of a given foosman geom, using that geom's local frame.

#     forward_offset: how far *in front* of the player (in model units).
#                     ~ 2–4 is a good range for this table scale.
#     """
#     model, data = env.model, env.data

#     g_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
#     if g_id < 0:
#         raise RuntimeError(f"Geom {geom_name!r} not found")

#     # World position and orientation of this geom
#     gpos = np.array(data.geom_xpos[g_id], copy=True)
#     gmat = np.array(data.geom_xmat[g_id], copy=True).reshape(3, 3)

#     # For these foosmen, local +Y (or -Y) is usually "forward" or "backward";
#     # adjust sign if you see the ball spawning behind instead of in front.
#     local_forward = gmat[:, 1]  # try +gmat[:,1] if this is wrong

#     offset_world = local_forward * forward_offset

#     # Ball joints
#     jx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
#     jy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
#     if jx < 0 or jy < 0:
#         raise RuntimeError("ball_x or ball_y joint not found")

#     qx = model.jnt_qposadr[jx]
#     qy = model.jnt_qposadr[jy]
#     vx = model.jnt_dofadr[jx]
#     vy = model.jnt_dofadr[jy]

#     # Place ball in front of the player
#     data.qpos[qx] = gpos[0] + offset_world[0]
#     data.qpos[qy] = gpos[1] + offset_world[1]
#     data.qvel[vx] = 0.0
#     data.qvel[vy] = 0.0

#     mujoco.mj_forward(model, data)

#     print(
#         f"[KICK TEST] Placed ball near {geom_name}: "
#         f"x={data.qpos[qx]:.4f}, y={data.qpos[qy]:.4f}, "
#         f"forward_offset={forward_offset}"
#     )


def place_ball_near_geom(env, geom_name="y_mid_guy3", forward_offset=3.0):
    model, data = env.model, env.data

    g_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if g_id < 0:
        raise RuntimeError(f"Geom {geom_name!r} not found")

    gpos = np.array(data.geom_xpos[g_id], copy=True)
    gmat = np.array(data.geom_xmat[g_id], copy=True).reshape(3, 3)

    # try gmat[:,1] as forward; flip sign later if needed
    local_forward = gmat[:, 1]
    offset_world = local_forward * forward_offset

    j_free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    if j_free < 0:
        raise RuntimeError("ball_free joint not found")

    qbase = model.jnt_qposadr[j_free]
    vbase = model.jnt_dofadr[j_free]

    # Position: world coords
    data.qpos[qbase + 0] = gpos[0] + offset_world[0]
    data.qpos[qbase + 1] = gpos[1] + offset_world[1]
    data.qpos[qbase + 2] = gpos[2] + offset_world[2]

    # Keep orientation identity for now
    data.qpos[qbase + 3] = 1.0
    data.qpos[qbase + 4] = 0.0
    data.qpos[qbase + 5] = 0.0
    data.qpos[qbase + 6] = 0.0

    # Zero velocities
    data.qvel[vbase : vbase + 6] = 0.0

    mujoco.mj_forward(model, data)

    print(
        f"[KICK TEST] Placed ball near {geom_name}: "
        f"x={data.qpos[qbase+0]:.4f}, y={data.qpos[qbase+1]:.4f}"
    )



def main():
    # Turn on verbose_mode so your env prints contact / termination debug
    env = FoosballEnv(antagonist_model=None, verbose_mode=True, debug_free_ball=True)
    import mujoco

    # spin the y_mid_rotation joint directly
    j_mid_rot = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "y_mid_rotation")
    mid_rot_dof = env.model.jnt_dofadr[j_mid_rot]
    env.data.qvel[mid_rot_dof] = 30.0  # big angular velocity

    for t in range(40):
        mujoco.mj_step(env.model, env.data)
        ball_pos, ball_vel = env._get_ball_obs()
        print(f"t={t:02d} ball=({ball_pos[0]:.4f}, {ball_pos[1]:.4f}), vel={ball_vel}")

    # Standard reset (will give the ball some serve velocity, but we overwrite below)
    obs, _ = env.reset()
    
    
    model, data = env.model, env.data

    # Find the ball geom
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    if ball_geom_id < 0:
        raise RuntimeError("Geom 'ball' not found")

    ball_body_id = model.geom_bodyid[ball_geom_id]
    print("[MASS DEBUG] ball_body_id =", ball_body_id)
    print("[MASS DEBUG] ball body mass =", model.body_mass[ball_body_id])

    # Also print total body inertial parameters for sanity
    print("[MASS DEBUG] body_inertia =", model.body_inertia[ball_body_id])


    # 1) Show all actuators so you can see which index is the mid-rod rotation
    list_actuators(env)

    # 2) Move the ball right in front of one of the yellow mid guys
    # OLD (causes TypeError)
    place_ball_near_geom(env, geom_name="y_mid_guy3", forward_offset=-0.015)


    # 3) Choose which protagonist actuator index to slam.
    #    After you look at the actuator list printed above, set this to the
    #    index whose name looks like "y_mid_*rotation*".
    #
    #    IMPORTANT: this index must be < env.protagonist_action_size (usually 8),
    #    because env.step() only writes the first 8 values into data.ctrl
    #    for the protagonist.
    MID_ROD_ROT_IDX = 5  # <-- update this after you see the printed actuator list

    if MID_ROD_ROT_IDX >= env.protagonist_action_size:
        raise RuntimeError(
            f"MID_ROD_ROT_IDX={MID_ROD_ROT_IDX} is >= protagonist_action_size="
            f"{env.protagonist_action_size}. Pick a smaller index."
        )

    print(f"[KICK TEST] Using protagonist action index {MID_ROD_ROT_IDX} for mid-rod rotation")

    # 4) Rollout: spin that rod hard for a bunch of steps and watch contacts + ball motion
    T = 40
    for t in range(T):
        a = np.zeros(env.protagonist_action_size, dtype=np.float32)
        a[MID_ROD_ROT_IDX] = 20.0  # big spin

        obs, r, terminated, truncated, info = env.step(a)

        print(
            f"t={t:02d} "
            f"ball=({info['ball_x']:.4f}, {info['ball_y']:.4f}) "
            f"reward={r:.3f}"
        )

        if terminated:
            print("[KICK TEST] Episode terminated")
            break

    env.close()


if __name__ == "__main__":
    test_ball_force()
    main()

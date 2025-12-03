import mujoco
import numpy as np

# Load model and data
model = mujoco.MjModel.from_xml_path("foosball_sim/v2/minimal_foosball_stage1.xml")
data = mujoco.MjData(model)

# Convenience ids / addresses
ball_qpos_addr = model.jnt_qposadr[model.joint("ball_free").id]
ball_geom_id = model.geom("ball").id
player_geom_id = model.geom("player_1_geom").id
rod_motor_id = model.actuator("rod_1_motor").id

ball_body_id = model.body("ball_body").id
rod_body_id = model.body("rod_1").id
rod_slide_addr = model.jnt_qposadr[model.joint("rod_1_slide").id]

print("Geom names:", [model.geom(i).name for i in range(model.ngeom)])

# Initial ball position
data.qpos[ball_qpos_addr : ball_qpos_addr+3] = np.array([0.0, 0.02, 0.10])
mujoco.mj_forward(model, data)

# Parameters
freq = 1.0     # 1 Hz → period 1s
amp = 20.0     # stronger drive
ball_radius = 0.02
player_radius = 0.015
contact_dist = ball_radius + player_radius
near_factor = 1.2
near_threshold = near_factor * contact_dist

T = 400  # number of steps

# Debug tracking variables
first_ball_player_contact_step = None
last_ball_player_contact_step = None
first_ground_contact_step = None

first_near_without_contact_step = None

max_abs_x = 0.0
max_abs_x_step = None
max_abs_x_pos = None

first_ball_player_contact_state = None
first_ground_contact_state = None
first_near_without_contact_state = None

def get_body_linear_vel(data, body_id):
    """Return world linear velocity of a body for both old/new MuJoCo APIs."""
    # Newer MuJoCo: xvelp exists and is (nbody, 3)
    if hasattr(data, "xvelp"):
        return data.xvelp[body_id].copy()
    # Older MuJoCo: use cvel (nbody, 6) → last 3 entries are linear vel
    if hasattr(data, "cvel"):
        return data.cvel[body_id][3:].copy()
    # Fallback: zero (should not really happen)
    return np.zeros(3)

for step in range(T):
    t = step * model.opt.timestep
    data.ctrl[rod_motor_id] = amp * np.sin(2*np.pi*freq*t)

    mujoco.mj_step(model, data)

    # Positions
    ball_pos = data.geom_xpos[ball_geom_id].copy()
    player_pos = data.geom_xpos[player_geom_id].copy()

    # 3D distance
    diff3d = ball_pos - player_pos
    dist3d = np.linalg.norm(diff3d)

    # Drift tracking in x
    abs_x = abs(ball_pos[0])
    if abs_x > max_abs_x:
        max_abs_x = abs_x
        max_abs_x_step = step
        max_abs_x_pos = ball_pos.copy()

    # Contact inspection
    ncon = data.ncon
    ball_player_contacts = 0
    ground_contacts = 0
    pairs = []

    for i in range(ncon):
        con = data.contact[i]
        g1 = model.geom(con.geom1).name
        g2 = model.geom(con.geom2).name
        pairs.append((g1, g2))

        if ((con.geom1 == ball_geom_id and con.geom2 == player_geom_id) or
            (con.geom2 == ball_geom_id and con.geom1 == player_geom_id)):
            ball_player_contacts += 1

        if ((g1 == "ground" and g2 == "ball") or
            (g2 == "ground" and g1 == "ball")):
            ground_contacts += 1

    # Record ball–player contact events
    if ball_player_contacts > 0:
        if first_ball_player_contact_step is None:
            first_ball_player_contact_step = step
            first_ball_player_contact_state = (ball_pos.copy(), player_pos.copy())
        last_ball_player_contact_step = step

    # Record ground contact
    if ground_contacts > 0 and first_ground_contact_step is None:
        first_ground_contact_step = step
        first_ground_contact_state = (ball_pos.copy(), player_pos.copy())

    # "Near miss" in 3D (no actual ball–player contact)
    if ball_player_contacts == 0 and dist3d < near_threshold:
        if first_near_without_contact_step is None:
            first_near_without_contact_step = step
            first_near_without_contact_state = (
                ball_pos.copy(), player_pos.copy(), dist3d
            )
        print(
            f"NEAR (no contact) at step {step}: "
            f"dist3d={dist3d:.4f}, ball_pos={ball_pos.round(3)}, "
            f"player_pos={player_pos.round(3)}"
        )

    # Occasional state print + whenever there is contact
    if step % 10 == 0 or ball_player_contacts > 0 or ground_contacts > 0:
        rod_y = data.qpos[rod_slide_addr]
        print(
            f"step {step:03d} | "
            f"t={t:.3f} "
            f"ball_pos={ball_pos.round(3)} "
            f"player_pos={player_pos.round(3)} "
            f"rod_y={rod_y:.4f} "
            f"ncon={ncon}, "
            f"ball_player_contacts={ball_player_contacts}, "
            f"ground_contacts={ground_contacts}"
        )

        # Only compute velocities when needed so we don't do extra work every step
        if ball_player_contacts > 0 or ground_contacts > 0:
            ball_vel = get_body_linear_vel(data, ball_body_id)
            rod_vel = get_body_linear_vel(data, rod_body_id)

            if ball_player_contacts > 0:
                print(
                    f"  >>> BALL–PLAYER CONTACT at step {step}: "
                    f"ball_vel={ball_vel.round(3)}, rod_vel={rod_vel.round(3)}"
                )

            if ground_contacts > 0:
                print(
                    f"  >>> BALL–GROUND CONTACT at step {step}: "
                    f"ball_vel={ball_vel.round(3)}"
                )

# ===========================
# Summary section
# ===========================
print("\n========== DEBUG SUMMARY ==========")

if first_ball_player_contact_step is not None:
    print(
        f"Ball–player contact first seen at step {first_ball_player_contact_step}, "
        f"last seen at step {last_ball_player_contact_step}."
    )
    bp_ball_pos, bp_player_pos = first_ball_player_contact_state
    print(f"  First contact positions: ball={bp_ball_pos}, player={bp_player_pos}")
else:
    print("No ball–player contacts detected at all.")

if first_ground_contact_step is not None:
    print(f"Ball–ground contact first seen at step {first_ground_contact_step}.")
    g_ball_pos, g_player_pos = first_ground_contact_state
    print(f"  First ground-contact positions: ball={g_ball_pos}, player={g_player_pos}")
else:
    print("No ball–ground contacts detected at all.")

if first_near_without_contact_step is not None:
    nb_ball_pos, nb_player_pos, nb_dist = first_near_without_contact_state
    print(
        f"First NEAR (3D) without ball–player contact at step {first_near_without_contact_step}, "
        f"dist3d={nb_dist:.4f}"
    )
    print(f"  Positions: ball={nb_ball_pos}, player={nb_player_pos}")
else:
    print("No 'near without contact' situation detected (3D threshold not crossed).")

print(
    f"Max |ball.x| over the run was {max_abs_x:.6f} at step {max_abs_x_step}, "
    f"position={max_abs_x_pos}"
)
print("===================================\n")

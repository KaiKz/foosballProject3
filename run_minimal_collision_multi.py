import mujoco
import numpy as np

MODEL_PATH = "foosball_sim/v2/minimal_collision_multi.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("Geom names:", [model.geom(i).name for i in range(model.ngeom)])

# --- IDs we care about ---
ball_geom_id = model.geom("ball").id
block_left_geom_id = model.geom("block_left_geom").id
block_left_body_id = model.body("block_left").id

# actuator for the sliding block
block_left_act_id = model.actuator("block_left_ctrl").id
ctrl_idx = block_left_act_id  # data.ctrl is indexed by actuator id

for step in range(300):
    # --- DRIVE THE BLOCK TOWARD THE BALL ---
    # actuator is type="position", so ctrl is the *target joint position*
    # joint range is [-0.5, 0.5]; starting body y = -0.5
    # set target to +0.5 to sweep it through the ball at y=0
    data.ctrl[ctrl_idx] = 0.5

    mujoco.mj_step(model, data)

    # ball height
    ball_z = data.geom_xpos[ball_geom_id][2]
    # block_left world y position
    block_left_y = data.xpos[block_left_body_id][1]

    # contact debug
    ncon = data.ncon
    ball_contacts = 0
    contact_pairs = []

    for i in range(ncon):
        con = data.contact[i]
        g1 = model.geom(con.geom1).name
        g2 = model.geom(con.geom2).name
        involves_ball = (con.geom1 == ball_geom_id or con.geom2 == ball_geom_id)
        if involves_ball:
            ball_contacts += 1
            contact_pairs.append((g1, g2))

    if step % 10 == 0 or ball_contacts > 0:
        print(
            f"step {step:03d} | ball_z={ball_z:.3f} "
            f"block_left_y={block_left_y:.3f} "
            f"ncon={ncon}, ball_contacts={ball_contacts}, pairs={contact_pairs}"
        )

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("foosball_sim/v2/minimal_collision.xml")
data = mujoco.MjData(model)


# Inspect available geoms if you want
print("Geom names:", [model.geom(i).name for i in range(model.ngeom)])

# Body ids
ball_body_id = model.body("ball_body").id
block_body_id = model.body("block").id

# Use the real ball geom name: 'ball'
ball_geom_id = model.geom("ball").id      # <-- changed from "ball_geom"

for step in range(300):
    mujoco.mj_step(model, data)

    ball_z = float(data.xpos[ball_body_id][2])
    block_z = float(data.xpos[block_body_id][2])

    ncon = data.ncon
    ball_contacts = 0

    for i in range(ncon):
        con = data.contact[i]
        if con.geom1 == ball_geom_id or con.geom2 == ball_geom_id:
            ball_contacts += 1

    print(
        f"step {step:03d} | "
        f"ball_z={ball_z:.3f} block_z={block_z:.3f} "
        f"ncon={ncon}, ball_contacts={ball_contacts}"
    )
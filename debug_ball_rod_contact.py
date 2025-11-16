# debug_ball_rod_contact.py
import mujoco
import numpy as np
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv

env = FoosballEnv(
    antagonist_model=None,
    verbose_mode=True,      # so CONTACT DEBUG prints
    debug_free_ball=True,   # also relaxes ball friction & stiffness
)
obs, info = env.reset()

model = env.model
data = env.data

# Ensure positions are up-to-date
mujoco.mj_forward(model, data)

# ---------- Find ball joints ----------
ball_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
ball_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
ball_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_z")  # if exists

x_qpos_adr = model.jnt_qposadr[ball_x_id]
y_qpos_adr = model.jnt_qposadr[ball_y_id]
if ball_z_id >= 0:
    z_qpos_adr = model.jnt_qposadr[ball_z_id]
else:
    z_qpos_adr = None

# ---------- Pick a player geom ----------
player_geom_name = "y_mid_guy3"   # from your XML
g_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, player_geom_name)

# Use WORLD coordinates:
gx, gy, gz = data.geom_xpos[g_id].copy()
print(f"[DEBUG] player geom {player_geom_name} world pos =", gx, gy, gz)

# ---------- Move ball to exactly that xyz ----------
data.qpos[x_qpos_adr] = gx
data.qpos[y_qpos_adr] = gy
if z_qpos_adr is not None:
    data.qpos[z_qpos_adr] = gz

mujoco.mj_forward(model, data)

# Optional: print ball geom world pos too
# (find ball geom by name containing "ball" or whatever it is)
for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    if name and "ball" in name.lower():
        print("[DEBUG] candidate ball geom:", g, name, "xpos=", data.geom_xpos[g])

# ---------- Spin rod hard ----------
action = np.zeros(env.protagonist_action_size, dtype=np.float32)
action[0] = 20.0  # some large torque/force on that rod

for t in range(10):
    obs, reward, terminated, truncated, info = env.step(action)
    # CONTACT DEBUG2 is already called inside step() for first few steps

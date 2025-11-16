# debug_rods_move.py
import numpy as np
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv
import mujoco

env = FoosballEnv(antagonist_model=None, verbose_mode=False)
obs, info = env.reset()

for t in range(20):
    # Big positive push on one linear joint, others zero
    action = np.zeros(env.protagonist_action_size, dtype=np.float32)
    action[0] = 20.0  # first rod linear joint

    obs, reward, terminated, truncated, info = env.step(action)

    # Print that rod’s qpos and qvel
    model = env.model
    data = env.data
    jname = "y_goal_linear"  # or whatever the first joint actually is in your XML
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    qpos_adr = model.jnt_qposadr[j_id]
    qvel_adr = model.jnt_dofadr[j_id]

    print(
        f"t={t} qpos={float(data.qpos[qpos_adr]):.4f}, "
        f"qvel={float(data.qvel[qvel_adr]):.4f}"
    )

#!/usr/bin/env python3
import os

import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Reuse the same SIM_PATH that FoosballEnv uses
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import SIM_PATH


def init_mujoco():
    xml_file = SIM_PATH
    print(f"[Step1a] Loading MuJoCo model from: {xml_file}")
    if not os.path.exists(xml_file):
        raise FileNotFoundError(
            f"MuJoCo XML not found at {xml_file}. "
            f"Set SIM_PATH env var or fix the path."
        )

    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)

    # Find the free joint of the ball
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    if ball_joint < 0:
        raise RuntimeError(
            "Joint 'ball_free' not found in the model. Check your MJCF names."
        )

    ball_qpos_adr = model.jnt_qposadr[ball_joint]  # start index in qpos
    ball_qvel_adr = model.jnt_dofadr[ball_joint]   # start index in qvel

    # Forward once to get default state
    mujoco.mj_forward(model, data)

    return model, data, ball_qpos_adr, ball_qvel_adr


def main():
    model, data, ball_qpos_adr, ball_qvel_adr = init_mujoco()

    # Put ball at default position, zero velocity
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    sim_x, sim_y = [], []
    dt = model.opt.timestep
    T = 5.0  # seconds to simulate
    steps = int(T / dt)
    print(f"[Step1a] Simulating {T:.2f}s with dt={dt:.6f}, steps={steps}")

    for _ in range(steps):
        mujoco.mj_step(model, data)
        bx = float(data.qpos[ball_qpos_adr + 0])
        by = float(data.qpos[ball_qpos_adr + 1])
        sim_x.append(bx)
        sim_y.append(by)

    sim_x = np.array(sim_x)
    sim_y = np.array(sim_y)
    t = np.arange(steps) * dt

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title("Ball X trajectory at rest")
    plt.plot(t, sim_x, label="sim x")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Ball Y trajectory at rest")
    plt.plot(t, sim_y, label="sim y")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

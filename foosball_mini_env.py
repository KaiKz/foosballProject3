import os
from typing import Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class FoosballMiniEnv(gym.Env):
    """
    Super-simplified foosball env for debugging:
      - One controllable rod (slide joint along axis).
      - Ball:
          * Mode 1: slide joints 'ball_x' and 'ball_y', OR
          * Mode 2: freejoint 'ball_free' on body 'ball'.
      - Reward: move ball toward +x (opponent goal).
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        sim_path: str | None = "/Users/kaikaizhang/foosballProject3/foosball_sim/v2/foosball_sim.xml",
        render_mode: str | None = None,
        frame_skip: int = 5,
        max_steps: int = 300,
    ):
        super().__init__()

        # ---------- MuJoCo model / data ----------
        if sim_path is None:
            _dir_path = os.path.dirname(__file__)
            default_path = os.path.join(
                _dir_path, "..", "..", "foosball_sim", "v2", "foosball_sim.xml"
            )
            sim_path = os.environ.get("SIM_PATH", default_path)

        self.model = mujoco.MjModel.from_xml_path(sim_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.step_count = 0
        self.dt = self.model.opt.timestep

        # ========== BALL SETUP ==========
        # Try slide joints 'ball_x' and 'ball_y'
        self.use_slide_ball = False
        self.use_free_ball = False

        try:
            self.ball_x_joint_name = "ball_x"
            self.ball_y_joint_name = "ball_y"

            self.ball_x_joint_id = self.model.joint(self.ball_x_joint_name).id
            self.ball_y_joint_id = self.model.joint(self.ball_y_joint_name).id

            self.ball_x_qpos_adr = int(self.model.jnt_qposadr[self.ball_x_joint_id])
            self.ball_y_qpos_adr = int(self.model.jnt_qposadr[self.ball_y_joint_id])

            self.ball_x_qvel_adr = int(self.model.jnt_dofadr[self.ball_x_joint_id])
            self.ball_y_qvel_adr = int(self.model.jnt_dofadr[self.ball_y_joint_id])

            self.use_slide_ball = True
            print("[FoosballMiniEnv] Using ball slide joints 'ball_x' / 'ball_y'")
        except KeyError:
            # Fallback: locate ball via its collision geom 'ball_phys'
            self.ball_geom_name = "ball_phys"
            ball_geom_id = self.model.geom(self.ball_geom_name).id

            # BODY that actually owns the collision geom
            ball_body_id = self.model.geom_bodyid[ball_geom_id]
            self.ball_body_id = int(ball_body_id)
            self.ball_body_name = self.model.body(self.ball_body_id).name

            # Joint that controls that body (should be 'ball_free')
            self.ball_free_joint_name = "ball_free"
            self.ball_free_joint_id = self.model.joint(self.ball_free_joint_name).id

            # qpos / qvel adr for that joint
            self.ball_free_qpos_adr = int(self.model.jnt_qposadr[self.ball_free_joint_id])
            self.ball_free_qvel_adr = int(self.model.jnt_dofadr[self.ball_free_joint_id])

            self.use_free_ball = True
            print(
                f"[FoosballMiniEnv] Using freejoint '{self.ball_free_joint_name}' "
                f"on body '{self.ball_body_name}' (via geom '{self.ball_geom_name}')"
            )



        # previous ball pos for reward & finite-diff velocity if needed
        self._prev_ball_x = 0.0
        self._prev_ball_y = 0.0

        # ========== ROD SETUP ==========
        self.rod_joint_name = "y_mid_linear"
        self.rod_joint_id = self.model.joint(self.rod_joint_name).id
        self.rod_qpos_adr = int(self.model.jnt_qposadr[self.rod_joint_id])
        self.rod_qvel_adr = int(self.model.jnt_dofadr[self.rod_joint_id])

        rod_ctrl_index = None
        for i in range(self.model.nu):
            a = self.model.actuator(i)
            j_id = int(a.trnid[0])  # joint id that actuator drives
            if j_id == self.rod_joint_id:
                rod_ctrl_index = i
                break

        if rod_ctrl_index is None:
            raise RuntimeError(f"Could not find actuator for joint {self.rod_joint_name}")

        self.rod_ctrl_index = rod_ctrl_index
        print(
            f"[FoosballMiniEnv] Using rod joint '{self.rod_joint_name}' "
            f"with actuator index {self.rod_ctrl_index}"
        )

        # ========== SPACES ==========
        # Action: 1D continuous – desired rod position command (servo)
        act_low = np.array([-1.0], dtype=np.float32)
        act_high = np.array([1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Observation: [ball_x, ball_y, ball_vx, ball_vy, rod_x, rod_vx]
        obs_low = np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32
        )
        obs_high = -obs_low
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._viewer = None

    # ---------- Helper: ball state ----------

    def _get_ball_xy(self) -> Tuple[float, float]:
        if self.use_slide_ball:
            x = float(self.data.qpos[self.ball_x_qpos_adr])
            y = float(self.data.qpos[self.ball_y_qpos_adr])
            return x, y
        else:
            pos = self.data.xpos[self.ball_body_id]
            return float(pos[0]), float(pos[1])


    def _get_ball_vxy(self):
        ball_x, ball_y = self._get_ball_xy()
        if not hasattr(self, "_prev_ball_x") or not hasattr(self, "_prev_ball_y"):
            self._prev_ball_x = ball_x
            self._prev_ball_y = ball_y
            return 0.0, 0.0

        vx = (ball_x - self._prev_ball_x) / self.dt
        vy = (ball_y - self._prev_ball_y) / self.dt

        self._prev_ball_x = ball_x
        self._prev_ball_y = ball_y

        return float(vx), float(vy)


    # ---------- Helper: rod state ----------

    def _get_rod_x(self) -> float:
        return float(self.data.qpos[self.rod_qpos_adr])

    def _get_rod_vx(self) -> float:
        return float(self.data.qvel[self.rod_qvel_adr])

    # ---------- Helper: observation, reward, termination ----------

    def _get_obs(self) -> np.ndarray:
        bx, by = self._get_ball_xy()
        bvx, bvy = self._get_ball_vxy()
        rx = self._get_rod_x()
        rvx = self._get_rod_vx()
        obs = np.array([bx, by, bvx, bvy, rx, rvx], dtype=np.float32)
        return obs

    def _is_ball_out(self) -> bool:
        # crude bounds; adjust once you inspect real ranges
        bx, by = self._get_ball_xy()
        return not (-20.0 <= bx <= 20.0 and -20.0 <= by <= 20.0)

    def _compute_reward(self) -> float:
        bx, _ = self._get_ball_xy()
        delta_x = bx - self._prev_ball_x
        self._prev_ball_x = bx

        time_penalty = -0.05
        return float(time_penalty + delta_x)

    # ---------- Gym API ----------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # --- rod initial state ---
        self.data.qpos[self.rod_qpos_adr] = 0.0
        self.data.qvel[self.rod_qvel_adr] = 0.0

        # --- ball initial state ---
        if self.use_slide_ball:
            # ball_x, ball_y slide joints
            self.data.qpos[self.ball_x_qpos_adr] = 0.0
            self.data.qpos[self.ball_y_qpos_adr] = 0.0

            self.data.qvel[self.ball_x_qvel_adr] = 0.0  # tiny push along +x
            self.data.qvel[self.ball_y_qvel_adr] = 0.0
        else:
            # ---------- FREEJOINT BALL RESET (ball_free on ball_body) ----------
            adr = self.ball_free_qpos_adr
            vadr = self.ball_free_qvel_adr

            # 1) identity quat
            self.data.qpos[adr + 0] = 1.0
            self.data.qpos[adr + 1] = 0.0
            self.data.qpos[adr + 2] = 0.0
            self.data.qpos[adr + 3] = 0.0

            mujoco.mj_forward(self.model, self.data)
                    # --- DEBUG: dump key body positions at reset ---
            def _dump_body(name):
                try:
                    bid = self.model.body(name).id
                    print(f"[DEBUG xpos] {name}: {self.data.xpos[bid]}")
                except KeyError:
                    print(f"[DEBUG xpos] body '{name}' not found")

            print("===== DEBUG POSITIONS AT RESET =====")
            for n in ["ball_body", "y_mid_rod", "y_mid_guy3", "ground", "table"]:
                _dump_body(n)
            print("====================================")


            # 2) Pick a reference player on the yellow mid rod
            #    (any mid guy works; 3 is roughly center)
            guy_body_name = "y_mid_guy3"
            guy_body_id = self.model.body(guy_body_name).id
            gx, gy, gz = self.data.xpos[guy_body_id]

            # 3) Place ball slightly in front of that player, at about same height
            #    - small offset along x so they don't start interpenetrating
            #    - tiny lift in z to avoid exploding contact at reset
            for k in range(6):
                self.data.qvel[vadr + k] = 0.0

            mujoco.mj_forward(self.model, self.data)

            # store DOF slice for debugging only
            jid = self.ball_free_joint_id
            dof0 = self.model.jnt_dofadr[jid]
            if jid + 1 < self.model.njnt:
                dofN = self.model.jnt_dofadr[jid + 1]
            else:
                dofN = self.model.nv
            self.ball_free_dof_slice = slice(dof0, dofN)
            print("ball qvel at reset:", self.data.qvel[dof0:dofN])

            # (optional) keep a DOF slice around for debugging
            jid = self.model.joint(self.ball_free_joint_name).id
            dof0 = self.model.jnt_dofadr[jid]
            if jid + 1 < self.model.njnt:
                dofN = self.model.jnt_dofadr[jid + 1]
            else:
                dofN = self.model.nv
            self.ball_free_dof_slice = slice(dof0, dofN)
            print("ball qvel at reset:", self.data.qvel[dof0:dofN])
            # NOTE: DO NOT zero qvel here again – we already did it above.


        mujoco.mj_forward(self.model, self.data)
        # in reset(), after mj_forward or after you set qpos
        jid = self.model.joint(self.ball_free_joint_name).id
        dof0 = self.model.jnt_dofadr[jid]
        if jid + 1 < self.model.njnt:
            dofN = self.model.jnt_dofadr[jid + 1]
        else:
            dofN = self.model.nv
        self.ball_free_dof_slice = slice(dof0, dofN)
        print("ball qvel at reset:", self.data.qvel[dof0:dofN])
        self.data.qvel[self.ball_free_dof_slice] = 0


        # init prev_ball_x / y for reward & velocity sanity
        bx, by = self._get_ball_xy()
        self._prev_ball_x = bx
        self._prev_ball_y = by

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action):
        # ensure 1D scalar
        if isinstance(action, np.ndarray):
            u = float(np.clip(action[0], -1.0, 1.0))
        else:
            u = float(np.clip(action, -1.0, 1.0))

        # control -> correct actuator
        self.data.ctrl[self.rod_ctrl_index] = u

        # simulate
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        if self.use_free_ball:
            q = self.data.qpos[self.ball_free_qpos_adr : self.ball_free_qpos_adr + 7]
            v = self.data.qvel[self.ball_free_qvel_adr : self.ball_free_qvel_adr + 6]
            print("[DEBUG freejoint qpos]", q)
            print("[DEBUG freejoint qvel]", v)

        ball_contacts = 0
        pairs = []

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = self.model.geom(con.geom1).name
            g2 = self.model.geom(con.geom2).name
            involves_ball = ("ball" in (g1, g2))  # or stricter: g1 == self._ball_geom_name, etc.
            if involves_ball:
                ball_contacts += 1
                pairs.append((g1, g2))

        # Optional: print for first episode or every N steps
        if ball_contacts > 0:
            print(f"[FoosballMiniEnv] contacts with ball: {pairs}")
                    
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = self.model.geom(con.geom1).name
            g2 = self.model.geom(con.geom2).name
            if "ball" in g1 or "ball" in g2:
                print("[contact] ball with:", g1, "<->", g2, "dist", con.dist)
                break

        self.step_count += 1

        reward = self._compute_reward()
        terminated = self._is_ball_out()
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        bx, by = float(obs[0]), float(obs[1])
        rx = float(obs[4])

        info = {
            "ball_x": bx,
            "ball_y": by,
            "rod_x": rx,
        }

        return obs, reward, terminated, truncated, info

    # ---------- Rendering ----------

    def render(self):
        if self.render_mode != "human":
            return

        if self._viewer is None:
            from mujoco import viewer
            self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

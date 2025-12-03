import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
from mujoco import mj_name2id, mjtObj


class FoosballSingleRodEnv(gym.Env):
    """
    Minimal MuJoCo foosball env for SAC/TQC on a *non-trivial* task.

    - Single controllable rod moving along +y/-y.
    - Ball starts near the rod, slightly randomized.
    - Reward: push ball toward +y, time + control penalties.
    - Goal: ball moves forward by at least `goal_delta_y` from its start.

    Info dict includes:
        - ball_pos
        - ball_vel
        - rod_y
        - rod_y_vel
        - is_goal
        - had_contact
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 1.0}

    def __init__(
        self,
        xml_path: str = "foosball_sim/v2/minimal_foosball_stage1.xml",
        render_mode: Optional[str] = None,
        episode_length: int = 150,
        frame_skip: int = 2,
        ball_body_name: str = "ball_body",
        player_body_name: str = "player_1",
        rod_joint_name: str = "rod_1_slide",
        rod_actuator_name: str = "rod_1_motor",
    ):
        super().__init__()

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = episode_length
        self.step_count = 0

        # ---------- IDs from names ----------
        self.ball_body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, ball_body_name)
        self.player_body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, player_body_name)
        self.rod_joint_id = mj_name2id(self.model, mjtObj.mjOBJ_JOINT, rod_joint_name)
        self.rod_actuator_id = mj_name2id(self.model, mjtObj.mjOBJ_ACTUATOR, rod_actuator_name)

        if self.ball_body_id < 0:
            raise ValueError(f"Ball body '{ball_body_name}' not found in model.")
        if self.player_body_id < 0:
            raise ValueError(f"Player body '{player_body_name}' not found in model.")
        if self.rod_joint_id < 0:
            raise ValueError(f"Rod joint '{rod_joint_name}' not found in model.")
        if self.rod_actuator_id < 0:
            raise ValueError(f"Rod actuator '{rod_actuator_name}' not found in model.")

        # Rod qpos / qvel addresses
        self.rod_qpos_adr = self.model.jnt_qposadr[self.rod_joint_id]
        self.rod_qvel_adr = self.model.jnt_dofadr[self.rod_joint_id]

        # ---------- Task / reward hyperparameters ----------
        self.max_rod_ctrl = 5.0          # motor strength
        self.delta_y_scale = 10.0        # reward for pushing ball in +y
        self.contact_bonus = 0.2         # reward per contact
        self.ctrl_cost_coeff = 0.001     # penalize large actions
        self.time_penalty = 0.01         # per-step cost
        self.goal_bonus = 60.0           # NEW: a bit smaller than 50/200, still meaningful

        # NEW: goal is defined *relative* to starting ball y
        self.goal_delta_y = 0.12         # need to move forward at least 0.12m
        self.goal_x_limit = 0.40         # must be roughly between walls
        self.start_ball_y = 0.0          # will be set in reset()

        # ---------- Spaces ----------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # [ball_pos(3), ball_vel(3), rod_y, rod_y_vel] = 8-dim
        obs_high = np.ones(8, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        self.prev_ball_y = 0.0
        self._viewer = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ball_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        ball_pos: data.xpos[ball_body_id]  (world position)
        ball_vel: linear vel, from xvelp if available, else cvel
        """
        ball_pos = self.data.xpos[self.ball_body_id].copy()

        if hasattr(self.data, "xvelp"):
            ball_vel = self.data.xvelp[self.ball_body_id].copy()
        else:
            ball_vel = self.data.cvel[self.ball_body_id, 3:].copy()

        return ball_pos, ball_vel

    def _get_rod_state(self) -> Tuple[float, float]:
        rod_y = self.data.qpos[self.rod_qpos_adr]
        rod_y_vel = self.data.qvel[self.rod_qvel_adr]
        return float(rod_y), float(rod_y_vel)

    def _get_obs(self) -> np.ndarray:
        ball_pos, ball_vel = self._get_ball_state()
        rod_y, rod_y_vel = self._get_rod_state()
        obs = np.concatenate(
            [ball_pos, ball_vel, np.array([rod_y, rod_y_vel], dtype=np.float64)]
        )
        return obs.astype(np.float32)

    def _is_ball_out_of_bounds(self, ball_pos: np.ndarray) -> bool:
        """
        Simple bounds based on your table geometry.
        """
        x, y, z = ball_pos
        if z < 0.0 or z > 0.5:
            return True
        if abs(x) > 0.5:
            return True
        if abs(y) > 0.9:
            return True
        return False

    def _has_ball_player_contact(self) -> bool:
        """
        Detect ball-player contact via geom→body mapping.
        """
        ball_body = self.ball_body_id
        player_body = self.player_body_id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2
            body1 = self.model.geom_bodyid[g1]
            body2 = self.model.geom_bodyid[g2]
            if ((body1 == ball_body and body2 == player_body)
                    or (body1 == player_body and body2 == ball_body)):
                return True
        return False

    def _is_goal(self, ball_pos: np.ndarray) -> bool:
        """
        Goal if ball moves forward by at least goal_delta_y from its start,
        stays roughly centered in x, and is above the ground.
        """
        x, y, z = ball_pos
        dy = y - self.start_ball_y
        return (dy > self.goal_delta_y) and (abs(x) < self.goal_x_limit) and (z > 0.0)

    # ------------------------------------------------------------------
    # RL API
    # ------------------------------------------------------------------

    def reset(
        self, *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        self.step_count = 0

        mujoco.mj_resetData(self.model, self.data)

        rng = np.random.default_rng(seed)

        # ----- Initialize ball near rod, slightly random -----
        ball_body = self.ball_body_id
        num_jnt = self.model.body_jntnum[ball_body]
        jntadr = self.model.body_jntadr[ball_body]

        if num_jnt > 0:
            jnt_id = jntadr
            jnt_type = self.model.jnt_type[jnt_id]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            qvel_adr = self.model.jnt_dofadr[jnt_id]

            if jnt_type == 0:  # free joint
                # qpos: [x, y, z, qw, qx, qy, qz]
                self.data.qpos[qpos_adr + 0] = 0.0
                self.data.qpos[qpos_adr + 1] = rng.uniform(-0.1, 0.1)
                self.data.qpos[qpos_adr + 2] = 0.10

                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array(
                    [1.0, 0.0, 0.0, 0.0]
                )

                # small initial velocity
                self.data.qvel[qvel_adr : qvel_adr + 3] = np.array(
                    [0.0, 0.0, 0.0]
                )

        # ----- Reset rod near center -----
        self.data.qpos[self.rod_qpos_adr] = 0.0
        self.data.qvel[self.rod_qvel_adr] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # NEW: record starting ball y to define relative goal
        ball_pos, _ = self._get_ball_state()
        self.prev_ball_y = ball_pos[1]
        self.start_ball_y = ball_pos[1]

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl_val = float(action[0]) * self.max_rod_ctrl
        self.data.ctrl[self.rod_actuator_id] = ctrl_val

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        ball_pos, ball_vel = self._get_ball_state()
        rod_y, rod_y_vel = self._get_rod_state()

        # ----- Reward -----
        delta_y = ball_pos[1] - self.prev_ball_y
        self.prev_ball_y = ball_pos[1]

        had_contact = self._has_ball_player_contact()

        progress_reward = self.delta_y_scale * delta_y
        contact_reward = self.contact_bonus if had_contact else 0.0
        ctrl_penalty = self.ctrl_cost_coeff * float(np.square(action).sum())
        step_penalty = self.time_penalty

        reward = progress_reward + contact_reward - ctrl_penalty - step_penalty

        # Termination
        terminated = False
        truncated = False

        is_goal = self._is_goal(ball_pos)
        if is_goal:
            reward += self.goal_bonus
            terminated = True

        if self._is_ball_out_of_bounds(ball_pos):
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        info: Dict[str, Any] = {
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "rod_y": rod_y,
            "rod_y_vel": rod_y_vel,
            "is_goal": is_goal,
            "had_contact": had_contact,
        }

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode != "human":
            return
        try:
            import mujoco.viewer as viewer
        except ImportError:
            return
        if self._viewer is None:
            self._viewer = viewer.launch_passive(self.model, self.data)
        else:
            self._viewer.sync()

    def close(self):
        self._viewer = None

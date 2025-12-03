import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
from mujoco import mj_name2id, mjtObj


class FoosballMultiRodEnv(gym.Env):
    """
    Multi-rod foosball env.

    - One agent controls several rods (1D slide each).
    - Action: R-dimensional (one scalar per rod).
    - Obs: ball_pos(3), ball_vel(3), rod_y(R), rod_y_vel(R).
    - Reward: push ball in +y, small contact bonus, time/control penalties,
      big bonus on "goal" (ball moved forward enough from start).
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 1.0}

    def __init__(
        self,
        xml_path: str = "foosball_sim/v2/multirod_foosball.xml",
        render_mode: Optional[str] = None,
        episode_length: int = 150,
        frame_skip: int = 2,
        ball_body_name: str = "ball_body",
        player_body_names: Optional[List[str]] = None,
        rod_joint_names: Optional[List[str]] = None,
        rod_actuator_names: Optional[List[str]] = None,
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

        # ---- Defaults for names (2-rod case) ----
        if player_body_names is None:
            player_body_names = ["player_1", "player_2"]
        if rod_joint_names is None:
            rod_joint_names = ["rod_1_slide", "rod_2_slide"]
        if rod_actuator_names is None:
            rod_actuator_names = ["rod_1_motor", "rod_2_motor"]

        assert len(player_body_names) == len(rod_joint_names) == len(rod_actuator_names), \
            "player_body_names, rod_joint_names, and rod_actuator_names must have same length."

        self.n_rods = len(rod_joint_names)

        # ---- IDs from names ----
        self.ball_body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, ball_body_name)
        if self.ball_body_id < 0:
            raise ValueError(f"Ball body '{ball_body_name}' not found in model.")

        self.player_body_ids = []
        for name in player_body_names:
            bid = mj_name2id(self.model, mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Player body '{name}' not found in model.")
            self.player_body_ids.append(bid)

        self.rod_joint_ids = []
        for name in rod_joint_names:
            jid = mj_name2id(self.model, mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Rod joint '{name}' not found in model.")
            self.rod_joint_ids.append(jid)

        self.rod_actuator_ids = []
        for name in rod_actuator_names:
            aid = mj_name2id(self.model, mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Rod actuator '{name}' not found in model.")
            self.rod_actuator_ids.append(aid)

        # qpos / qvel addresses for each rod
        self.rod_qpos_adrs = [self.model.jnt_qposadr[jid] for jid in self.rod_joint_ids]
        self.rod_qvel_adrs = [self.model.jnt_dofadr[jid] for jid in self.rod_joint_ids]

        # ---- Task / reward hyperparameters (same spirit as single rod) ----
        self.max_rod_ctrl = 5.0
        self.delta_y_scale = 10.0
        self.contact_bonus = 0.2
        self.ctrl_cost_coeff = 0.001
        self.time_penalty = 0.01
        self.goal_bonus = 60.0

        self.goal_delta_y = 0.12
        self.goal_x_limit = 0.40
        self.start_ball_y = 0.0

        # ---- Spaces ----
        # Action: one scalar per rod (clamped to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_rods,),
            dtype=np.float32,
        )

        # Observation: ball_pos(3), ball_vel(3), rod_y(R), rod_y_vel(R)
        obs_dim = 6 + 2 * self.n_rods
        obs_high = np.ones(obs_dim, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        self.prev_ball_y = 0.0
        self._viewer = None

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _get_ball_state(self) -> Tuple[np.ndarray, np.ndarray]:
        ball_pos = self.data.xpos[self.ball_body_id].copy()
        if hasattr(self.data, "xvelp"):
            ball_vel = self.data.xvelp[self.ball_body_id].copy()
        else:
            ball_vel = self.data.cvel[self.ball_body_id, 3:].copy()
        return ball_pos, ball_vel

    def _get_rod_state(self) -> Tuple[np.ndarray, np.ndarray]:
        rod_y = np.array([self.data.qpos[adr] for adr in self.rod_qpos_adrs], dtype=np.float64)
        rod_y_vel = np.array([self.data.qvel[adr] for adr in self.rod_qvel_adrs], dtype=np.float64)
        return rod_y, rod_y_vel

    def _get_obs(self) -> np.ndarray:
        ball_pos, ball_vel = self._get_ball_state()
        rod_y, rod_y_vel = self._get_rod_state()
        obs = np.concatenate([ball_pos, ball_vel, rod_y, rod_y_vel])
        return obs.astype(np.float32)

    def _is_ball_out_of_bounds(self, ball_pos: np.ndarray) -> bool:
        x, y, z = ball_pos
        if z < 0.0 or z > 0.5:
            return True
        if abs(x) > 0.5:
            return True
        if abs(y) > 0.9:
            return True
        return False

    def _has_ball_player_contact(self) -> bool:
        """True if ball is touching ANY player body."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2
            body1 = self.model.geom_bodyid[g1]
            body2 = self.model.geom_bodyid[g2]

            # ball vs any player body
            if body1 == self.ball_body_id and body2 in self.player_body_ids:
                return True
            if body2 == self.ball_body_id and body1 in self.player_body_ids:
                return True
        return False

    def _is_goal(self, ball_pos: np.ndarray) -> bool:
        x, y, z = ball_pos
        dy = y - self.start_ball_y
        return (dy > self.goal_delta_y) and (abs(x) < self.goal_x_limit) and (z > 0.0)

    # --------------------------------------------------
    # RL API
    # --------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        self.step_count = 0

        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)

        # Ball initial state (similar to single-rod)
        ball_body = self.ball_body_id
        num_jnt = self.model.body_jntnum[ball_body]
        jntadr = self.model.body_jntadr[ball_body]
        if num_jnt > 0:
            jnt_id = jntadr
            jnt_type = self.model.jnt_type[jnt_id]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            qvel_adr = self.model.jnt_dofadr[jnt_id]
            if jnt_type == 0:  # free joint
                self.data.qpos[qpos_adr + 0] = 0.0
                self.data.qpos[qpos_adr + 1] = rng.uniform(-0.1, 0.1)
                self.data.qpos[qpos_adr + 2] = 0.10
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array(
                    [1.0, 0.0, 0.0, 0.0]
                )
                self.data.qvel[qvel_adr : qvel_adr + 3] = np.array([0.0, 0.0, 0.0])

        # Reset all rods to center
        for qadr, nadr in zip(self.rod_qpos_adrs, self.rod_qvel_adrs):
            self.data.qpos[qadr] = 0.0
            self.data.qvel[nadr] = 0.0

        mujoco.mj_forward(self.model, self.data)

        ball_pos, _ = self._get_ball_state()
        self.prev_ball_y = ball_pos[1]
        self.start_ball_y = ball_pos[1]

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray):
        self.step_count += 1

        # clip and apply all rod actions
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        for k, aid in enumerate(self.rod_actuator_ids):
            ctrl_val = float(action[k]) * self.max_rod_ctrl
            self.data.ctrl[aid] = ctrl_val

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        ball_pos, ball_vel = self._get_ball_state()
        rod_y, rod_y_vel = self._get_rod_state()

        # Reward
        delta_y = ball_pos[1] - self.prev_ball_y
        self.prev_ball_y = ball_pos[1]

        had_contact = self._has_ball_player_contact()

        progress_reward = self.delta_y_scale * delta_y
        contact_reward = self.contact_bonus if had_contact else 0.0
        ctrl_penalty = self.ctrl_cost_coeff * float(np.square(action).sum())
        step_penalty = self.time_penalty

        reward = progress_reward + contact_reward - ctrl_penalty - step_penalty

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

    # --------------------------------------------------
    # Rendering
    # --------------------------------------------------

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

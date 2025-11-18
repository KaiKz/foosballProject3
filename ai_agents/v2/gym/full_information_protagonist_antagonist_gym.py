import math
import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ai_agents.v2.gym.mujoco_table_render_mixin import MujocoTableRenderMixin

DIRECTION_CHANGE = 1
TABLE_MAX_Y_DIM = 65
BALL_STOPPED_COUNT_THRESHOLD = 200
MAX_STEPS = 40

# Reward shaping: pretend the goal is closer than the physical one
REWARD_GOAL_Y_FRACTION = 0.2  # 20% of the full distance; tweak as needed


# Calculate project root and build relative path to simulation XML
_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_default_path = os.path.join(_dir_path, "foosball_sim", "v2", "foosball_sim.xml")
SIM_PATH = os.environ.get("SIM_PATH", _default_path)

F32 = np.float32
INF32 = np.finfo(np.float32).max

RODS = ["_goal_", "_def_", "_mid_", "_attack_"]


class FoosballEnv(MujocoTableRenderMixin, gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, antagonist_model=None, play_until_goal=False, verbose_mode=False,  debug_free_ball=False):
        super(FoosballEnv, self).__init__()

        xml_file = SIM_PATH
        print("[FoosballEnv] Loading XML from:", xml_file)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        # --- BALL INDEXES (FREE JOINT VERSION) ---
        self.ball_free_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
        )
        if self.ball_free_joint < 0:
            raise RuntimeError("Joint 'ball_free' not found in the model")

        self.ball_qpos_adr = self.model.jnt_qposadr[self.ball_free_joint]
        self.ball_qvel_adr = self.model.jnt_dofadr[self.ball_free_joint]
                
        
        # ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        # ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        # ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")

        # print("[BODY DEBUG] ball geom body   =", self.model.geom_bodyid[ball_geom_id])
        # print("[BODY DEBUG] ball_x joint body=", self.model.jnt_bodyid[ball_x_id])
        # print("[BODY DEBUG] ball_y joint body=", self.model.jnt_bodyid[ball_y_id])
        # print("[OPT DEBUG] disableflags before =", self.model.opt.disableflags)
        # self.model.opt.disableflags = 0

        # print("[OPT DEBUG] disableflags after  =", self.model.opt.disableflags)

        # DEBUG: force-enable contact for all geoms
        for g in range(self.model.ngeom):
            self.model.geom_contype[g] = 1
            self.model.geom_conaffinity[g] = 1
            
        print("=== GEOM CONTACT DEBUG ===")
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            ct = self.model.geom_contype[g]
            ca = self.model.geom_conaffinity[g]
            print(f"geom {g:2d} name={name} contype={ct} conaffinity={ca}")
            

        # ---------- GLOBAL DEBUG: DOF / GEOM / OPT PARAMS (NO MODIFICATIONS HERE) ----------
        # print("[GLOBAL DEBUG] max dof_damping before =", float(self.model.dof_damping.max()))
        # print("[GLOBAL DEBUG] nonzero dof_damping idx =", np.nonzero(self.model.dof_damping)[0])

        # print("[GLOBAL DEBUG] max dof_frictionloss before =", float(self.model.dof_frictionloss.max()))
        # print("[GLOBAL DEBUG] nonzero dof_frictionloss idx =", np.nonzero(self.model.dof_frictionloss)[0])

        # Just peek at the first few geom frictions
        for g in range(min(5, self.model.ngeom)):
            old = np.array(self.model.geom_friction[g], copy=True)
        #     print(f"[GLOBAL DEBUG] geom {g} friction = {old}")

        # print("[OPT DEBUG] timestep        =", self.model.opt.timestep)
        # print("[OPT DEBUG] viscosity       =", self.model.opt.viscosity)
        # print("[OPT DEBUG] density         =", self.model.opt.density)

        # # ---------- BALL-SPECIFIC DEBUG & FIXES ----------

        if debug_free_ball:
            # Only in kick_ball_test / debugging
            self._relax_ball_joint_friction()
            self._relax_ball_joint_stiffness()
            self._debug_ball_dofs()
            # self._debug_ball_geoms()
            # self._disable_ball_contacts()



        # ---------- ENV STATE ----------
        self.simulation_time = 0.0
        self._debug_step_counter = 0

        self.num_rods_per_player = 4
        self.num_players = 2
        self.num_rods = self.num_rods_per_player * self.num_players  # Total rods

        self.protagonist_action_size = self.num_rods_per_player * 2  # 8 actions for protagonist
        self.antagonist_action_size = self.num_rods_per_player * 2   # 8 actions for antagonist

        action_high = np.ones(self.protagonist_action_size, dtype=F32)

        self.rotation_action_space = spaces.Box(
            low=F32(-2.5) * action_high, high=F32(2.5) * action_high, dtype=F32
        )
        self.goal_linear_action_space = spaces.Box(
            low=F32(-10.0) * action_high, high=F32(10.0) * action_high, dtype=F32
        )
        self.def_linear_action_space = spaces.Box(
            low=F32(-20.0) * action_high, high=F32(20.0) * action_high, dtype=F32
        )
        self.mid_linear_action_space = spaces.Box(
            low=F32(-7.0) * action_high, high=F32(7.0) * action_high, dtype=F32
        )
        self.attack_linear_action_space = spaces.Box(
            low=F32(-12.0) * action_high, high=F32(12.0) * action_high, dtype=F32
        )

        # TEMP overall action space (same bounds for all protagonist actions)
        self.action_space = spaces.Box(
            low=F32(-20.0) * action_high, high=F32(20.0) * action_high, dtype=F32
        )

        # obs_dim = 38
        obs_dim = 36
        self.observation_space = spaces.Box(
            low=np.full((obs_dim,), -INF32, dtype=F32),
            high=np.full((obs_dim,), INF32, dtype=F32),
            dtype=F32,
        )

        self.viewer = None

        self._healthy_reward = 1.0
        self._ctrl_cost_weight = 0.005
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (-80, 80)
        self.max_no_progress_steps = 15

        self.prev_ball_y = None
        self.no_progress_steps = 0
        self.ball_stopped_count = 0

        self.antagonist_model = antagonist_model
        self.play_until_goal = play_until_goal
        self.verbose_mode = verbose_mode
        
                # ---------------- NEW: protagonist direction & last-ball-y -------------
        # Let protagonist always try to score toward +y for now.
        self._direction_sign_for_protagonist = 1.0  # or -1.0 if you flip sides
        self._last_ball_y = 0.0

    # -------------------------------------------------------------------------
    # BASIC SETUP / RESET / STEP
    # -------------------------------------------------------------------------
    def _reset_ball_to_center(self):
        # import mujoco

        # ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        # ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")

        # if ball_x_id < 0 or ball_y_id < 0:
        #     raise RuntimeError("ball_x or ball_y joint not found in model")

        # x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
        # y_qpos_adr = self.model.jnt_qposadr[ball_y_id]

        # # Put it roughly in the middle of the table
        # self.data.qpos[x_qpos_adr] = 0.0
        # self.data.qpos[y_qpos_adr] = 0.0

        # # optional: small positive z if you later add ball_z
        # mujoco.mj_forward(self.model, self.data)
            # Position at center
        self.data.qpos[self.ball_x_dof] = 0.0
        self.data.qpos[self.ball_y_dof] = 0.0

        self.data.qvel[self.ball_x_dof] = 0.0
        self.data.qvel[self.ball_y_dof] = 0.0

        self._last_ball_y = 0.0


    def set_antagonist_model(self, antagonist_model):
        self.antagonist_model = antagonist_model

    # def reset(self, *, seed=None, options=None):
    #     super().reset(seed=seed)
    #     mujoco.mj_resetData(self.model, self.data)

    #     # 1) Center the ball (or keep your randomization if you like)
    #     ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    #     ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")

    #     x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
    #     y_qpos_adr = self.model.jnt_qposadr[ball_y_id]

    #     # Centered or lightly randomized
    #     self.data.qpos[x_qpos_adr] = 0.0
    #     self.data.qpos[y_qpos_adr] = 0.0

    #     # 2) Give it a small random “serve” velocity in +y or -y
    #     x_qvel_adr = self.model.jnt_dofadr[ball_x_id]
    #     y_qvel_adr = self.model.jnt_dofadr[ball_y_id]

    #     # e.g. mainly down table (y), small lateral x
    #     vx = self.np_random.uniform(-0.5, 0.5)
    #     vy = self.np_random.uniform(1.0, 2.0)  # towards one goal
    #     self.data.qvel[x_qvel_adr] = vx
    #     self.data.qvel[y_qvel_adr] = vy

    #     mujoco.mj_forward(self.model, self.data)

    #     self.simulation_time = 0.0
    #     self.prev_ball_y = self.data.qpos[y_qpos_adr]
    #     self.no_progress_steps = 0
    #     self.ball_stopped_count = 0
    #     self._debug_step_counter = 0

    #     self._direction_sign_for_protagonist = 1.0
    #     self._last_ball_y = self.data.qpos[y_qpos_adr]

    #     return self._get_obs(), {}
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Center the ball in 2D; choose some reasonable z (same as XML pos)
        base_qpos = self.ball_qpos_adr
        base_qvel = self.ball_qvel_adr

        self.data.qpos[base_qpos + 0] = 0.0  # x
        self.data.qpos[base_qpos + 1] = 0.0  # y
        self.data.qpos[base_qpos + 2] = 0.08  # z, matches XML pos (0 0 0.08)

        # Identity orientation
        self.data.qpos[base_qpos + 3] = 1.0  # qw
        self.data.qpos[base_qpos + 4] = 0.0  # qx
        self.data.qpos[base_qpos + 5] = 0.0  # qy
        self.data.qpos[base_qpos + 6] = 0.0  # qz

        # 2D serve velocity
        vx = self.np_random.uniform(-0.5, 0.5)
        vy = self.np_random.uniform(1.0, 1500.0)

        self.data.qvel[base_qvel + 0] = vx
        self.data.qvel[base_qvel + 1] = vy
        self.data.qvel[base_qvel + 2] = 0.0  # vz
        self.data.qvel[base_qvel + 3] = 0.0  # wx
        self.data.qvel[base_qvel + 4] = 0.0  # wy
        self.data.qvel[base_qvel + 5] = 0.0  # wz

        mujoco.mj_forward(self.model, self.data)

        self.simulation_time = 0.0
        ball_pos, _ = self._get_ball_obs()
        self.prev_ball_y = ball_pos[1]
        self.no_progress_steps = 0
        self.ball_stopped_count = 0
        self._debug_step_counter = 0

        self._direction_sign_for_protagonist = 1.0
        self._last_ball_y = self.prev_ball_y

        return self._get_obs(), {}



    def step(self, protagonist_action):
        protagonist_action = np.clip(
            protagonist_action, self.action_space.low, self.action_space.high
        )

        if self._debug_step_counter == 1:
            _, vel0 = self._get_ball_obs()
            print("[STEP DEBUG] BEFORE first mj_step, ball_vel =", vel0)

        antagonist_observation = self._get_antagonist_obs()

        if self.antagonist_model is not None:
            antagonist_action, _ = self.antagonist_model.predict(antagonist_observation)
            antagonist_action = np.clip(antagonist_action, -1.0, 1.0)
            antagonist_action = self._adjust_antagonist_action(antagonist_action)
        else:
            antagonist_action = np.zeros(self.antagonist_action_size, dtype=F32)

        # Apply controls
        self.data.ctrl[: self.protagonist_action_size] = protagonist_action
        self.data.ctrl[
            self.protagonist_action_size : self.protagonist_action_size
            + self.antagonist_action_size
        ] = antagonist_action

        ball_pos_before = np.array(self._get_ball_obs()[0][:2], dtype=float)

        mujoco.mj_step(self.model, self.data)
        if self._debug_step_counter < 5:
            self._debug_ball_forces()
            # self._debug_ball_forces_2d()
            # self._debug_ball_contacts()

        if self._debug_step_counter == 0:
            pos1, vel1 = self._get_ball_obs()
            print("[STEP DEBUG] AFTER first mj_step, ball_vel =", vel1)

        if self._debug_step_counter == 0:
            _, vel1 = self._get_ball_obs()
            print("[STEP DEBUG] AFTER first mj_step, ball_vel =", vel1)

        self.simulation_time += self.model.opt.timestep

        obs = self._get_obs().astype(F32)
        reward =  float(self._compute_step_reward(protagonist_action))

        terminated = self.terminated

        ball_pos_after = np.array(self._get_ball_obs()[0][:2], dtype=float)
        delta = np.linalg.norm(ball_pos_after - ball_pos_before)

        if self._debug_step_counter < 20:
            print(
                f"[DEBUG PHYSICS] step={self._debug_step_counter} "
                f"ball_before={ball_pos_before} ball_after={ball_pos_after} Δ={delta}"
            )

        try:
            ball_pos, _ = self._get_ball_obs()
            ball_x, ball_y = ball_pos
        except Exception as e:
            print(f"[FoosballEnv DEBUG] _get_ball_obs() failed: {e}")
            ball_x = ball_y = float("nan")

        info = {
            "ball_x": float(ball_x),
            "ball_y": float(ball_y),
            "reward": float(reward),
        }

        self._last_ball_y = ball_y  # update after computing reward
        self._debug_step_counter += 1
        if self._debug_step_counter <= 20:
            print(
                f"[FoosballEnv DEBUG] step={self._debug_step_counter} "
                f"ball_x={ball_x:.3f} ball_y={ball_y:.3f} reward={reward:.3f}"
            )

        return obs, reward, bool(terminated), False, info
    
    def _check_goal_scored(self, ball_pos):
        """
        Decide whether the protagonist scored or conceded,
        taking into account the direction sign.
        """
        ball_x, ball_y = ball_pos
        forward_sign = self._direction_sign_for_protagonist

        # Protagonist tries to score at y = +TABLE_MAX_Y_DIM if forward_sign = +1
        # and at y = -TABLE_MAX_Y_DIM if forward_sign = -1.
        winning_goal = (
            (forward_sign > 0 and ball_y >= TABLE_MAX_Y_DIM) or
            (forward_sign < 0 and ball_y <= -TABLE_MAX_Y_DIM)
        )

        # Own-goal is the opposite side
        losing_goal = (
            (forward_sign > 0 and ball_y <= -TABLE_MAX_Y_DIM) or
            (forward_sign < 0 and ball_y >= TABLE_MAX_Y_DIM)
        )

        return winning_goal, losing_goal


    # -------------------------------------------------------------------------
    # BALL-SPECIFIC DEBUG / FIX HELPERS
    # -------------------------------------------------------------------------
    def _debug_ball_forces_2d(self):
        import mujoco

        for joint_name in ["ball_x", "ball_y"]:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if j_id < 0:
                print(f"[FORCE DEBUG] joint {joint_name} not found")
                continue

            dof = self.model.jnt_dofadr[j_id]

            bias       = float(self.data.qfrc_bias[dof])
            passive    = float(self.data.qfrc_passive[dof])
            constraint = float(self.data.qfrc_constraint[dof])
            actuator   = float(self.data.qfrc_actuator[dof])
            applied    = float(self.data.qfrc_applied[dof])
            qvel       = float(self.data.qvel[dof])
            qacc       = float(self.data.qacc[dof])

            print(
                f"[FORCE DEBUG] {joint_name} (dof={dof}) | "
                f"bias={bias:+.6f}, passive={passive:+.6f}, "
                f"constraint={constraint:+.6f}, actuator={actuator:+.6f}, "
                f"applied={applied:+.6f}, qvel={qvel:+.6f}, qacc={qacc:+.6f}"
            )

    # def _debug_ball_contacts(self):
    #     import mujoco

    #     # Find ball body
    #     j_ball_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    #     if j_ball_x < 0:
    #         print("[CONTACT DEBUG2] ball_x joint not found")
    #         return

    #     ball_body = self.model.jnt_bodyid[j_ball_x]

    #     if self.data.ncon == 0:
    #         print("[CONTACT DEBUG2] no contacts this step")
    #         return

    #     print(f"[CONTACT DEBUG2] ncon = {self.data.ncon}")
    #     for i in range(self.data.ncon):
    #         c = self.data.contact[i]
    #         g1, g2 = c.geom1, c.geom2
    #         b1 = self.model.geom_bodyid[g1]
    #         b2 = self.model.geom_bodyid[g2]

    #         # Only show contacts that involve the ball body
    #         if b1 != ball_body and b2 != ball_body:
    #             continue

    #         n = np.array(c.frame[:3])     # contact normal
    #         dist = c.dist                 # penetration (negative = overlapping)

    #         g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
    #         g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)

    #         # projection of normal on world x axis (ball_x direction)
    #         proj_on_x = n[0]

    #         print(
    #             f"[CONTACT DEBUG2] con#{i} "
    #             f"{g1_name}({g1}) vs {g2_name}({g2}) | "
    #             f"dist={dist:.6f}, normal={n}, proj_on_x={proj_on_x:.3f}"
    #         )
    
    
    def _debug_ball_contacts(self):
        import mujoco

        j_ball_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        if j_ball_x < 0:
            print("[CONTACT DEBUG2] ball_x joint not found")
            return

        ball_body = self.model.jnt_bodyid[j_ball_x]

        if self.data.ncon == 0:
            print("[CONTACT DEBUG2] no contacts this step")
            return

        print(f"[CONTACT DEBUG2] ncon = {self.data.ncon}")
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1 = self.model.geom_bodyid[g1]
            b2 = self.model.geom_bodyid[g2]

            # Only show contacts that involve the ball body
            if b1 != ball_body and b2 != ball_body:
                continue

            n = np.array(c.frame[:3])     # contact normal
            dist = c.dist                 # penetration (negative = overlapping)

            g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)

            print(
                f"[CONTACT DEBUG2] con#{i} "
                f"{g1_name}({g1}) vs {g2_name}({g2}) | "
                f"dist={dist:.6f}, normal={n}"
            )




    def _disable_ball_contacts(self):
        import mujoco

        ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        if ball_x_id < 0:
            print("[CONTACT DEBUG] ball_x joint not found")
            return

        ball_body_id = self.model.jnt_bodyid[ball_x_id]

        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == ball_body_id:
                print(f"[CONTACT DEBUG] disabling contact for geom {g}")
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0


    def _debug_ball_forces(self):
        """Print MuJoCo generalized forces acting on ball_x DOF."""
        j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        if j_id < 0:
            print("[FORCE DEBUG] ball_x joint not found")
            return

        dof = self.model.jnt_dofadr[j_id]

        # These arrays are length nv (number of DOFs)
        bias       = float(self.data.qfrc_bias[dof])
        passive    = float(self.data.qfrc_passive[dof])
        constraint = float(self.data.qfrc_constraint[dof])
        actuator   = float(self.data.qfrc_actuator[dof])
        applied    = float(self.data.qfrc_applied[dof])

        print(
            f"[FORCE DEBUG] dof={dof} | "
            f"bias={bias:+.6f}, passive={passive:+.6f}, "
            f"constraint={constraint:+.6f}, actuator={actuator:+.6f}, "
            f"applied={applied:+.6f}"
        )


    def _relax_ball_joint_friction(self):
        """Real fix: relax only ball_x / ball_y frictionloss."""
        model = self.model

        j_ball_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        j_ball_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")

        if j_ball_x < 0 or j_ball_y < 0:
            print("[BALL FIX] ball_x or ball_y joint not found, skipping friction fix")
            return

        dof_x = model.jnt_dofadr[j_ball_x]
        dof_y = model.jnt_dofadr[j_ball_y]

        print(
            "[BALL FIX] dof_frictionloss before:",
            float(model.dof_frictionloss[dof_x]),
            float(model.dof_frictionloss[dof_y]),
        )

        # Remove or greatly reduce just the ball frictionloss
        model.dof_frictionloss[dof_x] = 0.0  # or something small like 0.1
        model.dof_frictionloss[dof_y] = 0.0

        print(
            "[BALL FIX] dof_frictionloss after:",
            float(model.dof_frictionloss[dof_x]),
            float(model.dof_frictionloss[dof_y]),
        )

    def _debug_ball_dofs(self):
        for name in ["ball_x", "ball_y", "ball_z"]:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id < 0:
                print(f"[BALL DEBUG] joint {name} not found")
                continue

            dof_adr = self.model.jnt_dofadr[j_id]
            damping = self.model.dof_damping[dof_adr]
            friction = self.model.dof_frictionloss[dof_adr]
            armature = self.model.dof_armature[dof_adr]
            print(
                f"[BALL DEBUG] {name}: dof={dof_adr}, "
                f"damping={damping}, frictionloss={friction}, armature={armature}"
            )

    def _debug_ball_geoms(self):
        """List and zero friction for all geoms on the ball body (and print)."""
        for name in ["ball_x", "ball_y"]:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id < 0:
                print(f"[BALL GEOM DEBUG] joint {name} not found")
                continue

            body_id = self.model.jnt_bodyid[j_id]
            print(f"[BALL GEOM DEBUG] {name}: joint {j_id}, body {body_id}")

            for g_id in range(self.model.ngeom):
                if self.model.geom_bodyid[g_id] == body_id:
                    old_fric = np.array(self.model.geom_friction[g_id], copy=True)
                    print(f"    geom {g_id}: old friction = {old_fric}")
                    self.model.geom_friction[g_id] = np.array([0.0, 0.0, 0.0])
                    print(
                        f"    geom {g_id}: new friction = "
                        f"{self.model.geom_friction[g_id]}"
                    )

    # def _kick_ball_x(self, vx=2.0):
    #     ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    #     if ball_x_id < 0:
    #         raise RuntimeError("ball_x joint 'ball_x' not found in model")

    #     dof_adr = self.model.jnt_dofadr[ball_x_id]
    #     print("[DEBUG] ball_x joint id:", ball_x_id, "dof index:", dof_adr)

    #     self.data.qvel[dof_adr] = vx
    #     print(
    #         "[DEBUG] set ball_x qvel to",
    #         vx,
    #         " -> self.data.qvel[dof_adr] =",
    #         self.data.qvel[dof_adr],
    #     )
    def _kick_ball_x(self, vx=2.0):
        ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        if ball_x_id < 0:
            raise RuntimeError("ball_x joint 'ball_x' not found in model")

        dof_adr = self.model.jnt_dofadr[ball_x_id]
        print("[DEBUG] ball_x joint id:", ball_x_id, "dof index:", dof_adr)

        self.data.qvel[dof_adr] = vx
        print(
            "[DEBUG] set ball_x qvel to",
            vx,
            " -> self.data.qvel[dof_adr] =",
            self.data.qvel[dof_adr],
        )


    def _relax_ball_joint_stiffness(self):
        """Remove spring stiffness from ball_x / ball_y joints so the ball can coast."""
        for name in ["ball_x", "ball_y"]:
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id < 0:
                print(f"[BALL FIX] joint {name} not found for stiffness fix")
                continue

            old_stiff = float(self.model.jnt_stiffness[j_id])
            print(f"[BALL FIX] {name} jnt_stiffness before = {old_stiff}")
            self.model.jnt_stiffness[j_id] = 0.0
            print(f"[BALL FIX] {name} jnt_stiffness after  = {self.model.jnt_stiffness[j_id]}")


    # -------------------------------------------------------------------------
    # OBSERVATIONS
    # -------------------------------------------------------------------------
    def _get_ball_obs(self):
        # For the free joint:
        # qpos: [x, y, z, qw, qx, qy, qz]
        # qvel: [vx, vy, vz, wx, wy, wz]
        base_qpos = self.ball_qpos_adr
        base_qvel = self.ball_qvel_adr

        x = self.data.qpos[base_qpos + 0]
        y = self.data.qpos[base_qpos + 1]
        # z = self.data.qpos[base_qpos + 2]  # if you ever want it

        vx = self.data.qvel[base_qvel + 0]
        vy = self.data.qvel[base_qvel + 1]
        # vz = self.data.qvel[base_qvel + 2]

        ball_pos = [x, y]
        ball_vel = [vx, vy]
        return ball_pos, ball_vel

    # def _get_ball_obs(self):
    #     ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
    #     ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")
    #     # ball_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_z")

    #     x_qpos_adr = self.model.jnt_qposadr[ball_x_id]
    #     y_qpos_adr = self.model.jnt_qposadr[ball_y_id]
    #     # z_qpos_adr = self.model.jnt_qposadr[ball_z_id]

    #     x_qvel_adr = self.model.jnt_dofadr[ball_x_id]
    #     y_qvel_adr = self.model.jnt_dofadr[ball_y_id]
    #     # z_qvel_adr = self.model.jnt_dofadr[ball_z_id]

    #     ball_pos = [
    #         self.data.qpos[x_qpos_adr],
    #         self.data.qpos[y_qpos_adr],
    #         # self.data.qpos[z_qpos_adr],
    #     ]
    #     ball_vel = [
    #         self.data.qvel[x_qvel_adr],
    #         self.data.qvel[y_qvel_adr],
    #         # self.data.qvel[z_qvel_adr],
    #     ]

    #     return ball_pos, ball_vel

    def _get_antagonist_obs(self):
        # TODO: fill in if you want antagonist observations
        return self._get_obs().copy()

    def _get_obs(self):
        ball_pos, ball_vel = self._get_ball_obs()

        rod_slide_positions = []
        rod_slide_velocities = []
        rod_rotate_positions = []
        rod_rotate_velocities = []

        for player in ["y", "b"]:
            for rod in RODS:
                # Linear joints
                slide_joint_name = f"{player}{rod}linear"
                slide_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, slide_joint_name
                )
                slide_qpos_adr = self.model.jnt_qposadr[slide_joint_id]
                slide_qvel_adr = self.model.jnt_dofadr[slide_joint_id]
                rod_slide_positions.append(self.data.qpos[slide_qpos_adr])
                rod_slide_velocities.append(self.data.qvel[slide_qvel_adr])

                # Rotational joints
                rotate_joint_name = f"{player}{rod}rotation"
                rotate_joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, rotate_joint_name
                )
                rotate_qpos_adr = self.model.jnt_qposadr[rotate_joint_id]
                rotate_qvel_adr = self.model.jnt_dofadr[rotate_joint_id]
                rod_rotate_positions.append(self.data.qpos[rotate_qpos_adr])
                rod_rotate_velocities.append(self.data.qvel[rotate_qvel_adr])

        obs = np.concatenate(
            [
                ball_pos,
                ball_vel,
                rod_slide_positions,
                rod_slide_velocities,
                rod_rotate_positions,
                rod_rotate_velocities,
            ]
        ).astype(F32, copy=False)

        assert obs.shape == self.observation_space.shape, (
            f"Observation shape {obs.shape} does not match "
            f"observation space shape {self.observation_space.shape}"
        )

        return obs

    # -------------------------------------------------------------------------
    # ACTION ADJUSTMENTS / REWARD / TERMINATION
    # -------------------------------------------------------------------------

    def _adjust_antagonist_action(self, antagonist_action):
        # simple mirror
        return -antagonist_action.copy()

    def euclidean_goal_distance(self, x, y):
        # target point (0, TABLE_MAX_Y_DIM)
        return math.sqrt((x - 0.0) ** 2 + (y - TABLE_MAX_Y_DIM) ** 2)

    # def compute_reward(self, protagonist_action):
    #     ball_obs = self._get_ball_obs()
    #     ball_x = ball_obs[0][0]
    #     ball_y = ball_obs[0][1]

    #     inverse_distance_to_goal = 300 - self.euclidean_goal_distance(ball_x, ball_y)
    #     if ball_y > TABLE_MAX_Y_DIM:
    #         inverse_distance_to_goal = 0.0

    #     ctrl_cost = self.control_cost(protagonist_action)  # currently unused but kept

    #     victory = 1000 * DIRECTION_CHANGE if ball_y > TABLE_MAX_Y_DIM else 0
    #     loss = -1000 * DIRECTION_CHANGE if ball_y < -1.0 * TABLE_MAX_Y_DIM else 0

    #     reward = loss + victory + inverse_distance_to_goal + ctrl_cost

    #     return reward
    def _compute_step_reward(self, protagonist_action):
        ball_pos, ball_vel = self._get_ball_obs()
        ball_x, ball_y = ball_pos

        # Forward progress since last step
        forward_sign = self._direction_sign_for_protagonist  # +1 or -1
        delta_y = forward_sign * (ball_y - self._last_ball_y)
        delta_y = max(delta_y, 0.0)  # only reward forward

        # Strongly reward actual forward motion
        progress_reward = 50.0 * delta_y

        # Much smaller distance-based shaping toward a *virtual* closer goal
        virtual_goal_y = forward_sign * (REWARD_GOAL_Y_FRACTION * TABLE_MAX_Y_DIM)
        dist = abs(virtual_goal_y - ball_y)
        distance_reward = 5.0 / (1.0 + dist)  # ~0–5

        # Penalize large actions a bit
        control_cost = 0.001 * float(np.sum(np.square(protagonist_action)))

        # Goal bonuses/penalties still use the real physical goal (TABLE_MAX_Y_DIM)
        winning_goal, losing_goal = self._check_goal_scored(ball_pos)
        victory_reward = 1000.0 if winning_goal else 0.0
        own_goal_penalty = -1000.0 if losing_goal else 0.0

        reward = (
            progress_reward
            + distance_reward
            + victory_reward
            + own_goal_penalty
            - control_cost
        )
        return reward





    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    def control_cost(self, action):
        # L1 norm penalty
        control_cost = self._ctrl_cost_weight * np.sum(np.abs(action)) * -1.0
        return control_cost

    @property
    def is_healthy(self):
        # ball_z = self._get_ball_obs()[0][2]
        # min_z, max_z = self._healthy_z_range
        # return min_z < ball_z < max_z
        return True


    def _is_ball_moving(self):
        _, ball_vel = self._get_ball_obs()
        return np.linalg.norm(ball_vel) > 0.01

    def _determine_progression(self):
        # ball_y = self._get_ball_obs()[0][1]

        # if self.prev_ball_y is not None:
        #     if ball_y > self.prev_ball_y:
        #         self.no_progress_steps = 0
        #     else:
        #         self.no_progress_steps += 1

        # self.prev_ball_y = ball_y
        """
        Track whether the ball has been essentially still for many steps.
        This will drive the 'ball_stagnant' termination.
        """
        if self._is_ball_moving():
            self.ball_stopped_count = 0
        else:
            self.ball_stopped_count += 1


    # @property
    # def terminated(self):
    #     self._determine_progression()

    #     # ball_stagnant turned off for now
    #     ball_stagnant = False
    #     self.ball_stopped_count = 0

    #     over_max_steps = self.simulation_time >= MAX_STEPS
    #     unhealthy = not self.is_healthy
    #     no_progress = self.no_progress_steps >= self.max_no_progress_steps

    #     ball_pos, _ = self._get_ball_obs()
    #     ball_x, ball_y, _ = ball_pos

    #     victory = ball_y < -TABLE_MAX_Y_DIM or ball_y > TABLE_MAX_Y_DIM

    #     if victory:
    #         print("Victory")
    #         print(f"Ball x: {ball_x}, Ball y: {ball_y}")

    #     terminated = (
    #         unhealthy
    #         or (no_progress and not self.play_until_goal)
    #         or over_max_steps
    #     ) if self._terminate_when_unhealthy else False

    #     if self.verbose_mode and terminated:
    #         print("Terminated")
    #         print(
    #             f"Unhealthy: {unhealthy}, No progress: {no_progress}, "
    #             f"Victory: {victory}, Ball stagnant: {ball_stagnant}"
    #         )
    #         print("x: ", ball_x, "y: ", ball_y)

    #     return terminated


    @property
    def terminated(self):
        # Update ball_stopped_count
        self._determine_progression()

        # Check basic conditions
        ball_pos, _ = self._get_ball_obs()
        ball_x, ball_y = ball_pos  # <-- only 2D now

        unhealthy = not self.is_healthy

        # Goal condition: ball crosses either end of the table in y
        goal_scored = (ball_y < -TABLE_MAX_Y_DIM) or (ball_y > TABLE_MAX_Y_DIM)

        # Ball stuck: has barely moved for many steps
        ball_stagnant = self.ball_stopped_count >= BALL_STOPPED_COUNT_THRESHOLD

        # Max episode duration in *seconds* of simulated time
        max_episode_seconds = 10.0  # tune as you like
        over_max_time = self.simulation_time >= max_episode_seconds

        terminated = (
            unhealthy
            or goal_scored
            or ball_stagnant
            or over_max_time
        ) if self._terminate_when_unhealthy else False

        if self.verbose_mode and terminated:
            print("Terminated")
            print(
                f"Unhealthy: {unhealthy}, Goal scored: {goal_scored}, "
                f"Ball stagnant: {ball_stagnant}, Over max time: {over_max_time}"
            )
            print("x: ", ball_x, "y: ", ball_y)

        return terminated



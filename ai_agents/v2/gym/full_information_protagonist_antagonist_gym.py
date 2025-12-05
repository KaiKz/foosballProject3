import math
import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import mujoco.viewer as mj_viewer

from ai_agents.v2.gym.mujoco_table_render_mixin import MujocoTableRenderMixin

DIRECTION_CHANGE = 1

# Ball “goal line” is inside the walls (walls are at ~±63)
GOAL_LINE_Y = 60.0              # center of goal line along +y / -y
TABLE_MAX_Y_DIM = GOAL_LINE_Y   # keep this for legacy uses

BALL_STOPPED_COUNT_THRESHOLD = 200

# Reward shaping: pretend the goal is closer than the physical one
REWARD_GOAL_Y_FRACTION = 0.5    # half-way to the real goal, tweak as you like

# Big terminal rewards
GOAL_REWARD = 1000.0
OWN_GOAL_PENALTY = -1000.0

MAX_STEPS = 40  # if you still use this elsewhere

# Goal line position (along y) and width (along x)
GOAL_LINE_Y = 60.0          # where the goal line is in y
GOAL_HALF_WIDTH = 10.0      # posts at x ∈ [-10, +10]  (tune this)


# Consider the ball "stuck" if it's moving slower than this or barely changing position
STAGNANT_VEL_EPS = 0.15      # was 5e-3; much looser
STAGNANT_POS_EPS = 0.002     # a bit stricter on positional motion
STAGNANT_STEPS   = 40        # how many consecutive steps before we call it stagnant

# Calculate project root and build relative path to simulation XML
_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_default_path = os.path.join(_dir_path, "foosball_sim", "v2", "foosball_sim.xml")
SIM_PATH = os.environ.get("SIM_PATH", _default_path)

F32 = np.float32
INF32 = np.finfo(np.float32).max

RODS = ["_goal_", "_def_", "_mid_", "_attack_"]


class FoosballEnv(MujocoTableRenderMixin, gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, antagonist_model=None, play_until_goal=False,
                 verbose_mode=False, debug_free_ball=False, render_mode="human"):
        super(FoosballEnv, self).__init__()
        self.render_mode = render_mode
        self.viewer = None
        self._offscreen = None 

        xml_file = SIM_PATH
        print("[FoosballEnv] Loading XML from:", xml_file)

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        # --- Cache ball + player geoms for reset logic ---
        self.ball_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_phys"
        )

        PLAYER_GEOM_PREFIXES = ["y_attack_guy", "y_mid_guy", "y_def_guy",
                                "b_attack_guy", "b_mid_guy", "b_def_guy"]

        self.player_geom_ids = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if not name:
                continue
            if any(name.startswith(p) for p in PLAYER_GEOM_PREFIXES):
                self.player_geom_ids.append(gid)
        self.attack_slide_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "y_attack_linear"  # <-- use correct name
        )
        self.attack_slide_qpos_adr = self.model.jnt_qposadr[self.attack_slide_joint]
        print("\n=== ACTUATOR ↔ CTRL MAPPING ===")
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"ctrl[{i}] -> actuator '{name}'")
        # --- BALL INDEXES (PLANAR: ball_x, ball_y) ---
        self.ball_x_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x"
        )
        self.ball_y_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y"
        )

        if self.ball_x_joint < 0 or self.ball_y_joint < 0:
            raise RuntimeError("Joints 'ball_x'/'ball_y' not found in the model")

        self.ball_x_qpos_adr = self.model.jnt_qposadr[self.ball_x_joint]
        self.ball_y_qpos_adr = self.model.jnt_qposadr[self.ball_y_joint]

        self.ball_x_qvel_adr = self.model.jnt_dofadr[self.ball_x_joint]
        self.ball_y_qvel_adr = self.model.jnt_dofadr[self.ball_y_joint]
            
        # ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        # ball_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        # ball_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_y")

        # print("[BODY DEBUG] ball geom body   =", self.model.geom_bodyid[ball_geom_id])
        # print("[BODY DEBUG] ball_x joint body=", self.model.jnt_bodyid[ball_x_id])
        # print("[BODY DEBUG] ball_y joint body=", self.model.jnt_bodyid[ball_y_id])
        # print("[OPT DEBUG] disableflags before =", self.model.opt.disableflags)
        # self.model.opt.disableflags = 0

        # print("[OPT DEBUG] disableflags after  =", self.model.opt.disableflags)

        # # DEBUG: force-enable contact for all geoms
        # for g in range(self.model.ngeom):
        #     self.model.geom_contype[g] = 1
        #     self.model.geom_conaffinity[g] = 1
            
        # print("=== GEOM CONTACT DEBUG ===")
        # for g in range(self.model.ngeom):
        #     name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
        #     ct = self.model.geom_contype[g]
        #     ca = self.model.geom_conaffinity[g]
        #     print(f"geom {g:2d} name={name} contype={ct} conaffinity={ca}")
            

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
            low=-1.0,
            high=1.0,
            shape=(self.protagonist_action_size,),
            dtype=F32,
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

        # -------------------- RENDERING -------------------- #
    def render(self):
        """
        Explicit MuJoCo viewer integration.
        - 'human': opens an interactive window using mujoco.viewer
        - 'rgb_array': returns an image as a numpy array
        """
        if self.render_mode == "human":
            if self.viewer is None:
                # This opens the interactive GLFW window
                self.viewer = mj_viewer.launch_passive(self.model, self.data)

                # ------------- set top-down camera once -------------
                cam = self.viewer.cam

                # Try to look at the table body if it exists, otherwise (0,0,0)
                try:
                    table_body_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, "table"
                    )
                    center = self.data.body_xpos[table_body_id].copy()
                except Exception:
                    center = np.array([0.0, 0.0, 0.0], dtype=float)

                cam.lookat[:] = center  # where the camera focuses
                cam.distance = 1.2      # zoom level (tweak)
                cam.azimuth = 0.0       # rotation around z (doesn't matter much top-down)
                cam.elevation = -90.0   # <- straight down

            else:
                # Just redraw the current scene
                self.viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            # Offscreen renderer for SB3-style usage
            if self._offscreen is None:
                self._offscreen = mujoco.Renderer(self.model, 800, 600)
            self._offscreen.update_scene(self.data)
            img = self._offscreen.render()
            return img

        else:
            # No-op if render_mode is None or unknown
            return None

    def close(self):
        # Clean up viewer and renderer
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self._offscreen is not None:
            self._offscreen.close()
            self._offscreen = None
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
        # self.data.qpos[self.ball_x_dof] = 0.0
        # self.data.qpos[self.ball_y_dof] = 0.0

        # self.data.qvel[self.ball_x_dof] = 0.0
        # self.data.qvel[self.ball_y_dof] = 0.0

        self.data.qpos[self.ball_x_qpos_adr] = 0.0
        self.data.qpos[self.ball_y_qpos_adr] = 0.0

        self.data.qvel[self.ball_x_qvel_adr] = 0.0
        self.data.qvel[self.ball_y_qvel_adr] = 0.0



        mujoco.mj_forward(self.model, self.data)

    def _debug_ball_contacts(self):
        """
        Print all contacts involving the ball's collision geom 'ball_phys'.
        """
        import mujoco

        ball_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_phys"
        )
        if ball_geom_id < 0:
            print("[CONTACT DEBUG] ball_phys geom not found")
            return

        if self.data.ncon == 0:
            print(f"[CONTACT DEBUG] step={self._debug_step_counter} no contacts")
            return

        print(f"[CONTACT DEBUG] step={self._debug_step_counter}, ncon={self.data.ncon}")
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            # Only show contacts involving the ball geom
            if g1 != ball_geom_id and g2 != ball_geom_id:
                continue

            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)

            print(
                f"  con#{i}: {name1}({g1}) <-> {name2}({g2}) | "
                f"dist={c.dist:.6f}, normal={np.array(c.frame[:3])}"
            )



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

        mujoco.mj_forward(self.model, self.data)
        DEBUG_FORCE_OVERLAP = False


        # --- Find attack guy pose ---
        attack_guy_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "y_attack_guy2"
        )
        if attack_guy_geom_id < 0:
            # Fallback: old "table center" behavior
            attack_guy_x = 0.0
            attack_guy_y = 0.0
            attack_guy_z = 0.08
        else:
            guy_xyz = self.data.geom_xpos[attack_guy_geom_id].copy()
            attack_guy_x = float(guy_xyz[0])
            attack_guy_y = float(guy_xyz[1])
            attack_guy_z = float(guy_xyz[2])



        # --- Compute safe offset: ball_radius + guy_radius + margin ---
        ball_radius = float(self.model.geom_size[self.ball_geom_id][0])
        guy_radius = float(self.model.geom_rbound[attack_guy_geom_id])

        # guy_type = self.model.geom_type[attack_guy_geom_id]
        # guy_size = self.model.geom_size[attack_guy_geom_id]

        # if guy_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        #     guy_radius = float(guy_size[0])
        # elif guy_type in (mujoco.mjtGeom.mjGEOM_CAPSULE,
        #                 mujoco.mjtGeom.mjGEOM_CYLINDER):
        #     guy_radius = float(guy_size[0])
        # elif guy_type == mujoco.mjtGeom.mjGEOM_BOX:
        #     guy_radius = float(guy_size[1])  # half-length in y
        # else:
        #     guy_radius = 0.05

        margin = 0  # start small
        safe_offset = ball_radius + guy_radius + margin

        bp_x = self.ball_x_qpos_adr
        bp_y = self.ball_y_qpos_adr
        bv_x = self.ball_x_qvel_adr
        bv_y = self.ball_y_qvel_adr

        # Place ball in front of the attack guy (x, y only)
        self.data.qpos[bp_x] = attack_guy_x 
        self.data.qpos[bp_y] = attack_guy_y +safe_offset

        # Zero planar velocity
        self.data.qvel[bv_x] = 0.0
        self.data.qvel[bv_y] = 0.0



        mujoco.mj_forward(self.model, self.data)
        if DEBUG_FORCE_OVERLAP:
            # 1) Force ball *center* to exactly attack guy's geom position
            self.data.qpos[bp + 0] = attack_guy_x
            self.data.qpos[bp + 1] = attack_guy_y
            self.data.qpos[bp + 2] = attack_guy_z

            # keep orientation / velocities as you already set them
            mujoco.mj_forward(self.model, self.data)

            # 2) Print contacts involving the ball
            print("=== DEBUG: AFTER FORCED OVERLAP ===")
            print("ball geom_xpos:", self.data.geom_xpos[self.ball_geom_id])
            print("guy  geom_xpos:", self.data.geom_xpos[attack_guy_geom_id])
            print("ncon:", self.data.ncon)

            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = c.geom1, c.geom2
                if g1 != self.ball_geom_id and g2 != self.ball_geom_id:
                    continue
                g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
                print(
                    f"  contact {i}: {g1_name}({g1}) <-> {g2_name}({g2}), dist={c.dist}"
                )

            # 3) Early-return so the usual contact-resolution + bookkeeping doesn't run
            #    (we only care about contacts for this test)
            return self._get_obs(), {}

        # --- Contact resolution loop: remove ball–player overlap ---
        def ball_in_contact_with_player(model, data):
            for i in range(data.ncon):
                c = data.contact[i]
                g1, g2 = c.geom1, c.geom2
                if ((g1 == self.ball_geom_id and g2 in self.player_geom_ids) or
                    (g2 == self.ball_geom_id and g1 in self.player_geom_ids)):
                    return True, c
            return False, None

        max_tries = 50
        step = 0.01
        tries = 0

        while True:
            touching, c = ball_in_contact_with_player(self.model, self.data)
            if not touching:
                break

            # c.dist < 0 -> penetration depth
            # c.frame[:3] is contact normal in world coords.
            # Move ball along normal to fix penetration.
            penetration = -float(c.dist)  # positive
            eps = 1e-3
            shift = penetration + eps

            n = np.array(c.frame[:3], dtype=float)  # contact normal
            self.data.qpos[bp + 0] += shift * n[0]
            self.data.qpos[bp + 1] += shift * n[1]
            self.data.qpos[bp + 2] += shift * n[2]

            mujoco.mj_forward(self.model, self.data)

            tries += 1
            if tries >= max_tries:
                if self.verbose_mode:
                    print("[WARN] couldn't fully resolve ball-player contact")
                break

            
        if self.verbose_mode:
            print("[RESET DEBUG] after reposition + player-resolution:")
            self._debug_ball_pos()

            # print all ball contacts with ANY geom
            if self.data.ncon == 0:
                print("[RESET DEBUG] no contacts at reset")
            else:
                print(f"[RESET DEBUG] ncon={self.data.ncon}")
                for i in range(self.data.ncon):
                    c = self.data.contact[i]
                    g1, g2 = c.geom1, c.geom2
                    if g1 != self.ball_geom_id and g2 != self.ball_geom_id:
                        continue
                    g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                    g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
                    print(
                        f"   ball contact: {g1_name}({g1}) <-> {g2_name}({g2}) "
                        f"dist={c.dist:.6f}"
                    )

        # --- Final book-keeping as before ---
        self.simulation_time = 0.0
        ball_pos, _ = self._get_ball_obs()
        self.prev_ball_y = ball_pos[1]
        self.no_progress_steps = 0
        self.ball_stopped_count = 0
        self._debug_step_counter = 0

        self._direction_sign_for_protagonist = 1.0
        self._last_ball_y = self.prev_ball_y

        return self._get_obs(), {}



    def _hacky_kick(self):
        """
        Apply a manual impulse to the ball when the attack rod's site comes
        close in the horizontal plane. This is just for debugging.
        """
        import mujoco
        ball_xy = self._get_ball_xy()

        rod_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attack_rod_site")
        if rod_site_id < 0:
            # print("[HACK KICK] attack_rod_site not found")
            return

        rod_xy = self.data.site_xpos[rod_site_id][:2].copy()
        dist_xy = np.linalg.norm(rod_xy - ball_xy)

        # Use something on the order of the ball radius / player foot size.
        # Given your scale, 0.03–0.05 is a reasonable start.
        if dist_xy < 0.05:
            self.data.qvel[self.ball_y_qvel_adr] += 3.0 * self._direction_sign_for_protagonist

            print(f"[HACK KICK] impulse vy applied, dist_xy={dist_xy:.4f}")
        # else:
        #     print(f"[HACK KICK] rod too far: dist_xy={dist_xy:.4f}")


    def _apply_actions(self, action, side="protagonist"):
        """
        Map action in [-1, 1] to actual MuJoCo controls
        for the 4 rods (goal, def, mid, attack), each with
        [linear, rotation] -> 8 controls total.

        side = "protagonist" -> yellow (y_*)
        side = "antagonist"  -> black  (b_*)
        """

        # Ensure we have a NumPy array in [-1, 1]
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        goal_linear_range   = 10.0
        def_linear_range    = 20.0
        mid_linear_range    = 7.0
        attack_linear_range = 12.0
        rot_range           = 2.5

        scaled = np.empty(self.protagonist_action_size, dtype=np.float32)

        # 0: goal linear
        scaled[0] = a[0] * goal_linear_range
        # 1: goal rotation
        scaled[1] = a[1] * rot_range

        # 2: def linear
        scaled[2] = a[2] * def_linear_range
        # 3: def rotation
        scaled[3] = a[3] * rot_range

        # 4: mid linear
        scaled[4] = a[4] * mid_linear_range
        # 5: mid rotation
        scaled[5] = a[5] * rot_range

        # 6: attack linear
        scaled[6] = a[6] * attack_linear_range
        # 7: attack rotation
        scaled[7] = a[7] * rot_range

        # Decide where in ctrl[] to write:
        # actuators are ordered as:
        #   0–7  : y_* (yellow)
        #   8–15 : b_* (black)
        if side == "protagonist":
            start = 0
        elif side == "antagonist":
            start = self.protagonist_action_size  # = 8
        else:
            raise ValueError(f"Unknown side={side}")

        self.data.ctrl[start : start + self.protagonist_action_size] = scaled


    def step(self, protagonist_action):
        protagonist_action = np.clip(
            protagonist_action, self.action_space.low, self.action_space.high
        )

        # --------- PROTAGONIST ACTIONS (YELLOW) ---------
        self._apply_actions(protagonist_action, side="protagonist")

        # --------- ANTAGONIST ACTIONS (BLACK) ----------
        # Build observation for antagonist (can be the same as protagonist for now)
        antagonist_observation = self._get_antagonist_obs()

        if self.antagonist_model is not None:
            # SB3-style predict
            antagonist_raw_action, _ = self.antagonist_model.predict(
                antagonist_observation, deterministic=True
            )
            antagonist_raw_action = np.clip(
                antagonist_raw_action, -1.0, 1.0
            ).astype(np.float32)

            # Optional mirror (you already have this helper)
            antagonist_action = self._adjust_antagonist_action(antagonist_raw_action)
        else:
            # If no opponent policy provided, keep black rods still
            antagonist_action = np.zeros(self.antagonist_action_size, dtype=np.float32)

        # Apply to black rods
        self._apply_actions(antagonist_action, side="antagonist")

        # --------- PHYSICS STEP AS BEFORE ----------
        attack_guy_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "y_attack_guy2"
        )
        ball_pos_before = np.array(self._get_ball_obs()[0][:2], dtype=float)

        mujoco.mj_step(self.model, self.data)
        
        # print("[NCON DEBUG]", self.data.ncon)
        # if self._debug_step_counter < 10:
            # q_attack = float(self.data.qpos[self.attack_slide_qpos_adr])
            # print(f"[ATTACK DEBUG] step={self._debug_step_counter}, attack_slide_q={q_attack}")
        # if self.verbose_mode:
        #     ball_pos_world = self.data.geom_xpos[self.ball_geom_id].copy()
        #     guy_pos_world  = self.data.geom_xpos[attack_guy_geom_id].copy()
        #     dist = np.linalg.norm(ball_pos_world - guy_pos_world)
        #     print(f"[RESET GEO DEBUG] ball={ball_pos_world}, attack_guy={guy_pos_world}, dist={dist}")



        # if self._debug_step_counter == 1:
        #     _, vel0 = self._get_ball_obs()
        #     print("[STEP DEBUG] BEFORE first mj_step, ball_vel =", vel0)

        # antagonist_observation = self._get_antagonist_obs()

        # if self.antagonist_model is not None:
        #     antagonist_action, _ = self.antagonist_model.predict(antagonist_observation)
        #     antagonist_action = np.clip(antagonist_action, -1.0, 1.0)
        #     antagonist_action = self._adjust_antagonist_action(antagonist_action)
        # else:
        #     antagonist_action = np.zeros(self.antagonist_action_size, dtype=F32)

        # # Apply controls
        # self.data.ctrl[: self.protagonist_action_size] = protagonist_action
        # self.data.ctrl[
        #     self.protagonist_action_size : self.protagonist_action_size
        #     + self.antagonist_action_size
        # ] = antagonist_action

        if self.verbose_mode and self._debug_step_counter < 50:
            self._debug_ball_forces()
            self._debug_ball_pos()  
            # self._debug_ball_forces_2d()
            self._debug_ball_contacts()

        if self.verbose_mode and self._debug_step_counter == 0:
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

        # if self._debug_step_counter < 20:
            # print(
            #     f"[DEBUG PHYSICS] step={self._debug_step_counter} "
            #     f"ball_before={ball_pos_before} ball_after={ball_pos_after} Δ={delta}"
            # )

        try:
            ball_pos, _ = self._get_ball_obs()
            ball_x, ball_y = ball_pos
        except Exception as e:
            # print(f"[FoosballEnv DEBUG] _get_ball_obs() failed: {e}")
            ball_x = ball_y = float("nan")

        info = {
            "ball_x": float(ball_x),
            "ball_y": float(ball_y),
            "reward": float(reward),
        }

        self._last_ball_y = ball_y  # update after computing reward
        self._debug_step_counter += 1
        # mujoco.mj_step(self.model, self.data)
        # if self._debug_step_counter <= 20:
        #     print(
        #         f"[FoosballEnv DEBUG] step={self._debug_step_counter} "
        #         f"ball_x={ball_x:.3f} ball_y={ball_y:.3f} reward={reward:.3f}"
        #     )

        return obs, reward, bool(terminated), False, info
    
    def _check_goal_scored(self, ball_pos):
        """
        Decide whether the protagonist scored or conceded,
        taking into account:
          - direction sign (which side they're attacking)
          - a fixed goal line in y
          - a finite goal width in x (posts)
        """
        ball_x, ball_y = ball_pos
        forward_sign = self._direction_sign_for_protagonist  # +1 or -1

        # Must be between the posts to count as a goal
        in_goal_x = abs(ball_x) <= GOAL_HALF_WIDTH

        # Protagonist scores at +GOAL_LINE_Y if forward_sign=+1,
        # and at -GOAL_LINE_Y if forward_sign=-1.
        winning_goal = (
            in_goal_x and (
                (forward_sign > 0 and ball_y >= GOAL_LINE_Y) or
                (forward_sign < 0 and ball_y <= -GOAL_LINE_Y)
            )
        )

        # Own-goal is the opposite side, also between posts
        losing_goal = (
            in_goal_x and (
                (forward_sign > 0 and ball_y <= -GOAL_LINE_Y) or
                (forward_sign < 0 and ball_y >= GOAL_LINE_Y)
            )
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
    
    
    # def _debug_ball_contacts(self):
    #     import mujoco

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

    #         print(
    #             f"[CONTACT DEBUG2] con#{i} "
    #             f"{g1_name}({g1}) vs {g2_name}({g2}) | "
    #             f"dist={dist:.6f}, normal={n}"
    #         )




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
            # print("[FORCE DEBUG] ball_x joint not found")
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
        x  = self.data.qpos[self.ball_x_qpos_adr]
        y  = self.data.qpos[self.ball_y_qpos_adr]
        vx = self.data.qvel[self.ball_x_qvel_adr]
        vy = self.data.qvel[self.ball_y_qvel_adr]

        ball_pos = [x, y]
        ball_vel = [vx, vy]
        return ball_pos, ball_vel


    def _get_ball_xy(self):
        """
        Convenience helper: return ball (x, y) as a NumPy array.
        """
        ball_pos, _ = self._get_ball_obs()
        return np.array(ball_pos, dtype=float)  # [x, y]
    def _debug_ball_pos(self):
        """
        Debug printout of ball position and velocity.
        """
        ball_pos, ball_vel = self._get_ball_obs()
        print(
            f"[BALL POS DEBUG] step={self._debug_step_counter} "
            f"pos=({ball_pos[0]:.4f}, {ball_pos[1]:.4f}), "
            f"vel=({ball_vel[0]:.4f}, {ball_vel[1]:.4f})"
        )


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

    def _compute_step_reward(self, protagonist_action):
        ball_pos, ball_vel = self._get_ball_obs()
        ball_x, ball_y = ball_pos

        forward_sign = self._direction_sign_for_protagonist  # +1 or -1

        # 1) Progress along scoring direction
        actual_delta_y = forward_sign * (ball_y - self._last_ball_y)
        extra_delta_y = max(actual_delta_y, 0.0)
        progress_reward = 50.0 * extra_delta_y  # tweak scaling as needed

        # 2) Distance-based shaping toward a *virtual* closer goal
        virtual_goal_y = forward_sign * (REWARD_GOAL_Y_FRACTION * GOAL_LINE_Y)
        dist = abs(virtual_goal_y - ball_y)
        distance_reward = 5.0 / (1.0 + dist)  # ~0–5, smoother near goal

        # 3) Penalize large actions slightly
        control_cost = 0.001 * float(np.sum(np.square(protagonist_action)))

        # 4) True goal or own-goal?
        winning_goal, losing_goal = self._check_goal_scored(ball_pos)
        victory_reward = GOAL_REWARD if winning_goal else 0.0
        own_goal_penalty = OWN_GOAL_PENALTY if losing_goal else 0.0

        # 5) Small reward for keeping ball moving
        speed = np.linalg.norm(ball_vel)
        speed_reward = 0.1 * speed

        reward = (
            progress_reward
            + distance_reward
            + victory_reward
            + own_goal_penalty
            + speed_reward
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
        """
        Track whether the ball has effectively stopped / is stuck.
        We look at BOTH velocity magnitude and position changes,
        and treat it as 'stagnant' if EITHER suggests it's basically not moving.
        """
        (ball_x, ball_y), ball_vel = self._get_ball_obs()
        speed = float(np.linalg.norm(ball_vel))

        if self.prev_ball_y is None:
            self.prev_ball_y = ball_y

        pos_delta = abs(ball_y - self.prev_ball_y)

        vel_still = speed < STAGNANT_VEL_EPS
        pos_still = pos_delta < STAGNANT_POS_EPS

        # 👈 key change: OR instead of AND
        if vel_still or pos_still:
            self.ball_stopped_count += 1
        else:
            self.ball_stopped_count = 0

        if self.verbose_mode and self._debug_step_counter % 50 == 0:
            print(
                f"[STAGNATION DEBUG] step={self._debug_step_counter} "
                f"speed={speed:.4f}, pos_delta={pos_delta:.4f}, "
                f"vel_still={vel_still}, pos_still={pos_still}, "
                f"ball_stopped_count={self.ball_stopped_count}"
            )

        self.prev_ball_y = ball_y




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
        ball_x, ball_y = ball_pos  # only 2D now

        unhealthy = not self.is_healthy

        # Goal condition using _check_goal_scored
        winning_goal, losing_goal = self._check_goal_scored(ball_pos)
        goal_scored = winning_goal or losing_goal

        # Ball stuck: has barely moved for many steps
        ball_stagnant = self.ball_stopped_count >= BALL_STOPPED_COUNT_THRESHOLD

        # Max episode duration in *seconds* of simulated time
        max_episode_seconds = 10.0  # tune as you like
        over_max_time = self.simulation_time >= max_episode_seconds

        if self.play_until_goal:
            # Only end on goal or unhealthy
            terminated = unhealthy or goal_scored
        else:
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




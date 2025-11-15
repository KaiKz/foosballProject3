import os
import time
import numpy as np
import mujoco

# glfw is optional at runtime: we only use it if FOOSBALL_RENDER=1
try:
    import glfw
except Exception:  # keep import failures from killing headless training
    glfw = None


class MujocoTableRenderMixin:
    """
    Headless-safe renderer mixin.

    Default: FOOSBALL_RENDER!=1  -> no window created; render() is a no-op.
    Opt-in:  FOOSBALL_RENDER=1   -> create a GLFW window (requires DISPLAY or Xvfb).
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # Core state
        self.viewer = None
        self.window = None
        self.first_render = True
        self.glfw_initialized = False

        # Headless by default unless explicitly enabled
        self.headless = os.environ.get("FOOSBALL_RENDER", "0") != "1"

        # If user asked for rendering, try to initialize GLFW
        if not self.headless:
            self._initialize_glfw()

    def _initialize_glfw(self):
        if glfw is None:
            raise RuntimeError("glfw not available but FOOSBALL_RENDER=1 was set")

        # On headless servers you need DISPLAY (or run via xvfb-run). Fail early with a clear message.
        if os.environ.get("DISPLAY") in (None, ""):
            raise RuntimeError(
                "FOOSBALL_RENDER=1 but DISPLAY is missing. "
                "Either run with `xvfb-run` or unset FOOSBALL_RENDER (use headless)."
            )

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        self.glfw_initialized = True

    def render(self, mode='human'):
        # In headless mode, do nothing (training stays on GPU; no X11/GLFW).
        if self.headless:
            return

        if not self.glfw_initialized:
            # Safety: if user toggled FOOSBALL_RENDER mid-run
            self._initialize_glfw()

        if self.first_render:
            # Create window
            self.window = glfw.create_window(1280, 720, "Foosball Simulation", None, None)
            if not self.window:
                if self.glfw_initialized:
                    glfw.terminate()
                    self.glfw_initialized = False
                raise RuntimeError("Could not create GLFW window")

            # Center window if possible
            monitor = glfw.get_primary_monitor()
            if monitor is not None:
                video_mode = glfw.get_video_mode(monitor)
                window_width, window_height = 1280, 720
                if video_mode is not None:
                    glfw.set_window_pos(
                        self.window,
                        (video_mode.size.width - window_width) // 2,
                        (video_mode.size.height - window_height) // 2
                    )

            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            # Mujoco visualization state
            self.cam = mujoco.MjvCamera()
            self.opt = mujoco.MjvOption()
            mujoco.mjv_defaultCamera(self.cam)
            mujoco.mjv_defaultOption(self.opt)

            # Camera tuning
            self.cam.azimuth = 180.0
            self.cam.elevation = -40.0
            self.cam.distance = 100.0
            self.cam.lookat[:] = np.array([0, 0, 1.0])

            # Scene & context
            self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
            self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

            self.first_render = False

        if self.window and not glfw.window_should_close(self.window):
            glfw.make_context_current(self.window)

            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scn
            )

            mujoco.mjr_render(viewport, self.scn, self.ctx)

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            time.sleep(0.02)
        else:
            self.close()

    def close(self):
        # Free Mujoco contexts if they exist
        if hasattr(self, 'ctx') and self.ctx is not None:
            self.ctx.free()
            self.ctx = None
        if hasattr(self, 'scn') and self.scn is not None:
            self.scn.free()
            self.scn = None

        # Destroy window / terminate glfw only if we initialized it
        if self.window is not None:
            try:
                glfw.destroy_window(self.window)
            except Exception:
                pass
            self.window = None

        if self.glfw_initialized:
            try:
                glfw.terminate()
            finally:
                self.glfw_initialized = False

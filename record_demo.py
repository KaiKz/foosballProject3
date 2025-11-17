import argparse
import numpy as np
import imageio
import mujoco

from stable_baselines3 import SAC
from sb3_contrib import TQC

from ai_agents.v2.gym.full_information_protagonist_antagonist_gym import FoosballEnv


def make_env(antagonist_model=None):
    print("[record_demo] Creating FoosballEnv...")
    env = FoosballEnv(antagonist_model=antagonist_model)
    print("[record_demo] Env created.")
    if hasattr(env, "xml_file"):
        print("[record_demo] Env xml_file:", getattr(env, "xml_file", None))
    return env


def load_model(algo: str, model_path: str, env):
    """
    Load a trained SAC or TQC model from disk.
    """
    algo = algo.lower()
    print(f"[record_demo] Loading model: algo={algo}, path={model_path}")
    if algo == "sac":
        model = SAC.load(model_path, env=env)
    elif algo == "tqc":
        model = TQC.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported algo: {algo} (use 'sac' or 'tqc')")
    print("[record_demo] Model loaded.")
    return model


def create_topdown_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """
    Simple top-down camera using only supported MjvCamera fields.
    """
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultFreeCamera(model, cam)

    # Start from model center, but force lookat height near table/ball
    cam.lookat[:] = model.stat.center
    cam.lookat[2] = 0.2  # around table top / ball height

    # Bring camera close enough so the ball isn't microscopic
    cam.distance = 6.0   # you can tweak: 5.0 closer, 7.0 farther

    # True top-down
    cam.elevation = -90.0
    cam.azimuth = 0.0

    print(
        "[CAMERA CONFIG] lookat=", cam.lookat,
        " distance=", cam.distance,
        " azimuth=", cam.azimuth,
        " elevation=", cam.elevation,
    )
    return cam


def run_and_record(
    algo: str,
    model_path: str,
    video_path: str,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    total_steps: int = 900,
    self_play: bool = False,
):
    print("[record_demo] Starting run_and_record...")

    # Load protagonist model first
    print("[record_demo] Loading protagonist model...")
    protagonist_model = load_model(algo, model_path, env=None)

    antagonist_model = None
    if self_play:
        print("[record_demo] Using same model as antagonist for self-play.")
        antagonist_model = protagonist_model

    # Create env (with or without antagonist)
    env = make_env(antagonist_model=antagonist_model)

    if env.model.ncam > 0:
        for cam_id in range(env.model.ncam):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            print(f"[CAMERA DEBUG] cam_id={cam_id} name={name}")
    else:
        print("[CAMERA DEBUG] No cameras defined in XML; using custom top-down camera.")

    # === BALL VISIBILITY SETTINGS ===
    ball_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    print("[DEBUG] ball geom id:", ball_geom_id)
    print("[DEBUG] ball size before:", env.model.geom_size[ball_geom_id])
    print("[DEBUG] ball rgba before:", env.model.geom_rgba[ball_geom_id])

    # Make ball clearly visible but not ridiculous
    env.model.geom_size[ball_geom_id][0] = 0.04  # radius (was 0.02)
    env.model.geom_rgba[ball_geom_id] = np.array(
        [1.0, 0.0, 1.0, 1.0], dtype=np.float32
    )  # bright magenta

    print("[DEBUG] ball size after:", env.model.geom_size[ball_geom_id])
    print("[DEBUG] ball rgba after:", env.model.geom_rgba[ball_geom_id])

    # Ensure ball z is slightly above table at reset
    def lift_ball_on_reset():
        if hasattr(env, "ball_qpos_adr"):
            base = env.ball_qpos_adr
            z_before = float(env.data.qpos[base + 2])
            env.data.qpos[base + 2] = 0.08  # your env.reset default
            mujoco.mj_forward(env.model, env.data)
            print(
                "[DEBUG] Lifted ball z at reset. "
                f"z_before={z_before:.4f}, z_after={float(env.data.qpos[base+2]):.4f}"
            )

    # Bind env to protagonist model
    protagonist_model.set_env(env)
    model = protagonist_model

    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise RuntimeError("FoosballEnv does not expose .model and .data for MuJoCo rendering")

    print(f"[record_demo] Creating mujoco.Renderer(width={width}, height={height})")
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    topdown_cam = create_topdown_camera(env.model)

    print(f"[record_demo] Opening video writer at {video_path} (fps={fps})")
    writer = imageio.get_writer(video_path, fps=fps)

    try:
        # Initial reset
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out

        lift_ball_on_reset()

        for step in range(1, total_steps + 1):
            # === ACTION ===
            action, _ = model.predict(obs, deterministic=True)
            if step <= 10:
                print("[DEBUG ACTION]", step, action[:8])

            # === STEP ENV ===
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            # DEBUG ball full (x, y, z)
            if hasattr(env, "ball_qpos_adr"):
                base = env.ball_qpos_adr
                bx = float(env.data.qpos[base + 0])
                by = float(env.data.qpos[base + 1])
                bz = float(env.data.qpos[base + 2])
            else:
                bx = by = bz = float("nan")

            if isinstance(info, dict):
                if step <= 20 or step % 30 == 0:
                    print(
                        f"[DEMO] step={step} "
                        f"reward={reward:.3f} "
                        f"ball_x={info.get('ball_x', float('nan')):.3f} "
                        f"ball_y={info.get('ball_y', float('nan')):.3f} "
                        f"(z={bz:.3f})"
                    )

            # === RENDER FRAME (fixed camera) ===
            renderer.update_scene(env.data, camera=topdown_cam)
            frame = renderer.render()
            frame = np.asarray(frame, dtype=np.uint8)
            writer.append_data(frame)

            # === EPISODE END ===
            if done:
                reset_out = env.reset()
                if isinstance(reset_out, tuple) and len(reset_out) == 2:
                    obs, _ = reset_out
                else:
                    obs = reset_out
                lift_ball_on_reset()
                # Re-use same camera config (table geometry isn't changing)
                # If you *want* to recompute, you could recreate the camera here.

        print(f"[record_demo] Finished {total_steps} steps total.")
    finally:
        print("[record_demo] Closing writer, renderer, and env...")
        writer.close()
        renderer.close()
        env.close()
        print(f"[record_demo] Saved demo video to {video_path}")


def main():
    print("[record_demo] __main__ starting")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=["sac", "tqc"],
        required=True,
        help="Which trained algorithm to use.",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Use the same model for antagonist (self-play).",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model .zip (e.g., best_model_fixed.zip).",
    )
    parser.add_argument(
        "--video-path",
        default="foosball_demo.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS of the output video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Video width in pixels (must be <= MuJoCo offwidth).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels (must be <= MuJoCo offheight).",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=900,
        help="Total environment steps to record (30 steps ~ 1s at 30fps).",
    )

    args = parser.parse_args()
    print("[record_demo] Parsed args:", args)

    run_and_record(
        algo=args.algo,
        model_path=args.model_path,
        video_path=args.video_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        total_steps=args.total_steps,
        self_play=args.self_play,
    )

    print("[record_demo] Done.")


if __name__ == "__main__":
    print("[record_demo] File executed as script, entering main()")
    main()

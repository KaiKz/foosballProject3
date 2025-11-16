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

    # Load protagonist model first (we need it either way)
    print("[record_demo] Loading protagonist model...")
    protagonist_model = load_model(algo, model_path, env=None)

    antagonist_model = None
    if self_play:
        print("[record_demo] Using same model as antagonist for self-play.")
        antagonist_model = protagonist_model  # or load a different zip if you have one

    # Now create env *after* we know antagonist_model
    env = make_env(antagonist_model=antagonist_model)

    # Re-bind env to protagonist_model so SB3 knows the space
    protagonist_model.set_env(env)
    model = protagonist_model


    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise RuntimeError("FoosballEnv does not expose .model and .data for MuJoCo rendering")

    print(f"[record_demo] Creating mujoco.Renderer(width={width}, height={height})")
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    print(f"[record_demo] Opening video writer at {video_path} (fps={fps})")
    writer = imageio.get_writer(video_path, fps=fps)

    try:
        # Initial reset
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out  # Gymnasium style: (obs, info)
        else:
            obs = reset_out

        for step in range(1, total_steps + 1):
            # === CHOOSE ACTION ===
            # DEBUG mode: zero actions (to see baseline env behavior)
            # action = np.zeros(env.action_space.shape, dtype=np.float32)

            # NORMAL mode: use the trained policy
            action, _ = model.predict(obs, deterministic=True)
            if step <= 10:
                print("[DEBUG ACTION]", step, action[:8]) 

            # === STEP ENV ===
            step_out = env.step(action)

            # Gymnasium style vs old gym:
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            # === PRINT DEBUG INFO ===
            if isinstance(info, dict):
                if step <= 20 or step % 30 == 0:
                    print(
                        f"[DEMO] step={step} "
                        f"reward={reward:.3f} "
                        f"ball_x={info.get('ball_x', float('nan')):.3f} "
                        f"ball_y={info.get('ball_y', float('nan')):.3f}"
                    )

            # === RENDER FRAME ===
            renderer.update_scene(env.data)
            frame = renderer.render()
            frame = np.asarray(frame, dtype=np.uint8)
            writer.append_data(frame)

            # === HANDLE EPISODE END ===
            if done:
                reset_out = env.reset()
                if isinstance(reset_out, tuple) and len(reset_out) == 2:
                    obs, _ = reset_out
                else:
                    obs = reset_out


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
    )

    print("[record_demo] Done.")


if __name__ == "__main__":
    print("[record_demo] File executed as script, entering main()")
    main()

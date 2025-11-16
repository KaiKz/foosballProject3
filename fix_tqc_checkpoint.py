# fix_tqc_checkpoint.py
import argparse
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

PREFIX = "_orig_mod."

def strip_orig_mod_prefix(state_dict):
    """
    Take a state_dict (dict of param_name -> tensor) and
    remove leading '_orig_mod.' if present.
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith(PREFIX):
            new_k = k[len(PREFIX):]
        else:
            new_k = k
        new_sd[new_k] = v
    return new_sd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", required=True, help="Compiled TQC .zip (with _orig_mod.* keys)")
    parser.add_argument("--out-path", required=True, help="Output fixed .zip")
    args = parser.parse_args()

    print(f"[fix_tqc_checkpoint] Loading {args.in_path}")
    data, params, pytorch_variables = load_from_zip_file(args.in_path, device="cpu")

    # params is a dict like {"policy": state_dict, "policy.optimizer": state_dict, ...}
    for name, value in list(params.items()):
        if isinstance(value, dict):
            print(f"[fix_tqc_checkpoint] Fixing param group: {name}")
            params[name] = strip_orig_mod_prefix(value)

    print(f"[fix_tqc_checkpoint] Saving fixed checkpoint to {args.out_path}")
    save_to_zip_file(args.out_path, data=data, params=params, pytorch_variables=pytorch_variables)
    print("[fix_tqc_checkpoint] Done")

if __name__ == "__main__":
    main()

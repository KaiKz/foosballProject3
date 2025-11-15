import io
import zipfile
import torch
from pathlib import Path

# 1) Paths: change old_path if your best_model.zip lives elsewhere
old_path = Path("/Users/kaikaizhang/Downloads/foosballpart2/models/0/tqc/best_model/best_model.zip")
new_path = old_path.with_name("best_model_fixed.zip")

print(f"[fix] Reading from: {old_path}")
print(f"[fix] Writing to:   {new_path}")

if not old_path.exists():
    raise FileNotFoundError(f"Could not find {old_path}")

with zipfile.ZipFile(old_path, "r") as zin, zipfile.ZipFile(new_path, "w") as zout:
    for info in zin.infolist():
        name = info.filename
        data = zin.read(name)

        # We only need to touch the policy weights file.
        # In SB3 2.x this is called 'policy.pth'.
        if name.endswith("policy.pth"):
            print(f"[fix] Patching '{name}'")

            # Load the state dict from bytes
            buf = io.BytesIO(data)
            state_dict = torch.load(buf, map_location="cpu")

            # Show a few original keys for sanity
            some_keys = list(state_dict.keys())[:5]
            print("[fix]  Sample original keys:")
            for k in some_keys:
                print("       ", k)

            # Build a new state dict with '_orig_mod.' removed from keys
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v

            # Show a few new keys for sanity
            some_new_keys = list(new_state_dict.keys())[:5]
            print("[fix]  Sample new keys:")
            for k in some_new_keys:
                print("       ", k)

            # Save patched state dict back to bytes
            out_buf = io.BytesIO()
            torch.save(new_state_dict, out_buf)
            data = out_buf.getvalue()

        # Write (possibly patched) data to new zip
        zout.writestr(info, data)

print("[fix] Done. New file saved as:", new_path)

import os
os.environ.setdefault("STABLEWM_HOME", os.path.expanduser("~/.stable-wm"))

import stable_worldmodel as swm

ds = swm.data.HDF5Dataset("libero", num_steps=1, frameskip=1, keys_to_load=["pixels"])
sample = ds[0]
print("keys:", list(sample.keys()))
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: type={type(v).__name__}, value={v}")

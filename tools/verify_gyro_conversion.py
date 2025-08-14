import os
import random
import torch
import json
from safetensors import safe_open
from kernel.codecs.gyrowt import decode_gyro_tensor


def verify_gyro_conversion():
    # Updated to use root directory structure (no original/ folder)
    model_dir = "memories/models/gpt-oss-20b"
    
    # Check for multi-part safetensors first
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Multi-part safetensors - use first file for verification
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        first_file = list(set(index_data.get("weight_map", {}).values()))[0]
        src = os.path.join(model_dir, first_file)
    else:
        # Single file
        src = os.path.join(model_dir, "model.safetensors")
    
    dst = os.path.join(model_dir, "gyro")  # per-file output

    with safe_open(src, framework="pt", device="cpu") as fs:
        keys = list(fs.keys())

    random.seed(0)
    probe = [
        k
        for k in keys
        if any(
            s in k
            for s in (".qkv.", "embedding.weight", "unembedding.weight", "mlp1_weight.blocks", "mlp1_weight.scales")
        )
    ]
    probe = probe[:8]  # sample

    for k in probe:
        # original bytes → tensor
        with safe_open(src, framework="pt", device="cpu") as fs:
            a = fs.get_tensor(k).cpu()

        # gyro bytes → decode → tensor
        gfile = os.path.join(dst, f"{k.replace('/', '_')}.safetensors")
        with safe_open(gfile, framework="pt", device="cpu") as fg:
            blob = fg.get_tensor(f"{k}.gyro")
            meta = fg.get_tensor(f"{k}.meta").cpu().numpy().tobytes()
            b = decode_gyro_tensor(k, blob, meta, device="cpu").cpu()

        assert a.shape == b.shape and a.dtype == b.dtype, (k, a.shape, b.shape, a.dtype, b.dtype)
        if a.dtype.is_floating_point:
            # byte-level equality for storage types we encoded losslessly
            ok = torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        else:
            ok = torch.equal(a, b)
        print(k, "OK" if ok else "MISMATCH")


if __name__ == "__main__":
    verify_gyro_conversion()

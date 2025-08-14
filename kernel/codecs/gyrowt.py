"""Weight codecs: lossless conversion from OSS checkpoints to GyroSI format.

This module provides CPU-only, streaming routines to convert safetensors checkpoints
to gyroscopic format. The conversion is lossless and preserves all model knowledge.
It is designed to run on machines with limited RAM by processing tensors in chunks.

The module supports both single-file and per-tensor conversion modes, with
compression options for optimal storage.

Download GPT-OSS
python -c "from kernel.codecs.gyrowt import download_model; download_model('openai/gpt-oss-20b', 'memories/models/gpt-oss-20b')"

Convert to Gyro
python -c "from kernel.codecs.gyrowt import convert_checkpoint_dir_to_gyro; convert_checkpoint_dir_to_gyro('memories/models/gpt-oss-20b')"

Pack to one file
python -c "from kernel.codecs.gyrowt import pack_gyro_safetensors; pack_gyro_safetensors('memories/models/gpt-oss-20b/gyro', 'memories/models/gpt-oss-20b/model.gyro.safetensors')" 

"""

from __future__ import annotations

import os
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Any
import sys
import zlib

import numpy as np
import torch

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None  # type: ignore[assignment]

# Defer gyro_head imports to avoid circular dependency
from typing import Callable

compute_exon_from_state: Callable[[int], int] | None = None
fold: Callable[[int, int], int] | None = None
GENE_Mic_S: int | None = None

def _import_gyro_head() -> None:
    """Lazy import of gyro_head functions to avoid circular imports."""
    global compute_exon_from_state, fold, GENE_Mic_S
    if compute_exon_from_state is None:
        import importlib
        gyro_head = importlib.import_module("kernel.gyro_head")
        compute_exon_from_state = gyro_head.compute_exon_from_state
        fold = gyro_head.fold
        GENE_Mic_S = gyro_head.GENE_Mic_S


def _verify_gyro_conversion_lazy():
    """Lazy import and call verification function."""
    import importlib
    verify_module = importlib.import_module("tools.verify_gyro_conversion")
    verify_module.main()


def download_model(model_name: str, model_dir: str) -> None:
    """Download model from Hugging Face Hub if not already present.
    
    Downloads to root directory, excluding original/ and metal/ folders.
    Handles multi-part safetensors files (model-00000-of-00002.safetensors, etc.)
    
    Args:
        model_name: Name of the model on Hugging Face Hub
        model_dir: Local directory to store the model
    """
    config_path = os.path.join(model_dir, "config.json")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    
    # Check if model is already complete
    if os.path.exists(config_path) and os.path.exists(index_path):
        print(f"Model {model_name} already exists at {model_dir}")
        return
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading {model_name} to {model_dir}...")
    
    # Download using huggingface-cli with resume capability, excluding original/ and metal/ folders
    cmd = [
        "huggingface-cli", "download", model_name,
        "--local-dir", model_dir,
        "--resume-download",
        "--exclude", "original/*",
        "--exclude", "metal/*"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully downloaded {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {model_name}: {e}")
        raise
    except FileNotFoundError:
        print("Error: huggingface-cli not found. Please install with: pip install huggingface_hub[cli]")
        raise


def discover_weight_keys(safetensors_path: str) -> tuple[str, str | None]:
    """Discover embedding and optional unembedding keys in a safetensors file."""
    if safe_open is None:
        raise RuntimeError("safetensors not available. Install with `pip install safetensors`.")
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        emb_candidates = [k for k in keys if k.endswith("embedding.weight") or k.endswith("embed_tokens.weight")]
        if not emb_candidates:
            raise RuntimeError("No embedding weight found in checkpoint")
        emb_key = emb_candidates[0]
        # Common names for output projection
        unemb_candidates = [
            k
            for k in keys
            if k.endswith("lm_head.weight") or k.endswith("unembedding.weight") or k.endswith("output.weight")
        ]
        unemb_key = unemb_candidates[0] if unemb_candidates else None
        return emb_key, unemb_key


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "discover_weight_keys",
    "encode_gyro_tensor",
    "decode_gyro_tensor",
    "convert_checkpoint_to_single_gyro_pack",
    "convert_checkpoint_dir_to_gyro",
]


# ------------------------------
# Lossless Gyro safetensors codec
# ------------------------------

_TORCH_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}

_STR_TO_TORCH_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _TORCH_DTYPE_TO_STR.items()}


# Add at the top with other imports
try:
    import zstandard as zstd
    _have_zstd = True
except ImportError:
    zstd = None  # type: ignore[assignment]
    _have_zstd = False


def encode_gyro_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Losslessly encode a tensor as compressed 6-byte cells + meta (uint8 JSON)."""
    # Get environment variables for codec configuration
    compressor = os.environ.get("GYRO_CODEC", "zstd" if _have_zstd else "zlib").lower()
    comp_level = int(os.environ.get("GYRO_LEVEL", "3"))
    apply_psi = int(os.environ.get("GYRO_PSI", "0")) != 0

    # Prepare tensor for encoding
    cpu = tensor.detach().cpu().contiguous()
    dtype_str = _TORCH_DTYPE_TO_STR.get(cpu.dtype)
    if dtype_str is None:
        raise RuntimeError(f"Unsupported dtype for gyro codec: {cpu.dtype}")

    # Get raw bytes regardless of dtype
    raw_bytes = cpu.view(torch.uint8).contiguous().numpy().tobytes()
    n_raw = len(raw_bytes)

    # Apply ψ isomorphism if requested (XOR 0xAA)
    if apply_psi:
        raw_bytes = bytes(b ^ 0xAA for b in raw_bytes)

    # Calculate padding needed to make byte count divisible by 6 (for 48-bit cells)
    pad = (6 - (n_raw % 6)) % 6
    if pad > 0:
        raw_bytes += b"\x00" * pad

    # Compress the bytes
    if compressor == "zstd" and _have_zstd and zstd is not None:
        cctx = zstd.ZstdCompressor(level=comp_level)
        comp = cctx.compress(raw_bytes)
    else:
        comp = zlib.compress(raw_bytes, level=comp_level)

    # Create metadata dictionary
    meta_dict = {
        "dtype": dtype_str,
        "shape": list(cpu.shape),
        "codec": "gyro-llp-v1",
        "compressor": compressor,
        "n_raw": n_raw,
        "psi": 1 if apply_psi else 0,
        "level": comp_level,
    }

    # Create blob tensor
    blob = torch.tensor(np.frombuffer(comp, dtype=np.uint8), dtype=torch.uint8)

    # Create metadata tensor
    meta = torch.tensor(np.frombuffer(json.dumps(meta_dict).encode("utf-8"), dtype=np.uint8), dtype=torch.uint8)

    return blob, meta


def decode_gyro_tensor(name: str, blob: torch.Tensor, meta: bytes | None, device: str) -> torch.Tensor:
    """Decode a gyro-compressed tensor back to torch.Tensor (lossless).

    Supports both legacy zlib format and new gyro-llp-v1 format.
    """
    if meta is None:
        raise RuntimeError(f"Missing meta for gyro tensor {name}")

    meta_dict = json.loads(meta.decode("utf-8"))
    dtype_str = meta_dict["dtype"]
    shape = tuple(meta_dict["shape"])
    codec = meta_dict.get("codec", "zlib")
    compressor = meta_dict.get("compressor", "zlib")

    target_dtype = _STR_TO_TORCH_DTYPE.get(dtype_str)
    if target_dtype is None:
        raise RuntimeError(f"Unsupported dtype in meta for {name}: {dtype_str}")

    comp_bytes = bytes(memoryview(blob.detach().cpu().numpy()))

    # Decompress based on codec
    if codec == "gyro-llp-v1":
        # Use the specified compressor
        if compressor == "zstd" and zstd:
            dctx = zstd.ZstdDecompressor()
            raw_with_pad = dctx.decompress(comp_bytes)
        else:
            raw_with_pad = zlib.decompress(comp_bytes)

        # Remove padding
        n_raw = meta_dict["n_raw"]
        raw = raw_with_pad[:n_raw]

        # Reverse ψ isomorphism if applied
        if meta_dict.get("psi", 0) == 1:
            raw = bytes(b ^ 0xAA for b in raw)
    else:
        # Legacy format
        raw = zlib.decompress(comp_bytes) if codec == "zlib" else comp_bytes

    # Convert to numpy array with correct dtype
    np_dtype = {
        torch.bfloat16: np.uint16,  # bfloat16 raw storage
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
    }[target_dtype]

    np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
    if target_dtype is torch.bfloat16:
        signed_arr = np_arr.view(np.int16)
        t = torch.from_numpy(signed_arr)
        t = t.view(torch.bfloat16)
    else:
        t = torch.from_numpy(np_arr)

    return t.to(torch.device(device))


# -----------------------------------------------------------------------------
# OSS Checkpoint Conversion (Schema-Aware)
# -----------------------------------------------------------------------------


def _read_oss_schema(model_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load config.json and dtypes.json if present."""
    cfg_path = Path(model_dir) / "config.json"
    dt_path = Path(model_dir) / "dtypes.json"
    cfg = json.load(open(cfg_path, "r", encoding="utf-8")) if cfg_path.exists() else {}
    dts = json.load(open(dt_path, "r", encoding="utf-8")) if dt_path.exists() else {}
    return cfg, dts


def _quant_pair_name(key: str) -> tuple[str, str] | None:
    """Returns (role, peer_key) if this key participates in FP4+UE8 pairing; else None."""
    if key.endswith(".blocks"):
        peer = key[:-7] + "scales"
        return ("blocks", peer)
    if key.endswith(".scales"):
        peer = key[:-7] + "blocks"
        return ("scales", peer)
    return None


def _infer_grouping(blocks_shape: tuple[int, ...], scales_shape: tuple[int, ...]) -> dict[str, Any]:
    """Infer quantization grouping parameters from tensor shapes."""
    info = {"group_axis": "col", "group_cols": None}
    if len(blocks_shape) == 2 and len(scales_shape) == 2:
        rows_b, cols_b_packed = blocks_shape
        rows_s, cols_s = scales_shape
        if rows_b == rows_s and cols_s > 0:
            # packed nibbles → real cols ≈ cols_b_packed * 2
            real_cols = cols_b_packed * 2
            if real_cols % cols_s == 0:
                info["group_cols"] = int(real_cols // cols_s)
    return info


def _augment_meta_for_oss(name: str, t: torch.Tensor, meta_json: dict[str, Any], dtypes: dict[str, Any], shapes_cache: dict[str, Any]) -> dict[str, Any]:
    """Augment meta JSON with logical dtype and quantization pairing info."""
    logical = dtypes.get(name) or dtypes.get(name.split(".")[-1])
    if logical:
        meta_json["logical_dtype"] = str(logical)
    # record shape for pairing inference later
    shapes_cache[name] = tuple(t.shape)
    qp = _quant_pair_name(name)
    if qp:
        role, peer = qp
        meta_json.setdefault("quant_pair", {"role": role, "peer": peer, "schema": "fp4+ue8"})
    return meta_json


def _compute_sha256_file(filepath: str, chunk_size: int = 1 << 20) -> str:
    """Compute SHA256 hash of a file."""
    import hashlib

    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_checkpoint_to_single_gyro_pack(
    safetensors_path: str, out_file: str, model_dir: str | None = None
) -> str:
    """Write one .safetensors pack with <key>.gyro and <key>.meta entries per tensor."""
    if safe_open is None:
        raise RuntimeError("safetensors not available. pip install safetensors")

    cfg, dts = _read_oss_schema(model_dir or str(Path(safetensors_path).parent))
    meta_header = {
        "gyro_pack": "1",
        "source": str(safetensors_path),
        "safetensors_sha256": _compute_sha256_file(safetensors_path),
        "config": json.dumps(cfg) if cfg else "",
        "dtypes": json.dumps(dts) if dts else "",
        "codec": "zlib",
        "version": "1",
    }

    tensors_out: dict[str, torch.Tensor] = {}
    shapes_cache: dict[str, tuple[int, ...]] = {}

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            blob, meta = encode_gyro_tensor(t)
            tensors_out[f"{key}.gyro"] = blob

            # augment meta JSON with logical dtype hint if present
            try:
                meta_json = json.loads(meta.numpy().tobytes().decode("utf-8"))
            except Exception:
                meta_json = {"dtype": str(t.dtype), "shape": list(t.shape), "codec": "zlib"}

            # augment with logical dtype + quant pair stub
            meta_json = _augment_meta_for_oss(key, t, meta_json, dts, shapes_cache)

            # if this key has a quant peer and the peer shape is known, add inferred grouping
            qp = _quant_pair_name(key)
            if qp:
                _, peer = qp
                peer_shape = shapes_cache.get(peer)
                if peer_shape:
                    if key.endswith(".blocks"):
                        info = _infer_grouping(tuple(t.shape), peer_shape)
                    else:
                        info = _infer_grouping(peer_shape, tuple(t.shape))
                    meta_json["quant_pair"].update(info)

            meta_bytes = json.dumps(meta_json).encode("utf-8")
            meta = torch.tensor(bytearray(meta_bytes), dtype=torch.uint8)
            tensors_out[f"{key}.meta"] = meta

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(tensors_out, out_file, metadata=meta_header)
    return out_file


def convert_checkpoint_dir_to_gyro(checkpoint_dir: str, output_subdir: str = "gyro") -> str:
    """Losslessly convert each tensor to its own small .safetensors file (schema-aware).
    
    Handles both single-file and multi-part safetensors (using model.safetensors.index.json).
    """
    if safe_open is None:
        raise RuntimeError("safetensors not available. pip install safetensors")

    in_dir = Path(checkpoint_dir)
    out_dir = in_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    cfg, dts = _read_oss_schema(str(in_dir))

    # Check for multi-part safetensors using index.json
    index_path = in_dir / "model.safetensors.index.json"
    if index_path.exists():
        # Multi-part safetensors - read from index
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        # Get unique filenames from the weight_map
        input_files = list(set(index_data.get("weight_map", {}).values()))
        input_files = [in_dir / fname for fname in input_files if fname.endswith(".safetensors")]
    else:
        # Single file or legacy format
        input_files = [
            p for p in in_dir.iterdir()
            if p.suffix == ".safetensors" and not p.name.endswith(".gyro.safetensors")
        ]
    shapes_cache: dict[str, tuple[int, ...]] = {}

    for st_path in input_files:
        header = {
            "gyro": "1",
            "codec": "zlib",
            "source": str(st_path),
            "safetensors_sha256": _compute_sha256_file(str(st_path)),
            "config": json.dumps(cfg),
            "dtypes": json.dumps(dts),
            "version": "1",
        }
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                blob, meta = encode_gyro_tensor(t)

                # augment meta JSON with logical dtype hint if known
                try:
                    meta_json = json.loads(meta.numpy().tobytes().decode("utf-8"))
                except Exception:
                    meta_json = {"dtype": str(t.dtype), "shape": list(t.shape), "codec": "zlib"}

                # augment with logical dtype + quant pair stub
                meta_json = _augment_meta_for_oss(key, t, meta_json, dts, shapes_cache)

                # if this key has a quant peer and the peer shape is known, add inferred grouping
                qp = _quant_pair_name(key)
                if qp:
                    _, peer = qp
                    peer_shape = shapes_cache.get(peer)
                    if peer_shape:
                        if key.endswith(".blocks"):
                            info = _infer_grouping(tuple(t.shape), peer_shape)
                        else:
                            info = _infer_grouping(peer_shape, tuple(t.shape))
                        meta_json["quant_pair"].update(info)

                meta_bytes = json.dumps(meta_json).encode("utf-8")
                meta = torch.tensor(bytearray(meta_bytes), dtype=torch.uint8)

                safe_name = key.replace("/", "_")
                out_file = out_dir / f"{safe_name}.safetensors"
                # Convert all metadata values to strings for safetensors compatibility
                str_header = {k: str(v) if not isinstance(v, str) else v for k, v in header.items()}
                save_file({f"{key}.gyro": blob, f"{key}.meta": meta}, str(out_file), metadata=str_header)
    return str(out_dir)


# Add CLI functionality
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Convert OSS model weights to gyro format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input safetensors file or directory")
    parser.add_argument("--output", "-o", type=str, help="Output path (defaults to input_dir/gyro)")
    parser.add_argument(
        "--single-file", "-s", action="store_true", help="Convert to a single file instead of per-tensor files"
    )
    parser.add_argument(
        "--codec", type=str, choices=["zstd", "zlib"], default="zstd", help="Compression codec to use (default: zstd)"
    )
    parser.add_argument("--level", type=int, default=3, help="Compression level (default: 3)")
    parser.add_argument(
        "--psi", type=int, choices=[0, 1], default=0, help="Apply ψ isomorphism (XOR 0xAA) (default: 0)"
    )
    parser.add_argument("--verify", "-v", action="store_true", help="Run verification after conversion")

    args = parser.parse_args()

    # Set environment variables for codec configuration
    os.environ["GYRO_CODEC"] = args.codec
    os.environ["GYRO_LEVEL"] = str(args.level)
    os.environ["GYRO_PSI"] = str(args.psi)

    input_path = args.input

    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Determine if input is a file or directory
    is_file = os.path.isfile(input_path)

    if is_file and args.single_file:
        # Single file to single file
        output_path = args.output or input_path.replace(".safetensors", ".gyro.safetensors")
        print(f"Converting {input_path} to single gyro file {output_path}...")
        convert_checkpoint_to_single_gyro_pack(input_path, output_path)
    elif is_file and not args.single_file:
        # Single file to directory
        output_dir = args.output or os.path.join(os.path.dirname(input_path), "gyro")
        print(f"Converting {input_path} to per-tensor gyro files in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        if safe_open is None:
            raise RuntimeError("safetensors not available. Install with `pip install safetensors`.")
        with safe_open(input_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                blob, meta = encode_gyro_tensor(t)
                safe_name = key.replace("/", "_")
                out_file = os.path.join(output_dir, f"{safe_name}.safetensors")
                from safetensors.torch import save_file

                save_file(
                    {f"{key}.gyro": blob, f"{key}.meta": meta}, out_file, metadata={"gyro": "1", "source": input_path}
                )
    else:
        # Directory to directory (default)
        output_subdir = args.output or "gyro"
        print(f"Converting all safetensors in {input_path} to per-tensor gyro files in {input_path}/{output_subdir}...")
        output_dir = convert_checkpoint_dir_to_gyro(input_path, output_subdir)

    print("Conversion complete!")

    # Run verification if requested
    if args.verify:
        print("\nVerifying conversion...")
        try:
            _verify_gyro_conversion_lazy()
            print("Verification successful!")
        except Exception as e:
            print(f"Verification failed: {e}", file=sys.stderr)
            sys.exit(1)


def pack_gyro_safetensors(input_dir: str, output_file: str) -> None:
    """Pack all .safetensors files in input_dir into a single output_file (model.gyro.safetensors)."""
    import safetensors.torch
    from glob import glob
    import os
    tensor_dict = {}
    safetensors_files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    for fname in safetensors_files:
        try:
            tensors = safetensors.torch.load_file(fname)
            for k, v in tensors.items():
                tensor_dict[os.path.basename(fname) + ":" + k] = v
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    safetensors.torch.save_file(tensor_dict, output_file)
    print(f"Packed {len(safetensors_files)} files into {output_file}")

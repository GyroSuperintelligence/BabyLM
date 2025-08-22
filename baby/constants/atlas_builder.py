# baby/constants/atlas_builder.py
"""
GyroSI Atlas Builder — canonical 5 artifacts only (θ, ontology, epistemology, phenomenology, orbit_sizes).

No optional maps. No scoring/greedy/dominance. Deterministic representatives.
Default output directory: memories/public/meta/

Usage examples:
  python -m baby.constants.atlas_builder ontology
  python -m baby.constants.atlas_builder epistemology
  python -m baby.constants.atlas_builder phenomenology
  python -m baby.constants.atlas_builder all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray

# Local physics (GENEs + transforms)
from baby.kernel import governance


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DEFAULT_META_DIR = Path("memories/public/meta")
DEFAULT_META_DIR.mkdir(parents=True, exist_ok=True)

PATH_ONTOLOGY = DEFAULT_META_DIR / "ontology_keys.npy"
PATH_EPI      = DEFAULT_META_DIR / "epistemology.npy"
PATH_THETA    = DEFAULT_META_DIR / "theta.npy"
PATH_PHENO    = DEFAULT_META_DIR / "phenomenology_map.npy"
PATH_ORBITS   = DEFAULT_META_DIR / "orbit_sizes.npy"


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

class Progress:
    def __init__(self, label: str):
        self.label = label
        self.start = time.time()
        self.last = 0.0
        self.first = True

    def update(self, cur: int, total: Optional[int] = None, extra: str = "") -> None:
        now = time.time()
        if not self.first and now - self.last < 0.1 and (total is None or cur != total):
            return
        self.first = False
        self.last = now
        elapsed = now - self.start
        rate = cur / elapsed if elapsed > 0 else 0
        msg = f"\r{self.label}: {cur:,}"
        if total is not None:
            pct = 100.0 * cur / total if total else 0.0
            msg += f"/{total:,} ({pct:.1f}%)"
        msg += f" | {rate:.0f}/s | {elapsed:.1f}s"
        if extra:
            msg += f" | {extra}"
        print(msg + " " * 20, end="", flush=True)

    def done(self) -> None:
        elapsed = time.time() - self.start
        print(f"\r{self.label}: Done in {elapsed:.1f}s" + " " * 40)


def _open_memmap_int32(path: Path, shape: Tuple[int, ...]) -> NDArray[np.int32]:
    from numpy.lib.format import open_memmap
    mm = open_memmap(str(path), dtype=np.int32, mode="w+", shape=shape)  # type: ignore
    return mm  # type: ignore[return-value]


# -------------------------------------------------------------------
# Step 1: Ontology + θ
# -------------------------------------------------------------------

def build_ontology_and_theta(
    ontology_path: Path = PATH_ONTOLOGY,
    theta_path: Path = PATH_THETA,
) -> NDArray[np.uint64]:
    """
    Discover full manifold via BFS from GENE_Mac_S under 256 introns.
    Save:
      - ontology_keys.npy (sorted uint64)
      - theta.npy (float32, angular divergence from archetype)
    """
    EXPECTED_N = 788_986
    prog = Progress("Discovering ontology")

    origin_int = int(governance.tensor_to_int(governance.GENE_Mac_S))

    discovered: set[int] = {origin_int}
    frontier: List[int] = [origin_int]
    depth = 0
    layer_sizes: List[int] = []

    while frontier:
        nxt: set[int] = set()
        for s in frontier:
            # Exhaustive 256 introns from state s
            for intron in range(256):
                t = governance.apply_gyration_and_transform(s, intron)
                if t not in discovered:
                    discovered.add(t)
                    nxt.add(t)
        if not nxt:
            break
        frontier = list(nxt)
        depth += 1
        layer_sizes.append(len(frontier))
        prog.update(len(discovered), EXPECTED_N, extra=f"depth={depth}")
    prog.done()

    # Checks (measured ground truth)
    if len(discovered) != EXPECTED_N:
        raise RuntimeError(f"Expected {EXPECTED_N:,} states, found {len(discovered):,}")
    if depth != 6:
        raise RuntimeError(f"Expected diameter 6, found {depth}")

    keys = np.array(sorted(discovered), dtype=np.uint64)
    ontology_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(ontology_path, keys)

    # θ relative to archetype (Hamming → angle)
    N = keys.size
    acos_lut = np.arccos(1 - 2 * np.arange(49) / 48.0).astype(np.float32)
    theta = np.empty(N, dtype=np.float32)
    # vectorized bit_count via XOR then Python int bit_count in loop (fast enough)
    for i, s in enumerate(keys):
        h = int(s ^ origin_int).bit_count()
        theta[i] = acos_lut[h]
    np.save(theta_path, theta)

    print(f"[INFO] Saved ontology → {ontology_path} ({keys.size:,} states)")
    print(f"[INFO] Saved θ        → {theta_path} (float32, N={keys.size:,})")
    return keys


# -------------------------------------------------------------------
# Step 2: Epistemology (N×256)
# -------------------------------------------------------------------

def build_epistemology(
    ontology_path: Path = PATH_ONTOLOGY,
    epi_path: Path = PATH_EPI,
    theta_path: Path = PATH_THETA,  # read-only; ensures presence
) -> None:
    """
    Build N×256 transition table: ep[i, intron] = next_state_index.
    """
    if not ontology_path.exists():
        raise FileNotFoundError(f"Missing ontology: {ontology_path}")
    if not theta_path.exists():
        raise FileNotFoundError(f"Missing theta: {theta_path}")

    keys = np.load(ontology_path, mmap_mode="r")  # uint64[N]
    N = int(keys.size)
    epi_mm = _open_memmap_int32(epi_path, (N, 256))
    prog = Progress("Building epistemology")

    CHUNK = 10_000
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunk_states = keys[start:end]  # uint64[chunk]
        # compute all-256 successors for this chunk (vectorized)
        next_states_all = governance.apply_gyration_and_transform_all_introns(chunk_states)  # [chunk,256]
        # map successor integers back to indices via searchsorted
        idxs = np.searchsorted(keys, next_states_all, side="left")
        # verify all found
        if idxs.max() >= keys.size or not np.all(keys[idxs] == next_states_all):
            raise RuntimeError("Transition produced unknown state (ontology mismatch).")
        epi_mm[start:end, :] = idxs.astype(np.int32, copy=False)
        prog.update(end, N)
    # flush memmap
    try:
        epi_mm.flush()  # type: ignore[attr-defined]
    except Exception:
        pass
    prog.done()
    print(f"[INFO] Saved epistemology → {epi_path} (int32, shape=({N:,}, 256))")


# -------------------------------------------------------------------
# Step 3: Phenomenology (SCCs over full 256 introns) + Orbit sizes
# -------------------------------------------------------------------

def _tarjan_scc_full(
    ep: NDArray[np.int32],
    idx_to_state: NDArray[np.uint64],
) -> Tuple[NDArray[np.int32], Dict[int, int]]:
    """
    Tarjan SCC using all 256 introns as directed edges.
    Deterministic representative rule: the node with minimal STATE INTEGER in SCC.
    Returns:
      canonical: int32[N] mapping node -> representative index
      orbit_sizes: dict(rep_index -> size)
    """
    N = int(ep.shape[0])
    indices = np.full(N, -1, dtype=np.int32)
    low = np.zeros(N, dtype=np.int32)
    on_stack = np.zeros(N, dtype=bool)
    stack: List[int] = []
    canonical = np.full(N, -1, dtype=np.int32)
    orbit_sizes: Dict[int, int] = {}
    counter = 0

    def neighbors(v: int) -> NDArray[np.int32]:
        # All successors for all 256 introns
        return np.asarray(ep[v, :], dtype=np.int32)

    for root in range(N):
        if indices[root] != -1:
            continue
        # iterative DFS
        dfs_stack: List[Tuple[int, Any]] = [(root, iter(neighbors(root)))]
        indices[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while dfs_stack:
            v, it = dfs_stack[-1]
            try:
                while True:
                    w = int(next(it))
                    if indices[w] == -1:
                        indices[w] = low[w] = counter
                        counter += 1
                        stack.append(w)
                        on_stack[w] = True
                        dfs_stack.append((w, iter(neighbors(w))))
                        break
                    elif on_stack[w]:
                        if indices[w] < low[v]:
                            low[v] = indices[w]
                        continue
                    else:
                        continue
            except StopIteration:
                dfs_stack.pop()
                if dfs_stack:
                    parent_v, _ = dfs_stack[-1]
                    if low[v] < low[parent_v]:
                        low[parent_v] = low[v]
                # root of SCC?
                if low[v] == indices[v]:
                    comp: List[int] = []
                    while True:
                        n = stack.pop()
                        on_stack[n] = False
                        comp.append(n)
                        if n == v:
                            break
                    # deterministic representative: minimal state integer
                    comp_arr = np.array(comp, dtype=np.int32)
                    comp_states = idx_to_state[comp_arr]
                    rep_local_idx = int(np.argmin(comp_states))
                    rep_idx = int(comp_arr[rep_local_idx])
                    canonical[comp_arr] = rep_idx
                    orbit_sizes[rep_idx] = int(comp_arr.size)

    if not np.all(canonical >= 0):
        raise RuntimeError("Unassigned nodes after SCC computation")
    return canonical, orbit_sizes


def build_phenomenology_and_orbit_sizes(
    epi_path: Path = PATH_EPI,
    ontology_path: Path = PATH_ONTOLOGY,
    pheno_path: Path = PATH_PHENO,
    orbit_sizes_path: Path = PATH_ORBITS,
) -> None:
    """
    Build:
      - phenomenology_map.npy : int32[N] representative index for each node
      - orbit_sizes.npy      : uint32[N] orbit size for each node
    """
    if not epi_path.exists() or not ontology_path.exists():
        raise FileNotFoundError("Missing epistemology and/or ontology")
    ep = np.load(epi_path, mmap_mode="r")          # int32[N,256]
    keys = np.load(ontology_path, mmap_mode="r")   # uint64[N]
    N = int(keys.size)

    print("[INFO] Computing canonical phenomenology (all 256 introns)…")
    canonical, orbit_sizes = _tarjan_scc_full(ep, keys)

    # materialize orbit sizes per node
    sizes_arr = np.zeros(N, dtype=np.uint32)
    for i in range(N):
        rep = int(canonical[i])
        sizes_arr[i] = orbit_sizes[rep]

    np.save(pheno_path, canonical.astype(np.int32, copy=False))
    np.save(orbit_sizes_path, sizes_arr)

    # light stats
    unique_reps = np.unique(canonical)
    print(f"[INFO] Orbits: {unique_reps.size} (deterministic reps by min(state_int))")
    print(f"[INFO] Saved phenomenology → {pheno_path}")
    print(f"[INFO] Saved orbit sizes   → {orbit_sizes_path}")


# -------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------

def cmd_ontology(_: argparse.Namespace) -> None:
    build_ontology_and_theta(PATH_ONTOLOGY, PATH_THETA)

def cmd_epistemology(_: argparse.Namespace) -> None:
    if not PATH_ONTOLOGY.exists() or not PATH_THETA.exists():
        print("[INFO] Ontology/θ missing — building them first…")
        build_ontology_and_theta(PATH_ONTOLOGY, PATH_THETA)
    build_epistemology(PATH_ONTOLOGY, PATH_EPI, PATH_THETA)

def cmd_phenomenology(_: argparse.Namespace) -> None:
    if not PATH_EPI.exists() or not PATH_ONTOLOGY.exists():
        print("[INFO] Epistemology/ontology missing — building them first…")
        if not PATH_ONTOLOGY.exists() or not PATH_THETA.exists():
            build_ontology_and_theta(PATH_ONTOLOGY, PATH_THETA)
        build_epistemology(PATH_ONTOLOGY, PATH_EPI, PATH_THETA)
    build_phenomenology_and_orbit_sizes(PATH_EPI, PATH_ONTOLOGY, PATH_PHENO, PATH_ORBITS)

def cmd_all(_: argparse.Namespace) -> None:
    keys = build_ontology_and_theta(PATH_ONTOLOGY, PATH_THETA)
    build_epistemology(PATH_ONTOLOGY, PATH_EPI, PATH_THETA)
    build_phenomenology_and_orbit_sizes(PATH_EPI, PATH_ONTOLOGY, PATH_PHENO, PATH_ORBITS)
    print("[INFO] All artifacts built.")
    print("       -", PATH_ONTOLOGY)
    print("       -", PATH_EPI)
    print("       -", PATH_THETA)
    print("       -", PATH_PHENO)
    print("       -", PATH_ORBITS)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GyroSI Atlas Builder (canonical 5 maps)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ontology", help="Build ontology_keys.npy and theta.npy").set_defaults(func=cmd_ontology)
    sub.add_parser("epistemology", help="Build epistemology.npy (requires ontology, theta)").set_defaults(func=cmd_epistemology)
    sub.add_parser("phenomenology", help="Build phenomenology_map.npy and orbit_sizes.npy").set_defaults(func=cmd_phenomenology)
    sub.add_parser("all", help="Build all artifacts in order").set_defaults(func=cmd_all)
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

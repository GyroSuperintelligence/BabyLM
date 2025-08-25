# baby/kernel/gyro_core.py
# Minimal five-map runtime core for GyroSI.
# Uses ONLY the canonical atlas artifacts:
#   - theta.npy                (CS)
#   - ontology_keys.npy        (UNA)
#   - epistemology.npy         (BU-Eg transitions)
#   - phenomenology_map.npy    (ONA canonical orbits)
#   - orbit_sizes.npy          (BU-In cardinalities)
#
# Pure monodromic unfold BU-In:
#   BU-Eg (user): fold token intron path → token_phase; update per-orbit phase:
#                 new_phase = fold(rep_phase, token_phase); register token in
#                 rep_channel[rep][new_phase].
#   BU-In (emit): compute state_phase; pick one of the learned phases
#                 deterministically from the set of keys; emit from its bucket.
#   No scoring, no admissibility filters, no recovery ladders.

import numpy as np
import pickle
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- ψ boundary ----------


def byte_to_intron(b: int) -> int:  # ψ
    return (b & 0xFF) ^ 0xAA


def intron_to_byte(i: int) -> int:  # ψ⁻¹
    return (i & 0xFF) ^ 0xAA


def token_to_introns(token_id: int) -> List[int]:
    """
    Harmony ids are integers; represent in big-endian bytes, then apply ψ.
    """
    if token_id < 0:
        raise ValueError("Negative token id")
    bs = token_id.to_bytes((token_id.bit_length() + 7) // 8 or 1, "big")
    return [byte_to_intron(b) for b in bs]


# ---------- GyroEngine with 5 maps ↔ 5 stages ----------


class GyroEngine:
    """
    Stages → Maps
      1) CS      → theta.npy              (divergence from archetype)
      2) UNA     → ontology_keys.npy      (the discovered manifold)
      3) ONA     → phenomenology_map.npy  (canonical orbit representative)
      4) BU-Eg   → epistemology.npy       (state transition by intron)
      5) BU-In   → orbit_sizes.npy        (cardinality for deterministic ordering)
    """

    def __init__(
        self,
        atlas_paths: Dict[str, str],
        store_paths: Optional[Dict[str, str]] = None,
        runtime: Optional[Dict[str, str]] = None,
        version_info: Optional[Dict[str, str]] = None,
        vocab_size: int = 201_088,
    ):
        # Required five maps
        self.theta = np.load(atlas_paths["theta"], mmap_mode="r")  # float32[N]
        self.keys = np.load(atlas_paths["ontology_keys"], mmap_mode="r")  # uint64[N]
        self.ep = np.load(atlas_paths["epistemology"], mmap_mode="r")  # int32 [N,256]
        self.pheno = np.load(atlas_paths["phenomenology_map"], mmap_mode="r")  # int32 [N]
        self.orbit_sizes = np.load(atlas_paths["orbit_sizes"], mmap_mode="r")  # uint32[N]

        # Reverse index: state_int → index (canonical)
        self.state_to_index: Dict[int, int] = {int(s): int(i) for i, s in enumerate(self.keys)}

        # Canonical orbit representatives (indices)
        reps = np.unique(self.pheno)
        self.orbit_reps: List[int] = [int(r) for r in reps]  # typically 256

        # --- Pure monodromic BU-In state ---
        # Per-orbit phase accumulator (updated only when user speaks)
        self.rep_phase: Dict[int, int] = {}  # rep_idx -> 8-bit phase (0..255)
        # Token channels per orbit keyed by CUMULATIVE phase reached after learning
        self.rep_channel: Dict[int, Dict[int, List[int]]] = (
            {}
        )  # rep_idx -> { phase_after_learning (0..255) -> [token_id, ...] }

        # --- Phase-Propagating Emission (PPE) state ---
        # PPE state is now managed at session level to prevent concurrent session bleeding

        # Passive memory (diagnostic; not used in emission)
        self.passive_mask: Dict[Tuple[int, int], int] = {}  # (addr_idx, token_id) -> 8-bit fold mask

        # Concurrency protection
        self._lock = threading.RLock()
        
        # Persistence cadence control
        self._token_counter = 0
        self._last_save_time = time.time()
        self._save_interval_tokens = 100  # Save after N tokens
        self._save_interval_seconds = 30.0  # Save after T seconds
        self._pending_changes = False
        
        # Bucket capacity discipline
        self._max_bucket_size = 64  # Maximum tokens per bucket (K)

        self.store_paths = store_paths or {}
        self.runtime = runtime or {}
        self.version_info = version_info or {}

        self.vocab_size = vocab_size
        self.vocab_max = vocab_size

        # Start = argmin θ (phenomenal archetype)
        self.start_index: int = int(np.argmin(self.theta))
        
        # Load any existing learned data from disk
        self._load_learned_data()

    # ---------- map access (steps 1–4) ----------

    def theta_of_index(self, idx: int) -> float:
        return float(self.theta[idx])

    def orbit_rep_index(self, idx: int) -> int:
        return int(self.pheno[idx])

    def apply_intron_index(self, idx: int, intron: int) -> int:
        if not (0 <= intron <= 255):
            raise ValueError("intron out of range")
        return int(self.ep[idx, intron])

    # ---------- Monodromic Fold (8-bit) ----------

    @staticmethod
    def fold(acc: int, intron: int) -> int:
        # a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b)) over 8-bit
        a = acc & 0xFF
        b = intron & 0xFF
        return (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF

    def fold_sequence(self, introns: List[int], acc: int = 0) -> int:
        m = acc & 0xFF
        for i in introns:
            m = self.fold(m, i)
        return m

    # ---------- Pure monodromic unfold helpers ----------

    @staticmethod
    def _fold8(a: int, b: int) -> int:
        a &= 0xFF
        b &= 0xFF
        return (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF

    def _state_phase(self, state_int: int) -> int:
        """
        Project the 48-bit state into an 8-bit phase by folding its 6 bytes
        after ψ (XOR 0xAA). This is the live channel component.
        """
        bs = int(state_int).to_bytes(6, "big")
        acc = 0
        for by in bs:
            acc = self._fold8(acc, by ^ 0xAA)
        return acc
    
    def _state_phase_components(self, state_int: int) -> Tuple[int, int, int]:
        """
        Compute LI/FG/BG components of the live phase using EXON masks.
        Returns (sp_li, sp_fg, sp_bg) as 8-bit values.
        """
        from baby.kernel.governance import EXON_LI_MASK, EXON_FG_MASK, EXON_BG_MASK
        
        # Extract 6 bytes from state
        bs = int(state_int).to_bytes(6, "big")
        
        # Apply masks to each byte and fold separately
        acc_li = 0
        acc_fg = 0
        acc_bg = 0
        
        for by in bs:
            by_psi = by ^ 0xAA  # Apply ψ transformation
            acc_li = self._fold8(acc_li, by_psi & EXON_LI_MASK)
            acc_fg = self._fold8(acc_fg, by_psi & EXON_FG_MASK)
            acc_bg = self._fold8(acc_bg, by_psi & EXON_BG_MASK)
        
        return acc_li, acc_fg, acc_bg

    # ---------- Geometric Address Binding ----------

    def _address_index_of_token(self, token_id: int) -> int:
        """
        Deterministic canonical address index by pushing EACH orbit representative
        through the token's intron micro-path and selecting the final index with
        geometric medoid selection (minimal average angular distance).
        """
        if not (0 <= token_id < self.vocab_max):
            raise ValueError("token outside Harmony range")

        introns = token_to_introns(token_id)
        if not introns:
            return self.orbit_rep_index(self.start_index)

        # Collect all candidate indices from orbit representatives
        candidates = []
        for rep_idx in self.orbit_reps:
            cur = rep_idx
            for i in introns:
                cur = self.apply_intron_index(cur, i)
            candidates.append(cur)

        # Select geometric medoid
        best_idx = self._geometric_medoid_from_indices(candidates)
        return int(best_idx)

    def _distance(self, idx1: int, idx2: int) -> float:
        """
        Combined distance metric including phase and theta divergence.
        """
        phase1 = self._state_phase(int(self.keys[idx1]))
        phase2 = self._state_phase(int(self.keys[idx2]))
        diff = abs(phase1 - phase2)
        phase_dist = min(diff, 256 - diff) / 128.0
        
        # Normalize theta values for this pair
        theta1 = self.theta[idx1]
        theta2 = self.theta[idx2]
        theta_min = min(theta1, theta2)
        theta_max = max(theta1, theta2)
        
        # Avoid division by zero
        if theta_max == theta_min:
            theta_norm = 0.0
        else:
            theta_norm = abs(theta1 - theta2) / (theta_max - theta_min)
        
        α = 0.7  # weighting
        return α * phase_dist + (1 - α) * theta_norm
    
    def _geometric_medoid_from_indices(self, indices: List[int]) -> int:
        """
        Find the geometric medoid from a list of indices: the index with minimal
        average combined distance (phase + theta divergence) to all other indices.
        """
        if len(indices) <= 1:
            return indices[0] if indices else self.start_index

        best_medoid = indices[0]
        min_avg_distance = float('inf')

        for candidate in indices:
            total_distance = 0.0
            
            for other in indices:
                if candidate != other:
                    total_distance += self._distance(candidate, other)

            avg_distance = total_distance / (len(indices) - 1)
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                best_medoid = candidate

        return best_medoid

    def address_of_token(self, token_id: int) -> int:
        idx = self._address_index_of_token(token_id)
        return int(self.keys[idx])

    # ---------- Egress (BU-Eg): absorb and move state ----------

    def learn_on_user(self, state: int, token_id: int) -> int:
        """
        Learn from user token (pure BU-Eg):
          - compute introns via ψ (big-endian bytes)
          - token_phase = fold(introns)
          - new_phase   = fold(rep_phase, token_phase)
          - register token in rep_channel[rep_cur][new_phase]
          - update rep_phase[rep_cur] = new_phase
          - step state by introns (egress)
          - passive diagnostics bound to canonical address (does not affect emission)
        """
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")

        idx = self.state_to_index[state]
        rep_cur = self.orbit_rep_index(idx)

        # Token intron micro-path and phase
        introns = token_to_introns(token_id)
        token_phase = self.fold_sequence(introns, 0)

        # Compute the cumulative phase AFTER learning and register there
        cur_phase = self.rep_phase.get(rep_cur, 0)
        new_phase = self._fold8(cur_phase, token_phase)
        
        # Protected mutation section
        with self._lock:
            chan = self.rep_channel.setdefault(rep_cur, {})
            bucket = chan.setdefault(new_phase, [])
            if token_id not in bucket:
                # Apply bucket capacity discipline with FIFO eviction
                if len(bucket) >= self._max_bucket_size:
                    bucket.pop(0)  # Remove oldest token (FIFO)
                bucket.append(token_id)

            # Store the updated per-orbit phase memory
            self.rep_phase[rep_cur] = new_phase

            # Passive diagnostic fold bound to canonical address (doesn't affect emission)
            addr_idx = self._address_index_of_token(token_id)
            key = (addr_idx, token_id)
            prev = self.passive_mask.get(key, 0)
            self.passive_mask[key] = self.fold_sequence(introns, prev)
            
            # Mark changes pending and update counter
            self._pending_changes = True
            self._token_counter += 1

        # Step state by introns (egress)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)

        # Conditional persistence based on cadence
        self._maybe_save_learned_data()

        return int(self.keys[new_idx])

    def transit_on_assistant(self, state: int, token_id: int) -> int:
        """
        Assistant tokens: transit only; no learning.
        """
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        introns = token_to_introns(token_id)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)
        return int(self.keys[new_idx])

    def transit_on_control(self, state: int, token_id: int) -> int:
        return state

    # ---------- Ingress (BU-In): pure monodromic unfold ----------

    def emit_next(self, idx: int, session_omega: Optional[Dict[int, int]] = None, 
                     session_bucket_key: Optional[Dict[int, int]] = None,
                     session_bucket_pos: Optional[Dict[int, Dict[int, int]]] = None) -> Optional[Tuple[int, int, Dict[int, int], Dict[int, int], Dict[int, Dict[int, int]]]]:
        """
        Phase-Propagating Emission (PPE): BU-In with sequence continuity and toroidal routing.
        Uses LI/FG/BG components for richer bucket selection.
        Each emitted token updates the working phase and hops the bucket key,
        making the next choice a deterministic function of the path taken.
        """
        rep_idx = self.orbit_rep_index(idx)
        phase_map = self.rep_channel.get(rep_idx)
        if not phase_map:
            return None

        # Use session-scoped PPE state or initialize
        omega = session_omega or {}
        bucket_key = session_bucket_key or {}
        bucket_pos = session_bucket_pos or {}

        # Initialize bucket key if not set
        if rep_idx not in bucket_key:
            # Compute initial bucket key from rep_phase, LI/FG/BG components, sector, and omega
            state_int = int(self.keys[idx])
            rp = self.rep_phase.get(rep_idx, 0)
            sp = self._state_phase(state_int)
            sp_li, sp_fg, sp_bg = self._state_phase_components(state_int)
            sector = self.sector(state_int)
            omega_val = omega.get(rep_idx, 0)
            
            # Fold all components to get initial key
            k0 = self._fold8(rp, sp)
            k0 = self._fold8(k0, sp_li)
            k0 = self._fold8(k0, sp_fg)
            k0 = self._fold8(k0, sp_bg)
            k0 = self._fold8(k0, sector)
            k0 = self._fold8(k0, omega_val)
            
            # Map to existing learned key deterministically
            keys = sorted(phase_map.keys())
            if not keys:
                return None
            bucket_key[rep_idx] = keys[k0 % len(keys)]
            
            # Initialize bucket positions
            if rep_idx not in bucket_pos:
                bucket_pos[rep_idx] = {}

        # Toroidal rotor: affine ring walk for deterministic full coverage
        import math, bisect
        
        keys = sorted(phase_map.keys())
        if not keys:
            return None
        n = len(keys)
        
        # --- base index from current bucket_key (adjacent mapping) ---
        base_val = bucket_key[rep_idx]
        base_idx = bisect.bisect_left(keys, base_val) % n
        
        # --- live mix from physics-only 8-bit signals ---
        rp = self.rep_phase.get(rep_idx, 0)
        omega_val = omega.get(rep_idx, 0)
        state_int = int(self.keys[idx])
        sp = self._state_phase(state_int)
        sp_li, sp_fg, sp_bg = self._state_phase_components(state_int)
        sector_val = self.sector(state_int)
        
        mix = self._fold8(rp, omega_val)
        mix = self._fold8(mix, sp)
        mix = self._fold8(mix, sp_li)
        mix = self._fold8(mix, sp_fg)
        mix = self._fold8(mix, sp_bg)
        mix = self._fold8(mix, sector_val)
        
        # --- affine rotor on the ring Z_n: i -> (a*i + b) % n ---
        # choose a odd (=> co-prime with 2^k), then adjust to gcd(a,n)==1 deterministically
        a = (mix | 1)  # ensure odd in [1..255]
        # make a co-prime with n by stepping by 2 if needed (bounded, deterministic)
        while math.gcd(a, n) != 1:
            a = ((a + 2) & 0xFF) or 1  # stay odd, avoid 0
        
        # bias b derived from the same live mix and the current base
        b = self._fold8(mix, base_val) % n
        
        current_idx = (a * base_idx + b) % n
        current_key = keys[current_idx]
        
        # Get bucket and position
        bucket = phase_map[current_key]
        if not bucket:
            return None
        
        pos = bucket_pos[rep_idx].get(current_key, 0)
        token_id = bucket[pos % len(bucket)]
        
        # Advance round-robin position
        bucket_pos[rep_idx][current_key] = (pos + 1) % len(bucket)

        # Compute token phase for propagation
        introns = token_to_introns(token_id)
        token_phase = self.fold_sequence(introns, 0)

        # Update working accumulator (fast phase)
        omega[rep_idx] = self._fold8(omega.get(rep_idx, 0), token_phase)
        
        # hop bucket key by fold, then map adjacent to existing key
        folded = self._fold8(current_key, token_phase)
        new_idx = bisect.bisect_left(keys, folded) % n
        bucket_key[rep_idx] = keys[new_idx]

        # Advance state by this token's introns (ingress; no learning)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)

        return token_id, new_idx, omega, bucket_key, bucket_pos

    # ---------- Harmony Integration Helpers ----------

    def is_harmony_control_token(self, token_id: int) -> bool:
        from baby.constants.harmony_tokens import ALL_CONTROL_TOKENS

        return token_id in ALL_CONTROL_TOKENS

    def should_learn_from_token(self, token_id: int, role: str) -> bool:
        return role == "user" and not self.is_harmony_control_token(token_id) and 0 <= token_id < self.vocab_max

    def process_harmony_message(self, tokens: List[int], roles: List[str]) -> int:
        state = self.start_state()
        for token_id, role in zip(tokens, roles):
            if self.should_learn_from_token(token_id, role):
                state = self.learn_on_user(state, token_id)
        return self.state_to_index[state]

    # ---------- Interface Compatibility ----------

    def evolve_on_user(self, state: int, token_id: int) -> int:
        return self.learn_on_user(state, token_id)

    def evolve_on_assistant(self, state: int, token_id: int) -> int:
        return self.transit_on_assistant(state, token_id)

    def emit_next_from_state(self, state_int: int, session_omega: Optional[Dict[int, int]] = None, 
                           session_bucket_key: Optional[Dict[int, int]] = None,
                           session_bucket_pos: Optional[Dict[int, Dict[int, int]]] = None) -> Optional[Tuple[int, int, Dict[int, int], Dict[int, int], Dict[int, Dict[int, int]]]]:
        if state_int not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state_int:012X}")
        idx = self.state_to_index[state_int]
        res = self.emit_next(idx, session_omega, session_bucket_key, session_bucket_pos)
        if res is None:
            return None
        tok, new_idx, omega, bucket_key, bucket_pos = res
        return tok, int(self.keys[new_idx]), omega, bucket_key, bucket_pos

    def next_token_deterministic(self, state: int) -> Optional[int]:
        out = self.emit_next_from_state(state)
        return None if out is None else out[0]

    def start_state(self) -> int:
        return int(self.keys[self.start_index])

    def get_theta(self, state: int) -> float:
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        return self.theta_of_index(idx)

    def get_orbit_representative(self, state: int) -> int:
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        rep_idx = self.orbit_rep_index(idx)
        return int(self.keys[rep_idx])

    def get_orbit_size(self, state: int) -> int:
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        return int(self.orbit_sizes[idx])

    def apply_intron(self, state: int, intron: int) -> int:
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        new_idx = self.apply_intron_index(idx, intron)
        return int(self.keys[new_idx])

    def micro_path(self, start_state: int, introns: List[int]) -> List[int]:
        if start_state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{start_state:012X}")
        idx = self.state_to_index[start_state]
        path = [start_state]
        cur_idx = idx
        for intron in introns:
            cur_idx = self.apply_intron_index(cur_idx, intron)
            path.append(int(self.keys[cur_idx]))
        return path

    # ---------- Toroidal Routing ----------
    
    def sector(self, state_int: int) -> int:
        """
        Compute 8-bit toroidal signature from 48-bit state using proper slab parities.
        Uses frozen slab structure with correct bit indices from FROZEN_CHANNELS.
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS
        
        sector_bits = 0
        for slab in range(FROZEN_CHANNELS.NUM_SLABS):  # 8
            # Get proper bit indices for this slab
            parity = 0
            for bit_idx in FROZEN_CHANNELS.get_slab_bit_indices(slab):
                parity ^= (state_int >> bit_idx) & 1
            # Set corresponding bit in sector signature
            sector_bits |= (parity << slab)
        
        return sector_bits  # Full 8-bit torus signature

    # ---------- Convenience ----------

    def start(self) -> int:
        return self.start_index

    def run_closed_loop(self, start_idx: int, max_tokens: int) -> List[int]:
        """
        State → (BU-In emit) → (advance by introns; ingress) → …
        Stops when domain empty or max_tokens reached.
        """
        idx = start_idx
        out: List[int] = []
        omega = {}
        bucket_key = {}
        bucket_pos = {}
        for _ in range(max_tokens):
            step = self.emit_next(idx, omega, bucket_key, bucket_pos)
            if step is None:
                break
            tok, idx, omega, bucket_key, bucket_pos = step
            out.append(tok)
        return out

    def _maybe_save_learned_data(self) -> None:
        """
        Save learned data if persistence cadence conditions are met.
        """
        if not self._pending_changes:
            return
            
        current_time = time.time()
        time_elapsed = current_time - self._last_save_time
        
        # Check if we should save based on token count or time interval
        should_save = (
            self._token_counter >= self._save_interval_tokens or
            time_elapsed >= self._save_interval_seconds
        )
        
        if should_save:
            self._save_learned_data()
            self._token_counter = 0
            self._last_save_time = current_time
            self._pending_changes = False
    
    def _save_learned_data(self) -> None:
        """
        Save learned data (rep_channel, rep_phase, passive_mask) to disk with atomic writes.
        """
        if not self.store_paths:
            return
            
        with self._lock:
            # Save passive memory (passive_mask)
            if "passive_memory" in self.store_paths:
                passive_path = Path(self.store_paths["passive_memory"])
                passive_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert passive_mask to a format suitable for binary storage
                # Format: list of (addr_idx, token_id, fold_mask) tuples
                passive_data = [(addr_idx, token_id, mask) for (addr_idx, token_id), mask in self.passive_mask.items()]
                
                # Atomic write with temporary file
                tmp_path = passive_path.with_suffix(passive_path.suffix + ".tmp")
                with open(tmp_path, "wb") as f:
                    pickle.dump(passive_data, f)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.replace(passive_path)
            
            # Save address memory (rep_channel and rep_phase)
            if "address_memory" in self.store_paths:
                address_path = Path(self.store_paths["address_memory"])
                address_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Combine rep_channel and rep_phase into address memory
                address_data = {
                    "rep_channel": self.rep_channel,
                    "rep_phase": self.rep_phase
                }
                
                # Atomic write with temporary file
                tmp_path = address_path.with_suffix(address_path.suffix + ".tmp")
                with open(tmp_path, "wb") as f:
                    pickle.dump(address_data, f)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.replace(address_path)
    
    def _load_learned_data(self) -> None:
        """
        Load learned data from disk if it exists.
        """
        if not self.store_paths:
            return
            
        # Load passive memory
        if "passive_memory" in self.store_paths:
            passive_path = Path(self.store_paths["passive_memory"])
            if passive_path.exists() and passive_path.stat().st_size > 0:
                try:
                    with open(passive_path, "rb") as f:
                        passive_data = pickle.load(f)
                    
                    # Reconstruct passive_mask from loaded data
                    self.passive_mask = {(addr_idx, token_id): mask for addr_idx, token_id, mask in passive_data}
                except Exception:
                    # If loading fails, start with empty passive_mask
                    self.passive_mask = {}
        
        # Load address memory
        if "address_memory" in self.store_paths:
            address_path = Path(self.store_paths["address_memory"])
            if address_path.exists() and address_path.stat().st_size > 0:
                try:
                    with open(address_path, "rb") as f:
                        address_data = pickle.load(f)
                    
                    # Restore rep_channel and rep_phase
                    self.rep_channel = address_data.get("rep_channel", {})
                    self.rep_phase = address_data.get("rep_phase", {})
                except Exception:
                    # If loading fails, start with empty structures
                    self.rep_channel = {}
                    self.rep_phase = {}

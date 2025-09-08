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
        #                 coherently from the set of keys; emit from its bucket.
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
       5) BU-In   → orbit_sizes.npy        (cardinality for traceable ordering)
    """

    def __init__(
        self,
        atlas_paths: Dict[str, str],
        store_paths: Optional[Dict[str, str]] = None,
        runtime: Optional[Dict[str, str]] = None,
        version_info: Optional[Dict[str, str]] = None,
        vocab_size: int = 201_088,
        # Core physics switches - disable all secondary heuristics for testing
        enable_slab_routing: bool = False,
        enable_dof_jitter: bool = False,
        enable_egress_mask: bool = False,
        enable_refractory_gates: bool = False,
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
        # Now supports slab-specific channels: (rep_idx, slab_idx) -> { phase -> [token_id, ...] }
        self.rep_channel: Dict[Tuple[int, int], Dict[int, List[int]]] = (
            {}
        )  # (rep_idx, slab_idx) -> { phase_after_learning (0..255) -> [token_id, ...] }

        # --- Emission hygiene (path-native, non-competitive) ---
        self.emit_gate: Dict[int, int] = {}        # per-rep monodromic refractory gate (8-bit)
        self.last_token: Dict[int, int] = {}       # last emitted token per rep
        self.last_emit_tick: Dict[int, int] = {}   # last free_tick per rep

        # --- Phase-Propagating Emission (PPE) state ---
        # PPE state is now managed at session level to prevent concurrent session bleeding

        # Passive diagnostics removed (unused)

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

        # Core physics switches
        self.enable_slab_routing = enable_slab_routing
        self.enable_dof_jitter = enable_dof_jitter
        self.enable_egress_mask = enable_egress_mask
        self.enable_refractory_gates = enable_refractory_gates

        # Optional debug for slab/global path selection
        self.debug_slab = bool((self.runtime or {}).get("debug_slab", False))

        # Start = argmin θ (phenomenal archetype)
        self.start_index: int = int(np.argmin(self.theta))

        # Ensure store files exist if store paths are provided (create only if missing)
        self._ensure_store_files()

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
        """Legacy fold operation for backward compatibility."""
        # a ⋄ b = a ⊕ (b ⊕ (a ∧ ¬b)) over 8-bit
        a = acc & 0xFF
        b = intron & 0xFF
        return (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF

    def fold_sequence(self, introns: List[int], acc: int = 0) -> Tuple[int, int]:
        """Fold sequence returning (phase, amplitude) for interference analysis."""
        m = acc & 0xFF
        amp = 0
        for i in introns:
            m, amp = self._fold8(m, i)
        return m, amp

    # ---------- Pure monodromic unfold helpers ----------

    @staticmethod
    def _fold8(a: int, b: int) -> Tuple[int, int]:
        """Fold operation returning (phase, amplitude) for interference analysis."""
        a &= 0xFF
        b &= 0xFF
        res = (a ^ (b ^ (a & (~b & 0xFF)))) & 0xFF
        amp = bin(res).count('1')  # Non-zero bits as "coherence strength"
        return res, amp

    def _state_phase(self, state_int: int) -> Tuple[int, int]:
        """
        Project the 48-bit state into an 8-bit phase and velocity by folding its 6 bytes
        after ψ (XOR 0xAA). Returns (phase, velocity) for interference analysis.
        """
        bs = int(state_int).to_bytes(6, "big")
        acc = 0
        prev_acc = 0
        velocity = 0
        for by in bs:
            prev_acc = acc
            acc, _ = self._fold8(acc, by ^ 0xAA)
            velocity, _ = self._fold8(velocity, acc ^ prev_acc)  # Delta as "speed"
        return acc, velocity

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
            acc_li, _ = self._fold8(acc_li, by_psi & EXON_LI_MASK)
            acc_fg, _ = self._fold8(acc_fg, by_psi & EXON_FG_MASK)
            acc_bg, _ = self._fold8(acc_bg, by_psi & EXON_BG_MASK)

        return acc_li, acc_fg, acc_bg

    def token_phase(self, token_id: int) -> Tuple[int, int]:
        """Compute token phase and amplitude for interference analysis."""
        return self.fold_sequence(token_to_introns(token_id), 0)

    # ---------- Freedom kernel: six DoF + free-tick ----------

    def _free_tick(self) -> int:
        """
        Endogenous temporal phase: fold 6 bytes of the current monotonic clock
        through ψ into an 8-bit tick. This is *not* RNG; it's the physical time
        boundary coupled to the monodromic fold, giving non-deterministic but
        lawful motion (BU Egress→Ingress).
        """
        ns = time.time_ns()
        bs = ns.to_bytes(8, "big")[-6:]  # 6 bytes → 48-bit affinity
        acc = 0
        for b in bs:
            acc, _ = self._fold8(acc, b ^ 0xAA)  # ψ at the boundary
        return acc  # 0..255

    def _row_parity_fold(self, state_int: int, row: int, frame: Optional[int] = None) -> int:
        """
        Fold parity of a given tensor row across all (layer, [frame], col).
        If frame is None, both frames contribute; otherwise only that frame.
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS
        acc = 0
        for layer in range(FROZEN_CHANNELS.NUM_LAYERS):  # 4
            for fr in (range(FROZEN_CHANNELS.NUM_FRAMES) if frame is None else [frame]):  # 2 or 1
                for col in range(FROZEN_CHANNELS.NUM_COLS):  # 2
                    bit_idx = FROZEN_CHANNELS.get_bit_index(layer, fr, row, col)
                    bit = (state_int >> bit_idx) & 1
                    acc, _ = self._fold8(acc, (bit & 1) * 0x01)  # compress parity into 8-bit phase
        return acc

    def _six_dof(self, state_int: int) -> Tuple[int, int, int, int, int, int]:
        """
        Six freedoms (3 rotational + 3 translational) as 8-bit phases:

        - Rotational (rX, rY, rZ): per-row parity over *both* frames (frame-summed)
        - Translational (tX, tY, tZ): per-row *frame-difference* parity (frame 0 vs 1)

        Rows 0/1/2 correspond to the three spatial axes from GENE_Com_S.
        """
        # rotational: both frames together
        rX = self._row_parity_fold(state_int, row=0, frame=None)
        rY = self._row_parity_fold(state_int, row=1, frame=None)
        rZ = self._row_parity_fold(state_int, row=2, frame=None)

        # translational: difference between frames
        f0X = self._row_parity_fold(state_int, row=0, frame=0)
        f1X = self._row_parity_fold(state_int, row=0, frame=1)
        f0Y = self._row_parity_fold(state_int, row=1, frame=0)
        f1Y = self._row_parity_fold(state_int, row=1, frame=1)
        f0Z = self._row_parity_fold(state_int, row=2, frame=0)
        f1Z = self._row_parity_fold(state_int, row=2, frame=1)

        tX, _ = self._fold8(f0X, f1X)  # path-dependent difference via fold
        tY, _ = self._fold8(f0Y, f1Y)
        tZ, _ = self._fold8(f0Z, f1Z)

        return rX, rY, rZ, tX, tY, tZ

    def _slab_byte(self, state_int: int, slab_idx: int) -> int:
        """
        Compress the 6 bits of a slab into the low bits of a byte (contiguous),
        then apply ψ. This fixes the previous shift-by-min-index approach,
        which preserved gaps and bled geometry.
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS
        indices = FROZEN_CHANNELS.get_slab_bit_indices(slab_idx)
        b = 0
        for j, bit_idx in enumerate(indices):
            b |= ((state_int >> bit_idx) & 1) << j
        return (b ^ 0xAA) & 0xFF  # ψ

    # Address helpers removed (unused)

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
        token_phase, _ = self.fold_sequence(introns, 0)

        # Compute the cumulative phase AFTER learning and register there
        cur_phase = self.rep_phase.get(rep_cur, 0)
        new_phase, _ = self._fold8(cur_phase, token_phase)

        # Get state_int for slab computation
        state_int = int(self.keys[idx])

        # Protected mutation section
        with self._lock:
            # === SLAB-SPECIFIC CHANNELS ===
            # Each slab gets its own phase and channel based on state geometry

            for slab_idx in range(8):
                # Compute slab-specific phase from token and state geometry
                slab_byte = self._slab_byte(state_int, slab_idx)
                slab_phase, _ = self._fold8(token_phase, slab_byte)  # Slab-specific phase

                # Each slab maintains its own channel
                slab_chan = self.rep_channel.setdefault((rep_cur, slab_idx), {})
                bucket = slab_chan.setdefault(slab_phase, [])
                if token_id not in bucket:
                    # Apply bucket capacity discipline with FIFO eviction
                    if len(bucket) >= self._max_bucket_size:
                        bucket.pop(0)  # Remove oldest token (FIFO)
                    bucket.append(token_id)

            # Store the updated per-orbit phase memory
            self.rep_phase[rep_cur] = new_phase

            # (passive diagnostics removed)

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

    def emit_next(self, idx: int,
                  session_omega: Optional[Dict[Tuple[int,int], int]] = None,
                  session_bucket_key: Optional[Dict[Tuple[int,int], int]] = None,
                  session_bucket_pos: Optional[Dict[Tuple[int,int], Dict[int, int]]] = None,
                  session_monodromy: Optional[Dict[Tuple[int,int], int]] = None,
                  session_mask: Optional[int] = None,
                  session_slab_cursor: Optional[Dict[int, int]] = None
) -> Optional[Tuple[int, int, Dict[Tuple[int,int], int], Dict[Tuple[int,int], int], Dict[Tuple[int,int], Dict[int, int]], Dict[Tuple[int,int], int], Dict[int,int]]]:
        """
        Slab-first PPE: pick one active slab (head), route inside that slab only.
        Returns (token_id, new_idx, omega, bucket_key, bucket_pos, monodromy, slab_cursor)
        """
        rep_idx = self.orbit_rep_index(idx)
        state_int = int(self.keys[idx])

        # Session state (now keyed by (rep, slab))
        omega = session_omega or {}
        bucket_key = session_bucket_key or {}
        bucket_pos = session_bucket_pos or {}
        monodromy = session_monodromy or {}
        slab_cursor = session_slab_cursor or {}

        # If slab routing is disabled, use a flattened/global path keyed by (rep,-1)
        if not self.enable_slab_routing:
            # Build a global phase map across all slabs
            phase_map: Dict[int, List[int]] = {}
            for s in range(8):
                slab_chan = self.rep_channel.get((rep_idx, s), {})
                if not slab_chan:
                    continue
                for phase, toks in slab_chan.items():
                    if not toks:
                        continue
                    if phase not in phase_map:
                        phase_map[phase] = []
                    # extend and deduplicate order-preserving
                    seen = set(phase_map[phase])
                    for t in toks:
                        if t not in seen:
                            phase_map[phase].append(t)
                            seen.add(t)

            if not phase_map:
                return None

            # Use per-rep global key
            key = (rep_idx, -1)

            keys = sorted(phase_map.keys())
            n = len(keys)

            rp = self.rep_phase.get(rep_idx, 0)
            sp, sp_vel = self._state_phase(state_int)

            if key not in bucket_key:
                k0, _ = self._fold8(rp, sp)
                k0, _ = self._fold8(k0, omega.get(key, 0))
                bucket_key[key] = keys[k0 % n]
                if key not in bucket_pos:
                    bucket_pos[key] = {}

            import bisect
            base_val = bucket_key[key]
            base_idx = bisect.bisect_left(keys, base_val) % n

            # DoF jitter wiring (affects ring index and intra-bucket pos)
            rotor_seed = 0
            pos_jitter = 0
            tick = 0
            if self.enable_dof_jitter:
                rX, rY, rZ, tX, tY, tZ = self._six_dof(state_int)
                tick = self._free_tick()
                for v in (rX, rY, rZ, tX, tY, tZ, tick):
                    rotor_seed, _ = self._fold8(rotor_seed, v)
                pos_jitter = rotor_seed & 0x07

            current_idx = (base_idx + ((sp ^ rotor_seed) % n)) % n
            current_key = keys[current_idx]

            bucket = phase_map[current_key]
            if not bucket:
                # hop locally and try next time
                bucket_key[key] = keys[(current_idx + 1) % n]
                return None

            L = len(bucket)
            pos_map = bucket_pos.setdefault(key, {})
            base_pos = pos_map.get(current_key, 0)
            pos = (base_pos + pos_jitter) % L

            gate_acc = self.emit_gate.get(rep_idx, 0)
            last_tok = self.last_token.get(rep_idx, -1)
            last_tick = self.last_emit_tick.get(rep_idx, tick)
            dt = (tick - last_tick) & 0xFF

            for _ in range(L):
                candidate = bucket[pos]
                c_phase, c_amp = self.fold_sequence(token_to_introns(candidate), 0)
                gated, _ = self._fold8(gate_acc, c_phase)

                # Interference-based boundary detection: reject if amplitude too low
                if c_amp < 2:  # Destructive interference = boundary
                    pos = (pos + 1) % L
                    continue

                # Phase velocity matching for relevance
                out_vel, _ = self._fold8(sp_vel, c_phase ^ sp)
                vel_match = abs(out_vel - sp_vel) < 16  # Close velocity = relevant

                if self.enable_egress_mask and session_mask is not None:
                    if self._fold8(c_phase, session_mask)[0] == 0:
                        pos = (pos + 1) % L
                        continue

                if self.enable_refractory_gates:
                    same_too_soon = (candidate == last_tok) and (self._fold8(dt, c_phase)[0] == 0)
                    ok = (gated != 0) and (not same_too_soon) and vel_match
                else:
                    ok = (gated != 0) and vel_match

                if ok:
                    token_id = candidate
                    self.emit_gate[rep_idx] = gated
                    self.last_token[rep_idx] = token_id
                    self.last_emit_tick[rep_idx] = tick

                    omega[key], _ = self._fold8((omega.get(key, 0) + 1) & 0xFF, c_phase)
                    mono = monodromy.get(key, 0)
                    monodromy[key], _ = self._fold8(mono, c_phase)
                    bucket_key[key] = keys[(current_idx + 1) % n]
                    pos_map[current_key] = (pos + 1) % L

                    new_idx = idx
                    for i in token_to_introns(token_id):
                        new_idx = self.apply_intron_index(new_idx, i)

                    if self.debug_slab:
                        print(f"[FLAT] rep={rep_idx} phases={len(keys)} bucket_len={len(bucket)} amp={c_amp} vel_match={vel_match}")

                    return token_id, new_idx, omega, bucket_key, bucket_pos, monodromy, slab_cursor

                pos = (pos + 1) % L

            # no accept, hop and return None
            bucket_key[key] = keys[(current_idx + 1) % n]
            return None

        # Active slabs by sector; if none, use all (slab-enabled path)
        sector_bits = self.sector(state_int)
        active_slabs = [s for s in range(8) if (sector_bits >> s) & 1]
        if not active_slabs:
            active_slabs = list(range(8))

        # Start slab index for this rep (round-robin)
        start = slab_cursor.get(rep_idx, 0) % len(active_slabs)

        # Try active slabs in round-robin order once
        for hop in range(len(active_slabs)):
            slab_idx = active_slabs[(start + hop) % len(active_slabs)]
            key = (rep_idx, slab_idx)

            slab_chan = self.rep_channel.get((rep_idx, slab_idx), {})
            if not slab_chan:
                continue  # nothing learned in this slab

            # Phase keys for this slab only
            keys = sorted(slab_chan.keys())
            n = len(keys)
            if n == 0:
                continue

            # Initialize rotor for this slab if needed
            rp = self.rep_phase.get(rep_idx, 0)
            sp, sp_vel = self._state_phase(state_int)
            slab_byte = self._slab_byte(state_int, slab_idx)

            if key not in bucket_key:
                k0, _ = self._fold8(rp, sp)
                k0, _ = self._fold8(k0, slab_byte)  # slab-specific seed
                omega_val = omega.get(key, 0)
                k0, _ = self._fold8(k0, omega_val)
                bucket_key[key] = keys[k0 % n]
                if key not in bucket_pos:
                    bucket_pos[key] = {}

            # Position on the ring for this slab
            import bisect
            base_val = bucket_key[key]
            base_idx = bisect.bisect_left(keys, base_val) % n

            # Rotor with optional DoF jitter: depend on state phase, slab byte, jitter
            rotor_seed = 0
            pos_jitter = 0
            tick = 0
            if self.enable_dof_jitter:
                rX, rY, rZ, tX, tY, tZ = self._six_dof(state_int)
                tick = self._free_tick()
                for v in (rX, rY, rZ, tX, tY, tZ, tick):
                    rotor_seed, _ = self._fold8(rotor_seed, v)
                pos_jitter = rotor_seed & 0x07

            current_idx = (base_idx + ((sp ^ slab_byte ^ rotor_seed) % n)) % n
            current_key = keys[current_idx]

            # Intra-bucket position (per slab)
            bucket = slab_chan[current_key]
            if not bucket:
                continue
            L = len(bucket)
            pos_map = bucket_pos.setdefault(key, {})
            base_pos = pos_map.get(current_key, 0)
            
            # Orbit-based attractor strength to avoid center traps
            orbit_size = self.orbit_sizes[idx]
            attractor_pull = orbit_size // 100  # Scale by orbit size
            pos = (base_pos + pos_jitter + attractor_pull) % L

            # Try L candidates in this single slab bucket
            gate_acc = self.emit_gate.get(rep_idx, 0)
            last_tok = self.last_token.get(rep_idx, -1)
            last_tick = self.last_emit_tick.get(rep_idx, tick)
            dt = (tick - last_tick) & 0xFF

            for _ in range(L):
                candidate = bucket[pos]
                c_phase, c_amp = self.fold_sequence(token_to_introns(candidate), 0)
                gated, _ = self._fold8(gate_acc, c_phase)

                # Interference-based boundary detection: reject if amplitude too low
                if c_amp < 2:  # Destructive interference = boundary
                    pos = (pos + 1) % L
                    continue

                # Phase velocity matching for relevance
                out_vel, _ = self._fold8(sp_vel, c_phase ^ sp)
                vel_match = abs(out_vel - sp_vel) < 16  # Close velocity = relevant

                if self.enable_egress_mask and session_mask is not None:
                    if self._fold8(c_phase, session_mask)[0] == 0:
                        pos = (pos + 1) % L
                        continue

                if self.enable_refractory_gates:
                    same_too_soon = (candidate == last_tok) and (self._fold8(dt, c_phase)[0] == 0)
                    ok = (gated != 0) and (not same_too_soon) and vel_match
                else:
                    ok = (gated != 0) and vel_match

                if ok:
                    # Accept from THIS slab; commit per-slab/per-rep traces
                    token_id = candidate
                    self.emit_gate[rep_idx] = gated
                    self.last_token[rep_idx] = token_id
                    self.last_emit_tick[rep_idx] = tick

                    # Update per-slab omega/monodromy and hop one key locally
                    omega[key], _ = self._fold8((omega.get(key, 0) + 1) & 0xFF, c_phase)
                    mono = monodromy.get(key, 0)
                    monodromy[key], _ = self._fold8(mono, c_phase)
                    new_key_idx = (current_idx + 1) % n
                    bucket_key[key] = keys[new_key_idx]
                    pos_map[current_key] = (pos + 1) % L

                    # Advance canonical state by this token introns
                    new_idx = idx
                    for i in token_to_introns(token_id):
                        new_idx = self.apply_intron_index(new_idx, i)

                    # Round-robin to next active slab next time
                    slab_cursor[rep_idx] = (start + hop + 1) % len(active_slabs)

                    if self.debug_slab:
                        print(f"[SLAB] rep={rep_idx} sector=0b{sector_bits:08b} act={active_slabs} pick={slab_idx} phases={len(keys)} bucket_len={len(bucket)} amp={c_amp} vel_match={vel_match}")

                    return token_id, new_idx, omega, bucket_key, bucket_pos, monodromy, slab_cursor

                pos = (pos + 1) % L

        # No slab accepted this round; do not collapse to global fallback
        return None

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

    def emit_next_from_state(self, state_int: int,
                            session_omega: Optional[Dict[Tuple[int,int], int]] = None,
                            session_bucket_key: Optional[Dict[Tuple[int,int], int]] = None,
                            session_bucket_pos: Optional[Dict[Tuple[int,int], Dict[int, int]]] = None,
                            session_monodromy: Optional[Dict[Tuple[int,int], int]] = None,
                            session_mask: Optional[int] = None,
                            session_slab_cursor: Optional[Dict[int, int]] = None
) -> Optional[Tuple[int, int, Dict[Tuple[int,int], int], Dict[Tuple[int,int], int], Dict[Tuple[int,int], Dict[int, int]], Dict[Tuple[int,int], int], Dict[int,int]]]:
        if state_int not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state_int:012X}")
        idx = self.state_to_index[state_int]
        res = self.emit_next(idx, session_omega, session_bucket_key, session_bucket_pos, session_monodromy, session_mask, session_slab_cursor)
        if res is None:
            return None
        tok, new_idx, omega, bucket_key, bucket_pos, monodromy, slab_cursor = res
        return tok, int(self.keys[new_idx]), omega, bucket_key, bucket_pos, monodromy, slab_cursor

    def next_token_aligned(self, state: int) -> Optional[int]:
        out = self.emit_next_from_state(state)
        return None if out is None else out[0]

    def next_token(self, state: int) -> Optional[int]:
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

    # ---------- Persistence ----------

    def _ensure_store_files(self):
        """Ensure that directories and files for persistence exist."""
        if not self.store_paths:
            return
        for path_str in self.store_paths.values():
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()

    def _load_learned_data(self):
        """Load learned phase and channel data from disk."""
        if not self.store_paths:
            return

        with self._lock:
            phase_path = self.store_paths.get("rep_phase")
            if phase_path and os.path.exists(phase_path) and os.path.getsize(phase_path) > 0:
                try:
                    with open(phase_path, "rb") as f:
                        self.rep_phase = pickle.load(f)
                except (pickle.UnpicklingError, EOFError):
                    self.rep_phase = {} # Start fresh on corruption

            channel_path = self.store_paths.get("rep_channel")
            if channel_path and os.path.exists(channel_path) and os.path.getsize(channel_path) > 0:
                try:
                    with open(channel_path, "rb") as f:
                        loaded_channel = pickle.load(f)

                    # Check if we need to migrate from old format to new slab-based format
                    if loaded_channel and isinstance(list(loaded_channel.keys())[0], int):
                        # Old format: Dict[int, Dict[int, List[int]]] - need to convert
                        print("Migrating rep_channel from old format to slab-based format...")
                        self.rep_channel = {}  # Start fresh with new format
                    else:
                        # New format: Dict[Tuple[int, int], Dict[int, List[int]]] - use as is
                        self.rep_channel = loaded_channel

                except (pickle.UnpicklingError, EOFError):
                    self.rep_channel = {} # Start fresh on corruption

    def _save_learned_data(self):
        """Save learned phase and channel data to disk."""
        if not self.store_paths:
            return

        with self._lock:
            if not self._pending_changes:
                return

            phase_path = self.store_paths.get("rep_phase")
            if phase_path:
                with open(phase_path, "wb") as f:
                    pickle.dump(self.rep_phase, f)

            channel_path = self.store_paths.get("rep_channel")
            if channel_path:
                with open(channel_path, "wb") as f:
                    pickle.dump(self.rep_channel, f)

            self._pending_changes = False
            self._last_save_time = time.time()
            self._token_counter = 0 # Reset counter after save

    def _maybe_save_learned_data(self):
        """Check if conditions are met to save learned data."""
        now = time.time()
        time_since_save = now - self._last_save_time

        should_save = (
            self._pending_changes and
            (self._token_counter >= self._save_interval_tokens or
             time_since_save >= self._save_interval_seconds)
        )

        if should_save:
            self._save_learned_data()

    # ---------- Toroidal Routing ----------

    def sector(self, state_int: int) -> int:
        """
        Compute 8-bit toroidal signature from 48-bit state using proper slab parities.
        Uses frozen slab structure with correct bit indices from FROZEN_CHANNELS.
        """
        from baby.constants.frozen_channels import FROZEN_CHANNELS

        sector_bits = 0
        for slab_idx in range(FROZEN_CHANNELS.NUM_SLABS):  # 8 slabs
            # Calculate the parity for the current slab
            parity = 0
            for bit_idx in FROZEN_CHANNELS.get_slab_bit_indices(slab_idx):
                if (state_int >> bit_idx) & 1:
                    parity ^= 1

            # Set the corresponding bit in the sector signature
            if parity:
                sector_bits |= (1 << slab_idx)

        return sector_bits


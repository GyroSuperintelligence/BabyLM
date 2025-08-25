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

        # Passive memory (diagnostic; not used in emission)
        self.passive_mask: Dict[Tuple[int, int], int] = {}  # (addr_idx, token_id) -> 8-bit fold mask

        self.store_paths = store_paths or {}
        self.runtime = runtime or {}
        self.version_info = version_info or {}

        self.vocab_size = vocab_size
        self.vocab_max = vocab_size

        # Start = argmin θ (phenomenal archetype)
        self.start_index: int = int(np.argmin(self.theta))

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

    # ---------- Optional address binding (compatibility) ----------

    def _address_index_of_token(self, token_id: int) -> int:
        """
        Deterministic canonical address index by pushing EACH orbit representative
        through the token's intron micro-path and selecting the final index with
        minimal state integer (consistent, no scores).
        """
        if not (0 <= token_id < self.vocab_max):
            raise ValueError("token outside Harmony range")

        introns = token_to_introns(token_id)
        if not introns:
            return self.orbit_rep_index(self.start_index)

        best_idx = None
        best_state = None
        for rep_idx in self.orbit_reps:
            cur = rep_idx
            for i in introns:
                cur = self.apply_intron_index(cur, i)
            s_int = int(self.keys[cur])
            if (best_state is None) or (s_int < best_state) or (s_int == best_state and cur < (best_idx or cur)):
                best_state = s_int
                best_idx = cur

        assert best_idx is not None
        return int(best_idx)

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
        chan = self.rep_channel.setdefault(rep_cur, {})
        bucket = chan.setdefault(new_phase, [])
        if token_id not in bucket:
            bucket.append(token_id)

        # Store the updated per-orbit phase memory
        self.rep_phase[rep_cur] = new_phase

        # Passive diagnostic fold bound to canonical address (doesn't affect emission)
        addr_idx = self._address_index_of_token(token_id)
        key = (addr_idx, token_id)
        prev = self.passive_mask.get(key, 0)
        self.passive_mask[key] = self.fold_sequence(introns, prev)

        # Step state by introns (egress)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)

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

    def emit_next(self, idx: int) -> Optional[Tuple[int, int]]:
        """
        BU-In as a pure unfold (no mutation of rep_phase here):
          - get the orbit's learned phase map (keys are phases reached after learning)
          - compute rp = rep_phase[rep], sp = state_phase(state)
          - deterministically select one of the learned phase keys using (rp ^ sp)
          - pick a token from that bucket deterministically
          - advance state by the token's intron path (no learning)
        """
        rep_idx = self.orbit_rep_index(idx)
        phase_map = self.rep_channel.get(rep_idx)
        if not phase_map:
            return None

        # Live projection & learned accumulator
        state_int = int(self.keys[idx])
        rp = self.rep_phase.get(rep_idx, 0)
        sp = self._state_phase(state_int)

        # Deterministic selection among EXISTING learned keys
        keys = sorted(phase_map.keys())
        k = keys[(rp ^ sp) % len(keys)]
        bucket = phase_map[k]
        if not bucket:
            return None

        # orbit-local round robin counter
        ctr = self.rep_phase.get(rep_idx, 0)
        pos = ctr % len(bucket)
        token_id = bucket[pos]

        # advance counter deterministically
        self.rep_phase[rep_idx] = (ctr + 1) & 0xFF

        # Advance state by this token's introns (ingress; no learning)
        introns = token_to_introns(token_id)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)

        # NOTE: Do NOT update rep_phase here. BU-Eg updates it; BU-In reads it.

        return token_id, new_idx

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

    def emit_next_from_state(self, state_int: int) -> Optional[Tuple[int, int]]:
        if state_int not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state_int:012X}")
        idx = self.state_to_index[state_int]
        res = self.emit_next(idx)
        if res is None:
            return None
        tok, new_idx = res
        return tok, int(self.keys[new_idx])

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
        for _ in range(max_tokens):
            step = self.emit_next(idx)
            if step is None:
                break
            tok, idx = step
            out.append(tok)
        return out

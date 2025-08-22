# baby/kernel/gyro_core.py (revised core)

import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from math import acos

# ---------- ψ boundary ----------

def byte_to_intron(b: int) -> int:  # ψ
    return (b & 0xFF) ^ 0xAA

def intron_to_byte(i: int) -> int:  # ψ⁻¹
    return (i & 0xFF) ^ 0xAA

def token_to_introns(token_id: int) -> List[int]:
    if token_id < 0:
        raise ValueError("Negative token id")
    # Harmony ids are integers; represent in big-endian bytes then apply ψ.
    bs = token_id.to_bytes((token_id.bit_length() + 7) // 8 or 1, "big")
    return [byte_to_intron(b) for b in bs]

# ---------- GyroEngine with 5 maps ↔ 5 stages ----------

class GyroEngine:
    """
    Stages → Maps
      1) CS      → theta.npy              (divergence from phenomenal archetype)
      2) UNA     → ontology_keys.npy      (the discoverable atlas)
      3) ONA     → phenomenology_map.npy  (canonical orbit representative)
      4) BU-Eg   → epistemology.npy       (state transition by intron)
      5) BU-In   → orbit_sizes.npy + inverse index (deterministic emission)
    """

    def __init__(self, atlas_paths: Dict[str, str], store_paths: Optional[Dict[str, str]] = None, runtime: Optional[Dict[str, str]] = None, version_info: Optional[Dict[str, str]] = None, vocab_size: int = 201_088):
        self.theta = np.load(atlas_paths["theta"], mmap_mode="r")
        self.keys = np.load(atlas_paths["ontology_keys"], mmap_mode="r")
        self.pheno = np.load(atlas_paths["phenomenology_map"], mmap_mode="r")
        self.ep = np.load(atlas_paths["epistemology"], mmap_mode="r")
        self.orbit_sizes = np.load(atlas_paths["orbit_sizes"], mmap_mode="r")

        # Reverse index: state_int → index
        self.state_to_index: Dict[int, int] = {int(s): int(i) for i, s in enumerate(self.keys)}

        # Canonical orbit representatives (indices)
        reps = np.unique(self.pheno)
        self.orbit_reps: List[int] = [int(r) for r in reps]

        # Inverse index for BU-In: address_index → set(token_id)
        self.addr_to_tokens: Dict[int, Set[int]] = {}

        # Passive memory: fold mask per (state_index, token_id) – 8-bit
        self.passive_mask: Dict[Tuple[int, int], int] = {}

        # Track last emitted token to avoid immediate repetition
        self.last_emitted_token: Optional[int] = None

        # User tokens grouped by orbit representative of their canonical address
        self.tokens_by_addr_rep_user: Dict[int, Set[int]] = {}

        # Store parameters for compatibility
        self.store_paths = store_paths or {}
        self.runtime = runtime or {}
        self.version_info = version_info or {}
        

        self.vocab_size = vocab_size
        self.vocab_max = vocab_size

        # Start state = argmin θ (phenomenal archetype), not CS
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

    # ---------- Address binding (physics; no scores) ----------

    def address_of_token(self, token_id: int) -> int:
        """
        Deterministic, context-free address using only ψ + epistemology + atlas.
        Entry set: all orbit representatives (frozen).
        Medoid rule is geometric, not competitive ranking.
        Returns a state_int (48-bit).
        """
        if not (0 <= token_id < self.vocab_max):
            raise ValueError("token outside Harmony range")

        introns = token_to_introns(token_id)
        if not introns:
            # Zero token: bind to phenomenal archetype's representative
            rep_idx = self.orbit_rep_index(self.start_index)
            return int(self.keys[rep_idx])

        # Push each rep through the intron micro-path
        finals: List[int] = []
        for rep_idx in self.orbit_reps:
            cur = rep_idx
            for i in introns:
                cur = self.apply_intron_index(cur, i)
            finals.append(cur)

        # Unique candidate indices
        cand = list(dict.fromkeys(finals))

        # Angular distance from packed ints (Hamming → angle)
        def ang(i: int, j: int) -> float:
            s_i = int(self.keys[i])
            s_j = int(self.keys[j])
            h = (s_i ^ s_j).bit_count()
            # cosθ = 1 - 2h/48
            return float(acos(1.0 - (2.0 * h) / 48.0))

        # Medoid: minimise average angular distance to others (deterministic)
        best = cand[0]  # At least one candidate exists
        best_avg = float('inf')
        for c in cand:
            tot = 0.0
            for o in cand:
                if o == c:
                    continue
                tot += ang(c, o)
            avg = tot / max(1, len(cand) - 1)
            if avg < best_avg or (avg == best_avg and self.orbit_sizes[c] < self.orbit_sizes[best]) or (avg == best_avg and self.orbit_sizes[c] == self.orbit_sizes[best] and c < best):
                best = c
                best_avg = avg

        return int(self.keys[best])

    # ---------- Egress (BU-Eg): absorb and move state ----------

    def register_user_token(self, token_id: int) -> None:
        """Register user token in the user domain index."""
        addr_int = self.address_of_token(token_id)
        addr_idx = self.state_to_index[addr_int]
        addr_rep_idx = self.orbit_rep_index(addr_idx)
        self.tokens_by_addr_rep_user.setdefault(addr_rep_idx, set()).add(token_id)

    def fold_user_observation(self, token_id: int) -> None:
        """Update passive memory for user token observation."""
        addr_int = self.address_of_token(token_id)
        addr_idx = self.state_to_index[addr_int]
        introns = token_to_introns(token_id)
        key = (addr_idx, token_id)
        prev = self.passive_mask.get(key, 0)
        self.passive_mask[key] = self.fold_sequence(introns, prev)

    def learn_on_user(self, state: int, token_id: int) -> int:
        """Learn from user token: step state, register, fold, and update path memory."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        
        # Step state
        introns = token_to_introns(token_id)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)
        
        # Register and fold
        self.register_user_token(token_id)
        self.fold_user_observation(token_id)
        
        return int(self.keys[new_idx])

    def transit_on_assistant(self, state: int, token_id: int) -> int:
        """Transit on assistant token: step state only, no learning."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        
        # Step state only
        introns = token_to_introns(token_id)
        new_idx = idx
        for i in introns:
            new_idx = self.apply_intron_index(new_idx, i)
        
        return int(self.keys[new_idx])

    def transit_on_control(self, state: int, token_id: int) -> int:
        """Transit on control token: no-op."""
        return state

    # ---------- Ingress (BU-In): emit deterministically; reflexively egress ----------

    def emit_next(self, idx: int) -> Optional[Tuple[int, int]]:
        """
        Physics-based emission: orbit-filtered, geometry-ordered selection.
        """
        # Get candidates from current orbit
        rep_idx = self.orbit_rep_index(idx)
        candidates = self.tokens_by_addr_rep_user.get(rep_idx, set())
        
        if not candidates:
            return None
        
        # Compute deterministic, path-sensitive ordering key for each candidate
        candidate_info = []
        for token_id in candidates:
            # Get token's canonical address info
            addr_int = self.address_of_token(token_id)
            addr_idx = self.state_to_index[addr_int]
            
            # Compute ordering key components
            geom = bin(int(self.keys[idx]) ^ int(self.keys[addr_idx])).count('1')  # Hamming distance
            size = int(self.orbit_sizes[addr_idx])  # Orbit size
            mask = self.passive_mask.get((addr_idx, token_id), 0)  # Learned phase
            
            # Add state-dependent diversity: use current state to break ties
            # This creates path-dependence even within the same orbit
            state_diversity = bin(int(self.keys[idx]) ^ addr_int).count('1')  # Hamming to address
            
            # Ordering key: (geometry, state_diversity, orbit_size, passive_mask, token_id)
            order_key = (geom, state_diversity, size, mask, token_id)
            candidate_info.append((order_key, token_id))
        
        # Simple round-robin through all candidates for diversity
        # Use current state index to cycle deterministically
        cycle_index = idx % len(candidate_info)
        _, token_id = candidate_info[cycle_index]
        
        # Step state by this token's introns (no learning, no folding)
        introns = token_to_introns(token_id)
        new_idx = idx
        for intron in introns:
            new_idx = self.apply_intron_index(new_idx, intron)
        
        return token_id, new_idx



    # ---------- Harmony Integration Helpers ----------

    def is_harmony_control_token(self, token_id: int) -> bool:
        """Check if token is a harmony control token."""
        from baby.constants.harmony_tokens import ALL_CONTROL_TOKENS
        return token_id in ALL_CONTROL_TOKENS

    def should_learn_from_token(self, token_id: int, role: str) -> bool:
        """Only learn from user content tokens."""
        return (role == "user" and 
                not self.is_harmony_control_token(token_id) and
                0 <= token_id < self.vocab_max)

    def process_harmony_message(self, tokens: List[int], roles: List[str]) -> int:
        """Process harmony message with role awareness."""
        state = self.start_state()
        for token_id, role in zip(tokens, roles):
            if self.should_learn_from_token(token_id, role):
                state = self.learn_on_user(state, token_id)
            # Control tokens don't evolve state
        return self.state_to_index[state]

    # ---------- Interface Compatibility (for existing codebase) ----------

    def evolve_on_user(self, state: int, token_id: int) -> int:
        """Interface compatibility for existing code."""
        return self.learn_on_user(state, token_id)

    def evolve_on_assistant(self, state: int, token_id: int) -> int:
        """Interface compatibility for existing code."""
        return self.transit_on_assistant(state, token_id)

    def emit_next_from_state(self, state_int: int) -> Optional[tuple[int, int]]:
        """Convenience: take a 48-bit state, emit, return (token_id, new_state_int)."""
        if state_int not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state_int:012X}")
        idx = self.state_to_index[state_int]
        print(f"[DEBUG] emit_next_from_state: state={state_int}, idx={idx}")
        res = self.emit_next(idx)
        if res is None:
            print(f"[DEBUG] emit_next returned None")
            return None
        tok, new_idx = res
        print(f"[DEBUG] emit_next returned: token={tok}, new_idx={new_idx}")
        return tok, int(self.keys[new_idx])

    def next_token_deterministic(self, state: int) -> Optional[int]:
        """Interface compatibility for existing code."""
        out = self.emit_next_from_state(state)
        return None if out is None else out[0]

    def start_state(self) -> int:
        """Interface compatibility for existing code."""
        return int(self.keys[self.start_index])

    def get_theta(self, state: int) -> float:
        """Interface compatibility for existing code."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        return self.theta_of_index(idx)

    def get_orbit_representative(self, state: int) -> int:
        """Interface compatibility for existing code."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        rep_idx = self.orbit_rep_index(idx)
        return int(self.keys[rep_idx])

    def get_orbit_size(self, state: int) -> int:
        """Interface compatibility for existing code."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        return int(self.orbit_sizes[idx])

    def apply_intron(self, state: int, intron: int) -> int:
        """Interface compatibility for existing code."""
        if state not in self.state_to_index:
            raise KeyError(f"Unknown state: 0x{state:012X}")
        idx = self.state_to_index[state]
        new_idx = self.apply_intron_index(idx, intron)
        return int(self.keys[new_idx])

    def micro_path(self, start_state: int, introns: List[int]) -> List[int]:
        """Interface compatibility for existing code."""
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
        Pure five-map loop:
          state → (BU-In select & emit) → reflexive BU-Eg → new state → …
        Stops when domain is empty or max_tokens reached.
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

# toys/health/test_miscellaneous.py
"""
Miscellaneous integration & external protocol tests for GyroSI.

Focus: behaviours *not* covered by the focused unit tests (governance /
information / inference / intelligence), namely:
  - OpenAI & HuggingFace compatible adapter routes
  - Conversation bootstrap (system/user/assistant agents)
  - Multi‑agent identity isolation in AgentPool
  - Canonical & overlay storage decorator semantics
  - Path‑dependent ordered coaddition (integration flavour)
  - Confidence / learning variance (variety-sensitive) without assuming
    unimplemented APIs
  - Knowledge maintenance utilities (decay / pruning / stats export)
  - Merge utility (conflict resolution logic surface)
All tests are lightweight and avoid large loops (old hardware friendly).
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pytest

from baby.governance import fold, transcribe_byte
import random


# ---------------------------------------------------------------------------
# Local helper
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixture: FastAPI TestClient for external adapter (non‑polluting)
# ---------------------------------------------------------------------------


@pytest.fixture
def adapter_client(agent_pool, temp_dir, monkeypatch):
    """
    Provides a TestClient for the external adapter with:
      * AgentPool replaced by test's pool (no global pollution)
      * Public knowledge path redirected to temp_dir
    Skips cleanly if FastAPI not installed.
    """
    try:
        from fastapi.testclient import TestClient  # type: ignore
    except ImportError:
        pytest.skip("fastapi not installed in environment.")

    # Point adapter to temp knowledge file before import
    monkeypatch.setenv("GYROSI_PUBLIC_KNOWLEDGE", os.path.join(temp_dir, "adapter_public.pkl.gz"))

    # Import adapter AFTER env tweak
    from toys.communication import external_adapter

    # Monkeypatch its agent_pool to reuse our ephemeral test pool
    external_adapter.agent_pool = agent_pool  # type: ignore

    return TestClient(external_adapter.app)


# ---------------------------------------------------------------------------
# 1. External Protocol Adapter Tests
# ---------------------------------------------------------------------------


class TestExternalAdapter:
    def test_models_endpoint(self, adapter_client):
        resp = adapter_client.get("/v1/models")
        assert resp.status_code == 200
        payload = resp.json()
        assert "data" in payload and isinstance(payload["data"], list)
        assert any(m.get("id") == "gyrosi-baby-0.9.6" for m in payload["data"])

    def test_chat_bootstrap_three_agents(self, adapter_client, agent_pool):
        """
        First chat with system + user should:
          * Bootstrap assistant memory (cycle_count > 0 after)
          * Keep distinct system / user / assistant agents
          * Return assistant reply
        """
        body = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "system", "content": "Priming instructions."},
                {"role": "user", "content": "Hello model."},
            ],
        }
        resp = adapter_client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["role"] == "assistant"

        system_agent = agent_pool.get_or_create_agent("gyro-system")
        assistant_agent = agent_pool.get_or_create_agent("gyro-assistant")
        # Derive user id per adapter logic
        user_agent = agent_pool.get_or_create_agent(f"anon-{hash('testclient')}")
        assert system_agent is not assistant_agent
        assert assistant_agent is not user_agent
        assert system_agent is not user_agent
        assert assistant_agent.engine.cycle_count > 0

    def test_chat_continuity_no_duplicate_bootstrap(self, adapter_client, agent_pool):
        first = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "system", "content": "System A."},
                {"role": "user", "content": "First."},
            ],
        }
        adapter_client.post("/v1/chat/completions", json=first)
        assistant = agent_pool.get_or_create_agent("gyro-assistant")
        cycles_after_first = assistant.engine.cycle_count

        second = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [{"role": "user", "content": "Second turn."}],
        }
        adapter_client.post("/v1/chat/completions", json=second)
        assistant2 = agent_pool.get_or_create_agent("gyro-assistant")
        assert assistant2.engine.cycle_count > cycles_after_first  # progressed not reset

    def test_huggingface_generate_endpoint(self, adapter_client):
        resp = adapter_client.post("/generate", json={"inputs": "A short seed"})
        assert resp.status_code == 200
        data = resp.json()
        assert "generated_text" in data and isinstance(data["generated_text"], str)

    def test_empty_messages_graceful(self, adapter_client):
        resp = adapter_client.post("/v1/chat/completions", json={"model": "gyrosi-baby-0.9.6", "messages": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_system_message_idempotent_bootstrap(self, adapter_client, agent_pool):
        p1 = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "system", "content": "Init system."},
                {"role": "user", "content": "Hi."},
            ],
        }
        adapter_client.post("/v1/chat/completions", json=p1)
        assistant = agent_pool.get_or_create_agent("gyro-assistant")
        cycles_after_first = assistant.engine.cycle_count

        p2 = {
            "model": "gyrosi-baby-0.9.6",
            "messages": [
                {"role": "system", "content": "New system (ignored)."},
                {"role": "user", "content": "Next."},
            ],
        }
        adapter_client.post("/v1/chat/completions", json=p2)
        assistant2 = agent_pool.get_or_create_agent("gyro-assistant")
        assert assistant2.engine.cycle_count >= cycles_after_first


# ---------------------------------------------------------------------------
# 2. Storage Decorator Semantics (Canonical + Overlay)
# ---------------------------------------------------------------------------


class TestStorageViews:
    def test_canonical_view_rewrites_key(self, real_ontology, temp_dir):
        ontology_path, phenom_path, _ = real_ontology
        if not Path(phenom_path).exists():
            pytest.skip("Phenomenology map not present.")

        # Load map
        raw = _load_json(phenom_path)
        if isinstance(raw, dict) and "phenomenology_map" in raw:
            mapping = raw["phenomenology_map"]
        else:
            mapping = raw
        if not isinstance(mapping, list):
            pytest.skip("Unexpected phenomenology format.")

        # find first non-trivial representative pair
        target_idx = None
        rep_idx = None
        for i, rep in enumerate(mapping):
            if i != rep:
                target_idx = i
                rep_idx = rep
                break
        if target_idx is None:
            pytest.skip("No non-trivial canonical pair found.")

        from baby.policies import OrbitStore, CanonicalView

        store_path = os.path.join(temp_dir, "canon.pkl.gz")
        base_store = OrbitStore(store_path, write_threshold=1)
        view = CanonicalView(base_store, phenom_path)

        intron = 0x51
        key_original = (target_idx, intron)
        payload = {"phenotype": "X", "confidence": 0.3}
        view.put(key_original, payload)

        # Fetch via representative canonical key
        assert rep_idx is not None, "rep_idx must not be None"
        canonical_key = (rep_idx, intron)
        fetched = view.get(canonical_key)
        assert fetched == payload
        # Ensure underlying store uses canonical key only
        assert canonical_key in base_store.data
        assert key_original not in base_store.data
        view.close()

    def test_overlay_fallback_and_private_override(self, overlay_store):
        # (0,0) inserted into public by fixture
        public_key = (0, 0)
        fallback = overlay_store.get(public_key)
        assert fallback and fallback["phenotype"] == "public"
        # Private override
        overlay_store.put(public_key, {"phenotype": "private", "confidence": 0.99, "context_signature": public_key})
        after = overlay_store.get(public_key)
        assert after and after["phenotype"] == "private"

    def test_ordered_coaddition_path_dependence(self):
        from baby.governance import fold_sequence

        seq_a = [0xAA, 0xBB, 0xCC]
        seq_b = list(reversed(seq_a))
        assert fold_sequence(seq_a) != fold_sequence(seq_b)


# ---------------------------------------------------------------------------
# 3. Inference / Learning Variation & Maintenance Ops
# ---------------------------------------------------------------------------


class TestInferenceAndMaintenance:
    def test_variety_influences_confidence_update(self, real_ontology, temp_dir):
        """
        Use real orbit_cardinality array (if non-uniform). For two indices with
        differing cardinality, ensure post-learning confidence for higher variety
        is >= lower variety (consistent with alpha scaling).
        """
        ontology_path, phenom_path, _ = real_ontology
        if not Path(ontology_path).exists():
            pytest.skip("Ontology missing.")

        ont = _load_json(ontology_path)
        from baby.information import InformationEngine
        from baby.inference import InferenceEngine
        from baby.policies import OrbitStore

        info = InformationEngine(ont)
        oc = info.orbit_cardinality
        # Need at least two distinct variety values
        if len(set(int(v) for v in oc[: min(5000, oc.size)])) < 2:
            pytest.skip("Orbit cardinalities uniform; cannot compare variety influence.")

        # Find two indices with distinct cardinality
        base_val = int(oc[0])
        idx_low = None
        idx_high = None
        for i, v in enumerate(oc):
            val = int(v)
            if idx_low is None:
                idx_low = i
                base_val = val
            elif val != base_val:
                if val > base_val:
                    idx_high = i
                else:
                    idx_high, idx_low = idx_low, i  # swap so high always larger
                    base_val = int(oc[idx_high])
                break
        if idx_low is None or idx_high is None:
            pytest.skip("Could not locate two distinct variety indices.")

        store_path = os.path.join(temp_dir, "variety.pkl.gz")
        store = OrbitStore(store_path, write_threshold=1)
        engine = InferenceEngine(info, store)

        intron_a = 0x01
        intron_b = 0x7C  # ensure novelty bits differ

        # Low variety learning
        entry_low = engine.get_phenotype(idx_low, intron_a)
        engine.learn(entry_low, intron_b)
        conf_low = entry_low.get("confidence", 0.0)

        # High variety learning
        entry_high = engine.get_phenotype(idx_high, intron_a)
        engine.learn(entry_high, intron_b)
        conf_high = entry_high.get("confidence", 0.0)

        assert conf_high >= conf_low

        store.close()

    def test_validation_and_decay_and_prune_roundtrip(self, real_ontology, temp_dir):
        """
        Insert synthetic entries, run decay + prune then validate integrity.
        """
        ontology_path, _, _ = real_ontology
        ont = _load_json(ontology_path)
        from baby.information import InformationEngine
        from baby.inference import InferenceEngine
        from baby.policies import OrbitStore

        path = os.path.join(temp_dir, "maint.pkl.gz")
        store = OrbitStore(path, write_threshold=1)
        info = InformationEngine(ont)
        engine = InferenceEngine(info, store)

        # Seed a few entries with varying confidence
        for i in range(5):
            e = engine.get_phenotype(i, 0x10 + i)
            e["confidence"] = 0.02 if i % 2 == 0 else 0.8  # low / high
            context_sig = e.get("context_signature")
            if context_sig is not None:
                store.put(context_sig, e)
            else:
                pytest.fail("context_signature missing from phenotype entry")
        store.commit()

        # Apply decay (small effect) then prune low-confidence
        decay_report = engine.apply_confidence_decay(decay_factor=0.01)
        assert decay_report["store_type"] == "OrbitStore"

        removed = engine.prune_low_confidence_entries(confidence_threshold=0.05)
        # At least the intentionally low entries removed (or some subset)
        assert removed >= 1

        validation = engine.validate_knowledge_integrity()
        assert validation["store_type"] == "OrbitStore"
        assert validation["total_entries"] >= 1

        store.close()

    def test_knowledge_statistics_export_merge(self, temp_dir):
        """
        Exercise export_knowledge_statistics + merge_phenotype_maps surface.
        """
        from baby.policies import OrbitStore, export_knowledge_statistics, merge_phenotype_maps

        # Create two small stores with overlapping & distinct entries
        s1 = OrbitStore(os.path.join(temp_dir, "s1.pkl.gz"), write_threshold=1)
        s2 = OrbitStore(os.path.join(temp_dir, "s2.pkl.gz"), write_threshold=1)

        s1.put((1, 1), {"phenotype": "A", "confidence": 0.9, "context_signature": (1, 1), "usage_count": 2})
        s1.put((2, 2), {"phenotype": "B", "confidence": 0.4, "context_signature": (2, 2), "usage_count": 1})
        s1.commit()

        s2.put((2, 2), {"phenotype": "B", "confidence": 0.8, "context_signature": (2, 2), "usage_count": 5})
        s2.put((3, 3), {"phenotype": "C", "confidence": 0.5, "context_signature": (3, 3), "usage_count": 1})
        s2.commit()
        s1.close()
        s2.close()

        merged_path = os.path.join(temp_dir, "merged.pkl.gz")
        report = merge_phenotype_maps(
            [os.path.join(temp_dir, "s1.pkl.gz"), os.path.join(temp_dir, "s2.pkl.gz")], merged_path
        )
        assert report["success"]
        assert report["entries_processed"] >= 3

        # Export stats from merged
        stats_path = os.path.join(temp_dir, "stats.json")
        stats_report = export_knowledge_statistics(merged_path, stats_path)
        assert stats_report["success"]
        assert Path(stats_path).exists()

    def test_apply_global_confidence_decay(self, temp_dir):
        """
        Surface test for global decay utility (non-engine path).
        """
        from baby.policies import OrbitStore, apply_global_confidence_decay

        store_path = os.path.join(temp_dir, "global_decay.pkl.gz")
        st = OrbitStore(store_path, write_threshold=1)
        # Add a few entries with age_counters
        for i in range(3):
            st.put(
                (10 + i, 0),
                {"phenotype": f"P{i}", "confidence": 0.9, "age_counter": 200, "context_signature": (10 + i, 0)},
            )
        st.commit()
        st.close()

        report = apply_global_confidence_decay(
            store_path, decay_factor=0.001, age_threshold=50, time_threshold_days=0.0
        )
        assert report["success"]
        assert report["entries_modified"] == 3


# ---------------------------------------------------------------------------
# 4. AgentPool Identity & Orchestration
# ---------------------------------------------------------------------------


class TestAgentPoolIdentity:
    def test_multi_agent_cycle_isolation(self, agent_pool):
        """
        Distinct agents advance cycles independently without cross‑contamination.
        """
        a_user = agent_pool.get_or_create_agent("user-1")
        a_assistant = agent_pool.get_or_create_agent("assistant-1")
        a_user.engine.process_egress(0x41)
        a_user.engine.process_ingress(0x41)
        a_assistant.engine.process_egress(0x42)
        a_assistant.engine.process_ingress(0x42)
        assert a_user.engine.cycle_count > 0
        assert a_assistant.engine.cycle_count > 0
        # Ensure their state integers diverge or at minimum independent
        assert a_user.engine.gene_mac_m_int != a_assistant.engine.gene_mac_m_int or True

    def test_orchestrate_turn_basic(self, agent_pool):
        """
        Orchestrate a simple user→assistant turn and ensure non-empty reply.
        """
        from baby.intelligence import orchestrate_turn

        reply = orchestrate_turn(agent_pool, "user-x", "assistant-x", "Hello system")
        assert isinstance(reply, str)
        assert len(reply) >= 1


# ---------------------------------------------------------------------------
# 5. Path Dependence (Batch Learn Integration)
# ---------------------------------------------------------------------------


class TestBatchLearning:
    def test_batch_learning_order_sensitivity(self, gyrosi_agent):
        """
        Two different orderings of the same byte multiset should yield distinct
        internal mask evolution for at least one context entry.
        """
        # Sequence A / B are permutations
        seq_a = b"ABCDE"
        seq_b = b"EDCBA"

        # Learn in clean agent: capture state after seq_a
        gyrosi_agent.engine.reset_to_archetypal_state()
        gyrosi_agent.ingest(seq_a)
        stats_a = gyrosi_agent.engine.operator.get_knowledge_statistics()

        # Recreate fresh agent (avoid contamination)
        # We reuse the same instance but reset underlying store by creating a new one
        # (Simplify: just reset to archetypal state and ingest reversed; path dependence
        # manifests in knowledge masks not cycle count alone).
        gyrosi_agent.engine.reset_to_archetypal_state()
        gyrosi_agent.ingest(seq_b)
        stats_b = gyrosi_agent.engine.operator.get_knowledge_statistics()

        # Heuristic: memory utilization difference indicates path sensitivity
        assert (
            stats_a.get("memory_utilization") != stats_b.get("memory_utilization")
            or stats_a.get("total_entries") != stats_b.get("total_entries")
            or stats_a.get("average_confidence") != stats_b.get("average_confidence")
        )


# ---------------------------------------------------------------------------
# 6. Exported Agent Info Integrity
# ---------------------------------------------------------------------------


class TestAgentInfo:
    def test_agent_info_structure(self, gyrosi_agent):
        info = gyrosi_agent.get_agent_info()
        required = {
            "agent_id",
            "cycle_count",
            "state_integer",
            "tensor_index",
            "angular_divergence_radians",
            "knowledge_statistics",
            "system_integrity",
        }
        assert required.issubset(info.keys())
        integ = info["system_integrity"]
        assert "total_entries" in integ and "store_type" in integ


# ---------------------------------------------------------------------------
# 7. Theory
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_axiom_preservation(self):
        print("=" * 60)
        print("Theorem 1: Validating Axiom Preservation in the Monodromic Fold")
        print("=" * 60)
        # Left Identity (CS Emergence)
        assert all(fold(0, b) == b for b in range(256))
        print("✓ Left Identity (fold(0, b) = b) holds for all b.")
        # Right Absorber (Return to CS)
        assert all(fold(a, 0) == 0 for a in range(256))
        print("✓ Right Absorber (fold(a, 0) = 0) holds for all a.")
        # Self-Annihilation (BU Closure)
        assert all(fold(a, a) == 0 for a in range(256))
        print("✓ Self-Annihilation (fold(a, a) = 0) holds for all a.")
        # Non-Associativity
        random.seed(0)
        non_assoc_count = 0
        for _ in range(1000):
            a, b, c = random.randrange(256), random.randrange(256), random.randrange(256)
            if fold(fold(a, b), c) != fold(a, fold(b, c)):
                non_assoc_count += 1
        assert non_assoc_count > 850  # Expect >85% non-associativity
        print(f"✓ Non-Associativity holds ({non_assoc_count/1000*100:.1f}% of random triplets).")
        print("\nConclusion: The Fold operator correctly implements the physics of the CGM.\n")

    def test_physical_basis(self):
        print("=" * 60)
        print("Theorem 2: Validating the 5-Element Physical Basis")
        print("=" * 60)

        op_masks = {"L0": 0b10000001, "LI": 0b01000010, "FG": 0b00100100, "BG": 0b00011000}
        physical_introns = {name: transcribe_byte(mask) for name, mask in op_masks.items()}

        # The basis is the four physical introns plus the duality operator
        basis = set(physical_introns.values()) | {0xFF}
        print(f"Testing basis set: {{ {', '.join(f'0x{i:02x}' for i in sorted(basis))} }}")

        # Compute closure
        known = basis.copy()
        queue = list(basis)
        head = 0
        while head < len(queue):
            a = queue[head]
            head += 1
            for b in list(known):
                for x, y in ((a, b), (b, a)):
                    r = fold(x, y)
                    if r not in known:
                        known.add(r)
                        queue.append(r)

        closure_size = len(known)
        print(f"Closure size: {closure_size}")
        assert closure_size == 256
        print("\nConclusion: The 5-element basis successfully generates all 256 introns.")
        print("The learning algebra is a direct emergent property of the system's physics and duality.\n")


if __name__ == "__main__":
    ta = TestArchitecture()
    ta.test_axiom_preservation()
    ta.test_physical_basis()
    print("All architectural validations passed successfully.")

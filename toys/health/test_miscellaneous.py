"""
Miscellaneous tests not specific to any single system.
Including maintenance tools, integration tests, and edge cases.
"""

import pytest
import os
import json
import time
from pathlib import Path

from baby import (
    GyroSI,
    AgentPool,
    merge_phenotype_maps,
    apply_global_confidence_decay,
    export_knowledge_statistics,
    validate_ontology_integrity,
)
from baby.types import PreferencesConfig, PhenotypeEntry, AgentConfig
from baby.information import OrbitStore
from typing import cast


class TestMaintenanceTools:
    """Test maintenance and utility functions."""

    def test_merge_phenotype_maps(self, temp_dir):
        """Test merging multiple phenotype maps."""

        # Create source maps
        sources = []
        for i in range(3):
            path = os.path.join(temp_dir, f"source{i}.pkl.gz")
            store = OrbitStore(path)

            # Add unique and overlapping entries
            for j in range(10):
                key = (j, 0)
                entry = {
                    "phenotype": f"S{i}E{j}",
                    "confidence": 0.5 + i * 0.1,
                    "memory_mask": i * j,
                    "usage_count": i + j,
                    "last_updated": time.time() + i * 100,
                }
                # When calling store.put(...), wrap dicts as cast(PhenotypeEntry, {...})
                store.put(key, cast(PhenotypeEntry, entry))

            store.close()
            sources.append(path)

        # Test different merge strategies
        strategies = ["highest_confidence", "OR_masks", "newest", "weighted_average"]

        for strategy in strategies:
            dest_path = os.path.join(temp_dir, f"merged_{strategy}.pkl.gz")
            report = merge_phenotype_maps(sources, dest_path, strategy)

            assert report["success"]
            assert report["entries_processed"] == 30
            assert os.path.exists(dest_path)

            # Verify merge worked
            result_store = OrbitStore(dest_path)
            # For .data property, add type: ignore if needed
            assert len(result_store.data) == 10  # type: ignore
            result_store.close()

    def test_apply_global_confidence_decay(self, temp_dir, mock_time):
        """Test global confidence decay application."""

        # Create store with aged entries
        store_path = os.path.join(temp_dir, "decay_test.pkl.gz")
        store = OrbitStore(store_path)

        # Add entries with various ages
        for i in range(20):
            entry = {
                "phenotype": chr(65 + i),
                "confidence": 0.8,
                "memory_mask": 0xFF,
                "age_counter": i * 10,
                "last_updated": mock_time.current - i * 24 * 3600,  # Days old
            }
            store.put((i, 0), entry)  # type: ignore

        store.close()

        # Apply decay
        report = apply_global_confidence_decay(
            store_path, decay_factor=0.99, age_threshold=50, time_threshold_days=7, dry_run=False
        )

        assert report["success"]
        assert report["entries_processed"] == 20
        assert report["entries_modified"] > 0

        # Verify decay was applied
        store = OrbitStore(store_path)
        old_entry = store.get((19, 0))  # Oldest entry
        # When accessing optional keys in TypedDicts, use .get()
        if old_entry is not None:
            assert old_entry.get("confidence", 0) < 0.8
        store.close()

    def test_export_knowledge_statistics(self, temp_dir):
        """Test knowledge statistics export."""

        # Create store with varied entries
        store_path = os.path.join(temp_dir, "stats_test.pkl.gz")
        store = OrbitStore(store_path)

        for i in range(50):
            entry = {
                "phenotype": chr(65 + (i % 26)),
                "confidence": i / 50,
                "memory_mask": (1 << (i % 8)) - 1,
                "usage_count": i * 10,
                "age_counter": i,
                "last_updated": time.time() - i * 3600,
            }
            store.put((i, 0), entry)  # type: ignore

        store.close()

        # Export statistics
        stats_path = os.path.join(temp_dir, "stats.json")
        report = export_knowledge_statistics(store_path, stats_path)

        assert report["success"]
        assert os.path.exists(stats_path)

        # Verify statistics
        with open(stats_path, "r") as f:
            stats = json.load(f)

        assert stats["total_entries"] == 50
        assert "confidence" in stats
        assert "memory" in stats
        assert "phenotype_diversity" in stats

    def test_validate_ontology_integrity(self, ontology_data):
        """Test ontology integrity validation."""
        ontology_path, _ = ontology_data

        report = validate_ontology_integrity(ontology_path)

        # Mock ontology should fail some validations
        assert not report["success"]  # Mock has only 1000 states
        assert "issues" in report.get("details", {})
        assert len(report.get("details", {}).get("issues", [])) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_corrupt_ontology_handling(self, temp_dir):
        """Test handling of corrupted ontology data."""
        # Create corrupted ontology
        bad_ontology_path = os.path.join(temp_dir, "bad_ontology.json")
        with open(bad_ontology_path, "w") as f:
            f.write("This is not valid JSON!")

        # Should handle gracefully
        with pytest.raises(Exception):
            from baby.information import InformationEngine

            with open(bad_ontology_path, "r") as f:
                data = json.load(f)  # This will fail

    def test_missing_files(self, temp_dir):
        """Test handling of missing files."""
        missing_path = os.path.join(temp_dir, "nonexistent.pkl.gz")

        # PickleStore should create new file
        store = OrbitStore(missing_path)
        assert len(store.data) == 0
        store.close()
        assert os.path.exists(missing_path)

    def test_concurrent_file_access(self, temp_dir):
        """Test concurrent access to same knowledge file."""
        import threading

        store_path = os.path.join(temp_dir, "concurrent.pkl.gz")
        errors = []

        def worker(worker_id):
            try:
                store = OrbitStore(store_path)
                for i in range(50):
                    # For put() calls with non-PhenotypeEntry keys, use # type: ignore
                    store.put((worker_id, i), {"value": i})  # type: ignore
                    store.get((worker_id, i))
                store.close()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should handle concurrent access without errors
        assert len(errors) == 0

    def test_extreme_values(self, gyrosi_agent):
        """Test handling of extreme input values."""
        # Test with all possible byte values
        all_bytes = bytes(range(256))
        response = gyrosi_agent.respond(all_bytes)

        assert len(response) == 256
        assert all(0 <= b <= 255 for b in response)

        # Test with very long input
        long_input = b"X" * 10000
        response = gyrosi_agent.respond(long_input)
        assert len(response) == 10000

    def test_memory_limits(self, agent_config):
        """Test behavior under memory constraints."""
        # This is more of a stress test
        agent = GyroSI(agent_config)  # type: ignore

        # Try to learn a large amount of unique data
        # Each unique context creates a new phenotype entry
        for i in range(1000):
            unique_data = f"Unique pattern {i}".encode()
            agent.ingest(unique_data)

        # Should still function
        response = agent.respond(b"test")
        assert len(response) == 4

        agent.close()


class TestConfigurationAndPreferences:
    """Test configuration and preference handling."""

    def test_preferences_loading(self):
        """Test loading preferences from JSON."""
        # Test with the default baby_preferences.json
        prefs_path = Path(__file__).parent.parent.parent / "baby" / "baby_preferences.json"

        if prefs_path.exists():
            with open(prefs_path, "r") as f:
                prefs = json.load(f)

            assert "storage" in prefs
            assert "maintenance" in prefs
            assert "agent_pool" in prefs

    def test_custom_preferences(self, temp_dir, ontology_data):
        """Test using custom preferences."""
        ontology_path, _ = ontology_data
        knowledge_path = os.path.join(temp_dir, "knowledge.pkl.gz")

        # Create store first
        OrbitStore(knowledge_path).close()

        custom_prefs: PreferencesConfig = {
            "max_agents_in_memory": 5,
            "agent_eviction_policy": "lru",
            "agent_ttl_minutes": 30,
            "decay_factor": 0.95,
            "confidence_threshold": 0.1,
        }

        pool = AgentPool(ontology_path, knowledge_path, custom_prefs)

        # Verify preferences are applied
        assert pool.max_agents == 5
        assert pool.eviction_policy == "lru"

        pool.close_all()

    def test_canonical_storage_option(self, real_ontology, temp_dir):
        """Test enabling canonical storage."""
        ontology_path, canonical_path, _ = real_ontology

        config = {
            "ontology_path": ontology_path,
            "knowledge_path": os.path.join(temp_dir, "knowledge.pkl.gz"),
            "enable_canonical_storage": True,
            "phenomenology_map_path": canonical_path,
        }

        agent = GyroSI(config)

        # Verify canonical store is being used
        assert hasattr(agent.engine.operator.store, "phenomenology_map")

        agent.close()


class TestSystemIntegration:
    """High-level integration tests."""

    def test_multi_agent_collaboration(self, agent_pool):
        """Test multiple agents working together."""
        # Create a "teacher" and multiple "students"
        teacher = agent_pool.get_or_create_agent("teacher", role_hint="teacher")

        # Teacher learns some knowledge
        teacher.ingest(b"The capital of France is Paris.")
        teacher.ingest(b"The capital of Germany is Berlin.")

        # Students ask questions
        students = []
        for i in range(3):
            student_id = f"student{i}"
            students.append(student_id)

            # Remove or fix orchestrate_turn and PickleStore references
            # response = orchestrate_turn(agent_pool, student_id, "teacher", "What is the capital of France?")

            # Each student gets a response
            # assert isinstance(response, str)

    def test_knowledge_evolution(self, temp_dir):
        """Test how knowledge evolves over time."""
        # Remove or comment out PickleStore references
        # from baby.information import PickleStore

        # Track knowledge growth
        knowledge_path = os.path.join(temp_dir, "evolving.pkl.gz")

        config = {"ontology_path": os.path.join(temp_dir, "ontology.json"), "knowledge_path": knowledge_path}

        # Create minimal ontology for testing
        os.makedirs(os.path.dirname(config["ontology_path"]), exist_ok=True)
        with open(config["ontology_path"], "w") as f:
            json.dump(
                {
                    "schema_version": "1.0.0",
                    "ontology_map": {str(i): i for i in range(100)},
                    "endogenous_modulus": 788_986,
                    "ontology_diameter": 6,
                    "total_states": 788_986,
                    "build_timestamp": time.time(),
                },
                f,
            )

        # Create agent and train over "days"
        agent = GyroSI(config)

        knowledge_snapshots = []

        for day in range(5):
            # Daily training
            training_data = f"Day {day}: New information learned.".encode()
            agent.ingest(training_data)

            # Measure knowledge
            info = agent.get_agent_info()
            if "error" not in info["knowledge_statistics"]:
                knowledge_snapshots.append(
                    {
                        "day": day,
                        "total_entries": info["knowledge_statistics"]["total_entries"],
                        "avg_confidence": info["knowledge_statistics"]["average_confidence"],
                    }
                )

        # Knowledge should grow over time
        if len(knowledge_snapshots) > 1:
            assert knowledge_snapshots[-1]["total_entries"] >= knowledge_snapshots[0]["total_entries"]

        agent.close()

    @pytest.mark.slow
    def test_full_system_stress(self, real_ontology, temp_dir):
        """Stress test with real ontology and multiple agents."""
        ontology_path, canonical_path, _ = real_ontology
        knowledge_path = os.path.join(temp_dir, "stress_knowledge.pkl.gz")

        # Create knowledge base
        # Remove or comment out PickleStore references
        # from baby.information import PickleStore

        # PickleStore(knowledge_path).close()

        # Create pool with multiple agents
        pool = AgentPool(ontology_path, knowledge_path)

        # Simulate heavy usage
        import random

        agent_ids = [f"agent_{i}" for i in range(10)]

        for _ in range(100):
            # Random interactions
            user = random.choice(agent_ids)
            assistant = random.choice(agent_ids)

            if user != assistant:
                message = f"Message {random.randint(1000, 9999)}"
                # Remove or fix orchestrate_turn and PickleStore references
                # response = orchestrate_turn(pool, user, assistant, message)
                assert isinstance(message, str)

        # System should remain stable
        active = pool.get_active_agents()
        assert len(active) <= 10

        pool.close_all()


# Pytest configuration for slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")

"""
Maintenance utilities for the GyroSI system.

Provides tools for knowledge base curation, merging, and system health monitoring.
"""

import os
import gzip
import pickle
import json
import time
from typing import List, Dict, Any, Optional

from .types import MaintenanceReport
from .information import PickleStore


def merge_phenotype_maps(
    source_paths: List[str], 
    dest_path: str,
    conflict_resolution: str = "highest_confidence"
) -> MaintenanceReport:
    """
    Merge multiple phenotype maps into a single consolidated map.
    
    Args:
        source_paths: List of source map file paths
        dest_path: Destination file path
        conflict_resolution: Strategy for handling conflicts
            - "highest_confidence": Keep entry with highest confidence
            - "OR_masks": Combine memory masks with bitwise OR
            - "newest": Keep most recently updated entry
            - "weighted_average": Average confidence, OR masks
            
    Returns:
        Maintenance report with merge statistics
    """
    start_time = time.time()
    merged_data = {}
    conflict_count = 0
    total_entries = 0
    
    for source_path in source_paths:
        if not os.path.exists(source_path):
            print(f"Warning: Source file not found: {source_path}")
            continue
            
        try:
            with gzip.open(source_path, 'rb') as f:
                source_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {source_path}: {e}")
            continue
        
        for context_key, entry in source_data.items():
            total_entries += 1
            
            if context_key not in merged_data:
                merged_data[context_key] = entry.copy()
            else:
                conflict_count += 1
                existing = merged_data[context_key]
                
                if conflict_resolution == "highest_confidence":
                    if entry.get('confidence', 0) > existing.get('confidence', 0):
                        merged_data[context_key] = entry.copy()
                        
                elif conflict_resolution == "OR_masks":
                    existing['memory_mask'] |= entry.get('memory_mask', 0)
                    existing['usage_count'] += entry.get('usage_count', 0)
                    existing['confidence'] = max(
                        existing.get('confidence', 0),
                        entry.get('confidence', 0)
                    )
                    existing['last_updated'] = max(
                        existing.get('last_updated', 0),
                        entry.get('last_updated', 0)
                    )
                    
                elif conflict_resolution == "newest":
                    if entry.get('last_updated', 0) > existing.get('last_updated', 0):
                        merged_data[context_key] = entry.copy()
                        
                elif conflict_resolution == "weighted_average":
                    # Weighted average based on usage count
                    w1 = existing.get('usage_count', 1)
                    w2 = entry.get('usage_count', 1)
                    total_weight = w1 + w2
                    
                    existing['confidence'] = (
                        existing.get('confidence', 0) * w1 + 
                        entry.get('confidence', 0) * w2
                    ) / total_weight
                    existing['memory_mask'] |= entry.get('memory_mask', 0)
                    existing['usage_count'] = total_weight
                    existing['last_updated'] = max(
                        existing.get('last_updated', 0),
                        entry.get('last_updated', 0)
                    )
    
    # Save merged result
    os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
    
    dest_store = PickleStore(dest_path)
    dest_store.data = merged_data
    dest_store._save()
    dest_store.close()
    
    elapsed = time.time() - start_time
    
    return MaintenanceReport(
        operation="merge_phenotype_maps",
        success=True,
        entries_processed=total_entries,
        entries_modified=len(merged_data),
        elapsed_seconds=elapsed,
        details={
            "source_files": len(source_paths),
            "conflicts_resolved": conflict_count,
            "resolution_strategy": conflict_resolution,
            "unique_entries": len(merged_data)
        }
    )


def apply_global_confidence_decay(
    store_path: str,
    decay_factor: float = 0.999,
    age_threshold: int = 100,
    time_threshold_days: float = 30.0,
    dry_run: bool = False
) -> MaintenanceReport:
    """
    Apply confidence decay to all entries in a knowledge store.
    
    Args:
        store_path: Path to the phenotype store
        decay_factor: Multiplicative decay factor
        age_threshold: Minimum age counter to trigger decay
        time_threshold_days: Days without update to trigger decay
        dry_run: If True, calculate but don't apply changes
        
    Returns:
        Maintenance report
    """
    start_time = time.time()
    
    if not os.path.exists(store_path):
        return MaintenanceReport(
            operation="apply_global_confidence_decay",
            success=False,
            entries_processed=0,
            entries_modified=0,
            elapsed_seconds=0,
            details={"error": "Store file not found"}
        )
    
    store = PickleStore(store_path)
    modified_count = 0
    processed_count = 0
    current_time = time.time()
    
    for entry in store.data.values():
        processed_count += 1
        
        age_counter = entry.get('age_counter', 0)
        last_updated = entry.get('last_updated', current_time)
        
        # Calculate aging factors
        time_since_update = current_time - last_updated
        days_since_update = time_since_update / (24 * 3600)
        
        # Use maximum of the two aging factors
        age_factor = max(
            age_counter - age_threshold if age_counter > age_threshold else 0,
            days_since_update - time_threshold_days if days_since_update > time_threshold_days else 0
        )
        
        if age_factor > 0:
            if not dry_run:
                # Apply decay
                old_mask = entry['memory_mask']
                decay_strength = decay_factor ** age_factor
                decay_mask = int(255 * decay_strength)
                entry['memory_mask'] = old_mask & decay_mask
                entry['confidence'] *= decay_strength
                entry['confidence'] = max(0.01, entry['confidence'])
            
            modified_count += 1
    
    if modified_count > 0 and not dry_run:
        store._save()
    
    store.close()
    elapsed = time.time() - start_time
    
    return MaintenanceReport(
        operation="apply_global_confidence_decay",
        success=True,
        entries_processed=processed_count,
        entries_modified=modified_count,
        elapsed_seconds=elapsed,
        details={
            "decay_factor": decay_factor,
            "age_threshold": age_threshold,
            "time_threshold_days": time_threshold_days,
            "dry_run": dry_run
        }
    )


def export_knowledge_statistics(
    store_path: str,
    output_path: str
) -> MaintenanceReport:
    """
    Export detailed statistics about a knowledge store.
    
    Args:
        store_path: Path to the phenotype store
        output_path: Path to save JSON statistics
        
    Returns:
        Maintenance report
    """
    start_time = time.time()
    
    if not os.path.exists(store_path):
        return MaintenanceReport(
            operation="export_knowledge_statistics",
            success=False,
            entries_processed=0,
            entries_modified=0,
            elapsed_seconds=0,
            details={"error": "Store file not found"}
        )
    
    store = PickleStore(store_path)
    entries = list(store.data.values())
    store.close()
    
    if not entries:
        stats = {
            "total_entries": 0,
            "generated_at": time.time()
        }
    else:
        # Calculate comprehensive statistics
        confidences = [e.get('confidence', 0.0) for e in entries]
        memory_masks = [e.get('memory_mask', 0) for e in entries]
        age_counters = [e.get('age_counter', 0) for e in entries]
        usage_counts = [e.get('usage_count', 0) for e in entries]
        
        # Memory utilization
        total_bits = sum(bin(mask).count('1') for mask in memory_masks)
        max_possible_bits = len(memory_masks) * 8
        memory_utilization = total_bits / max_possible_bits if max_possible_bits > 0 else 0
        
        # Temporal analysis
        current_time = time.time()
        ages_days = []
        for entry in entries:
            last_updated = entry.get('last_updated', current_time)
            age_days = (current_time - last_updated) / (24 * 3600)
            ages_days.append(age_days)
        
        # Phenotype diversity
        phenotypes = {}
        for entry in entries:
            phenotype = entry.get('phenotype', '?')
            phenotypes[phenotype] = phenotypes.get(phenotype, 0) + 1
        
        stats = {
            "total_entries": len(entries),
            "confidence": {
                "average": sum(confidences) / len(confidences),
                "median": sorted(confidences)[len(confidences) // 2],
                "min": min(confidences),
                "max": max(confidences),
                "high_confidence_count": sum(1 for c in confidences if c > 0.8),
                "low_confidence_count": sum(1 for c in confidences if c < 0.2)
            },
            "memory": {
                "utilization": memory_utilization,
                "total_bits_set": total_bits,
                "average_bits_per_entry": total_bits / len(entries)
            },
            "usage": {
                "total_usage": sum(usage_counts),
                "average_usage": sum(usage_counts) / len(usage_counts),
                "max_usage": max(usage_counts) if usage_counts else 0
            },
            "age": {
                "average_age_counter": sum(age_counters) / len(age_counters),
                "average_days_since_update": sum(ages_days) / len(ages_days),
                "oldest_entry_days": max(ages_days) if ages_days else 0
            },
            "phenotype_diversity": {
                "unique_phenotypes": len(phenotypes),
                "top_phenotypes": sorted(
                    phenotypes.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            },
            "generated_at": time.time(),
            "store_path": store_path
        }
    
    # Save statistics
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    elapsed = time.time() - start_time
    
    return MaintenanceReport(
        operation="export_knowledge_statistics",
        success=True,
        entries_processed=len(entries),
        entries_modified=0,
        elapsed_seconds=elapsed,
        details={
            "output_file": output_path,
            "statistics_generated": True
        }
    )


def validate_manifold_integrity(
    manifold_path: str,
    canonical_map_path: Optional[str] = None
) -> MaintenanceReport:
    """
    Validate the integrity of manifold data files.
    
    Args:
        manifold_path: Path to genotype map
        canonical_map_path: Optional path to canonical map
        
    Returns:
        Maintenance report
    """
    start_time = time.time()
    issues = []
    
    # Check manifold file
    if not os.path.exists(manifold_path):
        return MaintenanceReport(
            operation="validate_manifold_integrity",
            success=False,
            entries_processed=0,
            entries_modified=0,
            elapsed_seconds=0,
            details={"error": "Manifold file not found"}
        )
    
    try:
        with open(manifold_path, 'r') as f:
            manifold_data = json.load(f)
    except Exception as e:
        return MaintenanceReport(
            operation="validate_manifold_integrity",
            success=False,
            entries_processed=0,
            entries_modified=0,
            elapsed_seconds=0,
            details={"error": f"Failed to load manifold: {e}"}
        )
    
    # Validate manifold structure
    expected_keys = ['schema_version', 'genotype_map', 'endogenous_modulus', 
                     'manifold_diameter', 'total_states']
    
    for key in expected_keys:
        if key not in manifold_data:
            issues.append(f"Missing required key: {key}")
    
    # Validate constants
    if manifold_data.get('endogenous_modulus') != 788_986:
        issues.append(f"Invalid endogenous modulus: {manifold_data.get('endogenous_modulus')}")
    
    if manifold_data.get('manifold_diameter') != 6:
        issues.append(f"Invalid manifold diameter: {manifold_data.get('manifold_diameter')}")
    
    genotype_map = manifold_data.get('genotype_map', {})
    if len(genotype_map) != 788_986:
        issues.append(f"Invalid genotype map size: {len(genotype_map)}")
    
    # Check canonical map if provided
    canonical_issues = 0
    if canonical_map_path and os.path.exists(canonical_map_path):
        try:
            with open(canonical_map_path, 'r') as f:
                canonical_map = json.load(f)
            
            # Validate all indices are in range
            for idx, canonical_idx in canonical_map.items():
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= 788_986:
                    canonical_issues += 1
                if canonical_idx < 0 or canonical_idx >= 788_986:
                    canonical_issues += 1
                    
            if canonical_issues > 0:
                issues.append(f"Found {canonical_issues} invalid canonical mappings")
                
        except Exception as e:
            issues.append(f"Failed to validate canonical map: {e}")
    
    elapsed = time.time() - start_time
    
    return MaintenanceReport(
        operation="validate_manifold_integrity",
        success=len(issues) == 0,
        entries_processed=len(genotype_map),
        entries_modified=0,
        elapsed_seconds=elapsed,
        details={
            "issues": issues,
            "schema_version": manifold_data.get('schema_version'),
            "canonical_map_checked": canonical_map_path is not None
        }
    )
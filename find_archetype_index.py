#!/usr/bin/env python3
"""
Find the new archetype index after CS refactoring.
Analyzes the corrected ontology to identify key archetypal states.
"""

import numpy as np
from pathlib import Path
from baby.governance import fold, tensor_to_int
from baby.information import InformationEngine

def analyze_archetype_index():
    """Find and analyze the new archetype index after CS refactoring."""
    
    # Load the corrected ontology
    meta_path = Path("memories/public/meta")
    ontology_path = meta_path / "ontology_keys.npy"
    theta_path = meta_path / "theta.npy"
    
    if not ontology_path.exists() or not theta_path.exists():
        print("❌ Ontology files not found. Please regenerate maps first.")
        return
    
    ontology = np.load(ontology_path, mmap_mode="r")
    theta = np.load(theta_path, mmap_mode="r")
    
    print(f"📊 Analyzing ontology with {len(ontology)} states")
    print(f"   Theta range: [{theta.min():.4f}, {theta.max():.4f}]")
    
    # Find CS state (minimum theta)
    cs_index = int(np.argmin(theta))
    cs_state = int(ontology[cs_index])
    cs_theta = float(theta[cs_index])
    
    print(f"\n🎯 CS State Analysis:")
    print(f"   Index: {cs_index}")
    print(f"   State: {cs_state} (0x{cs_state:012X})")
    print(f"   Theta: {cs_theta:.6f}")
    
    # Find archetype candidates (states with special properties)
    print(f"\n🔍 Archetype Analysis:")
    
    # 1. Find states with theta close to key values
    target_thetas = {
        "π/4 (UNA)": np.pi / 4,
        "π/2 (CS)": np.pi / 2,
        "3π/4": 3 * np.pi / 4,
        "π": np.pi
    }
    
    archetype_candidates = {}
    
    for name, target in target_thetas.items():
        # Find closest theta
        diff = np.abs(theta - target)
        closest_idx = int(np.argmin(diff))
        closest_theta = float(theta[closest_idx])
        closest_state = int(ontology[closest_idx])
        
        archetype_candidates[name] = {
            'index': closest_idx,
            'state': closest_state,
            'theta': closest_theta,
            'diff': float(diff[closest_idx])
        }
        
        print(f"   {name:12}: idx={closest_idx:6}, state=0x{closest_state:012X}, θ={closest_theta:.6f}, Δ={diff[closest_idx]:.6f}")
    
    # 2. Find states with special fold properties
    print(f"\n🧮 Fold Property Analysis:")
    
    # Test some key fold relationships
    test_values = [0, 1, 85, 170, 255]  # Including 0x55 (85) and 0xAA (170)
    
    special_states = []
    
    for a in test_values:
        for b in test_values:
            result = fold(a, b)
            if result in ontology:
                # Find this state in ontology
                state_indices = np.where(ontology == result)[0]
                if len(state_indices) > 0:
                    idx = state_indices[0]
                    special_states.append({
                        'fold_expr': f"fold({a}, {b})",
                        'result': result,
                        'index': idx,
                        'theta': float(theta[idx])
                    })
    
    # Remove duplicates and sort by theta
    unique_states = {}
    for state in special_states:
        key = state['result']
        if key not in unique_states or state['theta'] < unique_states[key]['theta']:
            unique_states[key] = state
    
    sorted_special = sorted(unique_states.values(), key=lambda x: x['theta'])
    
    print(f"   Found {len(sorted_special)} states with special fold properties:")
    for state in sorted_special[:10]:  # Show top 10
        print(f"     {state['fold_expr']:15} = 0x{state['result']:03X} (idx={state['index']:6}, θ={state['theta']:.6f})")
    
    # 3. Analyze tensor encoding consistency
    print(f"\n🔢 Tensor Encoding Analysis:")
    
    # Create InformationEngine to test tensor encoding
    ep_path = meta_path / "epistemology.npy"
    phenomap_path = meta_path / "phenomenology_map.npy"
    
    if ep_path.exists() and phenomap_path.exists():
        info_engine = InformationEngine(
            keys_path=str(ontology_path),
            ep_path=str(ep_path),
            phenomap_path=str(phenomap_path),
            theta_path=str(theta_path)
        )
        
        # Test some key states
        test_states = [cs_state, 0, 85, 170, 255]
        
        for state in test_states:
            if state in ontology:
                # Find index
                indices = np.where(ontology == state)[0]
                if len(indices) > 0:
                    idx = indices[0]
                    theta_val = float(theta[idx])
                    
                    # Test tensor encoding consistency
                    try:
                        # Convert state to tensor and back
                        tensor = info_engine.int_to_tensor(state)
                        recovered = tensor_to_int(tensor)
                        consistent = (recovered == state)
                        
                        print(f"     State 0x{state:03X}: idx={idx:6}, θ={theta_val:.6f}, encoding={'✓' if consistent else '✗'}")
                    except Exception as e:
                        print(f"     State 0x{state:03X}: idx={idx:6}, θ={theta_val:.6f}, encoding=ERROR ({e})")
    else:
        print("   ⚠️  Epistemology or phenomenology maps not found, skipping tensor encoding test")
    
    # 4. Recommend new archetype index
    print(f"\n🎯 Archetype Index Recommendation:")
    
    # The archetype should be a state that:
    # 1. Has good theta properties (close to key angles)
    # 2. Has consistent tensor encoding
    # 3. Is not the CS state (which is extra-phenomenal)
    
    # Prefer UNA state (π/4) as it's the primary phenomenal archetype
    una_candidate = archetype_candidates["π/4 (UNA)"]
    
    print(f"   Recommended Archetype Index: {una_candidate['index']}")
    print(f"   State: 0x{una_candidate['state']:012X}")
    print(f"   Theta: {una_candidate['theta']:.6f} (target: {np.pi/4:.6f})")
    print(f"   Difference from π/4: {una_candidate['diff']:.6f}")
    
    return {
        'archetype_index': una_candidate['index'],
        'archetype_state': una_candidate['state'],
        'archetype_theta': una_candidate['theta'],
        'cs_index': cs_index,
        'cs_state': cs_state,
        'cs_theta': cs_theta,
        'total_states': len(ontology)
    }

if __name__ == "__main__":
    result = analyze_archetype_index()
    if result:
        print(f"\n✅ Analysis complete. New archetype index: {result['archetype_index']}")
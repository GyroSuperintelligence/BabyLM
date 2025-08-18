#!/usr/bin/env python3
"""
Debug script to investigate why recovery ladder finds 0 candidates.
"""

import sys
sys.path.insert(0, '.')

from baby.kernel.gyro_core import GyroEngine
from baby.constants.harmony_tokens import ALL_CONTROL_TOKENS
from pathlib import Path

def create_real_engine():
    """Create a real GyroEngine with proper configuration."""
    import tempfile
    import os
    
    # Use real atlas files from memories/public/meta/
    project_root = Path(__file__).parent
    atlas_dir = project_root / "memories" / "public" / "meta"
    
    atlas_paths = {
        'epistemology': str(atlas_dir / 'epistemology.npy'),
        'ontology_keys': str(atlas_dir / 'ontology_keys.npy'), 
        'theta': str(atlas_dir / 'theta.npy'),
        'phenomenology_map': str(atlas_dir / 'phenomenology_map.npy'),
        'orbit_sizes': str(atlas_dir / 'orbit_sizes.npy')
    }
    
    # Create temporary store directory
    temp_dir = tempfile.mkdtemp()
    store_paths = {
        'address_memory': os.path.join(temp_dir, 'address_memory.dat'),
        'passive_memory': os.path.join(temp_dir, 'passive_memory.log')
    }
    
    runtime = {'max_nudges': 6}
    
    version_info = {
        'atlas_version': 'v1.2.0',
        'address_version': 'v1.1.0', 
        'config_version': 'v1.0.0'
    }
    
    return GyroEngine(atlas_paths, store_paths, runtime, version_info, vocab_size=50000)

def debug_recovery():
    """Debug recovery ladder behavior."""
    print("Debugging recovery ladder...")
    
    # Create engine
    engine = create_real_engine()
    
    # Get start state
    start_state = engine.start_state()
    print(f"Start state: 0x{start_state:012X}")
    
    # Check if start state is in state_to_index
    if start_state not in engine.state_to_index:
        print("ERROR: Start state not in state_to_index!")
        return
    
    start_idx = engine.state_to_index[start_state]
    start_orbit = engine.phenomenology_map[start_idx]
    print(f"Start state index: {start_idx}")
    print(f"Start state orbit: {start_orbit}")
    
    # Check orbit-to-tokens mapping
    print(f"\nChecking orbit-to-tokens mapping...")
    if hasattr(engine, '_orbit_to_tokens'):
        if start_orbit in engine._orbit_to_tokens:
            tokens_in_orbit = engine._orbit_to_tokens[start_orbit]
            print(f"Tokens already in orbit {start_orbit}: {len(tokens_in_orbit)}")
            if tokens_in_orbit:
                print(f"First few tokens: {tokens_in_orbit[:10]}")
        else:
            print(f"Orbit {start_orbit} not in _orbit_to_tokens yet")
    else:
        print("_orbit_to_tokens not initialized")
    
    # Try _get_tokens_in_orbit with detailed debugging
    print(f"\nCalling _get_tokens_in_orbit({start_orbit})...")
    
    # Check sweep state before call
    if hasattr(engine, '_sweep') and start_orbit in engine._sweep:
        pos, stride = engine._sweep[start_orbit]
        print(f"Sweep state before: pos={pos}, stride={stride}")
    else:
        print("No sweep state yet")
    
    orbit_tokens = engine._get_tokens_in_orbit(start_orbit)
    print(f"Found {len(orbit_tokens)} tokens in orbit {start_orbit}")
    if orbit_tokens:
        print(f"First few tokens: {orbit_tokens[:10]}")
    
    # Check sweep state after call
    if hasattr(engine, '_sweep') and start_orbit in engine._sweep:
        pos, stride = engine._sweep[start_orbit]
        print(f"Sweep state after: pos={pos}, stride={stride}")
    
    # Let's manually test a few tokens to see their orbits
    print(f"\nManual token orbit testing...")
    test_tokens = [0, 1, 2, 10, 100, 1000]
    orbit_counts = {}
    for token in test_tokens:
        try:
            token_address = engine.address_of_token(token)
            if token_address in engine.state_to_index:
                token_orbit = engine.phenomenology_map[engine.state_to_index[token_address]]
                print(f"Token {token}: address=0x{token_address:012X}, orbit={token_orbit}")
                orbit_counts[token_orbit] = orbit_counts.get(token_orbit, 0) + 1
                if token_orbit == start_orbit:
                    print(f"  -> Token {token} is in target orbit {start_orbit}!")
            else:
                print(f"Token {token}: address not in state_to_index")
        except Exception as e:
            print(f"Token {token}: error = {e}")
    
    print(f"\nOrbit distribution in test tokens: {orbit_counts}")
    
    # Let's try a broader search to find tokens in the target orbit
    print(f"\nSearching for tokens in orbit {start_orbit}...")
    found_tokens = []
    search_limit = 5000  # Search more tokens
    for token in range(search_limit):
        if token % 1000 == 0:
            print(f"  Searched {token} tokens so far...")
        try:
            token_address = engine.address_of_token(token)
            if token_address in engine.state_to_index:
                token_orbit = engine.phenomenology_map[engine.state_to_index[token_address]]
                if token_orbit == start_orbit:
                    found_tokens.append(token)
                    if len(found_tokens) >= 5:  # Stop after finding 5
                        break
        except Exception:
            continue
    
    print(f"Found {len(found_tokens)} tokens in orbit {start_orbit}: {found_tokens}")
    
    # Let's check the orbit range and distribution
    print(f"\nChecking orbit system...")
    print(f"Total orbit codes: {len(engine.orbit_codes)}")
    print(f"Min orbit code: {min(engine.orbit_codes)}")
    print(f"Max orbit code: {max(engine.orbit_codes)}")
    print(f"Start state orbit {start_orbit} in orbit_codes: {start_orbit in engine.orbit_codes}")
    
    # Check if start_state orbit has a representative
    if start_orbit in engine.orbit_representatives:
        rep_state_idx = engine.orbit_representatives[start_orbit]
        rep_state = engine.ontology_keys[rep_state_idx]
        print(f"Orbit {start_orbit} representative: state_idx={rep_state_idx}, state=0x{rep_state:012X}")
    else:
        print(f"Orbit {start_orbit} has no representative!")
    
    # Let's try a different start state - maybe use orbit 0 which has tokens
    print(f"\nTrying orbit 0 (which has tokens)...")
    if 0 in engine.orbit_representatives:
        orbit_0_rep_idx = engine.orbit_representatives[0]
        orbit_0_state = engine.ontology_keys[orbit_0_rep_idx]
        print(f"Orbit 0 representative: state_idx={orbit_0_rep_idx}, state=0x{orbit_0_state:012X}")
        
        # Test recovery with orbit 0 state
        orbit_0_tokens = engine._get_tokens_in_orbit(0)
        print(f"Tokens in orbit 0: {len(orbit_0_tokens)} (first 10: {orbit_0_tokens[:10]})")
        
        # Test recovery ladder with orbit 0 state
        recovery_candidates_0 = engine.recover_candidates(orbit_0_state, 6)
        print(f"Recovery candidates for orbit 0 state: {len(recovery_candidates_0)}")
    
    # Test admissibility of some tokens
    print(f"\nTesting admissibility...")
    test_tokens = orbit_tokens[:5] if orbit_tokens else [100, 200, 300, 400, 500]
    for token in test_tokens:
        if token not in ALL_CONTROL_TOKENS:
            try:
                is_admissible = engine.is_admissible(start_state, token)
                print(f"Token {token}: admissible = {is_admissible}")
            except Exception as e:
                print(f"Token {token}: error = {e}")
    
    # Test recovery levels individually
    print(f"\nTesting recovery levels...")
    for level in range(1, 6):
        method_name = f'_recovery_level_{level}'
        if hasattr(engine, method_name):
            method = getattr(engine, method_name)
            try:
                if level == 5:
                    candidates = method(start_state, 6)  # max_nudges for level 5
                else:
                    candidates = method(start_state)
                print(f"Level {level}: {len(candidates)} candidates")
                if candidates:
                    print(f"  First few: {candidates[:5]}")
            except Exception as e:
                print(f"Level {level}: error = {e}")
    
    # Test full recovery ladder
    print(f"\nTesting full recovery ladder...")
    try:
        candidates = engine.recover_candidates(start_state, 6)
        print(f"Recovery ladder found {len(candidates)} candidates")
        if candidates:
            print(f"Candidates: {candidates[:10]}")
    except Exception as e:
        print(f"Recovery ladder error: {e}")

if __name__ == "__main__":
    debug_recovery()
#!/usr/bin/env python3
import traceback
import sys
from pathlib import Path

sys.path.insert(0, '.')

try:
    from kernel.chat_oss import load_gyro_model
    
    print("Loading GyroHead model...")
    model = load_gyro_model(Path('memories/models/gpt-oss-20b'))
    
    print("Testing token conversion...")
    introns = model.token_to_introns(1)
    print(f"Introns for token 1: {introns}")
    
    print("Testing state transition...")
    current_state = model.CS_STATE_INDEX
    print(f"CS state: {current_state}")
    
    # Test applying an intron
    if introns:
        next_state = model._apply_intron_and_gate(current_state, introns[0])
        print(f"Next state after intron {introns[0]}: {next_state}")
    
    print("Testing orbit lookup...")
    orbit_rep = int(model.phenomenology[current_state])
    print(f"Orbit rep for CS state: {orbit_rep}")
    
    print(f"Total orbit candidates: {len(model._orbit_candidates)}")
    print(f"Sample orbit keys: {list(model._orbit_candidates.keys())[:10]}")
    
    # Test with a different state
    test_state = 0
    test_orbit = int(model.phenomenology[test_state])
    print(f"Orbit rep for state 0: {test_orbit}")
    
    if orbit_rep in model._orbit_candidates:
        candidates = model._orbit_candidates[orbit_rep]
        print(f"Candidates for orbit {orbit_rep}: {len(candidates)} tokens")
        if candidates:
            print(f"First candidate: {candidates[0]}")
    else:
        print(f"No candidates found for orbit {orbit_rep}")
        
    if test_orbit in model._orbit_candidates:
        candidates = model._orbit_candidates[test_orbit]
        print(f"Candidates for test orbit {test_orbit}: {len(candidates)} tokens")
    else:
        print(f"No candidates found for test orbit {test_orbit}")
        
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
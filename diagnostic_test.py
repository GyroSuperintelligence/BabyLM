#!/usr/bin/env python3
"""
Diagnostic test to trace token selection behavior and identify repetition causes.
"""

import sys
sys.path.insert(0, '.')

from baby.kernel.gyro_core import GyroEngine
from baby.constants.frozen_channels import FROZEN_CHANNELS

def run_diagnostic():
    """Run diagnostic test to trace token selection behavior."""
    print("=== Token Selection Diagnostic ===")
    
    # Initialize engine
    atlas_paths = {
        "theta": "memories/public/meta/theta.npy",
        "ontology_keys": "memories/public/meta/ontology_keys.npy",
        "epistemology": "memories/public/meta/epistemology.npy",
        "phenomenology_map": "memories/public/meta/phenomenology_map.npy",
        "orbit_sizes": "memories/public/meta/orbit_sizes.npy"
    }
    
    try:
        engine = GyroEngine(atlas_paths)
        print(f"Engine initialized successfully")
        print(f"Vocab size: {engine.vocab_size}")
        print(f"Atlas size: {len(engine.keys)}")
        
        # Get initial state
        start_state = engine.start_state()
        print(f"Start state: 0x{start_state:012X}")
        
        # Test sector computation
        sector = engine.sector(start_state)
        print(f"Sector (8-bit): {sector:08b} (0x{sector:02X})")
        
        # Test state phase components
        sp = engine._state_phase(start_state)
        sp_li, sp_fg, sp_bg = engine._state_phase_components(start_state)
        print(f"State phase: {sp}")
        print(f"LI/FG/BG components: {sp_li}, {sp_fg}, {sp_bg}")
        
        # Learn some tokens first to populate rep_channel
        print("\n=== Learning Phase ===")
        test_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                       25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # More diverse test tokens
        current_state = start_state
        
        for i, token in enumerate(test_tokens):
            try:
                new_state = engine.learn_on_user(current_state, token)
                print(f"Learned token {token}: 0x{current_state:012X} -> 0x{new_state:012X}")
                current_state = new_state
            except Exception as e:
                print(f"Failed to learn token {token}: {e}")
                break
        
        # Check rep_channel population
        total_learned = sum(len(phases) for phases in engine.rep_channel.values())
        print(f"Total learned phases across all orbits: {total_learned}")
        
        if total_learned == 0:
            print("❌ No tokens learned - cannot test emission")
            return False
        
        # Initialize session state
        session_omega = {}
        session_bucket_key = {}
        session_bucket_pos = {}
        
        # Reset to start state for emission testing
        current_state = start_state
        
        # Generate sequence and trace behavior
        print("\n=== Token Generation Trace ===")
        current_state = start_state
        tokens = []
        
        for step in range(20):  # Generate 20 tokens
            result = engine.emit_next_from_state(
                current_state, 
                session_omega, 
                session_bucket_key, 
                session_bucket_pos
            )
            
            if result is None:
                print(f"Step {step}: No token available")
                break
                
            token_id, new_state, new_omega, new_bucket_key, new_bucket_pos = result
            tokens.append(token_id)
            
            # Get orbit info
            state_idx = engine.state_to_index[current_state]
            rep_idx = engine.orbit_rep_index(state_idx)
            
            # Get bucket info
            phase_map = engine.rep_channel.get(rep_idx, {})
            available_keys = sorted(phase_map.keys()) if phase_map else []
            current_bucket_key = new_bucket_key.get(rep_idx, 'None')
            current_omega = new_omega.get(rep_idx, 0)
            
            # Debug rotation calculation
            if step >= 10 and available_keys:  # Only debug after some steps to see the pattern
                rep_phase = engine.rep_phase.get(rep_idx, 0)
                omega_val = session_omega.get(rep_idx, 0)
                sector_val = engine.sector(current_state)
                
                rot = engine._fold8(rep_phase, omega_val)
                rot = engine._fold8(rot, sector_val)
                rot = engine._fold8(rot, session_bucket_key.get(rep_idx, 0))
                
                base_key_idx = next((i for i, k in enumerate(available_keys) if k == session_bucket_key.get(rep_idx, 0)), 0)
                rotation_offset = (rot % len(available_keys))
                current_key_idx = (base_key_idx + rotation_offset) % len(available_keys)
                selected_key = available_keys[current_key_idx]
                
                print(f"         DEBUG: rot={rot}, base_idx={base_key_idx}, offset={rotation_offset}, selected_key={selected_key}")
                
                # Debug bucket contents and position
                if rep_idx in engine.rep_channel and selected_key in engine.rep_channel[rep_idx]:
                    bucket = engine.rep_channel[rep_idx][selected_key]
                    pos = session_bucket_pos[rep_idx].get(selected_key, 0)
                    print(f"         BUCKET DEBUG: key={selected_key}, bucket={bucket}, pos={pos}, bucket_size={len(bucket)}")
            
            print(f"Step {step:2d}: Token {token_id:6d} | State 0x{current_state:012X} -> 0x{new_state:012X}")
            print(f"         Rep {rep_idx:3d} | Omega {current_omega:3d} | Bucket key {current_bucket_key}")
            print(f"         Available keys: {available_keys[:5]}{'...' if len(available_keys) > 5 else ''}")
            
            # Check for immediate repetition
            if len(tokens) >= 2 and tokens[-1] == tokens[-2]:
                print(f"         *** IMMEDIATE REPETITION DETECTED ***")
            
            # Check for pattern repetition
            if len(tokens) >= 4:
                last_two = tokens[-2:]
                prev_two = tokens[-4:-2]
                if last_two == prev_two:
                    print(f"         *** 2-TOKEN PATTERN REPETITION: {last_two} ***")
            
            # Update state for next iteration
            current_state = new_state
            session_omega = new_omega
            session_bucket_key = new_bucket_key
            session_bucket_pos = new_bucket_pos
        
        print(f"\nGenerated tokens: {tokens}")
        
        # Analyze repetition patterns
        print("\n=== Repetition Analysis ===")
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        repeated_tokens = {k: v for k, v in token_counts.items() if v > 1}
        if repeated_tokens:
            print(f"Repeated tokens: {repeated_tokens}")
        else:
            print("No repeated tokens found")
        
        # Check for consecutive repetitions
        consecutive_reps = []
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                consecutive_reps.append((i-1, tokens[i]))
        
        if consecutive_reps:
            print(f"Consecutive repetitions at positions: {consecutive_reps}")
        else:
            print("No consecutive repetitions found")
            
        # Test phase map analysis
        print("\n=== Phase Map Analysis ===")
        state_idx = engine.state_to_index[current_state]
        rep_idx = engine.orbit_rep_index(state_idx)
        if rep_idx in engine.rep_channel:
            phase_map = engine.rep_channel[rep_idx]
            print(f"Rep {rep_idx} phase_map keys: {sorted(phase_map.keys())}")
            for key in sorted(phase_map.keys()):
                tokens_in_phase = len(phase_map[key])
                print(f"  Key {key}: {tokens_in_phase} tokens")
        
        # Test FROZEN_CHANNELS structure
        print("\n=== FROZEN_CHANNELS Verification ===")
        print(f"NUM_SLABS: {FROZEN_CHANNELS.NUM_SLABS}")
        for slab in range(min(3, FROZEN_CHANNELS.NUM_SLABS)):
            indices = FROZEN_CHANNELS.get_slab_bit_indices(slab)
            print(f"Slab {slab} bit indices: {indices}")
        
        return True
        
    except Exception as e:
        print(f"Error during diagnostic: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_diagnostic()
    sys.exit(0 if success else 1)
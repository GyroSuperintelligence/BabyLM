#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from baby.policies import OrbitStore, CanonicalView
from baby.intelligence import GyroSI

def test_a_enumeration():
    """Test (a): Can you enumerate any entries from the file you wrote?"""
    print("=== Test (a): Entry enumeration ===")
    
    BIN = "memories/public/knowledge/knowledge.bin"
    PHEN = "memories/public/meta/phenomenology_map.npy"
    
    try:
        cv = CanonicalView(OrbitStore(BIN), phenomenology_map_path=PHEN)
        n = 0
        for _, _ in cv.iter_entries():
            n += 1
            if n >= 5:
                break
        print("entries peek:", n)  # expect > 0
        return n > 0
    except Exception as e:
        print(f"Error in test (a): {e}")
        return False

def test_b_keys_for_state():
    """Test (b): Does iter_keys_for_state(rep) return something?"""
    print("\n=== Test (b): Keys for state ===")
    
    BIN = "memories/public/knowledge/knowledge.bin"
    PHEN = "memories/public/meta/phenomenology_map.npy"
    
    try:
        cv = CanonicalView(OrbitStore(BIN), phenomenology_map_path=PHEN)
        
        # find one state we know exists in the store
        state_idx = None
        for (s_idx, tok), _ in cv.iter_entries():
            state_idx = s_idx
            break
        print("sample state:", state_idx)
        
        if state_idx is None:
            print("No state found!")
            return False
        
        # now ask for keys for that same state
        ks = list(cv.iter_keys_for_state(state_idx))
        print("keys for that state:", len(ks))  # expect >= 1
        return len(ks) >= 1
    except Exception as e:
        print(f"Error in test (b): {e}")
        return False

def test_c_runtime_candidates():
    """Test (c): Does your runtime stack see candidates if you point it at your .bin?"""
    print("\n=== Test (c): Runtime candidates ===")
    
    BIN = "memories/public/knowledge/knowledge.bin"
    PHEN = "memories/public/meta/phenomenology_map.npy"
    
    try:
        # First get a sample state from the direct store
        cv = CanonicalView(OrbitStore(BIN), phenomenology_map_path=PHEN)
        state_idx = None
        for (s_idx, tok), _ in cv.iter_entries():
            state_idx = s_idx
            break
        
        if state_idx is None:
            print("No state found for runtime test!")
            return False
            
        print("sample state:", state_idx)
        
        # Now test with GyroSI runtime
        cfg = {
            "ontology_path": "memories/public/meta/ontology_keys.npy",
            "public_knowledge_path": BIN,
            "phenomenology_map_path": PHEN,
            "enable_phenomenology_storage": True,
            "preferences": {}
        }
        # Use absolute paths for GyroSI
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))
        cfg["ontology_path"] = os.path.join(base_path, cfg["ontology_path"])
        cfg["public_knowledge_path"] = os.path.join(base_path, cfg["public_knowledge_path"])
        cfg["phenomenology_map_path"] = os.path.join(base_path, cfg["phenomenology_map_path"])
        g = GyroSI(cfg, agent_id="debug")
        store = g.engine.operator.store  # OverlayView(ReadOnlyView(CanonicalView(.bin)), OrbitStore(private-empty))
        
        # use the same state we found above
        ks = list(store.iter_keys_for_state(state_idx))
        print("runtime sees:", len(ks))   # expect >= 1
        return len(ks) >= 1
    except Exception as e:
        print(f"Error in test (c): {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Candidate Sanity Checklist")
    print("=" * 50)
    
    results = []
    results.append(("a", test_a_enumeration()))
    results.append(("b", test_b_keys_for_state()))
    results.append(("c", test_c_runtime_candidates()))
    
    print("\n=== Summary ===")
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"Test {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Minimal test to verify basic agent functionality without loading large knowledge stores.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baby.intelligence import orchestrate_turn, AgentPool


def test_minimal_agent():
    """Test minimal agent functionality."""
    print("🔧 Minimal Agent Test")
    print("=" * 40)
    
    try:
        # Load preferences
        prefs_path = PROJECT_ROOT / "memories/memory_preferences.json"
        with open(prefs_path) as f:
            preferences = json.load(f)
        
        # Use minimal paths
        ontology_path = str(PROJECT_ROOT / "memories/public/meta/ontology_keys.npy")
        
        # Create a minimal knowledge store path (empty or small)
        minimal_knowledge_path = str(PROJECT_ROOT / "memories/public/knowledge/minimal_test.bin")
        
        print(f"📁 Using minimal knowledge path: {minimal_knowledge_path}")
        print(f"📁 Using ontology path: {ontology_path}")
        
        # Create minimal agent pool
        agent_pool = AgentPool(
            ontology_path=ontology_path,
            base_knowledge_path=minimal_knowledge_path,
            preferences=preferences,
            allowed_ids={"user", "system", "assistant"},
            allow_auto_create=False,
            private_agents_base_path=str(PROJECT_ROOT / "memories/private/agents"),
            base_path=PROJECT_ROOT,
        )
        
        # Ensure triad exists
        agent_pool.ensure_triad()
        
        print("✅ Agent pool created successfully!")
        print(f"📊 Active agents: {agent_pool.get_active_agents()}")
        
        # Test one simple conversation
        user_id = "user"
        assistant_id = "assistant"
        
        test_message = "Hello, what is AI?"
        print(f"\n📝 Testing: '{test_message}'")
        
        t0 = time.time()
        response = orchestrate_turn(
            pool=agent_pool,
            user_id=user_id,
            assistant_id=assistant_id,
            user_input=test_message,
            tokenizer_name=preferences["tokenizer"]["name"]
        )
        print(f"⏱️  Turn took {time.time() - t0:.2f}s")
        
        print(f"🤖 Response: {response}")
        
        print("\n✅ Minimal test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'agent_pool' in locals():
            agent_pool.close_all()


if __name__ == "__main__":
    test_minimal_agent() 
#!/usr/bin/env python3
"""
Test script that uses the external adapter directly to test our trained system.
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


def test_external_adapter_direct():
    """Test the external adapter functionality directly."""
    print("🌐 Testing External Adapter Directly")
    print("=" * 50)
    
    try:
        # Import the external adapter's agent pool setup
        from toys.communication.external_adapter import agent_pool
        
        print("✅ Using external adapter's agent pool")
        print(f"📊 Active agents: {agent_pool.get_active_agents()}")
        
        # Test simple conversation using the correct agent IDs
        user_id = "user"
        assistant_id = "assistant"
        
        # Test messages
        test_messages = [
            "Hello, what is artificial intelligence?",
            "Tell me about machine learning",
            "What is the capital of France?",
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n📝 Test {i}: '{message}'")
            
            # Use orchestrate_turn (same as external adapter)
            response = orchestrate_turn(
                pool=agent_pool,
                user_id=user_id,
                assistant_id=assistant_id,
                user_input=message,
                tokenizer_name="bert-base-uncased"
            )
            
            print(f"🤖 Response: {response}")
            
            # Small delay between messages
            time.sleep(0.5)
        
        print("\n✅ External adapter test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_simple_agent_creation():
    """Test creating a minimal agent pool."""
    print("\n🔧 Testing Simple Agent Creation")
    print("=" * 50)
    
    try:
        # Load preferences
        prefs_path = PROJECT_ROOT / "memories/memory_preferences.json"
        with open(prefs_path) as f:
            preferences = json.load(f)
        
        # Use a minimal knowledge path for testing
        base_knowledge_path = str(PROJECT_ROOT / "memories/public/knowledge/knowledge.bin")
        ontology_path = str(PROJECT_ROOT / "memories/public/meta/ontology_keys.npy")
        
        print(f"📁 Knowledge path: {base_knowledge_path}")
        print(f"📁 Ontology path: {ontology_path}")
        
        # Create minimal agent pool
        agent_pool = AgentPool(
            ontology_path=ontology_path,
            base_knowledge_path=base_knowledge_path,
            preferences=preferences,
            allowed_ids={"user", "system", "assistant"},
            allow_auto_create=False,  # Don't auto-create, use existing
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
        
        response = orchestrate_turn(
            pool=agent_pool,
            user_id=user_id,
            assistant_id=assistant_id,
            user_input=test_message,
            tokenizer_name="bert-base-uncased"
        )
        
        print(f"🤖 Response: {response}")
        
        print("\n✅ Simple agent creation test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'agent_pool' in locals():
            agent_pool.close_all()


def main():
    """Run tests."""
    print("🚀 BabyLM External Adapter Test")
    print("=" * 60)
    
    # Test 1: Use external adapter directly
    test_external_adapter_direct()
    
    # Test 2: Simple agent creation
    test_simple_agent_creation()
    
    print("\n🎉 All tests completed!")
    print("\n💡 To start the external adapter server, run:")
    print("   uvicorn toys.communication.external_adapter:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main() 
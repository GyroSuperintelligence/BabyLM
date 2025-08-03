#!/usr/bin/env python3
"""
Test script for agent interaction using external adapters.

This script demonstrates how to:
1. Initialize agents with trained knowledge
2. Use the external adapter API
3. Test different conversation scenarios
4. Verify the system is working properly
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baby.intelligence import AgentPool, orchestrate_turn
from baby.contracts import AgentConfig, PreferencesConfig


def load_preferences() -> Dict[str, Any]:
    """Load preferences from the canonical JSON file."""
    prefs_path = PROJECT_ROOT / "memories/memory_preferences.json"
    with open(prefs_path) as f:
        return json.load(f)


def create_agent_pool() -> AgentPool:
    """Create and configure the agent pool."""
    preferences = load_preferences()
    
    # Resolve paths relative to project root
    base_knowledge_path = str(PROJECT_ROOT / preferences["public_knowledge"]["path"])
    ontology_path = str(PROJECT_ROOT / "memories" / preferences["ontology"]["ontology_map_path"])
    phenomenology_path = str(PROJECT_ROOT / "memories" / preferences["ontology"]["phenomenology_map_path"])
    
    # Create agent pool with trained knowledge
    agent_pool = AgentPool(
        ontology_path=ontology_path,
        base_knowledge_path=base_knowledge_path,
        preferences=preferences,
        allowed_ids={"user", "system", "assistant"},
        allow_auto_create=True,
        private_agents_base_path=str(PROJECT_ROOT / "memories" / preferences["private_knowledge"]["base_path"]),
        base_path=PROJECT_ROOT,
    )
    
    # Ensure the triad of agents exists
    agent_pool.ensure_triad()
    
    return agent_pool


def test_direct_agent_interaction():
    """Test direct agent interaction without the external adapter."""
    print("🤖 Testing Direct Agent Interaction")
    print("=" * 50)
    
    agent_pool = create_agent_pool()
    
    try:
        # Test basic conversation using the correct agent IDs
        user_id = "user"  # Fixed agent ID
        assistant_id = "assistant"  # Fixed agent ID
        
        # Get agents (they should already exist from ensure_triad)
        user_agent = agent_pool.get(user_id)
        assistant_agent = agent_pool.get(assistant_id)
        
        print(f"✅ User agent: {user_agent.agent_id}")
        print(f"✅ Assistant agent: {assistant_agent.agent_id}")
        
        # Test a simple conversation
        test_messages = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Tell me about machine learning",
            "What is the capital of France?",
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n📝 Test {i}: '{message}'")
            
            # Use orchestrate_turn for proper conversation flow
            t0 = time.time()
            response = orchestrate_turn(
                pool=agent_pool,
                user_id=user_id,
                assistant_id=assistant_id,
                user_input=message,
                tokenizer_name="bert-base-uncased"
            )
            print(f"⏱️  Turn took {time.time() - t0:.2f}s")
            
            print(f"🤖 Response: {response}")
            # Cheap state snapshot (no full-store scan)
            print(f"⏱️  State: {user_agent.engine.get_state_info()}")
            
            # Small delay between messages
            time.sleep(0.5)
        
        print("\n✅ Direct agent interaction test completed!")
        
    except Exception as e:
        print(f"❌ Error in direct agent interaction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        agent_pool.close_all()


def test_external_adapter_simulation():
    """Simulate external adapter API calls."""
    print("\n🌐 Testing External Adapter Simulation")
    print("=" * 50)
    
    agent_pool = create_agent_pool()
    
    try:
        # Simulate OpenAI-compatible chat completion using correct agent IDs
        user_id = "user"  # Fixed agent ID (like external adapter)
        assistant_id = "assistant"  # Fixed agent ID
        
        # Simulate a chat completion request
        chat_request = {
            "model": "gyro-baby",
            "messages": [
                {"role": "user", "content": "Hello, what can you tell me about AI?"}
            ]
        }
        
        print(f"📤 Simulating chat completion request:")
        print(f"   User ID: {user_id}")
        print(f"   Model: {chat_request['model']}")
        print(f"   Message: {chat_request['messages'][0]['content']}")
        
        # Use orchestrate_turn (same as external adapter)
        response = orchestrate_turn(
            pool=agent_pool,
            user_id=user_id,
            assistant_id=assistant_id,
            user_input=chat_request['messages'][0]['content'],
            tokenizer_name="bert-base-uncased"
        )
        
        # Simulate OpenAI response format
        chat_response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat_request['model'],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        
        print(f"\n📥 Simulated OpenAI response:")
        print(f"   Response: {response}")
        print(f"   Response ID: {chat_response['id']}")
        
        # Test HuggingFace-style generation
        hf_request = {
            "inputs": "The future of artificial intelligence is"
        }
        
        print(f"\n📤 Simulating HF generation request:")
        print(f"   Input: {hf_request['inputs']}")
        
        # Generate continuation
        continuation = orchestrate_turn(
            pool=agent_pool,
            user_id=user_id,
            assistant_id=assistant_id,
            user_input=hf_request['inputs'],
            tokenizer_name="bert-base-uncased"
        )
        
        hf_response = {
            "generated_text": continuation
        }
        
        print(f"\n📥 Simulated HF response:")
        print(f"   Generated text: {hf_response['generated_text']}")
        
        print("\n✅ External adapter simulation completed!")
        
    except Exception as e:
        print(f"❌ Error in external adapter simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        agent_pool.close_all()


def test_agent_pool_management():
    """Test agent pool management and persistence."""
    print("\n🏪 Testing Agent Pool Management")
    print("=" * 50)
    
    agent_pool = create_agent_pool()
    
    try:
        # Test the triad of agents that should exist
        test_agents = [
            ("user", "user"),
            ("system", "system"),
            ("assistant", "assistant"),
        ]
        
        created_agents = []
        
        for agent_id, role in test_agents:
            agent = agent_pool.get(agent_id)
            if agent:
                created_agents.append(agent)
                print(f"✅ Retrieved agent: {agent.agent_id} (role: {role})")
            else:
                print(f"❌ Failed to retrieve agent: {agent_id}")
        
        # Test active agents list
        active_agents = agent_pool.get_active_agents()
        print(f"📊 Active agents: {len(active_agents)}")
        for agent_id in active_agents:
            print(f"   - {agent_id}")
        
        # Test agent info
        for agent in created_agents[:2]:  # Test first 2 agents
            info = agent.get_agent_info()
            print(f"\n📋 Agent info for {agent.agent_id}:")
            for key, value in info.items():
                print(f"   {key}: {value}")
        
        print("\n✅ Agent pool management test completed!")
        
    except Exception as e:
        print(f"❌ Error in agent pool management: {e}")
        import traceback
        traceback.print_exc()
    finally:
        agent_pool.close_all()


def main():
    """Run all tests."""
    print("🚀 BabyLM Agent Interaction Test Suite")
    print("=" * 60)
    
    # Test 1: Direct agent interaction
    test_direct_agent_interaction()
    
    # Test 2: External adapter simulation
    test_external_adapter_simulation()
    
    # Test 3: Agent pool management
    test_agent_pool_management()
    
    print("\n🎉 All tests completed!")
    print("\n💡 To start the external adapter server, run:")
    print("   uvicorn toys.communication.external_adapter:app --host 0.0.0.0 --port 8000 --reload")
    print("\n💡 Then you can make API calls to:")
    print("   POST http://localhost:8000/v1/chat/completions (OpenAI compatible)")
    print("   POST http://localhost:8000/generate (HuggingFace compatible)")


if __name__ == "__main__":
    main() 
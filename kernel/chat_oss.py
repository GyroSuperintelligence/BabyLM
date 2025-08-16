#!/usr/bin/env python3
"""
Interactive CPU chat using GyroHead physics-based language model.
This script uses the pure physics GyroSI implementation for text generation.
- Uses GyroHead for all language processing through physics operations
- Loads consolidated model.gyro.safetensors weights automatically
- Uses harmony response format for proper conversation
- No transformer operations - pure gyroscopic intelligence
- CPU-only physics-based inference

Example:
  .venv/bin/python kernel/chat_oss.py
"""

from __future__ import annotations

import argparse
import numpy as np
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional, TYPE_CHECKING, Union

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# For typing only; do not rely on these at runtime
try:
    from openai_harmony import Message as _HarmonyMessage
    from openai_harmony import Role as _HarmonyRole
except Exception:
    _HarmonyMessage = None  # type: ignore[assignment]
    _HarmonyRole = None  # type: ignore[assignment]

# Create stub classes for runtime usage
class _Role:
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"

if TYPE_CHECKING:
    from kernel.gyro_head import GyroHead
    from openai_harmony import Message as HarmonyMessage, StreamableParser, Role


def setup_harmony_format() -> tuple[Any, Any]:
    """Set up the harmony format for gpt-oss."""
    try:
        from openai_harmony import (
            SystemContent,
            Message,
            Role,
            Conversation,
            load_harmony_encoding,
            HarmonyEncodingName,
            ReasoningEffort,
        )
        import datetime

        # Load the harmony encoding for gpt-oss
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        # Get special token ids from encoding by encoding the special token strings
        stop_token_id = encoding.encode("<|return|>", allowed_special={"<|return|>"})[0] if encoding.encode("<|return|>", allowed_special={"<|return|>"}) else 200002
        channel_token_id = encoding.encode("<|channel|>", allowed_special={"<|channel|>"})[0] if encoding.encode("<|channel|>", allowed_special={"<|channel|>"}) else 200005
        message_token_id = encoding.encode("<|message|>", allowed_special={"<|message|>"})[0] if encoding.encode("<|message|>", allowed_special={"<|message|>"}) else 200008

        # Setup system message with required channels (analysis, commentary, final)
        system_message_content = (
            SystemContent.new()
            .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
            .with_reasoning_effort(ReasoningEffort.LOW)  # Low reasoning = no thinking mode
            .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
            .with_knowledge_cutoff("2024-06")
            .with_required_channels(["analysis", "commentary", "final"])  # All channels as per guide
        )

        system_message = Message.from_role_and_content("system", system_message_content)

        return encoding, system_message

    except ImportError as e:
        print(f"[error] Missing openai-harmony dependency: {e}")
        print("[error] Install with: pip install openai-harmony")
        return None, None


def load_gyro_model(model_path: Path) -> Optional[GyroHead]:
    """Load the GyroHead model for physics-based inference."""
    try:
        from kernel.gyro_head import GyroHead
        
        # Physics tables are in memories/ but model weights are in memories/models/gpt-oss-20b/
        memories_base = Path("memories")
        print(f"[info] Loading GyroHead model - physics from: {memories_base}")
        
        # Initialize GyroHead directly - let it handle weight decoding
        model = GyroHead(base_path=memories_base)
        
        print("[info] GyroHead model loaded successfully")
        return model
        
    except Exception as e:
        print(f"[error] Failed to load GyroHead model: {e}")
        return None


def generate_response(
    encoding: Any,
    gyro_model: 'GyroHead',
    conversation: List[Any],
    max_new_tokens: int = 256,
) -> Optional[str]:
    """Generate a response using GyroHead physics-based generation."""
    try:
        # Convert conversation to tokens using harmony encoding
        try:
            from openai_harmony import Conversation
        except ImportError as e:
            print(f"[error] Missing openai-harmony dependency in generate_response: {e}")
            print("[error] Install with: pip install openai-harmony")
            return None
        
        conv = Conversation.from_messages(conversation)
        # Import Role from harmony library for proper enum usage
        from openai_harmony import Role
        token_ids = encoding.render_conversation_for_completion(conv, Role.ASSISTANT)

        # Use proper state seeding method instead of manual token loop
        def seed_from_tokens(gyro_model: 'GyroHead', token_ids: List[int]) -> None:
            gyro_model.current_state_index = gyro_model.CS_STATE_INDEX
            # Reset path_memory to seed (not "itself")
            from kernel.gyro_head import GENE_Mic_S
            gyro_model.path_memory = GENE_Mic_S
            # Process tokens sequentially to advance the model state
            for i, token_id in enumerate(token_ids):
                if token_id < gyro_model.vocab_size:
                    # Use ingest_token to update state with the actual input token
                    gyro_model.ingest_token(token_id, pos=i)
        
        seed_from_tokens(gyro_model, token_ids)
        
        # Get all special tokens from encoding (no hard-coding)
        channel_token_id = encoding.encode("<|channel|>", allowed_special={"<|channel|>"})[0]
        message_token_id = encoding.encode("<|message|>", allowed_special={"<|message|>"})[0]
        stop_token_id = encoding.encode("<|return|>", allowed_special={"<|return|>"})[0]
        
        # Derive channel name tokens from encoding by rendering them
        # Render "<|channel|>final<|message|>" to get the token between channel and message
        temp_tokens = encoding.encode("<|channel|>final<|message|>", allowed_special={"<|channel|>", "<|message|>"})
        final_token_id = None
        for i, token in enumerate(temp_tokens):
            if token == channel_token_id and i + 2 < len(temp_tokens) and temp_tokens[i + 2] == message_token_id:
                final_token_id = temp_tokens[i + 1]
                break
        
        if final_token_id is None:
            print(f"[error] Could not derive 'final' channel token from encoding")
            return None
        
        # Generate tokens using proper GyroHead physics
        generated_tokens: List[int] = []
        
        # Start with proper Harmony message structure: <|channel|>final<|message|>
        harmony_prefix: List[int] = [channel_token_id, final_token_id, message_token_id]
        for i, tok in enumerate(harmony_prefix):
            gyro_model.ingest_token(tok, pos=i)
            generated_tokens.append(tok)
        

        
        # Generate content using GyroHead physics engine
        content_tokens_generated = 0
        max_content_tokens = min(50, max_new_tokens - len(harmony_prefix) - 1)  # Reserve space for stop token
        consecutive_special_tokens = 0  # Track consecutive special tokens to prevent infinite loops
        max_consecutive_special = 10  # Maximum consecutive special tokens before breaking
        
        for step in range(max_content_tokens):
            # step with real tensors + physics sieve
            try:
                prev = generated_tokens[-1] if generated_tokens else 0  # or your last assistant token
                best_token = gyro_model.generate_next_token(prev_token_id=prev, pos=step)
                
                # Validate token is within valid range
                if best_token is None or best_token < 0:
                    print(f"[warning] Invalid token generated: {best_token}")
                    break
                
                # Check if we should stop generation
                if best_token == stop_token_id or best_token >= gyro_model.vocab_size:
                    break
                    
                # Filter out special tokens but track consecutive occurrences
                if (best_token == channel_token_id or best_token == message_token_id):
                    consecutive_special_tokens += 1
                    if consecutive_special_tokens >= max_consecutive_special:
                        print(f"[warning] Too many consecutive special tokens, stopping generation")
                        break
                    continue
                else:
                    consecutive_special_tokens = 0  # Reset counter on valid token
                
                # Generate the selected token
                generated_tokens.append(best_token)
                content_tokens_generated += 1
                    # Path memory is now updated inside generate_next_token
                
                # Safety check: if no content tokens generated after many attempts, break
                if step > 20 and content_tokens_generated == 0:
                    print(f"[warning] No content tokens generated after {step} attempts")
                    break
                
            except Exception as e:
                print(f"[warning] Token generation error at step {step}: {e}")
                break
        
        # End with stop token
        generated_tokens.append(stop_token_id)
        
        # Parse tokens back to messages (exclude stop token from parser input)
        if generated_tokens and len(generated_tokens) > 0:
            try:
                # Remove stop token before parsing as recommended
                tokens_for_parsing: List[int] = generated_tokens[:-1] if generated_tokens[-1] == stop_token_id else generated_tokens
                
                # Validate tokens before parsing
                if not tokens_for_parsing:
                    return "No valid tokens to parse"
                
                # Check for minimum viable token sequence
                if len(tokens_for_parsing) < 3:  # At least channel + content + message tokens
                    return "Insufficient tokens for valid message"
                
                # Import Role from harmony library for proper enum usage
                from openai_harmony import Role
                messages: List[Any] = encoding.parse_messages_from_completion_tokens(tokens_for_parsing, Role.ASSISTANT)
                
                if not messages:
                    return "No messages parsed from tokens"
                
                # Return only 'final' channel messages, ignore analysis/commentary
                for i, m in enumerate(messages):
                    channel = getattr(m, 'channel', None)
                    
                    if channel == "final":
                        # Extract text content from final channel message
                        if hasattr(m, 'content'):
                            if hasattr(m.content, 'text'):
                                content_text = m.content.text
                                if content_text and content_text.strip():
                                    return content_text.strip()
                            elif isinstance(m.content, list) and len(m.content) > 0:
                                # Handle list of TextContent objects
                                if hasattr(m.content[0], 'text'):
                                    content_text = m.content[0].text
                                    if content_text and content_text.strip():
                                        return content_text.strip()
                                else:
                                    content_str = str(m.content[0])
                                    if content_str and content_str.strip():
                                        return content_str.strip()
                            else:
                                content_str = str(m.content)
                                if content_str and content_str.strip():
                                    return content_str.strip()
                
                # If no final channel message found, try to extract any content
                for i, m in enumerate(messages):
                    if hasattr(m, 'content'):
                        if hasattr(m.content, 'text') and m.content.text:
                            return f"[{getattr(m, 'channel', 'unknown')}] {m.content.text.strip()}"
                
                # If no final channel message found, return diagnostic
                return "No final channel message found in response"
                
            except Exception as e:
                print(f"[error] Token parsing failed: {e}")
                return f"Parsing error: {e}"
        else:
            return "No tokens generated"

    except Exception as e:
        print(f"[error] Generation failed: {e}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU chat using GyroHead physics-based language model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens per response (default: 32)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="memories/models/gpt-oss-20b",
        help="Path to model directory containing model.gyro.safetensors (default: memories/models/gpt-oss-20b)",
    )

    args = parser.parse_args()

    # Set up model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[error] Model directory not found: {model_path}")
        return 1

    # Set up harmony format
    encoding, system_message = setup_harmony_format()
    if encoding is None or system_message is None:
        return 1

    # Load GyroHead model
    model = load_gyro_model(model_path)
    if model is None:
        return 1

    # Initialize conversation
    conversation: list[Any] = [system_message]

    print("\n[chat] GyroHead ready! Type your message. Use /exit to quit.\n")
    print("[info] Model: GyroHead physics-based language model (CPU-only)")
    print("[info] Format: Harmony response format")
    print("[info] Implementation: Pure gyroscopic intelligence")
    print("[info] Weights: Consolidated model.gyro.safetensors")
    print()

    while True:
        try:
            user_input = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input == "" or user_input.lower() == "/exit":
            break

        # Add user message to conversation
        try:
            from openai_harmony import Message as HarmonyMessage, Role
            
            user_message = HarmonyMessage.from_role_and_content(Role.USER, user_input)
        except Exception as e:
            print(f"[error] Harmony library unavailable: {e}")
            return 1
        conversation.append(user_message)

        # Generate response
        response = generate_response(
            encoding=encoding,
            gyro_model=model,
            conversation=conversation,
            max_new_tokens=args.max_new_tokens,
        )

        if response:
            print(f"assistant> {response}")
            # Add assistant response to conversation
            # The response is already a parsed message content, so we need to create a proper Message
            try:
                from openai_harmony import Message as HarmonyMessage, Role
                
                assistant_message = HarmonyMessage.from_role_and_content(Role.ASSISTANT, response)
                conversation.append(assistant_message)
            except Exception as e:
                print(f"[warning] Could not add response to conversation: {e}")
                # Continue without adding to conversation for now
        else:
            print("assistant> [Error: Failed to generate response]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

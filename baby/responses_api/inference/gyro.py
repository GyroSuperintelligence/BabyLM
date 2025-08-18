# baby/responses_api/inference/gyro.py

import json
from pathlib import Path
from typing import Callable, Optional
from openai_harmony import StreamableParser, Role
from baby.kernel.gyro_core import GyroEngine
from baby.constants.harmony_tokens import START, CHANNEL, MESSAGE, END, RETURN


def setup_model(checkpoint: str, encoding, config_path: str) -> Callable[[list[int], float], int]:
    """
    Returns infer_next_token(tokens_so_far, temperature) -> next_token_id
    
    Args:
        checkpoint: Model checkpoint path (unused for GyroSI)
        encoding: Harmony encoding instance
        config_path: Path to GyroSI configuration file
        
    Returns:
        Function that takes token list and temperature, returns next token ID
    """
    # Load GyroSI configuration
    cfg = json.loads(Path(config_path).read_text())
    
    # Initialize GyroEngine with version validation (vocab_size derives from engine/tokenizer)
    engine = GyroEngine(
        atlas_paths=cfg["atlas"],
        store_paths=cfg["stores"],
        runtime=cfg.get("runtime", {}),
        version_info=cfg.get("version", {}),
    )
    
    # Create parser for Harmony message processing
    parser = StreamableParser(encoding, role=Role.ASSISTANT)
    
    # Track end sequence state for clean termination
    end_sequence_state = {"step": 0}  # 0=normal, 1=emit_end, 2=emit_return
    
    def reconstruct_state(tokens: list[int]) -> tuple[int, bool]:
        """
        Returns (current_state, currently_inside_assistant_message)
        Processes tokens into messages; egress on USER, ingress on ASSISTANT.
        """
        print(f"DEBUG: reconstruct_state called with {len(tokens)} tokens")
        state = engine.start_state()
        # Create a fresh parser instance instead of trying to reset
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        inside_assistant = False
        
        # Process tokens through the parser
        try:
            for i, token in enumerate(tokens):
                parser.process(token)
                if i < 10 or i >= len(tokens) - 10:  # Log first and last 10 tokens
                    print(f"DEBUG: Processed token {i}: {token} -> '{encoding.decode([token])}'")
        except Exception as e:
            print(f"DEBUG: Error processing tokens in reconstruct_state: {e}")
            return state, False
            
        print(f"DEBUG: Parser has {len(parser.messages)} messages")
        
        # Process completed messages
        last_role = None
        for i, message in enumerate(parser.messages):
            # Access role from message.author.role
            role = message.author.role
            
            # Extract text content from message
            text = "".join(part.text for part in message.content if hasattr(part, "text"))
            
            print(f"DEBUG: Message {i}: role={role}, text='{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            if role == Role.USER:
                # Retokenize text and evolve state (egress)
                if text:
                    user_tokens = encoding.encode(text)
                    print(f"DEBUG: USER message, evolving state with {len(user_tokens)} tokens")
                    for token_id in user_tokens:
                        old_state = state
                        state = engine.evolve_on_user(state, token_id)
                        if old_state != state:
                            print(f"DEBUG: State evolved: {old_state:012x} -> {state:012x}")
                        
                last_role = Role.USER
                
            elif role == Role.ASSISTANT:
                # Retokenize text and evolve state (ingress)
                if text:
                    assistant_tokens = encoding.encode(text)
                    print(f"DEBUG: ASSISTANT message, evolving state with {len(assistant_tokens)} tokens")
                    for token_id in assistant_tokens:
                        old_state = state
                        state = engine.evolve_on_assistant(state, token_id)
                        if old_state != state:
                            print(f"DEBUG: State evolved: {old_state:012x} -> {state:012x}")
                        
                last_role = Role.ASSISTANT
                
            elif role == Role.TOOL:
                # Retokenize text and evolve state (ingress, no fold)
                # Tool outputs are treated as ingress to keep learning semantics aligned
                if text:
                    tool_tokens = encoding.encode(text)
                    print(f"DEBUG: TOOL message, evolving state with {len(tool_tokens)} tokens")
                    for token_id in tool_tokens:
                        old_state = state
                        state = engine.evolve_on_assistant(state, token_id)
                        if old_state != state:
                            print(f"DEBUG: State evolved: {old_state:012x} -> {state:012x}")
                        
                last_role = Role.TOOL
                
        # Check if we're currently inside an assistant message
        inside_assistant = (last_role == Role.ASSISTANT) and not parser.is_message_closed
        
        print(f"DEBUG: Final state={state:012x}, inside_assistant={inside_assistant}, last_role={last_role}")
        return state, inside_assistant
        
    def infer_next_token(tokens: list[int], temperature: float = 0.0) -> int:
        """
        Generate next token using GyroSI deterministic selection.
        
        Args:
            tokens: List of token IDs processed so far
            temperature: Ignored (GyroSI is deterministic)
            
        Returns:
            Next token ID
        """
        print(f"DEBUG: infer_next_token called with {len(tokens)} tokens, end_sequence_state={end_sequence_state}")
        
        # Handle end sequence state machine
        if end_sequence_state["step"] == 1:
            end_sequence_state["step"] = 2
            print("DEBUG: Returning RETURN token")
            return RETURN
        elif end_sequence_state["step"] == 2:
            end_sequence_state["step"] = 0  # Reset for next conversation
            print("DEBUG: Returning END token")
            return END
            
        # Special case: If this is the first token of the response, start with Harmony message header
        if len(tokens) == 0 or (len(tokens) > 0 and not any(t == START for t in tokens)):
            print("DEBUG: Starting new assistant message with START token")
            return START
            
        # If we just emitted START, follow with ASSISTANT_ROLE_TOKEN
        if tokens and tokens[-1] == START:
            print("DEBUG: Emitting ASSISTANT_ROLE_TOKEN after START")
            return ASSISTANT_ROLE_TOKEN
            
        # If we just emitted ASSISTANT_ROLE_TOKEN, follow with CHANNEL token
        if tokens and tokens[-1] == ASSISTANT_ROLE_TOKEN:
            print("DEBUG: Emitting CHANNEL token after ASSISTANT_ROLE_TOKEN")
            return CHANNEL
            
        # If we just emitted CHANNEL, follow with FINAL_CHANNEL token (35644 for 'final' channel)
        if tokens and tokens[-1] == CHANNEL:
            print("DEBUG: Emitting FINAL_CHANNEL token after CHANNEL")
            return FINAL_CHANNEL
            
        # If we just emitted FINAL_CHANNEL, follow with MESSAGE token
        if tokens and tokens[-1] == FINAL_CHANNEL:
            print("DEBUG: Emitting MESSAGE token after FINAL_CHANNEL")
            return MESSAGE
            
        # Reconstruct current state from token history
        try:
            state, inside_assistant = reconstruct_state(tokens)
            print(f"DEBUG: Reconstructed state={state:012x}, inside_assistant={inside_assistant}")
        except Exception as e:
            # Fallback to clean termination on parsing errors
            print(f"DEBUG: Exception in reconstruct_state: {e}")
            end_sequence_state["step"] = 1
            return END
            
        # Generate candidate vocabulary excluding Harmony control tokens
        vocab_size = engine.vocab_size
        
        # Define Harmony control tokens to exclude during normal generation
        harmony_control_tokens = {START, CHANNEL, MESSAGE}  # Allow END and RETURN from state machine only
        
        # Use orbit→tokens index for efficient candidate gathering if available
        if hasattr(engine, '_orbit_to_tokens') and state in engine.state_to_index:
            # Get current state's orbit
            state_idx = engine.state_to_index[state]
            current_orbit = engine.phenomenology_map[state_idx]
            
            # Get tokens in current orbit
            if current_orbit in engine._orbit_to_tokens:
                orbit_tokens = engine._orbit_to_tokens[current_orbit]
                # Filter out Harmony control tokens and ensure vocab bounds
                candidate_vocab = [tok for tok in orbit_tokens 
                                 if 0 <= tok < vocab_size and tok not in harmony_control_tokens]
            else:
                # Fallback: use bounded lazy address materialization
                candidate_vocab = []
                orbit_tokens = engine._get_tokens_in_orbit(current_orbit)
                for tok in orbit_tokens:
                    if 0 <= tok < vocab_size and tok not in harmony_control_tokens:
                        candidate_vocab.append(tok)
        else:
            # Fallback: bounded scan excluding control tokens (limit to 512 for performance)
            scan_limit = min(512, vocab_size)
            candidate_vocab = [tok for tok in range(scan_limit) 
                             if tok not in harmony_control_tokens]
        
        # Get next token from GyroSI engine
        next_token = engine.next_token_deterministic(state, candidate_vocab)
        
        # Debug logging
        print(f"DEBUG: state={state:012x}, candidate_vocab_size={len(candidate_vocab)}, next_token={next_token}")
        if next_token is None:
            print(f"DEBUG: No admissible tokens found for state {state:012x}")
        
        if next_token is None:
            # Engine requests halt - begin end sequence
            end_sequence_state["step"] = 1
            return END
            
        # Validate token is in vocabulary
        if next_token >= vocab_size:
            # Invalid token - begin end sequence
            end_sequence_state["step"] = 1
            return END
            
        return next_token
        
    return infer_next_token


# Channel token IDs
FINAL_CHANNEL = 35644  # Token ID for 'final' channel


# Add a function to get the token ID for 'assistant' role
def get_assistant_role_token_id():
    """Get the token ID for 'assistant' role."""
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    # Tokenize 'assistant' to get its token ID
    tokens = encoding.encode('assistant')
    if tokens:
        return tokens[0]  # Return the first token
    return None

# Get the assistant role token ID
ASSISTANT_ROLE_TOKEN = get_assistant_role_token_id()
print(f"DEBUG: ASSISTANT_ROLE_TOKEN = {ASSISTANT_ROLE_TOKEN}")
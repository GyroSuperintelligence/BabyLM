# baby/responses_api/inference/gyro.py

import json
from pathlib import Path
from typing import Callable, Optional
from openai_harmony import StreamableParser, Role
from baby.kernel.gyro_core import GyroEngine
from baby.constants.harmony_tokens import END, START, RETURN, CHANNEL, MESSAGE, ALL_CONTROL_TOKENS


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
    
    # Initialize GyroEngine with version validation and vocab_size
    engine = GyroEngine(
        atlas_paths=cfg["atlas"],
        store_paths=cfg["stores"],
        runtime=cfg.get("runtime", {}),
        version_info=cfg.get("version", {}),
        vocab_size=int(encoding.vocab_size)
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
        state = engine.start_state()
        parser.reset()
        inside_assistant = False
        
        # Feed all tokens to parser
        for tok in tokens:
            parser.process(tok)
            
        # Process completed messages
        last_role = None
        for message in parser.messages:
            if message.role == Role.USER:
                # Extract text content from USER message
                text = "".join(part.text for part in message.content if hasattr(part, "text"))
                
                # Retokenize text and evolve state (egress)
                if text:
                    user_tokens = encoding.encode(text)
                    for token_id in user_tokens:
                        state = engine.evolve_on_user(state, token_id)
                        
                last_role = Role.USER
                
            elif message.role == Role.ASSISTANT:
                # Extract text content from ASSISTANT message
                text = "".join(part.text for part in message.content if hasattr(part, "text"))
                
                # Retokenize text and evolve state (ingress)
                if text:
                    assistant_tokens = encoding.encode(text)
                    for token_id in assistant_tokens:
                        state = engine.evolve_on_assistant(state, token_id)
                        
                last_role = Role.ASSISTANT
                
            elif message.role == Role.TOOL:
                # Extract text content from TOOL message
                text = "".join(part.text for part in message.content if hasattr(part, "text"))
                
                # Retokenize text and evolve state (ingress, no fold)
                # Tool outputs are treated as ingress to keep learning semantics aligned
                if text:
                    tool_tokens = encoding.encode(text)
                    for token_id in tool_tokens:
                        state = engine.evolve_on_assistant(state, token_id)
                        
                last_role = Role.TOOL
                
        # Check if we're currently inside an assistant message
        inside_assistant = (last_role == Role.ASSISTANT) and not parser.is_message_closed
        
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
        # Handle end sequence state machine
        if end_sequence_state["step"] == 1:
            end_sequence_state["step"] = 2
            return RETURN
        elif end_sequence_state["step"] == 2:
            end_sequence_state["step"] = 0  # Reset for next conversation
            return END
            
        # Reconstruct current state from token history
        try:
            state, inside_assistant = reconstruct_state(tokens)
        except Exception:
            # Fallback to clean termination on parsing errors
            end_sequence_state["step"] = 1
            return END
            
        # Generate candidate vocabulary excluding Harmony control tokens
        vocab_size = encoding.vocab_size
        
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
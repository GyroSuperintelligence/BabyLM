"""
LEB128 Integration with GyroSI Core Pipeline

This module integrates the LEB128 Ôåö GyroSI physics mapping into the core
learning and generation pipeline, enabling token-level state transitions
and endogenous compression.
"""

import numpy as np
from typing import List, Tuple, Optional, Iterator
from toys.experiments.leb128_physics import token_to_introns, introns_to_token, ¤ê, ¤ê_inv

class LEB128GyroSI:
    """Integrated LEB128 Ôåö GyroSI physics for core pipeline."""
    
    def __init__(self, epistemology: np.ndarray, phenomenology_map: Optional[np.ndarray] = None):
        self.epistemology = epistemology
        self.phenomenology_map = phenomenology_map
        self.token_stt_cache = {}  # Cache for token-level state transitions
        
    def apply_token_physics(self, state_index: int, token_id: int) -> int:
        """Apply a token's physics directly using LEB128 intron sequence."""
        introns = token_to_introns(token_id)
        
        current_state = state_index
        for intron in introns:
            current_state = self.epistemology[current_state, intron]
        
        return current_state
    
    def get_token_transition(self, state_index: int, token_id: int) -> int:
        """Get the final state after applying a token's intron sequence (cached)."""
        key = (state_index, token_id)
        
        if key not in self.token_stt_cache:
            self.token_stt_cache[key] = self.apply_token_physics(state_index, token_id)
        
        return self.token_stt_cache[key]
    
    def learn_token_leb128(self, token_id: int, state_index: int, store) -> dict:
        """
        Token-level learning using LEB128 physics.
        
        Args:
            token_id: Token ID from tokenizer
            state_index: Current state index
            store: Phenotype store
            
        Returns:
            Updated phenotype entry
        """
        # Get the final state after applying the token's physics
        final_state = self.apply_token_physics(state_index, token_id)
        
        # Get the last intron from the token's LEB128 sequence
        introns = token_to_introns(token_id)
        last_intron = introns[-1] if introns else 0
        
        # Create or get phenotype entry
        storage_key = (final_state, token_id)
        entry = store.get(storage_key)
        
        if entry is None:
            entry = {
                "key": (final_state, token_id),
                "mask": last_intron,  # Initialize with last intron
                "conf": 0.1
            }
        else:
            entry = dict(entry)  # Copy to avoid mutation
            
            # Update mask using Monodromic Fold with last intron
            old_mask = entry.get("mask", 0) & 0xFF
            new_mask = self._fold_mask(old_mask, last_intron)
            entry["mask"] = new_mask
        
        # Update confidence based on orbit cardinality
        if self.phenomenology_map is not None:
            orbit_id = int(self.phenomenology_map[final_state])
            orbit_size = self._get_orbit_cardinality(orbit_id)
            alpha = (1 / 6) * np.sqrt(orbit_size / 1000)  # Assuming max_variety = 1000
        else:
            alpha = 0.1  # Default learning rate
        
        current_conf = entry.get("conf", 0.1)
        new_conf = min(1.0, current_conf + (1 - current_conf) * alpha)
        entry["conf"] = new_conf
        
        # Store the updated entry
        store.put(storage_key, entry)
        
        return entry
    
    def generate_token_leb128(self, state_index: int, store, temperature: float = 1.0) -> int:
        """
        Generate next token using LEB128 physics and learned phenotypes.
        
        Args:
            state_index: Current state index
            store: Phenotype store
            temperature: Generation temperature (0.0 = deterministic, 1.0 = random)
            
        Returns:
            Generated token_id
        """
        # Get candidate tokens from store
        candidates = []
        
        # Scan store for phenotypes with this state_index
        for key, entry in store.iter_entries():
            if key[0] == state_index:  # Same state
                token_id = key[1]
                confidence = entry.get("conf", 0.1)
                mask = entry.get("mask", 0)
                
                # Calculate resonance with current state
                resonance = self._calculate_resonance(state_index, mask)
                score = confidence * resonance
                
                candidates.append((token_id, score))
        
        if not candidates:
            # No learned tokens for this state, generate random
            return self._generate_random_token()
        
        # Sort by score and apply temperature
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if temperature < 0.1:
            # Deterministic: return highest scoring token
            return candidates[0][0]
        else:
            # Probabilistic: sample based on scores and temperature
            scores = np.array([score for _, score in candidates])
            probs = scores ** (1 / temperature)
            probs = probs / np.sum(probs)
            
            chosen_idx = np.random.choice(len(candidates), p=probs)
            return candidates[chosen_idx][0]
    
    def _fold_mask(self, old_mask: int, intron: int) -> int:
        """Apply Monodromic Fold to mask using intron."""
        # Simple XOR fold (can be enhanced with more complex folding)
        return (old_mask ^ intron) & 0xFF
    
    def _calculate_resonance(self, state_index: int, mask: int) -> float:
        """Calculate resonance between state and mask."""
        # Simple resonance calculation (can be enhanced)
        state_bits = bin(state_index)[2:].zfill(8)[-8:]  # Last 8 bits
        mask_bits = bin(mask)[2:].zfill(8)
        
        # Count matching bits
        matches = sum(1 for a, b in zip(state_bits, mask_bits) if a == b)
        return matches / 8.0
    
    def _get_orbit_cardinality(self, orbit_id: int) -> int:
        """Get cardinality of orbit (simplified)."""
        # This should be replaced with actual orbit cardinality lookup
        return 100  # Default value
    
    def _generate_random_token(self) -> int:
        """Generate a random token ID."""
        # This should be replaced with actual vocabulary size
        return np.random.randint(1, 30000)  # BERT vocab size approx

# Integration with existing InferenceEngine
def integrate_leb128_physics(engine, epistemology: np.ndarray, phenomenology_map: Optional[np.ndarray] = None):
    """Integrate LEB128 physics into existing InferenceEngine."""
    
    leb128_engine = LEB128GyroSI(epistemology, phenomenology_map)
    
    # Override learn_token method
    def learn_token_leb128_wrapper(token_id: int, state_index: int, last_intron: int):
        return leb128_engine.learn_token_leb128(token_id, state_index, engine.store)
    
    engine.learn_token = learn_token_leb128_wrapper
    
    # Add token-level generation method
    def generate_token_leb128(state_index: int, temperature: float = 1.0):
        return leb128_engine.generate_token_leb128(state_index, engine.store, temperature)
    
    engine.generate_token_leb128 = generate_token_leb128
    
    return engine

# Stream processing with LEB128 physics
def process_text_stream_leb128(text_stream: Iterator[str], tokenizer, engine) -> Iterator[int]:
    """Process text stream using LEB128 physics."""
    
    leb128_engine = LEB128GyroSI(engine.epistemology, engine.phenomenology_map)
    current_state = 0  # Start from archetypal state
    
    for text in text_stream:
        for token_id in tokenizer.encode(text):
            # Learn the token
            entry = leb128_engine.learn_token_leb128(token_id, current_state, engine.store)
            
            # Update state
            current_state = leb128_engine.apply_token_physics(current_state, token_id)
            
            yield token_id

def generate_text_stream_leb128(engine, initial_prompt: str, max_tokens: int = 50) -> Iterator[str]:
    """Generate text stream using LEB128 physics."""
    
    leb128_engine = LEB128GyroSI(engine.epistemology, engine.phenomenology_map)
    current_state = 0  # Start from archetypal state
    
    # Process initial prompt
    for token_id in engine.tokenizer.encode(initial_prompt):
        current_state = leb128_engine.apply_token_physics(current_state, token_id)
    
    # Generate continuation
    for _ in range(max_tokens):
        token_id = leb128_engine.generate_token_leb128(current_state, engine.store)
        current_state = leb128_engine.apply_token_physics(current_state, token_id)
        
        yield engine.tokenizer.decode([token_id]) 

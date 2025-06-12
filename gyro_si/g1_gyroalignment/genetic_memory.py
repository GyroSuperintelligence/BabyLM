"""
Genetic memory module for G1 alignment system.
Manages core genetic patterns and their evolution.
"""

from typing import Dict, Any, List
import json
from pathlib import Path

class GeneticMemory:
    """Manages genetic memory for G1 alignment."""
    
    def __init__(self, patterns_dir: Path):
        self.patterns_dir = patterns_dir
        self.patterns: Dict[str, Any] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load genetic patterns from disk."""
        pattern_files = self.patterns_dir.glob("**/*.json")
        for file in pattern_files:
            with open(file, 'r') as f:
                self.patterns[file.stem] = json.load(f)

    def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """Retrieve a genetic pattern by ID."""
        return self.patterns.get(pattern_id, {})

    def store_pattern(self, pattern_id: str, pattern: Dict[str, Any]) -> None:
        """Store a new genetic pattern."""
        self.patterns[pattern_id] = pattern
        pattern_file = self.patterns_dir / f"{pattern_id}.json"
        with open(pattern_file, 'w') as f:
            json.dump(pattern, f, indent=2) 
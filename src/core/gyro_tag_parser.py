"""
gyro_tag_parser.py - TAG Expression Parser

Implements the TAG grammar validation and parsing per CORE-SPEC-04.
"""

import re
from typing import Dict, Optional


def validate_tag(tag: str) -> bool:
    """
    Validate TAG syntax per CORE-SPEC-04.
    
    Format: <temporal>.<invariant>[.<context>]
    temporal ∈ {previous, current, next}
    
    Args:
        tag: TAG expression to validate
        
    Returns:
        True if valid TAG syntax
    """
    if not isinstance(tag, str) or not tag:
        return False
    
    # TAG pattern: temporal.invariant[.context]
    pattern = r'^(previous|current|next)\.(gyrotensor_id|gyrotensor_com|gyrotensor_nest|gyrotensor_add|gyrotensor_quant)(?:\.[\w_]+)?$'
    
    return bool(re.match(pattern, tag))


def parse_tag(tag: str) -> Dict[str, str]:
    """
    Parse TAG expression into components.
    
    Args:
        tag: Valid TAG expression
        
    Returns:
        Dictionary with 'temporal', 'invariant', and optional 'context'
        
    Raises:
        ValueError: If TAG is invalid
    """
    if not validate_tag(tag):
        raise ValueError(f"Invalid TAG expression: {tag}")
    
    parts = tag.split('.')
    
    result = {
        'temporal': parts[0],
        'invariant': parts[1]
    }
    
    if len(parts) > 2:
        result['context'] = parts[2]
    
    return result
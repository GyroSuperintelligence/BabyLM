from typing import TypedDict, List, Optional, Dict, Any, Union


class PatternMetadata(TypedDict, total=False):
    index: int
    semantic: Optional[Union[str, List[str], int]]
    count: int
    first_cycle: Optional[int]
    last_cycle: Optional[int]
    resonance_class: str
    confidence: float


class FormatMetadata(TypedDict, total=False):
    format_uuid: str
    format_name: str
    cgm_version: str
    format_version: str
    stability: str
    compatibility: Dict[str, Any]
    metadata: Dict[str, Any]
    cgm_policies: Dict[str, Any]
    patterns: List[PatternMetadata]

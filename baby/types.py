from typing import TypedDict, List, Optional, Dict, Any, Union


class PatternMetadata(TypedDict, total=False):
    index: int
    character: Optional[Union[str, List[str], int]]
    description: Optional[str]
    type: Optional[str]
    count: int
    first_cycle: Optional[int]
    last_cycle: Optional[int]
    gyration_feature: str
    confidence: float


class FormatMetadata(TypedDict, total=False):
    format_uuid: str
    format_name: str
    format_version: str
    stability: str
    compatibility: Dict[str, Any]
    metadata: Dict[str, Any]
    cgm_policies: Dict[str, Any]
    patterns: List[PatternMetadata]

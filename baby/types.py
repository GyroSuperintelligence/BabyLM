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


class ChildRef(TypedDict):
    uuid: str
    name: Optional[str]


class ThreadMetadata(TypedDict, total=False):
    thread_uuid: str
    thread_name: Optional[str]
    agent_uuid: Optional[str]  # Deprecated: use 'privacy' instead
    parent_uuid: Optional[str]
    parent_name: Optional[str]
    children: List[ChildRef]
    format_uuid: str
    curriculum: Optional[str]
    tags: Optional[List[str]]
    created_at: str
    last_updated: str
    size_bytes: int
    privacy: str  # 'public' or 'private'


class GeneKeysMetadata(TypedDict):
    cycle: int
    pattern_index: int
    thread_uuid: str
    format_uuid: str
    event_type: str
    source_byte: int
    resonance: float
    created_at: str
    privacy: str  # 'public' or 'private'
    agent_uuid: Optional[str]  # Agent association/ownership only; not used for privacy logic

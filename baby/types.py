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


class ThreadMetadata(TypedDict, total=False):
    thread_uuid: str
    thread_name: Optional[str]
    agent_uuid: Optional[str]
    parent_uuid: Optional[str]
    parent_name: Optional[str]
    child_uuids: List[str]
    child_names: List[Optional[str]]
    format_uuid: str
    curriculum: Optional[str]
    tags: Optional[List[str]]
    created_at: str
    last_updated: str
    size_bytes: int


class GeneKeysMetadata(TypedDict):
    cycle: int
    pattern_index: int
    thread_uuid: str
    agent_uuid: Optional[str]
    format_uuid: str
    event_type: str
    source_byte: int
    resonance: float
    created_at: str

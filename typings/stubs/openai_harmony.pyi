# Type stubs for openai_harmony module
from typing import Any, List, Optional, Union, Dict, Literal

# String literal types for better compatibility
HarmonyEncodingNameType = Literal["harmony-gpt-oss"]
ReasoningEffort = Literal["low", "medium", "high"]
RoleType = Literal["system", "user", "assistant", "developer"]

# Enum-like class for HarmonyEncodingName
class HarmonyEncodingName:
    HARMONY_GPT_OSS: str = "harmony-gpt-oss"

# Enum-like classes for attribute access
class Role:
    SYSTEM: str = "system"
    USER: str = "user"
    ASSISTANT: str = "assistant"
    DEVELOPER: str = "developer"

class SystemContent:
    @classmethod
    def new(cls) -> 'SystemContent': ...
    def with_model_identity(self, identity: str) -> 'SystemContent': ...
    def with_reasoning_effort(self, effort: Union[ReasoningEffort, str]) -> 'SystemContent': ...
    def with_conversation_start_date(self, date: str) -> 'SystemContent': ...
    def with_knowledge_cutoff(self, cutoff: str) -> 'SystemContent': ...
    def with_required_channels(self, channels: List[str]) -> 'SystemContent': ...

class TextContent:
    text: str
    def __init__(self, text: str) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class Message:
    role: Union[str, RoleType]
    content: Union[str, List[TextContent], List[Any], SystemContent, Any]
    channel: Optional[str]
    
    @classmethod
    def from_role_and_content(cls, role: Union[RoleType, str], content: Union[str, SystemContent]) -> 'Message': ...
    def __getattr__(self, name: str) -> Any: ...

class Conversation:
    messages: List[Message]
    def __init__(self, messages: List[Message]) -> None: ...
    @classmethod
    def from_messages(cls, messages: List[Message]) -> 'Conversation': ...
    def to_tokens(self, encoding: Any) -> List[int]: ...

class StreamableParser:
    current_role: Optional[RoleType]
    current_channel: Optional[str]
    last_content_delta: str
    current_content_type: str
    current_recipient: Optional[str]
    current_content: str
    
    def __init__(self, encoding: Any, role: Union[RoleType, str]) -> None: ...
    def process(self, token: int) -> None: ...
    def parse_tokens(self, tokens: List[int]) -> List[Message]: ...

def load_harmony_encoding(name: Union[HarmonyEncodingNameType, str]) -> Any: ...
"""
state.py - Global state management for GyroSI Baby LM

Manages application state including active agent, threads, and UI state.
"""

from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from s4_intelligence.g1_intelligence_in import IntelligenceEngine
from s4_intelligence.g2_intelligence_eg import MessageStore


@dataclass
class Message:
    """Represents a single message in a thread."""

    id: str
    thread_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to format for storage."""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from format."""
        return cls(
            id=data["id"],
            thread_id=data["thread_id"],
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Thread:
    """Represents a conversation thread."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert thread to format for storage."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thread":
        """Create thread from format."""
        return cls(
            id=data["id"],
            title=data["title"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            message_count=data.get("message_count", 0),
        )


class AppState:
    """
    Central state management for the GyroSI Baby LM app.

    Manages:
    - Active IntelligenceEngine instance
    - Current thread and messages
    - UI state (processing, errors)
    - Settings and preferences
    """

    def __init__(self, base_path: str = "s2_information"):
        # Core engine state
        self.base_path = base_path
        self.current_agent: Optional[IntelligenceEngine] = None
        self.agent_uuid: Optional[str] = None
        self.message_store: Optional[MessageStore] = None

        # Thread state
        self.current_thread_id: Optional[str] = None
        self.threads: Dict[str, Thread] = {}
        self.current_messages: List[Message] = []

        # UI state
        self.processing: bool = False
        self.error_message: Optional[str] = None
        self.status_message: Optional[str] = None
        self.processing_stats: Dict[str, Any] = {}

        # Settings
        self.settings: Dict[str, Any] = {
            "encryption_enabled": True,
            "max_recent_messages": 250,
            "auto_save": True,
            "theme": "dark",
            "show_dev_info": False,
        }

        # Callbacks for UI updates
        self._update_callbacks: List[Callable] = []

    def add_update_callback(self, callback: Callable) -> None:
        """Add a callback to be called when state changes."""
        self._update_callbacks.append(callback)

    def remove_update_callback(self, callback: Callable) -> None:
        """Remove an update callback."""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)

    def _notify_updates(self) -> None:
        """Notify all registered callbacks of state change."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in update callback: {e}")

    def set_agent(self, agent_uuid: str, encryption_key: Optional[bytes] = None) -> None:
        """
        Set the active agent and initialize engines.

        Args:
            agent_uuid: UUID of the agent to activate
            encryption_key: Optional encryption key for the agent
        """
        try:
            # Close existing agent if any
            if self.current_agent:
                self.current_agent.close()

            # Initialize new agent
            self.agent_uuid = agent_uuid
            self.current_agent = IntelligenceEngine(
                agent_uuid=agent_uuid,
                base_path=self.base_path,
                encryption_enabled=self.settings["encryption_enabled"],
                gyrocrypt_key=encryption_key,
            )

            # Initialize message store
            self.message_store = MessageStore(agent_uuid=agent_uuid, base_path=self.base_path)

            # Load threads for this agent
            self.load_threads()

            self.error_message = None
            self._notify_updates()

        except Exception as e:
            self.error_message = f"Failed to set agent: {str(e)}"
            self._notify_updates()
            raise

    def create_thread(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation thread.

        Args:
            title: Optional title for the thread

        Returns:
            Thread ID
        """
        thread_id = str(uuid.uuid4())
        now = datetime.utcnow()

        thread = Thread(
            id=thread_id,
            title=title or f"Thread {now.strftime('%Y-%m-%d %H:%M')}",
            created_at=now,
            updated_at=now,
        )

        self.threads[thread_id] = thread
        self.current_thread_id = thread_id
        self.current_messages = []

        self._notify_updates()
        return thread_id

    def set_current_thread(self, thread_id: str) -> None:
        """
        Switch to a different thread.

        Args:
            thread_id: ID of the thread to switch to
        """
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")

        self.current_thread_id = thread_id
        self.load_thread_messages()
        self._notify_updates()

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to the current thread.

        Args:
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata

        Returns:
            Created message
        """
        # Ensure we have a current thread
        if not self.current_thread_id:
            self.create_thread()

        # At this point, we're guaranteed to have a valid thread_id
        assert self.current_thread_id is not None, "Thread ID should not be None here"

        message = Message(
            id=str(uuid.uuid4()),
            thread_id=self.current_thread_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )

        self.current_messages.append(message)

        # Update thread
        if self.current_thread_id in self.threads:
            thread = self.threads[self.current_thread_id]
            thread.updated_at = message.timestamp
            thread.message_count += 1

        # Save to message store
        if self.message_store and self.settings.get("auto_save", True):
            self.save_thread_messages()

        self._notify_updates()
        return message

    def process_user_input(self, text: str) -> None:
        """
        Process user text input through the IntelligenceEngine.

        Args:
            text: User input text
        """
        if not self.current_agent:
            self.error_message = "No agent selected"
            self._notify_updates()
            return

        self.processing = True
        self.error_message = None
        self._notify_updates()

        try:
            # Tokenize input (default: list of characters, but could be any tokenization)
            tokens = list(text)
            # Use new instance-based codec (agent+global format)
            data_bytes = self.current_agent.encode_tokens_to_bytes(tokens)

            # Add user message
            user_message = self.add_message("user", text)

            # Process through engine
            artifacts = self.current_agent.process_stream(data_bytes)

            # Store processing stats
            self.processing_stats = {
                "accepted_ops": len(artifacts.get("accepted_ops", [])),
                "resonances": len(artifacts.get("resonances", [])),
                "compressed_blocks": len(artifacts.get("compressed_blocks", [])),
                "pattern_promotions": len(artifacts.get("pattern_promotions", [])),
            }

            # Generate response
            response_bytes = self.current_agent.generate(b"", max_length=200)
            response_tokens = self.current_agent.decode_bytes_to_tokens(response_bytes)
            # Join tokens for display if they are strings
            response_text = (
                "".join(response_tokens)
                if response_tokens and isinstance(response_tokens[0], str)
                else str(response_tokens)
            )

            # Add assistant message
            self.add_message(
                "assistant", response_text, metadata={"processing_stats": self.processing_stats}
            )

        except Exception as e:
            self.error_message = f"Processing error: {str(e)}"

        finally:
            self.processing = False
            self._notify_updates()

    def process_file(self, file_path: str) -> None:
        """Process a file through the IntelligenceEngine using the codec."""
        if not self.current_agent:
            self.error_message = "No agent selected"
            self._notify_updates()
            return

        self.processing = True
        self.error_message = None
        self._notify_updates()

        try:
            # Read file
            with open(file_path, "rb") as f:
                data_bytes = f.read()

            file_name = file_path.split("/")[-1]

            # Try to decode the file content using the codec
            try:
                # Decode bytes to tokens using the agent's format
                tokens = self.current_agent.decode_bytes_to_tokens(data_bytes)

                # For display, join tokens (could be characters, words, etc.)
                content_preview = "".join(str(t) for t in tokens[:100])  # First 100 tokens
                if len(tokens) > 100:
                    content_preview += "..."

                # Add user message with decoded content
                self.add_message(
                    "user",
                    f"[File: {file_name}]\n{content_preview}",
                    metadata={
                        "file_path": file_path,
                        "file_size": len(data_bytes),
                        "token_count": len(tokens),
                    },
                )

            except Exception as decode_error:
                # If decode fails, show raw bytes info
                self.add_message(
                    "user",
                    f"[Binary file: {file_name}, {len(data_bytes)} bytes]",
                    metadata={"file_path": file_path, "file_size": len(data_bytes)},
                )

            # Process through engine (this learns patterns)
            artifacts = self.current_agent.process_stream(data_bytes)

            # Generate a response based on what was learned
            response_bytes = self.current_agent.generate(b"", max_length=50)
            response_tokens = self.current_agent.decode_bytes_to_tokens(response_bytes)
            response_text = "".join(str(t) for t in response_tokens)

            # Store processing stats
            self.processing_stats = {
                "file_name": file_name,
                "bytes_processed": len(data_bytes),
                "accepted_ops": len(artifacts.get("accepted_ops", [])),
                "resonances": len(artifacts.get("resonances", [])),
                "compressed_blocks": len(artifacts.get("compressed_blocks", [])),
                "pattern_promotions": len(artifacts.get("pattern_promotions", [])),
                "new_patterns": len([p for p in artifacts.get("pattern_promotions", [])]),
            }

            # Add meaningful response
            self.add_message(
                "assistant",
                f"Processed {file_name}. Learned {self.processing_stats['new_patterns']} new patterns.\n{response_text}",
                metadata={"processing_stats": self.processing_stats},
            )

        except Exception as e:
            self.error_message = f"File processing error: {str(e)}"

        finally:
            self.processing = False
            self._notify_updates()

    def load_threads(self) -> None:
        """Load threads for the current agent."""
        # In a real implementation, this would load from persistent storage
        # For now, we'll start with empty threads
        self.threads = {}

    def load_thread_messages(self) -> None:
        """Load messages for the current thread."""
        if not self.message_store or not self.current_thread_id:
            self.current_messages = []
            return

        try:
            message_dicts = self.message_store.load_recent(self.current_thread_id)
            self.current_messages = [Message.from_dict(m) for m in message_dicts]
        except Exception as e:
            print(f"Error loading messages: {e}")
            self.current_messages = []

    def save_thread_messages(self) -> None:
        """Save current thread messages."""
        if not self.message_store or not self.current_thread_id:
            return

        try:
            message_dicts = [m.to_dict() for m in self.current_messages]
            self.message_store.write_recent(self.current_thread_id, message_dicts)
        except Exception as e:
            print(f"Error saving messages: {e}")

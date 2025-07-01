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
        """Convert message to dictionary for storage."""
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
        """Create message from dictionary."""
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
        """Convert thread to dictionary for storage."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thread":
        """Create thread from dictionary."""
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
            self._load_threads()

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
        self._load_thread_messages()
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
            self._save_thread_messages()

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
            # Add user message
            user_message = self.add_message("user", text)

            # Process through engine
            data_bytes = text.encode("utf-8")
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
            response_text = response_bytes.decode("utf-8", errors="replace")

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
        """
        Process a file through the IntelligenceEngine.

        Args:
            file_path: Path to the file to process
        """
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

            # Add user message
            file_name = file_path.split("/")[-1]
            self.add_message(
                "user",
                f"[Uploaded file: {file_name}]",
                metadata={"file_path": file_path, "file_size": len(data_bytes)},
            )

            # Process through engine
            artifacts = self.current_agent.process_stream(data_bytes)

            # Store processing stats
            self.processing_stats = {
                "file_name": file_name,
                "bytes_processed": len(data_bytes),
                "accepted_ops": len(artifacts.get("accepted_ops", [])),
                "resonances": len(artifacts.get("resonances", [])),
                "compressed_blocks": len(artifacts.get("compressed_blocks", [])),
                "pattern_promotions": len(artifacts.get("pattern_promotions", [])),
            }

            # Add response
            self.add_message(
                "assistant",
                f"Processed {len(data_bytes)} bytes from {file_name}",
                metadata={"processing_stats": self.processing_stats},
            )

        except Exception as e:
            self.error_message = f"File processing error: {str(e)}"

        finally:
            self.processing = False
            self._notify_updates()

    def get_engine_state(self) -> Dict[str, Any]:
        """Get current state of the IntelligenceEngine."""
        if not self.current_agent:
            return {}
        return self.current_agent.get_state()

    def _load_threads(self) -> None:
        """Load threads for the current agent."""
        # In a real implementation, this would load from persistent storage
        # For now, we'll start with empty threads
        self.threads = {}

    def _load_thread_messages(self) -> None:
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

    def _save_thread_messages(self) -> None:
        """Save current thread messages."""
        if not self.message_store or not self.current_thread_id:
            return

        try:
            message_dicts = [m.to_dict() for m in self.current_messages]
            self.message_store.write_recent(self.current_thread_id, message_dicts)
        except Exception as e:
            print(f"Error saving messages: {e}")

    def clear_error(self) -> None:
        """Clear the current error message."""
        self.error_message = None
        self._notify_updates()

    def update_setting(self, key: str, value: Any) -> None:
        """Update a setting value."""
        self.settings[key] = value
        self._notify_updates()

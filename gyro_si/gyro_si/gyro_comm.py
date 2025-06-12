"""
Communication utilities for GyroSI system.
Provides message passing and subscription mechanisms.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    """Base message structure for inter-system communication."""
    sender: str
    recipient: str
    content: Any
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None

class MessageBus:
    """Message bus for inter-system communication."""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()

    async def send_message(self, message: Message) -> None:
        """Send a message to the bus."""
        await self._message_queue.put(message)

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Unsubscribe from a topic."""
        if topic in self._subscribers:
            self._subscribers[topic].remove(callback)

    async def process_messages(self) -> None:
        """Process messages in the queue."""
        while True:
            message = await self._message_queue.get()
            if message.recipient in self._subscribers:
                for callback in self._subscribers[message.recipient]:
                    await callback(message)
            self._message_queue.task_done()

# Global message bus instance
message_bus = MessageBus()

async def send_message(recipient: str, content: Any, sender: str = "system") -> None:
    """Send a message to a recipient."""
    message = Message(sender=sender, recipient=recipient, content=content)
    await message_bus.send_message(message)

def subscribe(topic: str, callback: Callable) -> None:
    """Subscribe to a topic."""
    message_bus.subscribe(topic, callback)

def unsubscribe(topic: str, callback: Callable) -> None:
    """Unsubscribe from a topic."""
    message_bus.unsubscribe(topic, callback) 
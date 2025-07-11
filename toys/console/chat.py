"""
Chat interface for Baby LM Console.
"""

import base64
from typing import Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from baby.intelligence import IntelligenceEngine
from .utils import create_header, create_footer, handle_error, format_bytes

console = Console()


def safe_format_uuid(uuid_val: Optional[str]) -> str:
    if uuid_val:
        return uuid_val[:8] + "..."
    return "None"


class ChatInterface:
    """Interactive chat interface for Baby LM."""

    def __init__(self, engine: IntelligenceEngine) -> None:
        self.engine = engine
        self.conversation_history: list[dict[str, str]] = []

    def display_conversation(self) -> None:
        """Display the current conversation history."""
        if not self.conversation_history:
            console.print("[dim]No conversation history yet. Start by typing a message![/]")
            return

        for entry in self.conversation_history[-10:]:  # Show last 10 messages
            if entry["sender"] == "user":
                console.print(Panel(entry["message"], title="👤 You", border_style="blue", title_align="left"))
            elif entry["sender"] == "ai":
                console.print(
                    Panel(Markdown(entry["message"]), title="🤖 BabyLM", border_style="magenta", title_align="left")
                )
            elif entry["sender"] == "system":
                console.print(
                    Panel(
                        f"[italic]{entry['message']}[/italic]",
                        title="🔧 System",
                        border_style="yellow",
                        title_align="left",
                    )
                )

    def display_stats(self) -> None:
        """Display current session statistics."""
        try:
            stats_text = Text()

            # Basic stats
            cycle_count = self.engine.inference_engine.cycle_counter
            thread_size = self.engine.current_thread_size
            thread_id = safe_format_uuid(self.engine.thread_uuid)

            stats_text.append("Cycle: ", style="dim")
            stats_text.append(f"{cycle_count:,}", style="bold cyan")
            stats_text.append(" | Thread Size: ", style="dim")
            stats_text.append(format_bytes(thread_size), style="bold yellow")
            stats_text.append(" | Thread: ", style="dim")
            stats_text.append(thread_id, style="dim")

            # Cache stats
            inference_engine = self.engine.inference_engine
            cache_hits = getattr(inference_engine, "_cache_hits", 0)
            cache_misses = getattr(inference_engine, "_cache_misses", 0)
            total_cache = cache_hits + cache_misses

            if total_cache > 0:
                cache_ratio = cache_hits / total_cache
                stats_text.append(" | Cache: ", style="dim")
                stats_text.append(f"{cache_ratio:.1%}", style="bold green" if cache_ratio > 0.5 else "bold red")

            console.print(stats_text)

        except Exception as e:
            console.print(f"[dim red]Error displaying stats: {e}[/dim red]")

    def process_message(self, message: str) -> Optional[str]:
        """Process a user message and return the AI response."""
        try:
            privacy = "private" if self.engine.agent_uuid else "public"

            # Process user input
            message_bytes = message.encode("utf-8")
            self.engine.process_input_stream(message_bytes, privacy=privacy)

            # Generate response
            response_length = min(len(message) * 2, 200)
            response_bytes = self.engine.generate_and_save_response(length=response_length, privacy=privacy)

            # Decode response
            response_text = response_bytes.decode("utf-8", errors="replace")
            return response_text

        except Exception as e:
            handle_error(console, "Failed to process message", e)
            return None

    def run(self) -> None:
        """Run the interactive chat session."""
        console.clear()
        console.print(create_header("💬 Chat Session"))

        # Initialize thread if needed
        if not self.engine.thread_uuid:
            privacy = "private" if self.engine.agent_uuid else "public"
            self.engine.start_new_thread(privacy=privacy)
            self.conversation_history.append(
                {"sender": "system", "message": f"Started new thread: {safe_format_uuid(self.engine.thread_uuid)}"}
            )

        # Display initial info
        mode = "🔒 Private" if self.engine.agent_uuid else "🌐 Public"
        console.print(
            Panel(
                f"Chat session active in [bold]{mode}[/] mode.\n"
                f"Commands: [bold]/new[/] (new thread), [bold]/stats[/] (show stats), [bold]/back[/] (return to menu)",
                border_style="green",
            )
        )

        while True:
            try:
                # Display conversation
                console.print("\n" + "─" * console.width)
                self.display_conversation()
                console.print("─" * console.width)

                # Display stats
                self.display_stats()
                console.print()

                # Get user input
                message = questionary.text(
                    "💬 Message:", qmark="", style=questionary.Style([("question", "fg:#ff0066 bold")])
                ).ask()

                if message is None:  # Ctrl+C
                    break

                message = message.strip()
                if not message:
                    continue

                # Handle commands
                if message.startswith("/"):
                    if message == "/back":
                        break
                    elif message == "/new":
                        privacy = "private" if self.engine.agent_uuid else "public"
                        self.engine.finalize_and_save_thread(privacy=privacy)
                        self.engine.start_new_thread(privacy=privacy)
                        self.conversation_history.append(
                            {"sender": "system", "message": f"Started new thread: {safe_format_uuid(self.engine.thread_uuid)}"}
                        )
                        continue
                    elif message == "/stats":
                        self.show_detailed_stats()
                        continue
                    else:
                        console.print(f"[red]Unknown command: {message}[/red]")
                        continue

                # Add user message to history
                self.conversation_history.append({"sender": "user", "message": message})

                # Process message with loading indicator
                with Live(
                    Panel(Spinner("dots", text="🧠 BabyLM is thinking..."), border_style="green"), refresh_per_second=4
                ):
                    response = self.process_message(message)

                if response:
                    self.conversation_history.append({"sender": "ai", "message": response})
                else:
                    self.conversation_history.append({"sender": "system", "message": "Failed to generate response."})

            except KeyboardInterrupt:
                break
            except Exception as e:
                handle_error(console, "Chat session error", e)

        # Finalize thread before exiting
        try:
            privacy = "private" if self.engine.agent_uuid else "public"
            self.engine.finalize_and_save_thread(privacy=privacy)
            console.print("[green]✅ Thread finalized and saved.[/]")
        except Exception as e:
            handle_error(console, "Failed to finalize thread", e)

    def show_detailed_stats(self) -> None:
        """Show detailed statistics in a panel."""
        try:
            stats_text = ""

            # Inference stats
            inference_engine = self.engine.inference_engine
            stats_text += f"🔄 Cycle Counter: {inference_engine.cycle_counter:,}\n"
            stats_text += f"📈 Recent Patterns: {len(inference_engine.recent_patterns)}/256\n"

            # Cache stats
            cache_hits = getattr(inference_engine, "_cache_hits", 0)
            cache_misses = getattr(inference_engine, "_cache_misses", 0)
            total_cache = cache_hits + cache_misses

            if total_cache > 0:
                cache_ratio = cache_hits / total_cache
                stats_text += f"🎯 Cache Hit Ratio: {cache_ratio:.1%} ({cache_hits}/{total_cache})\n"

            # Thread stats
            thread_id = safe_format_uuid(self.engine.thread_uuid)
            stats_text += f"🧵 Thread ID: {thread_id}\n"
            stats_text += f"📏 Thread Size: {format_bytes(self.engine.current_thread_size)}\n"
            stats_text += f"📝 Gene Keys: {len(self.engine.current_thread_keys):,}\n"

            # Session stats
            mode = "Private" if self.engine.agent_uuid else "Public"
            stats_text += f"🔒 Privacy Mode: {mode}\n"
            format_id = safe_format_uuid(self.engine.format_uuid)
            stats_text += f"🎨 Format ID: {format_id}\n"

            console.print(Panel(stats_text.strip(), title="📊 Detailed Statistics", border_style="cyan"))

            questionary.press_any_key_to_continue().ask()

        except Exception as e:
            handle_error(console, "Failed to show detailed stats", e)

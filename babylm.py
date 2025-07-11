#!/usr/bin/env python3
"""
babylm.py - Enhanced CLI for GyroSI Baby LM

A colorful, interactive command-line interface that provides both development
tools and user-friendly language model functionality.
"""

import argparse
import os
import sys
import logging
import readline
import atexit
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import textwrap

# Rich terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.layout import Layout
from rich.live import Live

try:
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    print("Warning: 'rich' library not installed. Install with: pip install rich")
    print("Falling back to basic output mode.\n")

# Import Baby LM components
from baby import initialize_intelligence_engine
from baby.information import assign_agent_uuid, list_formats, load_format
from baby.information import get_memory_preferences

# Use the fastest available JSON library (orjson > ujson > stdlib json)
try:
    import orjson as json

    def json_loads(s):
        if isinstance(s, str):
            s = s.encode("utf-8")
        return json.loads(s)

    def json_dumps(obj):
        return json.dumps(obj).decode("utf-8")

except ImportError:
    try:
        import ujson as json  # type: ignore

        json_loads = json.loads
        json_dumps = json.dumps
    except ImportError:
        import json

        json_loads = json.loads
        json_dumps = json.dumps


# Setup logging with color
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        self.COLORS.get(record.levelname, self.RESET)
        record.levelname = "{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


# Configure logging
log_level = os.environ.get("BABYLM_LOG_LEVEL", "INFO")
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(level=getattr(logging, log_level), handlers=[handler])
logger = logging.getLogger("babylm_cli")

# History file for readline
HISTORY_FILE = Path.home() / ".babylm_history"


def setup_readline():
    """Setup readline for better interactive experience"""
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass

    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, HISTORY_FILE)

    # Enable tab completion
    readline.parse_and_bind("tab: complete")


class BabyLMCLI:
    """Enhanced CLI for GyroSI Baby LM"""

    def __init__(self):
        self.engine = None
        self.current_agent = None
        self.agents = {}  # agent_uuid -> agent_name mapping
        self.interactive_mode = False
        self.developer_mode = False
        self.conversation_history = []

    def print_banner(self):
        """Print colorful banner"""
        if RICH_AVAILABLE and console is not None:
            banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════════╗
║                    GyroSI Baby LM v1.0.0                      ║
║         Quantum Physics-Inspired Language Model               ║
╚═══════════════════════════════════════════════════════════════╝[/bold cyan]
            """
            console.print(banner)
        else:
            print("=" * 60)
            print("         GyroSI Baby LM v1.0.0")
            print("    Quantum Physics-Inspired Language Model")
            print("=" * 60)

    def initialize_engine(self, agent_uuid: Optional[str] = None) -> bool:
        """Initialize or switch to a different agent"""
        try:
            if agent_uuid:
                # Switch to specific agent
                base_memories_dir = "memories"  # or the appropriate directory
                prefs = get_memory_preferences(base_memories_dir)
                assign_agent_uuid(agent_uuid, base_memories_dir=base_memories_dir, prefs=prefs)

            if RICH_AVAILABLE and console is not None:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    task = progress.add_task("Initializing intelligence engine...", total=None)
                    self.engine = initialize_intelligence_engine()
                    progress.update(task, completed=True)
            else:
                print("Initializing intelligence engine...")
                self.engine = initialize_intelligence_engine()

            if self.engine is not None:
                self.current_agent = self.engine.agent_uuid
                if RICH_AVAILABLE and console is not None:
                    agent_display = f"{self.current_agent[:8] if self.current_agent else 'None'}..."
                    console.print(f"[green]✓[/green] Agent initialized: [cyan]{agent_display}[/cyan]")
                else:
                    agent_display = f"{self.current_agent[:8] if self.current_agent else 'None'}..."
                    print(f"✓ Agent initialized: {agent_display}")
                return True
            return False

        except Exception:
            logger.error("Failed to initialize engine: {e}")
            if RICH_AVAILABLE and console is not None:
                console.print("[red]✗ Failed to initialize: {e}[/red]")
            else:
                print("✗ Failed to initialize: {e}")
            return False

    def list_agents(self):
        """List all available agents"""
        agents_dir = Path("memories/private/agents")
        if not agents_dir.exists():
            if RICH_AVAILABLE and console is not None:
                console.print("[yellow]No agents found.[/yellow]")
            else:
                print("No agents found.")
            return

        agents = []
        for shard_dir in agents_dir.glob("*"):
            if shard_dir.is_dir():
                for agent_dir in shard_dir.glob("agent-*"):
                    if agent_dir.is_dir():
                        agent_uuid = "-".join(agent_dir.name.split("-")[1:])
                        agents.append(agent_uuid)

        if RICH_AVAILABLE and console is not None:
            table = Table(title="Available Agents")
            table.add_column("Index", style="cyan", no_wrap=True)
            table.add_column("Agent UUID", style="magenta")
            table.add_column("Status", style="green")

            for i, agent_uuid in enumerate(agents, 1):
                status = "Active" if agent_uuid == self.current_agent else ""
                table.add_row(str(i), agent_uuid, status)

            console.print(table)
        else:
            print("\nAvailable Agents:")
            for i, agent_uuid in enumerate(agents, 1):
                status = " (Active)" if agent_uuid == self.current_agent else ""
                print("{i}. {agent_uuid}{status}")

    def create_new_agent(self):
        """Create a new agent interactively"""
        if RICH_AVAILABLE and console is not None:
            agent_name = Prompt.ask("Enter a name for the new agent", default="default")
            confirm = Confirm.ask("Create new agent '{agent_name}'?")
        else:
            agent_name = input("Enter a name for the new agent (default): ") or "default"
            confirm = input("Create new agent '{agent_name}'? (y/n): ").lower() == "y"

        if confirm:
            import uuid

            new_uuid = str(uuid.uuid4())

            if self.initialize_engine(new_uuid):
                # Save agent metadata
                self.agents[new_uuid] = agent_name
                self._save_agent_metadata(new_uuid, agent_name)

                if RICH_AVAILABLE and console is not None:
                    console.print("[green]✓ Created agent '{agent_name}' ({new_uuid[:8]}...)[/green]")
                else:
                    print("✓ Created agent '{agent_name}' ({new_uuid[:8]}...)")

    def _save_agent_metadata(self, agent_uuid: str, agent_name: str):
        """Save agent metadata"""
        metadata = {
            "agent_uuid": agent_uuid,
            "agent_name": agent_name,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
        }

        # Save to agent preferences
        prefs_path = Path("memories/private/agents/{agent_uuid[:2]}/agent-{agent_uuid}/preferences.json")
        prefs_path.parent.mkdir(parents=True, exist_ok=True)

        with open(prefs_path, "w") as f:
            f.write(json_dumps(metadata))

    def show_system_status(self):
        """Show comprehensive system status"""
        if not self.engine:
            if RICH_AVAILABLE and console is not None:
                console.print("[red]No engine initialized.[/red]")
            else:
                print("No engine initialized.")
            return

        if RICH_AVAILABLE and console is not None:
            # Create status panels
            agent_info = Panel(
                "[cyan]UUID:[/cyan] {self.engine.agent_uuid}\n"
                "[cyan]Format:[/cyan] {self.engine.format_uuid[:8]}...\n"
                "[cyan]Thread:[/cyan] {self.engine.thread_uuid[:8] if self.engine.thread_uuid else 'None'}...",
                title="Agent Information",
                border_style="cyan",
            )

            tensor_info = Panel(
                "[green]Shape:[/green] {self.engine.inference_engine.T.shape}\n"
                "[green]Cycles:[/green] {self.engine.inference_engine.cycle_counter}\n"
                "[green]Gene Stateless:[/green] 0x{gene_stateless:02X}",
                title="Tensor State",
                border_style="green",
            )

            self._get_pattern_statistics()
            pattern_info = Panel(
                "[magenta]Total:[/magenta] {pattern_stats['total']}\n"
                "[magenta]Labeled:[/magenta] {pattern_stats['labeled']}\n"
                "[magenta]Active:[/magenta] {pattern_stats['active']}",
                title="Pattern Statistics",
                border_style="magenta",
            )

            console.print(agent_info)
            console.print(tensor_info)
            console.print(pattern_info)
        else:
            print("\n=== System Status ===")
            if self.engine:
                print("Agent UUID: {self.engine.agent_uuid}")
                print("Format UUID: {self.engine.format_uuid[:8]}...")
                print("Thread UUID: {self.engine.thread_uuid[:8] if self.engine.thread_uuid else 'None'}...")
                print("Tensor Shape: {self.engine.inference_engine.T.shape}")
                print("Cycles: {self.engine.inference_engine.cycle_counter}")
                print("Gene Stateless: 0x{gene_stateless:02X}")

                self._get_pattern_statistics()
                print("\nPattern Statistics:")
                print("  Total: {pattern_stats['total']}")
                print("  Labeled: {pattern_stats['labeled']}")
                print("  Active: {pattern_stats['active']}")

    def _get_pattern_statistics(self) -> Dict[str, int]:
        """Get pattern statistics"""
        if not self.engine:
            return {"total": 0, "labeled": 0, "active": 0}

        if self.engine and self.engine.M:
            patterns = self.engine.M.get("patterns", [])
        else:
            patterns = []
        total = len(patterns)
        labeled = sum(1 for p in patterns if p.get("character") is not None)
        active = sum(1 for p in patterns if p.get("count", 0) > 0)

        return {"total": total, "labeled": labeled, "active": active}

    def interactive_chat(self):
        """Interactive chat mode"""
        if not self.engine:
            if not self.initialize_engine():
                return

        if RICH_AVAILABLE and console is not None:
            console.print("\n[bold cyan]Entering interactive chat mode[/bold cyan]")
            console.print("[dim]Type 'help' for commands, 'exit' to quit[/dim]\n")
        else:
            print("\nEntering interactive chat mode")
            print("Type 'help' for commands, 'exit' to quit\n")

        self.interactive_mode = True

        while self.interactive_mode:
            try:
                # Get user input
                if RICH_AVAILABLE and console is not None:
                    user_input = Prompt.ask("[bold]You[/bold]")
                else:
                    user_input = input("You: ")

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input[1:])
                    continue

                # Process as conversation
                self._process_conversation(user_input)

            except KeyboardInterrupt:
                print("\n")
                if RICH_AVAILABLE and console is not None:
                    if Confirm.ask("Exit chat mode?"):
                        break
                else:
                    if input("Exit chat mode? (y/n): ").lower() == "y":
                        break
            except Exception:
                logger.error("Error in chat: {e}")
                if RICH_AVAILABLE and console is not None:
                    console.print("[red]Error: {e}[/red]")
                else:
                    print("Error: {e}")

    def _handle_command(self, command: str):
        """Handle chat commands"""
        parts = command.split()
        cmd = parts[0].lower()

        commands = {
            "help": self._show_chat_help,
            "status": self.show_system_status,
            "clear": self._clear_conversation,
            "save": self._save_conversation,
            "load": lambda: self._load_conversation(parts[1] if len(parts) > 1 else None),
            "thread": self._show_thread_info,
            "pattern": lambda: self._show_pattern_info(int(parts[1]) if len(parts) > 1 else None),
            "dev": self._toggle_developer_mode,
            "exit": self._exit_chat,
            "quit": self._exit_chat,
        }

        if cmd in commands:
            commands[cmd]()
        else:
            if RICH_AVAILABLE and console is not None:
                console.print("[yellow]Unknown command: {cmd}[/yellow]")
            else:
                print("Unknown command: {cmd}")

    def _show_chat_help(self):
        """Show chat help"""
        help_text = """
Chat Commands:
  /help          - Show this help
  /status        - Show system status
  /clear         - Clear conversation history
  /save [name]   - Save conversation
  /load [name]   - Load conversation
  /thread        - Show current thread info
  /pattern [idx] - Show pattern information
  /dev           - Toggle developer mode
  /exit, /quit   - Exit chat mode
        """

        if RICH_AVAILABLE and console is not None:
            console.print(Panel(help_text, title="Chat Commands", border_style="cyan"))
        else:
            print(help_text)

    def _process_conversation(self, user_input: str):
        """Process user input as conversation"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        # Add to history
        self.conversation_history.append(
            {"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()}
        )

        # Process input through the engine
        input_bytes = user_input.encode("utf-8")

        if self.developer_mode and RICH_AVAILABLE and console is not None:
            # Show processing details
            with Live(refresh_per_second=4) as live:
                # Create layout for developer view
                layout = Layout()
                layout.split_column(
                    Layout(name="input", size=3), Layout(name="processing", size=10), Layout(name="output")
                )

                # Show input
                layout["input"].update(Panel("Input: {user_input}", border_style="cyan"))

                # Process with visualization
                import numpy as np  # noqa: F401

                plaintext, encrypted = self.engine.process_input_stream(input_bytes)

                # Show tensor state
                tensor_state = "Tensor norm: {np.linalg.norm(self.engine.inference_engine.T):.4f}\n"
                tensor_state += "Cycles: {self.engine.inference_engine.cycle_counter}\n"
                tensor_state += "Recent patterns: {self.engine.inference_engine.recent_patterns[-5:]}"

                layout["processing"].update(Panel(tensor_state, title="Processing", border_style="yellow"))
                live.update(layout)

                # Generate response
                response_length = min(200, len(user_input) * 2)  # Adaptive length
                response_bytes = self.engine.generate_and_save_response(response_length)

                # Show output
                try:
                    response_text = response_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    response_text = "[Binary response: {len(response_bytes)} bytes]"

                layout["output"].update(Panel(response_text, title="Response", border_style="green"))
                live.update(layout)
        else:
            # Normal processing
            if RICH_AVAILABLE and console is not None:
                with console.status("Processing...", spinner="dots"):
                    plaintext, encrypted = self.engine.process_input_stream(input_bytes)
                    response_length = min(200, len(user_input) * 2)
                    response_bytes = self.engine.generate_and_save_response(response_length)
            else:
                print("Processing...")
                plaintext, encrypted = self.engine.process_input_stream(input_bytes)
                response_length = min(200, len(user_input) * 2)
                response_bytes = self.engine.generate_and_save_response(response_length)

        # Display response
        try:
            response_text = response_bytes.decode("utf-8")
        except UnicodeDecodeError:
            response_text = "[Binary response: {len(response_bytes)} bytes]"

        # Add to history
        self.conversation_history.append(
            {"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()}
        )

        # Display
        if RICH_AVAILABLE and console is not None:
            console.print("\n[bold green]Baby LM[/bold green]: {response_text}\n")
        else:
            print("\nBaby LM: {response_text}\n")

    def _clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        if RICH_AVAILABLE and console is not None:
            console.print("[green]Conversation history cleared.[/green]")
        else:
            print("Conversation history cleared.")

    def _save_conversation(self, name: Optional[str] = None):
        """Save conversation to file"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        if not name:
            name = datetime.now().strftime("conversation_%Y%m%d_%H%M%S")

        filename = "conversations/{name}.json"
        Path("conversations").mkdir(exist_ok=True)

        with open(filename, "w") as f:
            f.write(
                json_dumps(
                    {
                        "agent_uuid": self.engine.agent_uuid,
                        "thread_uuid": self.engine.thread_uuid,
                        "timestamp": datetime.now().isoformat(),
                        "history": self.conversation_history,
                    }
                )
            )

        if RICH_AVAILABLE and console is not None:
            console.print("[green]Conversation saved to {filename}[/green]")
        else:
            print("Conversation saved to {filename}")

    def _load_conversation(self, name: Optional[str] = None):
        """Load conversation from file"""
        if not name:
            # List available conversations
            conv_dir = Path("conversations")
            if not conv_dir.exists():
                if RICH_AVAILABLE and console is not None:
                    console.print("[yellow]No saved conversations found.[/yellow]")
                else:
                    print("No saved conversations found.")
                return

            conversations = list(conv_dir.glob("*.json"))
            if not conversations:
                if RICH_AVAILABLE and console is not None:
                    console.print("[yellow]No saved conversations found.[/yellow]")
                else:
                    print("No saved conversations found.")
                return

            if RICH_AVAILABLE and console is not None:
                table = Table(title="Saved Conversations")
                table.add_column("Index", style="cyan")
                table.add_column("Name", style="magenta")
                table.add_column("Date", style="green")

                for i, conv in enumerate(conversations, 1):
                    table.add_row(
                        str(i), conv.stem, datetime.fromtimestamp(conv.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    )

                console.print(table)

                idx = Prompt.ask("Select conversation", choices=[str(i) for i in range(1, len(conversations) + 1)])
                name = conversations[int(idx) - 1].stem
            else:
                print("\nSaved Conversations:")
                for i, conv in enumerate(conversations, 1):
                    print("{i}. {conv.stem}")

                idx = input("Select conversation: ")
                name = conversations[int(idx) - 1].stem

        filename = "conversations/{name}.json"
        try:
            with open(filename, "r") as f:
                data = json_loads(f.read())

            self.conversation_history = data["history"]

            if RICH_AVAILABLE and console is not None:
                console.print("[green]Loaded conversation from {filename}[/green]")
                console.print("[dim]Contains {len(self.conversation_history)} messages[/dim]")
            else:
                print("Loaded conversation from {filename}")
                print("Contains {len(self.conversation_history)} messages")

            # Display conversation
            for msg in self.conversation_history[-5:]:  # Show last 5 messages
                if RICH_AVAILABLE and console is not None:
                    role_str = "You" if msg["role"] == "user" else "Baby LM"
                    content_str = msg['content']
                    console.print(f"[bold]{role_str}[/bold]: {content_str}")
                else:
                    print("{0}: {1}".format("You" if msg["role"] == "user" else "Baby LM", msg['content']))

        except Exception:
            logger.error("Failed to load conversation: {e}")
            if RICH_AVAILABLE and console is not None:
                console.print("[red]Failed to load conversation: {e}[/red]")
            else:
                print("Failed to load conversation: {e}")

    def _show_thread_info(self):
        """Show current thread information"""
        if self.engine is None or not self.engine.thread_uuid:
            if RICH_AVAILABLE and console is not None:
                console.print("[yellow]No active thread.[/yellow]")
            else:
                print("No active thread.")
            return

        stats = self.engine.get_thread_statistics()
        self.engine.get_thread_relationships(self.engine.thread_uuid)
        # Find current thread detail
        thread_detail = next((d for d in stats["thread_details"] if d["thread_uuid"] == self.engine.thread_uuid), None)
        thread_detail.get("thread_name") if thread_detail else "-"
        thread_detail.get("curriculum") if thread_detail else "-"
        ", ".join(thread_detail.get("tags") or []) if thread_detail and thread_detail.get("tags") else "-"

        if RICH_AVAILABLE and console is not None:
            info = """
[cyan]Thread UUID:[/cyan] {self.engine.thread_uuid}
[cyan]Name:[/cyan] {name}
[cyan]Curriculum:[/cyan] {curriculum}
[cyan]Tags:[/cyan] {tags}
[cyan]Size:[/cyan] {self.engine.current_thread_size} bytes
[cyan]Parent:[/cyan] {relationships['parent'] or 'None'}
[cyan]Children:[/cyan] {len(relationships['children'])}
[cyan]Gene Keys:[/cyan] {len(self.engine.current_thread_keys)}
            """
            console.print(Panel(info.strip(), title="Current Thread", border_style="cyan"))
        else:
            print("\nCurrent Thread:")
            print("  UUID: {self.engine.thread_uuid}")
            print("  Name: {name}")
            print("  Curriculum: {curriculum}")
            print("  Tags: {tags}")
            print("  Size: {self.engine.current_thread_size} bytes")
            print("  Parent: {relationships['parent'] or 'None'}")
            print("  Children: {len(relationships['children'])}")
            print("  Gene Keys: {len(self.engine.current_thread_keys)}")

    def _show_pattern_info(self, pattern_idx: Optional[int] = None):
        """Show pattern information"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        if pattern_idx is None:
            # Show most active patterns
            patterns = [(i, p) for i, p in enumerate(self.engine.M.get("patterns", []))]
            active_patterns = sorted(patterns, key=lambda x: x[1].get("count", 0), reverse=True)[:10]

            if RICH_AVAILABLE and console is not None:
                table = Table(title="Most Active Patterns")
                table.add_column("Index", style="cyan")
                table.add_column("Count", style="green")
                table.add_column("Semantic", style="magenta")
                table.add_column("Class", style="yellow")

                for idx, pattern in active_patterns:
                    table.add_row(  # type: ignore
                        str(idx),
                        str(pattern.get("count", 0)),
                        str(pattern.get("character", "") or "-"),
                        str(pattern.get("gyration_feature", "-")),
                    )

                console.print(table)
            else:
                print("\nMost Active Patterns:")
                for idx, pattern in active_patterns:
                    print(
                        "  {idx}: count={pattern.get('count', 0)}, "
                        "character={pattern.get('character', '') or '-'}, "
                        "class={pattern.get('gyration_feature', '-')}"
                    )
        else:
            # Show specific pattern
            if 0 <= pattern_idx < 256:
                stats = self.engine.get_pattern_statistics(pattern_idx)

                if RICH_AVAILABLE and console is not None:
                    info = """
[cyan]Index:[/cyan] {stats['pattern_index']}
[cyan]Semantic:[/cyan] {stats['character'] or 'None'}
[cyan]Count:[/cyan] {stats['count']}
[cyan]First Cycle:[/cyan] {stats['first_cycle'] or 'Never'}
[cyan]Last Cycle:[/cyan] {stats['last_cycle'] or 'Never'}
[cyan]Resonance Class:[/cyan] {stats['gyration_feature']}
[cyan]Confidence:[/cyan] {stats['confidence']:.4f}
[cyan]Current Resonance:[/cyan] {stats['current_resonance']:.4f if stats['current_resonance'] else 'N/A'}
                    """
                    console.print(Panel(info.strip(), title="Pattern {pattern_idx}", border_style="cyan"))

                    # Show contexts if available
                    if stats["contexts"]:
                        console.print("\n[bold]Pattern Contexts:[/bold]")
                        if stats["contexts"]["before"]:
                            console.print("  [cyan]Common predecessors:[/cyan]")
                            for pred_idx, count in stats["contexts"]["before"][:5]:
                                console.print("    Pattern {pred_idx}: {count} times")
                        if stats["contexts"]["after"]:
                            console.print("  [cyan]Common successors:[/cyan]")
                            for succ_idx, count in stats["contexts"]["after"][:5]:
                                console.print("    Pattern {succ_idx}: {count} times")
                else:
                    print("\nPattern {pattern_idx}:")
                    print("  Semantic: {stats['character'] or 'None'}")
                    print("  Count: {stats['count']}")
                    print("  Resonance Class: {stats['gyration_feature']}")
                    print("  Confidence: {stats['confidence']:.4f}")
            else:
                if RICH_AVAILABLE and console is not None:
                    console.print("[red]Invalid pattern index: {pattern_idx}[/red]")
                else:
                    print("Invalid pattern index: {pattern_idx}")

    def _toggle_developer_mode(self):
        """Toggle developer mode"""
        self.developer_mode = not self.developer_mode
        "enabled" if self.developer_mode else "disabled"

        if RICH_AVAILABLE and console is not None:
            console.print("[green]Developer mode {mode}[/green]")
        else:
            print("Developer mode {mode}")

    def _exit_chat(self):
        """Exit chat mode"""
        self.interactive_mode = False
        if RICH_AVAILABLE and console is not None:
            console.print("[cyan]Exiting chat mode...[/cyan]")
        else:
            print("Exiting chat mode...")

    def run_cli(self):
        """Main CLI entry point"""
        parser = argparse.ArgumentParser(
            description="GyroSI Baby LM - Enhanced CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(
                """
                Examples:
                  %(prog)s --chat                    # Start interactive chat
                  %(prog)s --agent list              # List all agents
                  %(prog)s --agent new               # Create new agent
                  %(prog)s --process "Hello world"   # Process text
                  %(prog)s --generate 100            # Generate 100 bytes
                  %(prog)s --thread list             # List threads
                  %(prog)s --dev --status            # Show detailed status
            """
            ),
        )

        # Mode selection
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument("--chat", "-c", action="store_true", help="Start interactive chat mode")
        mode_group.add_argument("--process", "-p", type=str, help="Process input text")
        mode_group.add_argument("--generate", "-g", type=int, help="Generate N bytes of output")

        # Agent management
        parser.add_argument("--agent", "-a", nargs="+", help="Agent operations: list, new, switch <uuid>")

        # Thread management
        parser.add_argument("--thread", "-t", nargs="+", help="Thread operations: list, show <uuid>, stats")

        # Format management
        parser.add_argument("--format", "-", nargs="+", help="Format operations: list, show <uuid>, discover <domain>")

        # System operations
        parser.add_argument("--status", "-s", action="store_true", help="Show system status")
        parser.add_argument("--dev", "-d", action="store_true", help="Enable developer mode")

        # I/O options
        parser.add_argument("--input-file", "-i", type=str, help="Input file path")
        parser.add_argument("--output-file", "-o", type=str, help="Output file path")

        # Other options
        parser.add_argument("--version", "-v", action="version", version="GyroSI Baby LM v1.0.0")
        parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")

        args = parser.parse_args()

        # Set developer mode
        self.developer_mode = args.dev

        # Show banner unless quiet
        if not args.quiet:
            self.print_banner()

        # Handle agent operations first
        if args.agent:
            self._handle_agent_operations(args.agent)
            return

        # Initialize engine if needed for other operations
        if args.status or args.process or args.generate or args.thread or args.format:
            if not self.initialize_engine():
                return

        # Handle status
        if args.status:
            self.show_system_status()
            return

        # Handle thread operations
        if args.thread:
            self._handle_thread_operations(args.thread)
            return

        # Handle format operations
        if args.format:
            self._handle_format_operations(args.format)
            return

        # Handle process
        if args.process or args.input_file:
            self._handle_process(args.process, args.input_file, args.output_file)
            return

        # Handle generate
        if args.generate:
            self._handle_generate(args.generate, args.output_file)
            return

        # Handle chat mode
        if args.chat:
            setup_readline()
            self.interactive_chat()
            return

        # Default: show help
        parser.print_help()

    def _handle_agent_operations(self, agent_args: List[str]):
        """Handle agent-related operations"""
        operation = agent_args[0].lower()

        if operation == "list":
            self.list_agents()
        elif operation == "new":
            self.create_new_agent()
        elif operation == "switch" and len(agent_args) > 1:
            agent_uuid = agent_args[1]
            if self.initialize_engine(agent_uuid):
                if RICH_AVAILABLE and console is not None:
                    console.print("[green]Switched to agent {agent_uuid[:8]}...[/green]")
                else:
                    print("Switched to agent {agent_uuid[:8]}...")
        else:
            if RICH_AVAILABLE and console is not None:
                console.print("[red]Unknown agent operation: {operation}[/red]")
            else:
                print("Unknown agent operation: {operation}")

    def _handle_thread_operations(self, thread_args: List[str]):
        """Handle thread-related operations"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        operation = thread_args[0].lower()

        if operation == "list":
            self._list_threads()
        elif operation == "show" and len(thread_args) > 1:
            self._show_thread(thread_args[1])
        elif operation == "stats":
            self._show_thread_stats()
        elif operation == "tree":
            self._show_thread_tree()
        elif operation == "export" and len(thread_args) > 1:
            self._export_thread(thread_args[1])
        else:
            if RICH_AVAILABLE and console is not None:
                console.print("[red]Unknown thread operation: {operation}[/red]")
            else:
                print("Unknown thread operation: {operation}")

    def _list_threads(self):
        """List all threads with rich formatting"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        stats = self.engine.get_thread_statistics()

        if RICH_AVAILABLE and console is not None:
            table = Table(
                title="Threads for Agent {self.engine.agent_uuid[:8] if self.engine.agent_uuid else 'None'}..."
            )
            table.add_column("Index", style="cyan", no_wrap=True)
            table.add_column("Thread UUID", style="magenta")
            table.add_column("Name", style="white")
            table.add_column("Curriculum", style="green")
            table.add_column("Tags", style="yellow")
            table.add_column("Size", style="green", justify="right")
            table.add_column("Parent", style="yellow")
            table.add_column("Children", style="blue", justify="center")

            for i, detail in enumerate(stats["thread_details"], 1):
                size_kb = detail["size_bytes"] / 1024
                parent = "✓" if detail["has_parent"] else "-"
                children = str(detail["child_count"]) if detail["has_children"] else "-"
                name = detail.get("thread_name") or "-"
                curriculum = detail.get("curriculum") or "-"
                tags = ", ".join(detail.get("tags") or []) if detail.get("tags") else "-"
                table.add_row(
                    str(i),
                    detail["thread_uuid"][:8] + "...",
                    name,
                    curriculum,
                    tags,
                    f"{size_kb:.1f} KB",
                    parent,
                    children,
                )

            console.print(table)

            # Summary
            console.print("\n[bold]Summary:[/bold]")
            console.print("  Total threads: {stats['total_threads']}")
            console.print("  Total size: {stats['total_size_bytes'] / 1024:.1f} KB")
            console.print("  Capacity usage: {stats['capacity_usage_percent']:.1f}%")
        else:
            print("\nThreads for Agent {self.engine.agent_uuid[:8] if self.engine.agent_uuid else 'None'}...")
            print("-" * 80)
            for i, detail in enumerate(stats["thread_details"], 1):
                name = detail.get("thread_name") or "-"
                curriculum = detail.get("curriculum") or "-"
                tags = ", ".join(detail.get("tags") or []) if detail.get("tags") else "-"
                size_kb = detail["size_bytes"] / 1024
                print(
                    f"{i}. {detail['thread_uuid'][:8]}... | Name: {name} | Curriculum: {curriculum} | Tags: {tags} | Size: {size_kb:.1f} KB"
                )
                if detail["has_parent"] or detail["has_children"]:
                    print(f"   Parent: {detail['has_parent']}, Children: {detail['child_count']}")

    def _show_thread(self, thread_uuid: str):
        """Show thread content with context"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        # Handle partial UUID
        if len(thread_uuid) < 36:
            # Find matching thread
            stats = self.engine.get_thread_statistics()
            matches = [d["thread_uuid"] for d in stats["thread_details"] if d["thread_uuid"].startswith(thread_uuid)]

            if len(matches) == 0:
                if RICH_AVAILABLE and console is not None:
                    console.print("[red]No thread found matching: {thread_uuid}[/red]")
                else:
                    print("No thread found matching: {thread_uuid}")
                return
            elif len(matches) > 1:
                if RICH_AVAILABLE and console is not None:
                    console.print("[yellow]Multiple threads match: {thread_uuid}[/yellow]")
                    for match in matches:
                        console.print("  - {match}")
                else:
                    print("Multiple threads match: {thread_uuid}")
                    for match in matches:
                        print("  - {match}")
                return

            thread_uuid = matches[0]

        # Load thread with context
        result = self.engine.load_thread_with_context(thread_uuid)

        if "error" in result:
            if RICH_AVAILABLE and console is not None:
                console.print("[red]{result['error']}[/red]")
            else:
                print(result["error"])
            return

        # Display thread content as NDJSON events
        events = result["content"]
        if RICH_AVAILABLE and console is not None:
            from rich.table import Table

            table = Table(title="Thread {thread_uuid[:8]}... ({result['size_bytes']} bytes)")
            table.add_column("#", style="dim", width=4)
            table.add_column("Type", style="cyan", width=8)
            table.add_column("Data (decoded)", style="white")
            for i, event in enumerate(events):
                typ = event.get("type", "?")
                data = event.get("data", b"")
                if isinstance(data, bytes):
                    try:
                        data_str = data.decode("utf-8")
                    except Exception:
                        data_str = str(data)
                else:
                    data_str = str(data)
                table.add_row(str(i + 1), typ, data_str)
            console.print(table)
        else:
            print("\nThread {thread_uuid[:8]}... ({result['size_bytes']} bytes)")
            print("-" * 60)
            for i, event in enumerate(events):
                typ = event.get("type", "?")
                data = event.get("data", b"")
                if isinstance(data, bytes):
                    try:
                        data_str = data.decode("utf-8")
                    except Exception:
                        data_str = str(data)
                else:
                    data_str = str(data)
                print("[{i+1:03}] {typ}: {data_str}")

        print("\nRelationships:")
        print("  Parent: {result['relationships']['parent'] or 'None'}")
        print("  Children: {len(result['relationships']['children'])}")

    def _show_thread_stats(self):
        """Show detailed thread statistics"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        stats = self.engine.get_thread_statistics()

        if RICH_AVAILABLE and console is not None:
            # Create statistics panels
            overview = Panel(
                "[cyan]Total Threads:[/cyan] {stats['total_threads']}\n"
                "[cyan]Total Size:[/cyan] {stats['total_size_bytes'] / 1024:.1f} KB\n"
                "[cyan]Capacity Usage:[/cyan] {stats['capacity_usage_percent']:.1f}%",
                title="Overview",
                border_style="cyan",
            )

            relationships = Panel(
                "[green]With Parents:[/green] {stats['relationship_stats']['threads_with_parents']}\n"
                "[green]With Children:[/green] {stats['relationship_stats']['threads_with_children']}\n"
                "[green]Isolated:[/green] {stats['relationship_stats']['isolated_threads']}",
                title="Relationships",
                border_style="green",
            )

            console.print(overview)
            console.print(relationships)

            # Size distribution
            if stats["thread_details"]:
                sizes = [d["size_bytes"] for d in stats["thread_details"]]
                avg_size = sum(sizes) / len(sizes)
                max_size = max(sizes)
                min_size = min(sizes)

                size_dist = Panel(
                    "[magenta]Average:[/magenta] {avg_size / 1024:.1f} KB\n"
                    "[magenta]Maximum:[/magenta] {max_size / 1024:.1f} KB\n"
                    "[magenta]Minimum:[/magenta] {min_size / 1024:.1f} KB",
                    title="Size Distribution",
                    border_style="magenta",
                )
                console.print(size_dist)
        else:
            print("\nThread Statistics")
            print("-" * 40)
            print("Total Threads: {stats['total_threads']}")
            print("Total Size: {stats['total_size_bytes'] / 1024:.1f} KB")
            print("Capacity Usage: {stats['capacity_usage_percent']:.1f}%")
            print("\nRelationships:")
            print("  With Parents: {stats['relationship_stats']['threads_with_parents']}")
            print("  With Children: {stats['relationship_stats']['threads_with_children']}")
            print("  Isolated: {stats['relationship_stats']['isolated_threads']}")

    def _show_thread_tree(self):
        """Show thread relationship tree"""
        if self.engine is None or not self.engine.thread_uuid:
            if RICH_AVAILABLE and console is not None:
                console.print("[yellow]No active thread.[/yellow]")
            else:
                print("No active thread.")
            return

        chain = self.engine.get_thread_chain(self.engine.thread_uuid)

        if RICH_AVAILABLE and console is not None:
            tree = Tree("Thread Hierarchy")

            # Build tree structure
            nodes = {}
            for uuid in chain:
                if uuid == self.engine.thread_uuid:
                    node = tree.add("[bold cyan]{uuid[:8]}... (CURRENT)[/bold cyan]")
                else:
                    node = tree.add("{uuid[:8]}...")
                nodes[uuid] = node

            console.print(tree)
        else:
            print("\nThread Hierarchy:")
            for i, uuid in enumerate(chain):
                indent = "  " * i
                if uuid == self.engine.thread_uuid:
                    print(f"{indent}→ {uuid[:8]}... (CURRENT)")
                else:
                    print(f"{indent}  {uuid[:8]}...")

    def _export_thread(self, thread_uuid: str):
        """Export thread content to file"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        content = self.engine.load_thread_content(thread_uuid)

        if content is None:
            if RICH_AVAILABLE and console is not None:
                console.print("[red]Thread {thread_uuid} not found[/red]")
            else:
                print("Thread {thread_uuid} not found")
            return

        filename = "thread_{thread_uuid[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # Ensure content is bytes before writing
        to_write = content
        if isinstance(content, list):
            import json

            to_write = (json.dumps(content, indent=2) + "\n").encode("utf-8")
        elif not isinstance(content, (bytes, bytearray)):
            to_write = str(content).encode("utf-8")

        with open(filename, "wb") as f:
            # Guarantee bytes: handle list, str, and fallback
            if isinstance(to_write, list):
                import json

                to_write = (json.dumps(to_write, indent=2) + "\n").encode("utf-8")
            elif not isinstance(to_write, (bytes, bytearray)):
                to_write = str(to_write).encode("utf-8")
            f.write(to_write)

        if RICH_AVAILABLE and console is not None:
            console.print("[green]Thread exported to {filename}[/green]")
        else:
            print("Thread exported to {filename}")

    def _handle_format_operations(self, format_args: List[str]):
        """Handle format-related operations"""
        operation = format_args[0].lower()

        if operation == "list":
            self._list_formats()
        elif operation == "show" and len(format_args) > 1:
            self._show_format(format_args[1])
        elif operation == "discover" and len(format_args) > 1:
            self._discover_format(format_args[1])
        elif operation == "compose" and len(format_args) > 2:
            self._compose_formats(format_args[1:])
        else:
            if RICH_AVAILABLE and console is not None:
                console.print("[red]Unknown format operation: {operation}[/red]")
            else:
                print("Unknown format operation: {operation}")

    def _list_formats(self):
        """List all available formats"""
        format_uuids = list_formats()

        if not format_uuids:
            if RICH_AVAILABLE and console is not None:
                console.print("[yellow]No formats found.[/yellow]")
            else:
                print("No formats found.")
            return

        if RICH_AVAILABLE and console is not None:
            table = Table(title="Available Formats")
            table.add_column("Index", style="cyan", no_wrap=True)
            table.add_column("UUID", style="magenta")
            table.add_column("Name", style="green")
            table.add_column("Stability", style="yellow")
            table.add_column("Usage", style="blue", justify="right")
            for i, format_uuid in enumerate(format_uuids, 1):
                base_memories_dir = self.engine.base_memories_dir if self.engine else "memories"
                format_data = load_format(format_uuid, base_memories_dir)
                if format_data:
                    table.add_row(
                        str(i),
                        format_uuid[:8] + "...",
                        format_data.get("format_name", "Unknown"),
                        format_data.get("stability", "Unknown"),
                        str(format_data.get("metadata", {}).get("usage_count", 0)),
                    )
            console.print(table)
        else:
            print("\nAvailable Formats:")
            for i, format_uuid in enumerate(format_uuids, 1):
                base_memories_dir = self.engine.base_memories_dir if self.engine else "memories"
                format_data = load_format(format_uuid, base_memories_dir)
                if format_data:
                    print("{i}. {format_uuid[:8]}... - {format_data.get('format_name', 'Unknown')}")

    def _show_format(self, format_uuid: str):
        """Show detailed format information"""
        # Handle partial UUID
        if len(format_uuid) < 36:
            format_uuids = list_formats()
            matches = [f for f in format_uuids if f.startswith(format_uuid)]

            if len(matches) == 0:
                if RICH_AVAILABLE and console is not None:
                    console.print("[red]No format found matching: {format_uuid}[/red]")
                else:
                    print("No format found matching: {format_uuid}")
                return
            elif len(matches) > 1:
                if RICH_AVAILABLE and console is not None:
                    console.print("[yellow]Multiple formats match: {format_uuid}[/yellow]")
                else:
                    print("Multiple formats match: {format_uuid}")
                return

            format_uuid = matches[0]

        base_memories_dir = self.engine.base_memories_dir if self.engine else "memories"
        format_data = load_format(format_uuid, base_memories_dir)
        if not format_data:
            if RICH_AVAILABLE and console is not None:
                console.print("[red]Format {format_uuid} not found[/red]")
            else:
                print("Format {format_uuid} not found")
            return

        if RICH_AVAILABLE and console is not None:
            # Basic info
            basic_info = """
[cyan]UUID:[/cyan] {format_data.get('format_uuid', 'Unknown')}
[cyan]Name:[/cyan] {format_data.get('format_name', 'Unknown')}
[cyan]Version:[/cyan] {format_data.get('format_version', 'Unknown')}
[cyan]CGM Version:[/cyan] {format_data.get('cgm_version', 'Unknown')}
[cyan]Stability:[/cyan] {format_data.get('stability', 'Unknown')}
            """
            console.print(Panel(basic_info.strip(), title="Format Information", border_style="cyan"))

            # Metadata
            format_data.get("metadata", {})
            meta_info = """
[green]Author:[/green] {meta.get('author', 'Unknown')}
[green]Description:[/green] {meta.get('description', 'None')}
[green]Created:[/green] {meta.get('created_at', 'Unknown')}
[green]Usage Count:[/green] {meta.get('usage_count', 0)}
[green]Tags:[/green] {', '.join(meta.get('tags', []))}
            """
            console.print(Panel(meta_info.strip(), title="Metadata", border_style="green"))

            # Pattern statistics
            patterns = format_data.get("patterns", [])
            sum(1 for p in patterns if p.get("character"))
            sum(1 for p in patterns if p.get("count", 0) > 0)

            pattern_info = """
[magenta]Total Patterns:[/magenta] {len(patterns)}
[magenta]Labeled:[/magenta] {labeled}
[magenta]Active:[/magenta] {active}
            """
            console.print(Panel(pattern_info.strip(), title="Pattern Statistics", border_style="magenta"))
        else:
            print("\nFormat: {format_data.get('format_uuid', 'Unknown')}")
            print("Name: {format_data.get('format_name', 'Unknown')}")
            print("Version: {format_data.get('format_version', 'Unknown')}")
            print("Stability: {format_data.get('stability', 'Unknown')}")
            print("Author: {format_data.get('metadata', {}).get('author', 'Unknown')}")
            print("Description: {format_data.get('metadata', {}).get('description', 'None')}")

    def _discover_format(self, domain: str):
        """Discover format for a domain"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        if RICH_AVAILABLE and console is not None:
            with console.status("Discovering format for domain '{domain}'...", spinner="dots"):
                for stability in ["stable", "beta", "experimental"]:
                    format_uuid = self.engine.select_stable_format(domain, stability)
                    if format_uuid:
                        base_memories_dir = self.engine.base_memories_dir if self.engine else "memories"
                        format_data = load_format(format_uuid, base_memories_dir)
                        console.print("[green]Found {stability} format:[/green]")
                        console.print("  UUID: {format_uuid}")
                        if format_data is not None:
                            print("  Name: {format_data.get('format_name', 'Unknown')}")
                        else:
                            print("  Name: Unknown")
                        return

            console.print("[yellow]No format found for domain '{domain}'[/yellow]")
        else:
            print("Discovering format for domain '{domain}'...")
            for stability in ["stable", "beta", "experimental"]:
                format_uuid = self.engine.select_stable_format(domain, stability)
                if format_uuid:
                    base_memories_dir = self.engine.base_memories_dir if self.engine else "memories"
                    format_data = load_format(format_uuid, base_memories_dir)
                    print("Found {stability} format:")
                    print("  UUID: {format_uuid}")
                    if format_data is not None:
                        print("  Name: {format_data.get('format_name', 'Unknown')}")
                    else:
                        print("  Name: Unknown")
                    return

            print("No format found for domain '{domain}'")

    def _compose_formats(self, format_uuids: List[str]):
        """Compose multiple formats"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        primary = format_uuids[0]
        secondary = format_uuids[1:]

        if RICH_AVAILABLE and console is not None:
            console.print("Composing formats:")
            console.print("  Primary: {primary}")
            for s in secondary:
                console.print("  Secondary: {s}")

            with console.status("Composing formats...", spinner="dots"):
                composed_uuid = self.engine.compose_formats(primary, secondary)

            if composed_uuid:
                console.print("[green]✓ Created composed format: {composed_uuid}[/green]")
            else:
                console.print("[red]✗ Failed to compose formats[/red]")
        else:
            print("Composing formats:")
            print("  Primary: {primary}")
            for s in secondary:
                print("  Secondary: {s}")

            composed_uuid = self.engine.compose_formats(primary, secondary)

            if composed_uuid:
                print("✓ Created composed format: {composed_uuid}")
            else:
                print("✗ Failed to compose formats")

    def _handle_process(self, text: Optional[str], input_file: Optional[str], output_file: Optional[str]):
        """Handle text processing"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        # Get input
        if text:
            data = text.encode("utf-8")
        elif input_file:
            try:
                with open(input_file, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                if RICH_AVAILABLE and console is not None:
                    console.print("[red]Input file not found: {input_file}[/red]")
                else:
                    print("Input file not found: {input_file}")
                return
        else:
            return

        if RICH_AVAILABLE and console is not None:
            with console.status("Processing {len(data)} bytes from {source}...", spinner="dots"):
                plaintext, encrypted = self.engine.process_input_stream(data)
            if self.engine.thread_uuid:
                console.print("[green]✓ Processed and saved to thread {self.engine.thread_uuid[:8]}...[/green]")
            else:
                console.print("[green]✓ Processed and saved to thread [unknown]...[/green]")
            if self.developer_mode:
                # Show processing details

                info = """
[cyan]Input size:[/cyan] {len(data)} bytes
[cyan]Thread UUID:[/cyan] {self.engine.thread_uuid if self.engine.thread_uuid else '[unknown]'}
[cyan]Cycles processed:[/cyan] {len(data)}
[cyan]Current cycle:[/cyan] {self.engine.inference_engine.cycle_counter}
[cyan]Recent patterns:[/cyan] {self.engine.inference_engine.recent_patterns[-5:]}"
                """
                console.print(Panel(info.strip(), title="Processing Details", border_style="yellow"))
        else:
            print("Processing {len(data)} bytes from {source}...")
            plaintext, encrypted = self.engine.process_input_stream(data)
            if self.engine.thread_uuid:
                print("✓ Processed and saved to thread {self.engine.thread_uuid[:8]}...")
            else:
                print("✓ Processed and saved to thread [unknown]...")

        # Save output if requested
        if output_file:
            with open(output_file, "wb") as f:
                f.write(encrypted)

            if RICH_AVAILABLE and console is not None:
                console.print("[green]✓ Saved encrypted output to {output_file}[/green]")
            else:
                print("✓ Saved encrypted output to {output_file}")

    def _handle_generate(self, length: int, output_file: Optional[str]):
        """Handle text generation"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        if RICH_AVAILABLE and console is not None:
            with console.status("Generating {length} bytes...", spinner="dots"):
                response = self.engine.generate_and_save_response(length)
            if self.engine.thread_uuid:
                console.print("[green]✓ Generated and saved to thread {self.engine.thread_uuid[:8]}...[/green]")
            else:
                console.print("[green]✓ Generated and saved to thread [unknown]...[/green]")
            # Try to display as text
            try:
                text = response.decode("utf-8")
                if len(text) > 200:
                    text = text[:200] + "..."
                console.print(Panel(text, title="Generated Response ({len(response)} bytes)", border_style="green"))
            except UnicodeDecodeError:
                console.print("[yellow]Generated binary response ({len(response)} bytes)[/yellow]")
        else:
            print("Generating {length} bytes...")
            response = self.engine.generate_and_save_response(length)
            if self.engine.thread_uuid:
                print("✓ Generated and saved to thread {self.engine.thread_uuid[:8]}...")
            else:
                print("✓ Generated and saved to thread [unknown]...")
            try:
                text = response.decode("utf-8")
                if len(text) > 200:
                    text = text[:200] + "..."
                print("\nGenerated response:\n{text}")
            except UnicodeDecodeError:
                print("Generated binary response ({len(response)} bytes)")

        # Save output if requested
        if output_file:
            with open(output_file, "wb") as f:
                f.write(response)

            if RICH_AVAILABLE and console is not None:
                console.print("[green]✓ Saved generated response to {output_file}[/green]")
            else:
                print("✓ Saved generated response to {output_file}")


def main():
    """Main entry point"""
    # Import numpy here to avoid issues if not installed
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("Error: NumPy is required. Install with: pip install numpy")
        sys.exit(1)

    # Create and run CLI
    cli = BabyLMCLI()

    try:
        cli.run_cli()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception:
        logger.error("Unexpected error: {e}", exc_info=True)
        if RICH_AVAILABLE and console is not None:
            console.print("\n[red]Error: {e}[/red]")
            console.print("[dim]Check babylm_cli.log for details[/dim]")
        else:
            print("\nError: {e}")
            print("Check babylm_cli.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()

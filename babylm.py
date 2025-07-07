#!/usr/bin/env python3
"""
babylm.py - Enhanced CLI for GyroSI Baby LM

A colorful, interactive command-line interface that provides both development
tools and user-friendly language model functionality.
"""

import argparse
import os
import sys
import json
import logging
import readline
import atexit
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
import textwrap
import shutil

# Rich terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich import print as rprint

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    print("Warning: 'rich' library not installed. Install with: pip install rich")
    print("Falling back to basic output mode.\n")

# Import Baby LM components
from baby import initialize_intelligence_engine
from baby.information import ensure_agent_uuid, assign_agent_uuid, list_formats, load_format, get_memory_preferences
from baby.governance import gene_stateless


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
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
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
                assign_agent_uuid(agent_uuid)

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
                    console.print(f"[green]✓[/green] Agent initialized: [cyan]{self.current_agent[:8]}...[/cyan]")
                else:
                    print(f"✓ Agent initialized: {self.current_agent[:8]}...")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            if RICH_AVAILABLE and console is not None:
                console.print(f"[red]✗ Failed to initialize: {e}[/red]")
            else:
                print(f"✗ Failed to initialize: {e}")
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
                print(f"{i}. {agent_uuid}{status}")

    def create_new_agent(self):
        """Create a new agent interactively"""
        if RICH_AVAILABLE and console is not None:
            agent_name = Prompt.ask("Enter a name for the new agent", default="default")
            confirm = Confirm.ask(f"Create new agent '{agent_name}'?")
        else:
            agent_name = input("Enter a name for the new agent (default): ") or "default"
            confirm = input(f"Create new agent '{agent_name}'? (y/n): ").lower() == "y"

        if confirm:
            import uuid

            new_uuid = str(uuid.uuid4())

            if self.initialize_engine(new_uuid):
                # Save agent metadata
                self.agents[new_uuid] = agent_name
                self._save_agent_metadata(new_uuid, agent_name)

                if RICH_AVAILABLE and console is not None:
                    console.print(f"[green]✓ Created agent '{agent_name}' ({new_uuid[:8]}...)[/green]")
                else:
                    print(f"✓ Created agent '{agent_name}' ({new_uuid[:8]}...)")

    def _save_agent_metadata(self, agent_uuid: str, agent_name: str):
        """Save agent metadata"""
        metadata = {
            "agent_uuid": agent_uuid,
            "agent_name": agent_name,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
        }

        # Save to agent preferences
        prefs_path = Path(f"memories/private/agents/{agent_uuid[:2]}/agent-{agent_uuid}/preferences.json")
        prefs_path.parent.mkdir(parents=True, exist_ok=True)

        with open(prefs_path, "w") as f:
            json.dump(metadata, f, indent=2)

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
                f"[cyan]UUID:[/cyan] {self.engine.agent_uuid}\n"
                f"[cyan]Format:[/cyan] {self.engine.format_uuid[:8]}...\n"
                f"[cyan]Thread:[/cyan] {self.engine.thread_uuid[:8] if self.engine.thread_uuid else 'None'}...",
                title="Agent Information",
                border_style="cyan",
            )

            tensor_info = Panel(
                f"[green]Shape:[/green] {self.engine.inference_engine.T.shape}\n"
                f"[green]Cycles:[/green] {self.engine.inference_engine.cycle_counter}\n"
                f"[green]Gene Stateless:[/green] 0x{gene_stateless:02X}",
                title="Tensor State",
                border_style="green",
            )

            pattern_stats = self._get_pattern_statistics()
            pattern_info = Panel(
                f"[magenta]Total:[/magenta] {pattern_stats['total']}\n"
                f"[magenta]Labeled:[/magenta] {pattern_stats['labeled']}\n"
                f"[magenta]Active:[/magenta] {pattern_stats['active']}",
                title="Pattern Statistics",
                border_style="magenta",
            )

            console.print(agent_info)
            console.print(tensor_info)
            console.print(pattern_info)
        else:
            print("\n=== System Status ===")
            if self.engine:
                print(f"Agent UUID: {self.engine.agent_uuid}")
                print(f"Format UUID: {self.engine.format_uuid[:8]}...")
                print(f"Thread UUID: {self.engine.thread_uuid[:8] if self.engine.thread_uuid else 'None'}...")
                print(f"Tensor Shape: {self.engine.inference_engine.T.shape}")
                print(f"Cycles: {self.engine.inference_engine.cycle_counter}")
                print(f"Gene Stateless: 0x{gene_stateless:02X}")

                pattern_stats = self._get_pattern_statistics()
                print(f"\nPattern Statistics:")
                print(f"  Total: {pattern_stats['total']}")
                print(f"  Labeled: {pattern_stats['labeled']}")
                print(f"  Active: {pattern_stats['active']}")

    def _get_pattern_statistics(self) -> Dict[str, int]:
        """Get pattern statistics"""
        if not self.engine:
            return {"total": 0, "labeled": 0, "active": 0}

        if self.engine and self.engine.M:
            patterns = self.engine.M["patterns"]
        else:
            patterns = []
        total = len(patterns)
        labeled = sum(1 for p in patterns if p.get("semantic") is not None)
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
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                if RICH_AVAILABLE and console is not None:
                    console.print(f"[red]Error: {e}[/red]")
                else:
                    print(f"Error: {e}")

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
                console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
            else:
                print(f"Unknown command: {cmd}")

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
                layout["input"].update(Panel(f"Input: {user_input}", border_style="cyan"))

                # Process with visualization
                import numpy as np

                plaintext, encrypted = self.engine.process_input_stream(input_bytes)

                # Show tensor state
                tensor_state = f"Tensor norm: {np.linalg.norm(self.engine.inference_engine.T):.4f}\n"
                tensor_state += f"Cycles: {self.engine.inference_engine.cycle_counter}\n"
                tensor_state += f"Recent patterns: {self.engine.inference_engine.recent_patterns[-5:]}"

                layout["processing"].update(Panel(tensor_state, title="Processing", border_style="yellow"))
                live.update(layout)

                # Generate response
                response_length = min(200, len(user_input) * 2)  # Adaptive length
                response_bytes = self.engine.generate_and_save_response(response_length)

                # Show output
                try:
                    response_text = response_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    response_text = f"[Binary response: {len(response_bytes)} bytes]"

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
            response_text = f"[Binary response: {len(response_bytes)} bytes]"

        # Add to history
        self.conversation_history.append(
            {"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()}
        )

        # Display
        if RICH_AVAILABLE and console is not None:
            console.print(f"\n[bold green]Baby LM[/bold green]: {response_text}\n")
        else:
            print(f"\nBaby LM: {response_text}\n")

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

        filename = f"conversations/{name}.json"
        Path("conversations").mkdir(exist_ok=True)

        with open(filename, "w") as f:
            json.dump(
                {
                    "agent_uuid": self.engine.agent_uuid,
                    "thread_uuid": self.engine.thread_uuid,
                    "timestamp": datetime.now().isoformat(),
                    "history": self.conversation_history,
                },
                f,
                indent=2,
            )

        if RICH_AVAILABLE and console is not None:
            console.print(f"[green]Conversation saved to {filename}[/green]")
        else:
            print(f"Conversation saved to {filename}")

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
                    print(f"{i}. {conv.stem}")

                idx = input("Select conversation: ")
                name = conversations[int(idx) - 1].stem

        filename = f"conversations/{name}.json"
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            self.conversation_history = data["history"]

            if RICH_AVAILABLE and console is not None:
                console.print(f"[green]Loaded conversation from {filename}[/green]")
                console.print(f"[dim]Contains {len(self.conversation_history)} messages[/dim]")
            else:
                print(f"Loaded conversation from {filename}")
                print(f"Contains {len(self.conversation_history)} messages")

            # Display conversation
            for msg in self.conversation_history[-5:]:  # Show last 5 messages
                role = "You" if msg["role"] == "user" else "Baby LM"
                if RICH_AVAILABLE and console is not None:
                    console.print(f"[bold]{role}[/bold]: {msg['content']}")
                else:
                    print(f"{role}: {msg['content']}")

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            if RICH_AVAILABLE and console is not None:
                console.print(f"[red]Failed to load conversation: {e}[/red]")
            else:
                print(f"Failed to load conversation: {e}")

    def _show_thread_info(self):
        """Show current thread information"""
        if self.engine is None or not self.engine.thread_uuid:
            if RICH_AVAILABLE and console is not None:
                console.print("[yellow]No active thread.[/yellow]")
            else:
                print("No active thread.")
            return

        stats = self.engine.get_thread_statistics()
        relationships = self.engine.get_thread_relationships(self.engine.thread_uuid)

        if RICH_AVAILABLE and console is not None:
            info = f"""
[cyan]Thread UUID:[/cyan] {self.engine.thread_uuid}
[cyan]Size:[/cyan] {self.engine.current_thread_size} bytes
[cyan]Parent:[/cyan] {relationships['parent'] or 'None'}
[cyan]Children:[/cyan] {len(relationships['children'])}
[cyan]Gene Keys:[/cyan] {len(self.engine.current_thread_keys)}
            """
            console.print(Panel(info.strip(), title="Current Thread", border_style="cyan"))
        else:
            print(f"\nCurrent Thread:")
            print(f"  UUID: {self.engine.thread_uuid}")
            print(f"  Size: {self.engine.current_thread_size} bytes")
            print(f"  Parent: {relationships['parent'] or 'None'}")
            print(f"  Children: {len(relationships['children'])}")
            print(f"  Gene Keys: {len(self.engine.current_thread_keys)}")

    def _show_pattern_info(self, pattern_idx: Optional[int] = None):
        """Show pattern information"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        if pattern_idx is None:
            # Show most active patterns
            patterns = [(i, p) for i, p in enumerate(self.engine.M["patterns"])]
            active_patterns = sorted(patterns, key=lambda x: x[1].get("count", 0), reverse=True)[:10]

            if RICH_AVAILABLE and console is not None:
                table = Table(title="Most Active Patterns")
                table.add_column("Index", style="cyan")
                table.add_column("Count", style="green")
                table.add_column("Semantic", style="magenta")
                table.add_column("Class", style="yellow")

                for idx, pattern in active_patterns:
                    table.add_row(
                        str(idx),
                        str(pattern.get("count", 0)),
                        pattern.get("semantic") or "-",
                        pattern.get("resonance_class", "-"),
                    )

                console.print(table)
            else:
                print("\nMost Active Patterns:")
                for idx, pattern in active_patterns:
                    print(
                        f"  {idx}: count={pattern.get('count', 0)}, "
                        f"semantic={pattern.get('semantic') or '-'}, "
                        f"class={pattern.get('resonance_class', '-')}"
                    )
        else:
            # Show specific pattern
            if 0 <= pattern_idx < 256:
                stats = self.engine.get_pattern_statistics(pattern_idx)

                if RICH_AVAILABLE and console is not None:
                    info = f"""
[cyan]Index:[/cyan] {stats['pattern_index']}
[cyan]Semantic:[/cyan] {stats['semantic'] or 'None'}
[cyan]Count:[/cyan] {stats['count']}
[cyan]First Cycle:[/cyan] {stats['first_cycle'] or 'Never'}
[cyan]Last Cycle:[/cyan] {stats['last_cycle'] or 'Never'}
[cyan]Resonance Class:[/cyan] {stats['resonance_class']}
[cyan]Confidence:[/cyan] {stats['confidence']:.4f}
[cyan]Current Resonance:[/cyan] {stats['current_resonance']:.4f if stats['current_resonance'] else 'N/A'}
                    """
                    console.print(Panel(info.strip(), title=f"Pattern {pattern_idx}", border_style="cyan"))

                    # Show contexts if available
                    if stats["contexts"]:
                        console.print("\n[bold]Pattern Contexts:[/bold]")
                        if stats["contexts"]["before"]:
                            console.print("  [cyan]Common predecessors:[/cyan]")
                            for pred_idx, count in stats["contexts"]["before"][:5]:
                                console.print(f"    Pattern {pred_idx}: {count} times")
                        if stats["contexts"]["after"]:
                            console.print("  [cyan]Common successors:[/cyan]")
                            for succ_idx, count in stats["contexts"]["after"][:5]:
                                console.print(f"    Pattern {succ_idx}: {count} times")
                else:
                    print(f"\nPattern {pattern_idx}:")
                    print(f"  Semantic: {stats['semantic'] or 'None'}")
                    print(f"  Count: {stats['count']}")
                    print(f"  Resonance Class: {stats['resonance_class']}")
                    print(f"  Confidence: {stats['confidence']:.4f}")
            else:
                if RICH_AVAILABLE and console is not None:
                    console.print(f"[red]Invalid pattern index: {pattern_idx}[/red]")
                else:
                    print(f"Invalid pattern index: {pattern_idx}")

    def _toggle_developer_mode(self):
        """Toggle developer mode"""
        self.developer_mode = not self.developer_mode
        mode = "enabled" if self.developer_mode else "disabled"

        if RICH_AVAILABLE and console is not None:
            console.print(f"[green]Developer mode {mode}[/green]")
        else:
            print(f"Developer mode {mode}")

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
        parser.add_argument("--format", "-f", nargs="+", help="Format operations: list, show <uuid>, discover <domain>")

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
                    console.print(f"[green]Switched to agent {agent_uuid[:8]}...[/green]")
                else:
                    print(f"Switched to agent {agent_uuid[:8]}...")
        else:
            if RICH_AVAILABLE and console is not None:
                console.print(f"[red]Unknown agent operation: {operation}[/red]")
            else:
                print(f"Unknown agent operation: {operation}")

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
                console.print(f"[red]Unknown thread operation: {operation}[/red]")
            else:
                print(f"Unknown thread operation: {operation}")

    def _list_threads(self):
        """List all threads with rich formatting"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        stats = self.engine.get_thread_statistics()

        if RICH_AVAILABLE and console is not None:
            table = Table(title=f"Threads for Agent {self.engine.agent_uuid[:8]}...")
            table.add_column("Index", style="cyan", no_wrap=True)
            table.add_column("Thread UUID", style="magenta")
            table.add_column("Size", style="green", justify="right")
            table.add_column("Parent", style="yellow")
            table.add_column("Children", style="blue", justify="center")

            for i, detail in enumerate(stats["thread_details"], 1):
                size_kb = detail["size_bytes"] / 1024
                parent = "✓" if detail["has_parent"] else "-"
                children = str(detail["child_count"]) if detail["has_children"] else "-"

                table.add_row(str(i), detail["thread_uuid"][:8] + "...", f"{size_kb:.1f} KB", parent, children)

            console.print(table)

            # Summary
            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f"  Total threads: {stats['total_threads']}")
            console.print(f"  Total size: {stats['total_size_bytes'] / 1024:.1f} KB")
            console.print(f"  Capacity usage: {stats['capacity_usage_percent']:.1f}%")
        else:
            print(f"\nThreads for Agent {self.engine.agent_uuid[:8]}...")
            print("-" * 60)

            for i, detail in enumerate(stats["thread_details"], 1):
                size_kb = detail["size_bytes"] / 1024
                print(f"{i}. {detail['thread_uuid'][:8]}... - {size_kb:.1f} KB")
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
                    console.print(f"[red]No thread found matching: {thread_uuid}[/red]")
                else:
                    print(f"No thread found matching: {thread_uuid}")
                return
            elif len(matches) > 1:
                if RICH_AVAILABLE and console is not None:
                    console.print(f"[yellow]Multiple threads match: {thread_uuid}[/yellow]")
                    for match in matches:
                        console.print(f"  - {match}")
                else:
                    print(f"Multiple threads match: {thread_uuid}")
                    for match in matches:
                        print(f"  - {match}")
                return

            thread_uuid = matches[0]

        # Load thread with context
        result = self.engine.load_thread_with_context(thread_uuid)

        if "error" in result:
            if RICH_AVAILABLE and console is not None:
                console.print(f"[red]{result['error']}[/red]")
            else:
                print(result["error"])
            return

        # Display thread content
        if RICH_AVAILABLE and console is not None:
            # Main thread
            try:
                content = result["content"].decode("utf-8")
                if len(content) > 500:
                    content = content[:500] + "..."

                console.print(
                    Panel(
                        content,
                        title=f"Thread {thread_uuid[:8]}... ({result['size_bytes']} bytes)",
                        border_style="cyan",
                    )
                )
            except UnicodeDecodeError:
                console.print(
                    Panel(
                        f"[Binary content: {result['size_bytes']} bytes]",
                        title=f"Thread {thread_uuid[:8]}...",
                        border_style="cyan",
                    )
                )

            # Relationships
            rels = result["relationships"]
            rel_info = f"Parent: {rels['parent'][:8] + '...' if rels['parent'] else 'None'}\n"
            rel_info += f"Children: {len(rels['children'])}"

            console.print(Panel(rel_info, title="Relationships", border_style="yellow"))

            # Related threads preview
            if result.get("related_threads"):
                tree = Tree(f"Thread Chain ({len(result['related_threads'])} related)")

                for related in result["related_threads"]:
                    try:
                        preview = related["content"].decode("utf-8")[:50] + "..."
                    except UnicodeDecodeError:
                        preview = f"[Binary: {related['size_bytes']} bytes]"

                    node_text = f"{related['thread_uuid'][:8]}... ({related['relationship']})"
                    tree.add(f"{node_text}\n[dim]{preview}[/dim]")

                console.print(tree)
        else:
            # Plain text display
            print(f"\nThread {thread_uuid[:8]}... ({result['size_bytes']} bytes)")
            print("-" * 60)

            try:
                content = result["content"].decode("utf-8")
                if len(content) > 500:
                    content = content[:500] + "..."
                print(content)
            except UnicodeDecodeError:
                print(f"[Binary content: {result['size_bytes']} bytes]")

            print(f"\nRelationships:")
            print(f"  Parent: {result['relationships']['parent'] or 'None'}")
            print(f"  Children: {len(result['relationships']['children'])}")

    def _show_thread_stats(self):
        """Show detailed thread statistics"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        stats = self.engine.get_thread_statistics()

        if RICH_AVAILABLE and console is not None:
            # Create statistics panels
            overview = Panel(
                f"[cyan]Total Threads:[/cyan] {stats['total_threads']}\n"
                f"[cyan]Total Size:[/cyan] {stats['total_size_bytes'] / 1024:.1f} KB\n"
                f"[cyan]Capacity Usage:[/cyan] {stats['capacity_usage_percent']:.1f}%",
                title="Overview",
                border_style="cyan",
            )

            relationships = Panel(
                f"[green]With Parents:[/green] {stats['relationship_stats']['threads_with_parents']}\n"
                f"[green]With Children:[/green] {stats['relationship_stats']['threads_with_children']}\n"
                f"[green]Isolated:[/green] {stats['relationship_stats']['isolated_threads']}",
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
                    f"[magenta]Average:[/magenta] {avg_size / 1024:.1f} KB\n"
                    f"[magenta]Maximum:[/magenta] {max_size / 1024:.1f} KB\n"
                    f"[magenta]Minimum:[/magenta] {min_size / 1024:.1f} KB",
                    title="Size Distribution",
                    border_style="magenta",
                )
                console.print(size_dist)
        else:
            print("\nThread Statistics")
            print("-" * 40)
            print(f"Total Threads: {stats['total_threads']}")
            print(f"Total Size: {stats['total_size_bytes'] / 1024:.1f} KB")
            print(f"Capacity Usage: {stats['capacity_usage_percent']:.1f}%")
            print(f"\nRelationships:")
            print(f"  With Parents: {stats['relationship_stats']['threads_with_parents']}")
            print(f"  With Children: {stats['relationship_stats']['threads_with_children']}")
            print(f"  Isolated: {stats['relationship_stats']['isolated_threads']}")

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
                    node = tree.add(f"[bold cyan]{uuid[:8]}... (CURRENT)[/bold cyan]")
                else:
                    node = tree.add(f"{uuid[:8]}...")
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
                console.print(f"[red]Thread {thread_uuid} not found[/red]")
            else:
                print(f"Thread {thread_uuid} not found")
            return

        filename = f"thread_{thread_uuid[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, "wb") as f:
            f.write(content)

        if RICH_AVAILABLE and console is not None:
            console.print(f"[green]Thread exported to {filename}[/green]")
        else:
            print(f"Thread exported to {filename}")

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
                console.print(f"[red]Unknown format operation: {operation}[/red]")
            else:
                print(f"Unknown format operation: {operation}")

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
                format_data = load_format(format_uuid)
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
                format_data = load_format(format_uuid)
                if format_data:
                    print(f"{i}. {format_uuid[:8]}... - {format_data.get('format_name', 'Unknown')}")

    def _show_format(self, format_uuid: str):
        """Show detailed format information"""
        # Handle partial UUID
        if len(format_uuid) < 36:
            format_uuids = list_formats()
            matches = [f for f in format_uuids if f.startswith(format_uuid)]

            if len(matches) == 0:
                if RICH_AVAILABLE and console is not None:
                    console.print(f"[red]No format found matching: {format_uuid}[/red]")
                else:
                    print(f"No format found matching: {format_uuid}")
                return
            elif len(matches) > 1:
                if RICH_AVAILABLE and console is not None:
                    console.print(f"[yellow]Multiple formats match: {format_uuid}[/yellow]")
                else:
                    print(f"Multiple formats match: {format_uuid}")
                return

            format_uuid = matches[0]

        format_data = load_format(format_uuid)
        if not format_data:
            if RICH_AVAILABLE and console is not None:
                console.print(f"[red]Format {format_uuid} not found[/red]")
            else:
                print(f"Format {format_uuid} not found")
            return

        if RICH_AVAILABLE and console is not None:
            # Basic info
            basic_info = f"""
[cyan]UUID:[/cyan] {format_data['format_uuid']}
[cyan]Name:[/cyan] {format_data['format_name']}
[cyan]Version:[/cyan] {format_data['format_version']}
[cyan]CGM Version:[/cyan] {format_data['cgm_version']}
[cyan]Stability:[/cyan] {format_data['stability']}
            """
            console.print(Panel(basic_info.strip(), title="Format Information", border_style="cyan"))

            # Metadata
            meta = format_data.get("metadata", {})
            meta_info = f"""
[green]Author:[/green] {meta.get('author', 'Unknown')}
[green]Description:[/green] {meta.get('description', 'None')}
[green]Created:[/green] {meta.get('created_at', 'Unknown')}
[green]Usage Count:[/green] {meta.get('usage_count', 0)}
[green]Tags:[/green] {', '.join(meta.get('tags', []))}
            """
            console.print(Panel(meta_info.strip(), title="Metadata", border_style="green"))

            # Pattern statistics
            patterns = format_data.get("patterns", [])
            labeled = sum(1 for p in patterns if p.get("semantic"))
            active = sum(1 for p in patterns if p.get("count", 0) > 0)

            pattern_info = f"""
[magenta]Total Patterns:[/magenta] {len(patterns)}
[magenta]Labeled:[/magenta] {labeled}
[magenta]Active:[/magenta] {active}
            """
            console.print(Panel(pattern_info.strip(), title="Pattern Statistics", border_style="magenta"))
        else:
            print(f"\nFormat: {format_data['format_uuid']}")
            print(f"Name: {format_data['format_name']}")
            print(f"Version: {format_data['format_version']}")
            print(f"Stability: {format_data['stability']}")
            print(f"Author: {format_data.get('metadata', {}).get('author', 'Unknown')}")
            print(f"Description: {format_data.get('metadata', {}).get('description', 'None')}")

    def _discover_format(self, domain: str):
        """Discover format for a domain"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        if RICH_AVAILABLE and console is not None:
            with console.status(f"Discovering format for domain '{domain}'...", spinner="dots"):
                for stability in ["stable", "beta", "experimental"]:
                    format_uuid = self.engine.select_stable_format(domain, stability)
                    if format_uuid:
                        format_data = load_format(format_uuid)
                        console.print(f"[green]Found {stability} format:[/green]")
                        console.print(f"  UUID: {format_uuid}")
                        if format_data is not None:
                            print(f"  Name: {format_data.get('format_name', 'Unknown')}")
                        else:
                            print(f"  Name: Unknown")
                        return

            console.print(f"[yellow]No format found for domain '{domain}'[/yellow]")
        else:
            print(f"Discovering format for domain '{domain}'...")
            for stability in ["stable", "beta", "experimental"]:
                format_uuid = self.engine.select_stable_format(domain, stability)
                if format_uuid:
                    format_data = load_format(format_uuid)
                    print(f"Found {stability} format:")
                    print(f"  UUID: {format_uuid}")
                    if format_data is not None:
                        print(f"  Name: {format_data.get('format_name', 'Unknown')}")
                    else:
                        print(f"  Name: Unknown")
                    return

            print(f"No format found for domain '{domain}'")

    def _compose_formats(self, format_uuids: List[str]):
        """Compose multiple formats"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        primary = format_uuids[0]
        secondary = format_uuids[1:]

        if RICH_AVAILABLE and console is not None:
            console.print(f"Composing formats:")
            console.print(f"  Primary: {primary}")
            for s in secondary:
                console.print(f"  Secondary: {s}")

            with console.status("Composing formats...", spinner="dots"):
                composed_uuid = self.engine.compose_formats(primary, secondary)

            if composed_uuid:
                console.print(f"[green]✓ Created composed format: {composed_uuid}[/green]")
            else:
                console.print(f"[red]✗ Failed to compose formats[/red]")
        else:
            print(f"Composing formats:")
            print(f"  Primary: {primary}")
            for s in secondary:
                print(f"  Secondary: {s}")

            composed_uuid = self.engine.compose_formats(primary, secondary)

            if composed_uuid:
                print(f"✓ Created composed format: {composed_uuid}")
            else:
                print(f"✗ Failed to compose formats")

    def _handle_process(self, text: Optional[str], input_file: Optional[str], output_file: Optional[str]):
        """Handle text processing"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        # Get input
        if text:
            data = text.encode("utf-8")
            source = "command line"
        elif input_file:
            try:
                with open(input_file, "rb") as f:
                    data = f.read()
                source = input_file
            except FileNotFoundError:
                if RICH_AVAILABLE and console is not None:
                    console.print(f"[red]Input file not found: {input_file}[/red]")
                else:
                    print(f"Input file not found: {input_file}")
                return
        else:
            return

        if RICH_AVAILABLE and console is not None:
            with console.status(f"Processing {len(data)} bytes from {source}...", spinner="dots"):
                plaintext, encrypted = self.engine.process_input_stream(data)
            if self.engine.thread_uuid:
                console.print(f"[green]✓ Processed and saved to thread {self.engine.thread_uuid[:8]}...[/green]")
            else:
                console.print(f"[green]✓ Processed and saved to thread [unknown]...[/green]")
            if self.developer_mode:
                # Show processing details
                import numpy as np

                recent_patterns = self.engine.inference_engine.recent_patterns
                if recent_patterns:
                    recent_patterns_str = recent_patterns[-5:]
                else:
                    recent_patterns_str = []
                info = f"""
[cyan]Input size:[/cyan] {len(data)} bytes
[cyan]Thread UUID:[/cyan] {self.engine.thread_uuid if self.engine.thread_uuid else '[unknown]'}
[cyan]Cycles processed:[/cyan] {len(data)}
[cyan]Current cycle:[/cyan] {self.engine.inference_engine.cycle_counter}
[cyan]Recent patterns:[/cyan] {recent_patterns_str}
                """
                console.print(Panel(info.strip(), title="Processing Details", border_style="yellow"))
        else:
            print(f"Processing {len(data)} bytes from {source}...")
            plaintext, encrypted = self.engine.process_input_stream(data)
            if self.engine.thread_uuid:
                print(f"✓ Processed and saved to thread {self.engine.thread_uuid[:8]}...")
            else:
                print(f"✓ Processed and saved to thread [unknown]...")

        # Save output if requested
        if output_file:
            with open(output_file, "wb") as f:
                f.write(encrypted)

            if RICH_AVAILABLE and console is not None:
                console.print(f"[green]✓ Saved encrypted output to {output_file}[/green]")
            else:
                print(f"✓ Saved encrypted output to {output_file}")

    def _handle_generate(self, length: int, output_file: Optional[str]):
        """Handle text generation"""
        if self.engine is None:
            print("Error: No engine initialized.")
            return

        import numpy as np

        if RICH_AVAILABLE and console is not None:
            with console.status(f"Generating {length} bytes...", spinner="dots"):
                response = self.engine.generate_and_save_response(length)
            if self.engine.thread_uuid:
                console.print(f"[green]✓ Generated and saved to thread {self.engine.thread_uuid[:8]}...[/green]")
            else:
                console.print(f"[green]✓ Generated and saved to thread [unknown]...[/green]")
            # Try to display as text
            try:
                text = response.decode("utf-8")
                if len(text) > 200:
                    text = text[:200] + "..."
                console.print(Panel(text, title=f"Generated Response ({len(response)} bytes)", border_style="green"))
            except UnicodeDecodeError:
                console.print(f"[yellow]Generated binary response ({len(response)} bytes)[/yellow]")
        else:
            print(f"Generating {length} bytes...")
            response = self.engine.generate_and_save_response(length)
            if self.engine.thread_uuid:
                print(f"✓ Generated and saved to thread {self.engine.thread_uuid[:8]}...")
            else:
                print(f"✓ Generated and saved to thread [unknown]...")
            try:
                text = response.decode("utf-8")
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"\nGenerated response:\n{text}")
            except UnicodeDecodeError:
                print(f"Generated binary response ({len(response)} bytes)")

        # Save output if requested
        if output_file:
            with open(output_file, "wb") as f:
                f.write(response)

            if RICH_AVAILABLE and console is not None:
                console.print(f"[green]✓ Saved generated response to {output_file}[/green]")
            else:
                print(f"✓ Saved generated response to {output_file}")


def main():
    """Main entry point"""
    # Import numpy here to avoid issues if not installed
    try:
        import numpy as np
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
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        if RICH_AVAILABLE and console is not None:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Check babylm_cli.log for details[/dim]")
        else:
            print(f"\nError: {e}")
            print("Check babylm_cli.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()

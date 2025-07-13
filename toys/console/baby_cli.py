#!/usr/bin/env python3
"""
Baby LM Console Interface - Private Agent Only Version
A beautiful, interactive CLI for the GyroSI Baby Language Model.
"""

import sys
from pathlib import Path
from typing import Optional

import os
import base64

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from baby.intelligence import IntelligenceEngine, initialize_intelligence_engine
from toys.console.chat import ChatInterface
from toys.console.dashboard import Dashboard
from toys.console.threads import ThreadManager
from toys.console.formats import FormatViewer
from toys.console.utils import create_header, create_footer, handle_error

console = Console()


class BabyLMCLI:
    """Main CLI application class."""
    
    def __init__(self) -> None:
        self.engine: Optional[IntelligenceEngine] = None
        self.chat_interface: Optional[ChatInterface] = None
        self.dashboard: Optional[Dashboard] = None
        self.thread_manager: Optional[ThreadManager] = None
        self.format_viewer: Optional[FormatViewer] = None
    
    def initialize_engine(self) -> bool:
        """Initialize the intelligence engine for private agent mode only."""
        console.print(create_header("🧠 Baby LM Initialization"))
        from baby.information import ensure_agent_uuid
        import json
        agent_name = None
        agent_uuid = None
        agent_secret = None
        baby_prefs_path = Path("baby/baby_preferences.json")
        # Check if agent exists
        if baby_prefs_path.exists():
            with open(baby_prefs_path, "r") as f:
                prefs = json.load(f)
            agent_name = prefs.get("agent_name")
            agent_secret = prefs.get("agent_secret")
            agent_uuid = ensure_agent_uuid()
        if agent_uuid:
            # Agent exists, offer to continue as this agent
            name_display = agent_name if agent_name else f"Agent {agent_uuid[:8]}..."
            choice = questionary.select(
                f"Continue as {name_display}?",
                choices=[
                    questionary.Choice(f"Continue as {name_display}", "continue"),
                    questionary.Choice("Create New Agent", "create"),
                    questionary.Choice("Exit", "exit"),
                ],
                style=questionary.Style([
                    ('question', 'fg:#ff0066 bold'),
                    ('pointer', 'fg:#ff0066 bold'),
                    ('choice', 'fg:#884444'),
                    ('selected', 'fg:#cc5454 bold'),
                ])
            ).ask()
            if choice == "exit" or choice is None:
                console.print("[yellow]Exited by user.[/]")
                return False
            elif choice == "continue":
                if not agent_secret:
                    console.print("[bold red]Error: Agent found but secret is missing in preferences. Cannot continue.[/]")
                    return False
                with console.status("[bold green]Loading agent..."):
                    self.engine = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=agent_secret)
                console.print(f"[green]✅ Loaded agent: [bold]{name_display}[/bold][/]")
            elif choice == "create":
                agent_name = questionary.text(
                    "Enter a name for your new agent (optional):",
                    style=questionary.Style([('question', 'fg:#ff0066 bold')])
                ).ask()
                with console.status("[bold green]Creating new agent..."):
                    agent_uuid = ensure_agent_uuid()
                    # Generate a new, secure secret
                    new_secret = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
                    Path("baby").mkdir(exist_ok=True)
                    prefs = {"agent_secret": new_secret}
                    if agent_name:
                        prefs["agent_name"] = agent_name
                    with open("baby/baby_preferences.json", "w") as f:
                        json.dump(prefs, f)
                    self.engine = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=new_secret)
                name_display = agent_name if agent_name else f"Agent {agent_uuid[:8]}..."
                console.print(f"[green]✅ New agent created: [bold]{name_display}[/bold][/]")
        else:
            # No agent exists, prompt to create one
            agent_name = questionary.text(
                "Enter a name for your new agent (optional):",
                style=questionary.Style([('question', 'fg:#ff0066 bold')])
            ).ask()
            with console.status("[bold green]Creating new agent..."):
                agent_uuid = ensure_agent_uuid()
                # Generate a new, secure secret
                new_secret = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
                Path("baby").mkdir(exist_ok=True)
                prefs = {"agent_secret": new_secret}
                if agent_name:
                    prefs["agent_name"] = agent_name
                with open("baby/baby_preferences.json", "w") as f:
                    json.dump(prefs, f)
                self.engine = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=new_secret)
            name_display = agent_name if agent_name else f"Agent {agent_uuid[:8]}..."
            console.print(f"[green]✅ New agent created: [bold]{name_display}[/bold][/]")
        # Consolidated subsystem initialization
        if self.engine:
            console.print("[green]Initializing CLI subsystems...[/]")
            self.chat_interface = ChatInterface(self.engine)
            self.dashboard = Dashboard(self.engine)
            self.thread_manager = ThreadManager(self.engine)
            self.format_viewer = FormatViewer(self.engine)
            return True
        return False
    
    def show_main_menu(self) -> None:
        """Display the main menu and handle navigation."""
        while True:
            console.clear()
            console.print(create_header("🧠 Baby LM Console"))
            # Display session info
            if self.engine:
                agent_id = self.engine.agent_uuid[:16] + "..." if self.engine.agent_uuid else "Private Mode"
                format_id = self.engine.format_uuid[:8] + "..." if self.engine.format_uuid else "None"
                info_text = Text()
                info_text.append("Session: ", style="dim")
                info_text.append("🔒 Private", style="bold cyan")
                info_text.append(f"\nAgent: ", style="dim")
                info_text.append(agent_id, style="dim")
                info_text.append(f"\nFormat: ", style="dim")
                info_text.append(format_id, style="dim")
                console.print(Panel(info_text, title="Session Info", border_style="blue"))
            # Main menu choices
            choices = [
                questionary.Choice("💬 Start Chat Session", "chat"),
                questionary.Choice("📊 Live Dashboard", "dashboard"),
                questionary.Choice("🧵 Browse Threads", "threads"),
                questionary.Choice("🎨 View Formats", "formats"),
                questionary.Choice("🔄 Reinitialize Agent", "reinit"),
                questionary.Choice("❌ Exit", "exit"),
            ]
            choice = questionary.select(
                "What would you like to do?",
                choices=choices,
                style=questionary.Style([
                    ('question', 'fg:#ff0066 bold'),
                    ('pointer', 'fg:#ff0066 bold'),
                    ('choice', 'fg:#884444'),
                    ('selected', 'fg:#cc5454 bold'),
                ])
            ).ask()
            if choice is None or choice == "exit":
                break
            elif choice == "chat" and self.chat_interface:
                self.chat_interface.run()
            elif choice == "dashboard" and self.dashboard:
                self.dashboard.run()
            elif choice == "threads" and self.thread_manager:
                self.thread_manager.run()
            elif choice == "formats" and self.format_viewer:
                self.format_viewer.run()
            elif choice == "reinit":
                if self.initialize_engine():
                    console.print("[green]✅ Agent reinitialized successfully![/]")
                    questionary.press_any_key_to_continue().ask()
    
    def run(self) -> None:
        """Main application entry point."""
        console.clear()
        try:
            if self.initialize_engine():
                self.show_main_menu()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Exiting gracefully.[/]")
        except Exception as e:
            handle_error(console, "Critical application error", e)
        finally:
            console.print(create_footer("Thank you for using Baby LM! 🧠✨"))

def main() -> None:
    """Entry point for the CLI application."""
    app = BabyLMCLI()
    app.run()

if __name__ == "__main__":
    main()
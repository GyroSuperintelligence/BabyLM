#!/usr/bin/env python3
"""
Baby LM Console Interface - Complete Rich/Questionary Version
A beautiful, interactive CLI for the GyroSI Baby Language Model.
"""

import sys
from pathlib import Path
from typing import Optional

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
        """Initialize the intelligence engine based on user choice."""
        console.print(create_header("🧠 Baby LM Initialization"))
        # Add a quote below the header
        console.print(Panel(
            Text('“Intelligence is not the ability to store information, but to know where to find it.”\n— Albert Einstein', justify="center"),
            border_style="magenta",
            style="dim",
        ))
        
        try:
            session_type = questionary.select(
                "Choose your session type:",
                choices=[
                    questionary.Choice("🌐 Public Session (No encryption, data is public)", "public"),
                    questionary.Choice("🔒 Private Session (Encrypted, requires secret)", "private"),
                    questionary.Choice("❌ Exit", "exit"),
                ],
                style=questionary.Style([
                    ('question', 'fg:#ff0066 bold'),
                    ('pointer', 'fg:#ff0066 bold'),
                    ('choice', 'fg:#884444'),
                    ('selected', 'fg:#cc5454 bold'),
                ])
            ).ask()

            if session_type is None or session_type == "exit":
                console.print("[yellow]Exited by user.[/]")
                return False

            if session_type == 'public':
                with console.status("[bold green]Initializing public session..."):
                    self.engine = initialize_intelligence_engine()
                console.print("[green]✅ Public session initialized successfully![/]")
            
            elif session_type == 'private':
                agent_secret = questionary.password(
                    "Enter your agent secret:",
                    style=questionary.Style([('question', 'fg:#ff0066 bold')])
                ).ask()
                
                if not agent_secret:
                    console.print("[red]Agent secret is required for private mode.[/]")
                    return False
                
                with console.status("[bold green]Initializing private session..."):
                    self.engine = initialize_intelligence_engine(agent_secret=agent_secret)
                console.print("[green]✅ Private session initialized successfully![/]")

            # Initialize subsystems
            if self.engine:
                self.chat_interface = ChatInterface(self.engine)
                self.dashboard = Dashboard(self.engine)
                self.thread_manager = ThreadManager(self.engine)
                self.format_viewer = FormatViewer(self.engine)
                return True
            
            return False

        except Exception as e:
            handle_error(console, "Failed to initialize engine", e)
            return False
    
    def show_main_menu(self) -> None:
        """Display the main menu and handle navigation."""
        while True:
            console.clear()
            console.print(create_header("🧠 Baby LM Console"))
            
            # Display session info
            if self.engine:
                mode = "🔒 Private" if self.engine.agent_uuid else "🌐 Public"
                agent_id = self.engine.agent_uuid[:16] + "..." if self.engine.agent_uuid else "Public Mode"
                format_id = self.engine.format_uuid[:8] + "..." if self.engine.format_uuid else "None"
                
                info_text = Text()
                info_text.append("Session: ", style="dim")
                info_text.append(mode, style="bold cyan" if self.engine.agent_uuid else "bold yellow")
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
                questionary.Choice("🔄 Reinitialize Engine", "reinit"),
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
                    console.print("[green]✅ Engine reinitialized successfully![/]")
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
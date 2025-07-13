"""
Thread management and browsing interface.
"""

from datetime import datetime
from typing import List, Dict, Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.json import JSON

import os
import json
from pathlib import Path

from baby.intelligence import IntelligenceEngine
from baby.information import load_gene_keys
from toys.console.utils import create_header, handle_error, format_bytes

console = Console()


class ThreadManager:
    """Thread management and browsing interface."""
    
    def __init__(self, engine: IntelligenceEngine) -> None:
        self.engine = engine
    
    def list_threads(self) -> List[Dict[str, Any]]:
        """Get list of available threads from the engine and sort them."""
        try:
            if not self.engine.agent_uuid:
                return []
            stats = self.engine.get_thread_statistics()
            thread_details = stats.get("thread_details", [])
            thread_details.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return thread_details
        except Exception as e:
            handle_error(console, "Failed to list threads", e)
            return []
    
    def display_threads_table(self, threads: List[Dict[str, Any]]) -> Table:
        """Create a table of threads."""
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("ID", style="dim", width=12)
        table.add_column("Name", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Size", justify="right")
        table.add_column("Children", justify="center")
        table.add_column("Tags", style="yellow")
        
        for thread in threads[:20]:  # Show max 20
            thread_id = thread["thread_uuid"][:8] + "..."
            thread_name = thread.get("thread_name") or "Unnamed"
            
            # Format creation date
            created_at = "Unknown"
            if thread.get("created_at"):
                try:
                    dt = datetime.fromisoformat(thread["created_at"].replace('Z', '+00:00'))
                    created_at = dt.strftime("%m/%d %H:%M")
                except Exception:
                    pass
            
            size = format_bytes(thread["size_bytes"])
            children = str(thread["child_count"])
            tags_list = thread.get("tags") or []
            tags = ", ".join(tags_list[:2])
            if len(tags_list) > 2:
                tags += "..."
            
            table.add_row(thread_id, thread_name, created_at, size, children, tags)
        
        return table
    
    def display_thread_details(self, thread: dict) -> bool:
        """Display thread details and offer actions. Returns True if user wants to resume chat."""
        console.clear()
        console.print(create_header(f"🧵 Thread Details: {thread['thread_uuid'][:8]}..."))
        # Load metadata
        metadata = self.engine.load_thread_metadata(thread["thread_uuid"])
        # Display metadata
        metadata_text = f"""🆔 Thread UUID: {thread['thread_uuid']}
  📝 Name: {metadata.get('thread_name') or 'Unnamed'}
  👤 Agent UUID: {metadata.get('agent_uuid') or 'N/A'}
  👨‍👩‍👧‍👦 Parent UUID: {metadata.get('parent_uuid') or 'None'}
  🌳 Children: {len(metadata.get('children', []))}
  🎨 Format UUID: {metadata.get('format_uuid') or 'N/A'}
  📚 Curriculum: {metadata.get('curriculum') or 'None'}
  🏷️  Tags: {metadata.get('tags') or 'None'}
  📅 Created: {metadata.get('created_at') or 'N/A'}
  🕒 Updated: {metadata.get('last_updated') or 'N/A'}
  📊 Size: {format_bytes(metadata.get('size_bytes', 0))}
  🔒 Privacy: {metadata.get('privacy') or 'private'}
"""
        console.print(Panel(metadata_text, title="📋 Metadata", border_style="cyan"))
        actions = [
            "💬 View Thread Content",
            "🧬 Gene Keys",
            "🔄 Resume Chat in this Thread",
            "🔙 Back to Thread List"
        ]
        while True:
            action = questionary.select(
                "What would you like to view?",
                choices=actions,
                style=questionary.Style([
                    ('question', 'fg:#ff0066 bold'),
                    ('pointer', 'fg:#ff0066 bold'),
                    ('choice', 'fg:#884444'),
                    ('selected', 'fg:#cc5454 bold'),
                ])
            ).ask()
            if action == "💬 View Thread Content":
                console.clear()
                self.view_thread_content(thread["thread_uuid"])
                questionary.press_any_key_to_continue().ask()
                console.clear()
            elif action == "🧬 Gene Keys":
                console.clear()
                self.view_gene_keys(thread["thread_uuid"])
                questionary.press_any_key_to_continue().ask()
                console.clear()
            elif action == "🔄 Resume Chat in this Thread":
                privacy = "private" if self.engine.agent_uuid else "public"
                self.engine.resume_thread(thread["thread_uuid"], privacy=privacy)
                return True
            elif action == "🔙 Back to Thread List":
                return False
    
    def view_thread_content(self, thread_uuid: str) -> None:
        """Display thread conversation content."""
        try:
            content = self.engine.load_thread_content(thread_uuid)
            if not content:
                console.print("[red]No content available for this thread.[/red]")
                questionary.press_any_key_to_continue().ask()
                return
            
            console.print(f"\n[bold]💬 Thread Content ({len(content)} events)[/bold]")
            console.print("─" * console.width)
            
            for i, event in enumerate(content[:20]):  # Show first 20 events
                event_type = event.get("type", "unknown")
                
                if event_type == "input":
                    data = event.get("data", b"")
                    if isinstance(data, bytes):
                        message = data.decode('utf-8', errors='replace')
                    else:
                        message = str(data)
                    
                    console.print(Panel(
                        message[:200] + ("..." if len(message) > 200 else ""),
                        title=f"👤 User (Event {i+1})",
                        border_style="blue"
                    ))
                
                elif event_type == "output":
                    data = event.get("data", b"")
                    if isinstance(data, bytes):
                        message = data.decode('utf-8', errors='replace')
                    else:
                        message = str(data)
                    
                    console.print(Panel(
                        message[:200] + ("..." if len(message) > 200 else ""),
                        title=f"🤖 AI (Event {i+1})",
                        border_style="magenta"
                    ))
            
            if len(content) > 20:
                console.print(f"[dim]... and {len(content) - 20} more events[/dim]")
            
            questionary.press_any_key_to_continue().ask()
            
        except Exception as e:
            handle_error(console, "Failed to view thread content", e)
    
    def view_gene_keys(self, thread_uuid: str) -> None:
        """Display gene keys for a thread."""
        try:
            gene_keys = load_gene_keys(
                thread_uuid,
                self.engine.memory_prefs,
                self.engine.agent_uuid,
                self.engine.agent_secret,
                base_memories_dir=self.engine.base_memories_dir
            )
            
            if not gene_keys:
                console.print("[red]No gene keys available for this thread.[/red]")
                questionary.press_any_key_to_continue().ask()
                return
            
            console.print(f"\n[bold]🧬 Gene Keys ({len(gene_keys)} entries)[/bold]")
            
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Cycle", justify="right")
            table.add_column("Pattern", justify="right")
            table.add_column("Character")
            table.add_column("Resonance", justify="right")
            table.add_column("Type")
            
            for key in gene_keys[:50]:  # Show first 50
                cycle = str(key.get("cycle", 0))
                pattern_idx = key.get("pattern_index", 0)
                character = self.engine.decode(pattern_idx) or f"#{pattern_idx}"
                resonance = f"{key.get('resonance', 0.0):.3f}"
                event_type = key.get("event_type", "unknown")
                
                table.add_row(cycle, str(pattern_idx), character, resonance, event_type)
            
            console.print(table)
            
            if len(gene_keys) > 50:
                console.print(f"[dim]... and {len(gene_keys) - 50} more entries[/dim]")
            
            questionary.press_any_key_to_continue().ask()
            
        except Exception as e:
            handle_error(console, "Failed to view gene keys", e)
    
    def view_thread_stats(self, thread_uuid: str) -> None:
        """Display thread statistics."""
        try:
            # Get relationships
            relationships = self.engine.get_thread_relationships(thread_uuid)
            thread_chain = self.engine.get_thread_chain(thread_uuid)
            
            stats_text = f"""🔗 RELATIONSHIPS
Parent: {relationships.get('parent') or 'None'}
Children: {len(relationships.get('children', []))}
Chain Length: {len(thread_chain)}

🧵 THREAD CHAIN"""
            
            for i, chain_uuid in enumerate(thread_chain):
                current = " (current)" if chain_uuid == thread_uuid else ""
                stats_text += f"\n{i+1}. {chain_uuid[:8]}...{current}"
            
            console.print(Panel(stats_text, title="📊 Thread Statistics", border_style="green"))
            questionary.press_any_key_to_continue().ask()
            
        except Exception as e:
            handle_error(console, "Failed to view thread stats", e)
    
    def resume_thread(self, thread_uuid: str) -> bool:
        """Resume an existing thread. Returns True if user wants to chat."""
        try:
            # Load agent secret from preferences
            baby_prefs_path = Path("baby/baby_preferences.json")
            agent_secret = None
            if baby_prefs_path.exists():
                with open(baby_prefs_path, "r") as f:
                    prefs = json.load(f)
                agent_secret = prefs.get("agent_secret")
            if not agent_secret:
                console.print("[bold red]Error: Agent secret missing. Cannot resume thread.[/]")
                return False
            privacy = "private" if self.engine.agent_uuid else "public"
            self.engine.resume_thread(thread_uuid, privacy=privacy)
            # Defensive check: did resume succeed?
            if self.engine.thread_uuid != thread_uuid:
                console.print(f"[bold red]Failed to resume thread {thread_uuid[:8]}... It may be corrupted or the secret is incorrect.[/]")
                return False
            console.print(f"[green]✅ Resumed thread {thread_uuid[:8]}...[/green]")
            # Ask user to jump to chat
            if questionary.confirm("Start chatting in this thread now?").ask():
                return True
            return False
        except Exception as e:
            handle_error(console, "Failed to resume thread", e)
            return False
    
    def run(self) -> bool:
        """Main thread manager loop. Returns True if user wants to resume chat."""
        while True:
            console.clear()
            console.print(create_header("🧵 Thread Management"))
            
            if not self.engine.agent_uuid:
                console.print(Panel(
                    "[yellow]Thread management is only available in private mode.[/yellow]",
                    border_style="yellow"
                ))
                questionary.press_any_key_to_continue().ask()
                return False
            
            threads = self.list_threads()
            
            if not threads:
                console.print(Panel(
                    "[yellow]No threads found. Start a chat session to create threads.[/yellow]",
                    border_style="yellow"
                ))
                questionary.press_any_key_to_continue().ask()
                return False
            
            # Display threads table
            table = self.display_threads_table(threads)
            console.print(Panel(table, title=f"📚 Available Threads ({len(threads)})", border_style="blue"))
            
            # Create choices for thread selection
            selected = questionary.select(
                "Select a thread to view details:",
                choices=[
                    *[f"{t['thread_uuid'][:8]}... - {t.get('thread_name', 'Unnamed')} ({format_bytes(t['size_bytes'])})" for t in threads],
                    "🔙 Back to Main Menu"
                ],
                style=questionary.Style([
                    ('question', 'fg:#ff0066 bold'),
                    ('pointer', 'fg:#ff0066 bold'),
                    ('choice', 'fg:#884444'),
                    ('selected', 'fg:#cc5454 bold'),
                ])
            ).ask()
            
            if selected == "🔙 Back to Main Menu":
                return False
            
            # Find the selected thread
            idx = [f"{t['thread_uuid'][:8]}... - {t.get('thread_name', 'Unnamed')} ({format_bytes(t['size_bytes'])})" for t in threads].index(selected)
            thread = threads[idx]
            
            # Show details and check if user wants to resume chat
            if self.display_thread_details(thread):
                return True
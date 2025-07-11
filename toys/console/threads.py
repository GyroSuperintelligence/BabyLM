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

from baby.intelligence import IntelligenceEngine
from baby.information import load_gene_keys
from toys.console.utils import create_header, handle_error, format_bytes

console = Console()


class ThreadManager:
    """Thread management and browsing interface."""
    
    def __init__(self, engine: IntelligenceEngine) -> None:
        self.engine = engine
    
    def list_threads(self) -> List[Dict[str, Any]]:
        """Get list of available threads."""
        try:
            if not self.engine.agent_uuid:
                return []
            
            stats = self.engine.get_thread_statistics()
            thread_details = stats.get("thread_details", [])
            
            # Sort by creation date (most recent first)
            for thread in thread_details:
                try:
                    metadata = self.engine.load_thread_metadata(thread["thread_uuid"])
                    thread["created_at"] = metadata.get("created_at", "")
                    thread["thread_name"] = metadata.get("thread_name")
                    thread["curriculum"] = metadata.get("curriculum")
                    thread["tags"] = metadata.get("tags", [])
                except Exception:
                    thread["created_at"] = ""
                    thread["thread_name"] = None
                    thread["curriculum"] = None
                    thread["tags"] = []
            
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
            tags = ", ".join(thread.get("tags", [])[:2])  # Show first 2 tags
            if len(thread.get("tags", [])) > 2:
                tags += "..."
            
            table.add_row(thread_id, thread_name, created_at, size, children, tags)
        
        return table
    
    def view_thread_details(self, thread_uuid: str) -> None:
        """Display detailed information about a thread."""
        try:
            console.clear()
            console.print(create_header(f"🧵 Thread Details: {thread_uuid[:8]}..."))
            
            # Load metadata
            metadata = self.engine.load_thread_metadata(thread_uuid)
            
            # Display metadata
            metadata_text = f"""🆔 Thread UUID: {thread_uuid}
📝 Name: {metadata.get('thread_name') or 'Unnamed'}
👤 Agent UUID: {metadata.get('agent_uuid') or 'N/A'}
👨‍👩‍👧‍👦 Parent UUID: {metadata.get('parent_uuid') or 'None'}
🌳 Children: {len(metadata.get('children', []))}
🎨 Format UUID: {metadata.get('format_uuid', 'N/A')}
📚 Curriculum: {metadata.get('curriculum') or 'None'}
🏷️  Tags: {', '.join(metadata.get('tags', []) or ['None'])}
📅 Created: {metadata.get('created_at', 'Unknown')}
🕒 Updated: {metadata.get('last_updated', 'Unknown')}
📊 Size: {format_bytes(metadata.get('size_bytes', 0))}
🔒 Privacy: {metadata.get('privacy', 'Unknown')}"""
            
            console.print(Panel(metadata_text, title="📋 Metadata", border_style="blue"))
            
            # Menu for detailed views
            choice = questionary.select(
                "What would you like to view?",
                choices=[
                    questionary.Choice("💬 Thread Content", "content"),
                    questionary.Choice("🧬 Gene Keys", "genes"),
                    questionary.Choice("📊 Statistics", "stats"),
                    questionary.Choice("↩️ Resume Thread", "resume"),
                    questionary.Choice("🔙 Back to List", "back"),
                ]
            ).ask()
            
            if choice == "content":
                self.view_thread_content(thread_uuid)
            elif choice == "genes":
                self.view_gene_keys(thread_uuid)
            elif choice == "stats":
                self.view_thread_stats(thread_uuid)
            elif choice == "resume":
                self.resume_thread(thread_uuid)
            
        except Exception as e:
            handle_error(console, "Failed to view thread details", e)
    
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
    
    def resume_thread(self, thread_uuid: str) -> None:
        """Resume an existing thread."""
        try:
            privacy = "private" if self.engine.agent_uuid else "public"
            self.engine.resume_thread(thread_uuid, privacy=privacy)
            console.print(f"[green]✅ Resumed thread {thread_uuid[:8]}...[/green]")
            questionary.press_any_key_to_continue().ask()
            
        except Exception as e:
            handle_error(console, "Failed to resume thread", e)
    
    def run(self) -> None:
        """Run the thread management interface."""
        while True:
            console.clear()
            console.print(create_header("🧵 Thread Management"))
            
            if not self.engine.agent_uuid:
                console.print(Panel(
                    "[yellow]Thread management is only available in private mode.[/yellow]",
                    border_style="yellow"
                ))
                questionary.press_any_key_to_continue().ask()
                return
            
            threads = self.list_threads()
            
            if not threads:
                console.print(Panel(
                    "[yellow]No threads found. Start a chat session to create threads.[/yellow]",
                    border_style="yellow"
                ))
                questionary.press_any_key_to_continue().ask()
                return
            
            # Display threads table
            table = self.display_threads_table(threads)
            console.print(Panel(table, title=f"📚 Available Threads ({len(threads)})", border_style="blue"))
            
            # Create choices for thread selection
            choices = []
            for thread in threads[:10]:  # Show max 10 in menu
                thread_id = thread["thread_uuid"][:8]
                thread_name = thread.get("thread_name") or "Unnamed"
                size = format_bytes(thread["size_bytes"])
                choices.append(questionary.Choice(
                    f"{thread_id}... - {thread_name} ({size})",
                    thread["thread_uuid"]
                ))
            
            choices.append(questionary.Choice("🔙 Back to Main Menu", "back"))
            
            choice = questionary.select(
                "Select a thread to view details:",
                choices=choices
            ).ask()
            
            if choice is None or choice == "back":
                break
            
            self.view_thread_details(choice)
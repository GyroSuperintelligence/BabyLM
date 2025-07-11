"""
Format and pattern viewing interface.
"""

from typing import List, Dict, Any, Union
from baby.types import FormatMetadata

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from baby.intelligence import IntelligenceEngine
from baby.information import list_formats, load_format
from toys.console.utils import create_header, handle_error

console = Console()


class FormatViewer:
    """Format and pattern viewing interface."""
    
    def __init__(self, engine: IntelligenceEngine) -> None:
        self.engine = engine
    
    def get_available_formats(self) -> List[str]:
        """Get list of available format UUIDs."""
        try:
            return list_formats(base_memories_dir=self.engine.base_memories_dir)
        except Exception as e:
            handle_error(console, "Failed to list formats", e)
            return []
    
    def display_formats_table(self, format_uuids: List[str]) -> Table:
        """Create a table of available formats."""
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Format ID", style="dim", width=12)
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Stability", style="yellow")
        table.add_column("Author", style="magenta")
        
        for format_uuid in format_uuids:
            try:
                format_data = load_format(format_uuid, base_memories_dir=self.engine.base_memories_dir)
                if format_data:
                    format_id = format_uuid[:8] + "..."
                    name = format_data.get("format_name", "Unknown")
                    version = format_data.get("format_version", "Unknown")
                    stability = format_data.get("stability", "Unknown")
                    author = format_data.get("metadata", {}).get("author", "Unknown")
                    
                    table.add_row(format_id, name, version, stability, author)
            except Exception:
                continue
        
        return table
    
    def view_format_details(self, format_uuid: str) -> None:
        """Display detailed information about a format."""
        try:
            console.clear()
            console.print(create_header(f"🎨 Format Details: {format_uuid[:8]}..."))
            
            format_data = load_format(format_uuid, base_memories_dir=self.engine.base_memories_dir)
            if not format_data:
                console.print("[red]Format not found.[/red]")
                questionary.press_any_key_to_continue().ask()
                return
            
            # Display format metadata
            metadata = format_data.get("metadata", {})
            compatibility = format_data.get("compatibility", {})
            
            details_text = f"""🆔 UUID: {format_uuid}
📝 Name: {format_data.get('format_name', 'Unknown')}
🔢 Version: {format_data.get('format_version', 'Unknown')}
⚡ Stability: {format_data.get('stability', 'Unknown')}
👤 Author: {metadata.get('author', 'Unknown')}
📅 Created: {metadata.get('created_at', 'Unknown')}
🕒 Updated: {metadata.get('last_updated', 'Unknown')}
🎯 Usage Count: {metadata.get('usage_count', 0):,}
✅ Validation: {metadata.get('validation_status', 'Unknown')}

🔗 COMPATIBILITY
📦 Min Version: {compatibility.get('min_format_version', 'N/A')}
📦 Max Version: {compatibility.get('max_format_version', 'N/A')}
📋 Dependencies: {len(compatibility.get('depends_on', []))}
⚠️  Conflicts: {len(compatibility.get('conflicts_with', []))}

🏷️  TAGS
{', '.join(metadata.get('tags', []) or ['None'])}

📖 DESCRIPTION
{metadata.get('description', 'No description available')}"""
            
            console.print(Panel(details_text, title="📋 Format Information", border_style="blue"))
            
            # Menu for detailed views
            choice = questionary.select(
                "What would you like to view?",
                choices=[
                    questionary.Choice("🧬 Pattern Mappings", "patterns"),
                    questionary.Choice("📊 Pattern Statistics", "stats"),
                    questionary.Choice("🔄 Set as Active Format", "activate"),
                    questionary.Choice("🔙 Back to List", "back"),
                ]
            ).ask()
            
            if choice == "patterns":
                self.view_pattern_mappings(format_data)
            elif choice == "stats":
                self.view_pattern_statistics(format_data)
            elif choice == "activate":
                self.activate_format(format_uuid)
            
        except Exception as e:
            handle_error(console, "Failed to view format details", e)
    
    def view_pattern_mappings(self, format_data: Union[Dict[str, Any], FormatMetadata]) -> None:
        """Display pattern-to-character mappings."""
        try:
            if not format_data or not isinstance(format_data, dict):
                patterns = []
            else:
                patterns = format_data.get("patterns") or []
            
            # Separate mapped and unmapped patterns
            mapped_patterns = [p for p in patterns if p is not None and isinstance(p, dict) and p.get("character")]
            unmapped_patterns = [p for p in patterns if p is not None and isinstance(p, dict) and not p.get("character")]
            
            console.print(f"\n[bold]🧬 Pattern Mappings[/bold]")
            console.print(f"Mapped: {len(mapped_patterns)}, Unmapped: {len(unmapped_patterns)}")
            
            # Show mapped patterns first (sorted by confidence)
            if mapped_patterns:
                mapped_patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                
                table = Table(show_header=True, header_style="bold blue", title="📝 Mapped Patterns")
                table.add_column("Index", justify="right")
                table.add_column("Character", style="cyan")
                table.add_column("Description")
                table.add_column("Count", justify="right")
                table.add_column("Confidence", justify="right")
                table.add_column("Feature", style="dim")
                
                for pattern in mapped_patterns[:30]:  # Show top 30
                    if not isinstance(pattern, dict):
                        continue
                    index = str(pattern.get("index", 0))
                    character = str(pattern.get("character", ""))
                    description = pattern.get("description", "")[:30]
                    count = f"{pattern.get('count', 0):,}"
                    confidence = f"{pattern.get('confidence', 0.0):.3f}"
                    feature = pattern.get("gyration_feature", "")[:15]
                    
                    # Color code confidence
                    if pattern.get('confidence', 0) > 0.7:
                        confidence = f"[green]{confidence}[/green]"
                    elif pattern.get('confidence', 0) > 0.3:
                        confidence = f"[yellow]{confidence}[/yellow]"
                    else:
                        confidence = f"[red]{confidence}[/red]"
                    
                    table.add_row(index, character, description, count, confidence, feature)
                
                console.print(table)
            
            # Show some unmapped patterns
            if unmapped_patterns and questionary.confirm("Show unmapped patterns?").ask():
                table = Table(show_header=True, header_style="bold red", title="❓ Unmapped Patterns")
                table.add_column("Index", justify="right")
                table.add_column("Count", justify="right") 
                table.add_column("Confidence", justify="right")
                table.add_column("Feature", style="dim")
                
                for pattern in unmapped_patterns[:20]:  # Show 20 unmapped
                    index = str(pattern.get("index", 0))
                    count = f"{pattern.get('count', 0):,}"
                    confidence = f"{pattern.get('confidence', 0.0):.3f}"
                    feature = pattern.get("gyration_feature", "")[:20]
                    
                    table.add_row(index, count, confidence, feature)
                
                console.print(table)
            
            questionary.press_any_key_to_continue().ask()
            
        except Exception as e:
            handle_error(console, "Failed to view pattern mappings", e)
    
    def view_pattern_statistics(self, format_data: Union[Dict[str, Any], FormatMetadata]) -> None:
        """Display pattern statistics."""
        try:
            if not format_data or not isinstance(format_data, dict):
                patterns = []
            else:
                patterns = format_data.get("patterns") or []
            
            # Calculate statistics
            total_patterns = len(patterns)
            mapped_count = len([p for p in patterns if p.get("character")])
            total_usage = sum(p.get("count", 0) for p in patterns)
            avg_confidence = sum(p.get("confidence", 0) for p in patterns) / max(total_patterns, 1)
            
            # Top patterns by usage
            top_patterns = sorted(patterns, key=lambda x: x.get("count", 0), reverse=True)[:10]
            
            stats_text = f"""📊 PATTERN STATISTICS

📈 Total Patterns: {total_patterns}
📝 Mapped Patterns: {mapped_count} ({mapped_count/total_patterns*100:.1f}%)
❓ Unmapped Patterns: {total_patterns - mapped_count}
🎯 Total Usage: {total_usage:,}
📊 Average Confidence: {avg_confidence:.3f}

🏆 TOP 10 MOST USED PATTERNS"""
            
            for i, pattern in enumerate(top_patterns):
                character = pattern.get("character", f"#{pattern.get('index', 0)}")
                count = pattern.get("count", 0)
                confidence = pattern.get("confidence", 0.0)
                stats_text += f"\n{i+1:2d}. {character:>4} - {count:,} uses (conf: {confidence:.3f})"
            
            console.print(Panel(stats_text, title="📊 Pattern Statistics", border_style="green"))
            questionary.press_any_key_to_continue().ask()
            
        except Exception as e:
            handle_error(console, "Failed to view pattern statistics", e)
    
    def activate_format(self, format_uuid: str) -> None:
        """Set a format as the active format."""
        try:
            # This would require modifying the engine to change formats
            # For now, just show a message
            console.print(Panel(
                f"[yellow]Format activation not implemented yet.\n"
                f"Current active format: {self.engine.format_uuid[:8]}...\n"
                f"Requested format: {format_uuid[:8]}...[/yellow]",
                title="⚠️ Not Implemented",
                border_style="yellow"
            ))
            questionary.press_any_key_to_continue().ask()
            
        except Exception as e:
            handle_error(console, "Failed to activate format", e)
    
    def run(self) -> None:
        """Run the format viewing interface."""
        while True:
            console.clear()
            console.print(create_header("🎨 Format Management"))
            
            format_uuids = self.get_available_formats()
            
            if not format_uuids:
                console.print(Panel(
                    "[yellow]No formats found.[/yellow]",
                    border_style="yellow"
                ))
                questionary.press_any_key_to_continue().ask()
                return
            
            # Display formats table
            table = self.display_formats_table(format_uuids)
            console.print(Panel(table, title=f"🎨 Available Formats ({len(format_uuids)})", border_style="blue"))
            
            # Show current active format
            current_format = self.engine.format_uuid
            console.print(f"\n[dim]Current active format: [bold]{current_format[:8]}...[/bold][/dim]")
            
            # Create choices for format selection
            choices = []
            for format_uuid in format_uuids:
                try:
                    format_data = load_format(format_uuid, base_memories_dir=self.engine.base_memories_dir)
                    if format_data:
                        format_id = format_uuid[:8]
                        name = format_data.get("format_name", "Unknown")
                        version = format_data.get("format_version", "Unknown")
                        
                        label = f"{format_id}... - {name} v{version}"
                        if format_uuid == current_format:
                            label += " (active)"
                        
                        choices.append(questionary.Choice(label, format_uuid))
                except Exception:
                    continue
            
            choices.append(questionary.Choice("🔙 Back to Main Menu", "back"))
            
            choice = questionary.select(
                "Select a format to view details:",
                choices=choices
            ).ask()
            
            if choice is None or choice == "back":
                break
            
            self.view_format_details(choice)
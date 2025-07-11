"""
Live dashboard for monitoring Baby LM internals.
"""

import time
from typing import Dict, Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn

from baby.intelligence import IntelligenceEngine
from toys.console.utils import create_header, handle_error, format_bytes

console = Console()


class Dashboard:
    """Live dashboard for monitoring model internals."""
    
    def __init__(self, engine: IntelligenceEngine) -> None:
        self.engine = engine
        self.running = False
    
    def create_stats_panel(self) -> Panel:
        """Create the statistics panel."""
        try:
            inference_engine = self.engine.inference_engine
            
            # Cache statistics
            cache_hits = getattr(inference_engine, '_cache_hits', 0)
            cache_misses = getattr(inference_engine, '_cache_misses', 0)
            total_cache = cache_hits + cache_misses
            cache_ratio = cache_hits / total_cache if total_cache > 0 else 0
            
            stats_text = f"""🔄 Cycle Counter: [bold cyan]{inference_engine.cycle_counter:,}[/]
🎯 Cache Hit Ratio: [bold {'green' if cache_ratio > 0.5 else 'red'}]{cache_ratio:.1%}[/] ({cache_hits}/{total_cache})
📈 Recent Patterns: [bold yellow]{len(inference_engine.recent_patterns)}[/]/256
🧵 Active Thread: [bold blue]{self.engine.thread_uuid[:8] if self.engine.thread_uuid else 'None'}...[/]
📏 Thread Size: [bold magenta]{format_bytes(self.engine.current_thread_size)}[/]
📝 Gene Keys: [bold green]{len(self.engine.current_thread_keys):,}[/]
🔒 Mode: [bold {'cyan' if self.engine.agent_uuid else 'yellow'}]{'Private' if self.engine.agent_uuid else 'Public'}[/]"""
            
            return Panel(stats_text, title="📊 Statistics", border_style="blue")
            
        except Exception as e:
            return Panel(f"[red]Error: {e}[/red]", title="📊 Statistics", border_style="red")
    
    def create_tensor_panel(self) -> Panel:
        """Create the tensor visualization panel."""
        try:
            tensor = self.engine.inference_engine.T
            
            # Create a simple text visualization of the 4x2x3x2 tensor
            tensor_text = ""
            
            for i in range(4):
                for j in range(2):
                    slice_data = tensor[i, j, :, :]  # 3x2 slice
                    tensor_text += f"T[{i},{j},:,:]\n"
                    
                    for row in range(3):
                        for col in range(2):
                            value = slice_data[row, col]
                            # Color code based on value
                            if value < -0.5:
                                color = "blue"
                            elif value < 0:
                                color = "cyan"
                            elif value < 0.5:
                                color = "green"
                            elif value < 1.0:
                                color = "yellow"
                            else:
                                color = "red"
                            
                            tensor_text += f"[{color}]{value:6.2f}[/] "
                        tensor_text += "\n"
                    tensor_text += "\n"
            
            return Panel(tensor_text.strip(), title="🧠 Epigenome Tensor", border_style="green")
            
        except Exception as e:
            return Panel(f"[red]Error: {e}[/red]", title="🧠 Epigenome Tensor", border_style="red")
    
    def create_resonance_panel(self) -> Panel:
        """Create the pattern resonance panel."""
        try:
            resonances = self.engine.inference_engine.compute_pattern_resonances()
            
            # Find top resonant patterns
            resonant_indices = [(i, dist) for i, dist in enumerate(resonances)]
            resonant_indices.sort(key=lambda x: x[1])  # Sort by distance (lower = more resonant)
            
            resonance_text = ""
            for i, (pattern_idx, distance) in enumerate(resonant_indices[:15]):
                character = self.engine.decode(pattern_idx) or f"#{pattern_idx}"
                resonance_pct = max(0, (1 - distance / 3.14159) * 100)
                
                # Create a progress bar
                bar_length = int(resonance_pct / 5)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                color = "green" if resonance_pct > 70 else "yellow" if resonance_pct > 40 else "red"
                resonance_text += f"{i+1:2d}. [{color}]{character:>4}[/] │{bar}│ {resonance_pct:5.1f}%\n"
            
            return Panel(resonance_text.strip(), title="🔊 Top Resonant Patterns", border_style="magenta")
            
        except Exception as e:
            return Panel(f"[red]Error: {e}[/red]", title="🔊 Pattern Resonance", border_style="red")
    
    def create_threads_table(self) -> Panel:
        """Create the threads capacity table."""
        try:
            stats = self.engine.get_thread_statistics()
            
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Thread ID", style="dim")
            table.add_column("Size", justify="right")
            table.add_column("Capacity", justify="right")
            table.add_column("Children", justify="center")
            
            thread_details = stats.get("thread_details", [])
            for thread in thread_details[:8]:  # Show top 8
                thread_id = thread["thread_uuid"][:8] + "..."
                size = format_bytes(thread["size_bytes"])
                capacity = f"{thread['capacity_percent']:.1f}%"
                children = str(thread["child_count"])
                
                # Color code capacity
                capacity_pct = thread['capacity_percent']
                if capacity_pct > 80:
                    capacity_style = "red"
                elif capacity_pct > 60:
                    capacity_style = "yellow"
                else:
                    capacity_style = "green"
                
                table.add_row(
                    thread_id,
                    size,
                    f"[{capacity_style}]{capacity}[/]",
                    children
                )
            
            return Panel(table, title="📚 Thread Capacity", border_style="yellow")
            
        except Exception as e:
            return Panel(f"[red]Error: {e}[/red]", title="📚 Thread Capacity", border_style="red")
    
    def create_memory_panel(self) -> Panel:
        """Create memory usage panel."""
        try:
            memory_text = ""
            
            # Basic memory info
            thread_stats = self.engine.get_thread_statistics()
            total_threads = thread_stats.get("total_threads", 0)
            total_size = thread_stats.get("total_size_bytes", 0)
            
            memory_text += f"🧵 Total Threads: [bold cyan]{total_threads:,}[/]\n"
            memory_text += f"📚 Thread Storage: [bold yellow]{format_bytes(total_size)}[/]\n"
            
            if total_threads > 0:
                avg_size = total_size / total_threads
                memory_text += f"📊 Avg Thread Size: [bold green]{format_bytes(avg_size)}[/]\n"
            
            # Cache info
            cache_size = len(getattr(self.engine.inference_engine, '_distance_cache', {}))
            memory_text += f"🔍 Cache Entries: [bold magenta]{cache_size:,}[/]\n"
            
            # Try to get process memory if psutil available
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_text += f"🖥️  Process Memory: [bold red]{memory_mb:.1f} MB[/]"
            except ImportError:
                memory_text += "🖥️  Process Memory: [dim]N/A (psutil not available)[/]"
            
            return Panel(memory_text, title="💾 Memory Usage", border_style="red")
            
        except Exception as e:
            return Panel(f"[red]Error: {e}[/red]", title="💾 Memory Usage", border_style="red")
    
    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="stats"),
            Layout(name="tensor")
        )
        
        layout["right"].split_column(
            Layout(name="resonance"),
            Layout(name="threads"),
            Layout(name="memory")
        )
        
        return layout
    
    def run(self) -> None:
        """Run the live dashboard."""
        console.clear()
        
        self.running = True
        layout = self.create_layout()
        
        try:
            with Live(layout, refresh_per_second=2, screen=True):
                while self.running:
                    # Update header
                    layout["header"].update(Panel(
                        "[bold]Baby LM Live Dashboard[/] - Press [bold red]Ctrl+C[/] to exit",
                        style="white on blue"
                    ))
                    
                    # Update panels
                    layout["stats"].update(self.create_stats_panel())
                    layout["tensor"].update(self.create_tensor_panel())
                    layout["resonance"].update(self.create_resonance_panel())
                    layout["threads"].update(self.create_threads_table())
                    layout["memory"].update(self.create_memory_panel())
                    
                    # Update footer
                    layout["footer"].update(Panel(
                        f"[dim]Last updated: {time.strftime('%H:%M:%S')} | "
                        f"Updates every 0.5s | Press Ctrl+C to exit[/]",
                        style="white on black"
                    ))
                    
                    time.sleep(0.5)
                    
        except KeyboardInterrupt:
            self.running = False
            console.print("\n[yellow]Dashboard closed.[/]")
        except Exception as e:
            handle_error(console, "Dashboard error", e)
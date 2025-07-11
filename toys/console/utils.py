"""
Utility functions for the Baby LM Console Interface.
"""

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback


def create_header(title: str) -> Panel:
    """Create a styled header panel."""
    header_text = Text(title, style="bold white")
    return Panel(
        header_text,
        style="white on blue",
        padding=(0, 1)
    )


def create_footer(message: str) -> Panel:
    """Create a styled footer panel."""
    footer_text = Text(message, style="dim white")
    return Panel(
        footer_text,
        style="white on black",
        padding=(0, 1)
    )


def handle_error(console: Console, message: str, exception: Optional[Exception] = None) -> None:
    """Handle and display errors in a consistent way."""
    error_text = f"[bold red]{message}[/bold red]"
    
    if exception:
        error_text += f"\n[red]{str(exception)}[/red]"
    
    console.print(Panel(
        error_text,
        title="❌ Error",
        border_style="red"
    ))
    
    # If you want debug tracebacks, add a debug flag to this function or use an environment variable.
    # Example:
    # if exception and debug:
    #     console.print(Traceback.from_exception(type(exception), exception, exception.__traceback__))


def format_bytes(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate a string to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_uuid(uuid_str: str, length: int = 8) -> str:
    """Format a UUID for display by taking first N characters."""
    if not uuid_str:
        return "None"
    return uuid_str[:length] + "..." if len(uuid_str) > length else uuid_str


def create_progress_bar(current: int, total: int, width: int = 20) -> str:
    """Create a simple ASCII progress bar."""
    if total == 0:
        return "░" * width
    
    progress = min(current / total, 1.0)
    filled = int(progress * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {progress:.1%}"


def validate_uuid_format(uuid_str: str) -> bool:
    """Validate UUID format."""
    if not uuid_str or not isinstance(uuid_str, str):
        return False
    
    # Basic UUID format check: 8-4-4-4-12 characters with hyphens
    parts = uuid_str.split('-')
    if len(parts) != 5:
        return False
    
    expected_lengths = [8, 4, 4, 4, 12]
    for part, expected_length in zip(parts, expected_lengths):
        if len(part) != expected_length or not all(c in '0123456789abcdefABCDEF' for c in part):
            return False
    
    return True


def style_text_by_confidence(text: str, confidence: float) -> str:
    """Style text based on confidence level."""
    if confidence > 0.8:
        return f"[bold green]{text}[/bold green]"
    elif confidence > 0.6:
        return f"[green]{text}[/green]"
    elif confidence > 0.4:
        return f"[yellow]{text}[/yellow]"
    elif confidence > 0.2:
        return f"[red]{text}[/red]"
    else:
        return f"[dim red]{text}[/dim red]"


def create_status_indicator(status: str) -> str:
    """Create a status indicator with emoji and color."""
    status_map = {
        "ready": ("✅", "green"),
        "processing": ("⚡", "yellow"),
        "error": ("❌", "red"),
        "warning": ("⚠️", "yellow"),
        "loading": ("🔄", "blue"),
        "success": ("✨", "green"),
        "private": ("🔒", "cyan"),
        "public": ("🌐", "yellow"),
    }
    
    emoji, color = status_map.get(status.lower(), ("❓", "white"))
    return f"[{color}]{emoji}[/{color}]"
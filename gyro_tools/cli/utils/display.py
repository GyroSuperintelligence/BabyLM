from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.progress import Progress, BarColumn, TextColumn
from ..config import CUSTOM_THEME

console = Console(theme=CUSTOM_THEME)

def show_panel(content, title="", style="info", subtitle=None):
    """Display a styled panel with content."""
    console.print(Panel(
        content,
        title=title,
        subtitle=subtitle,
        style=style,
        padding=(1, 2),
        box=box.ROUNDED
    ))

def show_error(message, title="Error"):
    """Display an error panel."""
    show_panel(f"[error]{message}[/error]", title=f"❌ {title}", style="error")

def show_success(message, title="Success"):
    """Display a success panel."""
    show_panel(f"[success]{message}[/success]", title=f"✅ {title}", style="success")

def show_info(message, title="Info"):
    """Display an info panel."""
    show_panel(f"[info]{message}[/info]", title=f"ℹ️ {title}", style="info")

def create_table(title, columns, rows, box_style=box.ROUNDED):
    """Create and return a styled table."""
    table = Table(title=f"[bold magenta]{title}[/bold magenta]", box=box_style)
    for col_name, col_style in columns:
        table.add_column(col_name, style=col_style)
    for row in rows:
        table.add_row(*row)
    return table

def show_progress(description, task_func, *args, **kwargs):
    """Show a progress bar while executing a task."""
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True
    ) as progress:
        task = progress.add_task(description=f"[cyan]{description}...", total=100)
        result = task_func(*args, **kwargs)
        progress.update(task, completed=100)
    return result
from rich.prompt import Confirm
import questionary
from ..config import BIOS_STYLE

def confirm_action(prompt, default=False):
    """Show a confirmation prompt."""
    return Confirm.ask(f"[warning]{prompt}[/warning]", default=default)

def select_from_list(title, choices, allow_back=True):
    """Show a selection list with optional back button."""
    if allow_back:
        choices = choices + [questionary.Separator(), "← Back"]
    
    result = questionary.select(
        title,
        choices=choices,
        style=BIOS_STYLE,
        use_indicator=True
    ).ask()
    
    if result == "← Back":
        return None
    return result

def get_text_input(prompt, default=""):
    """Get text input from user."""
    return questionary.text(prompt, default=default, style=BIOS_STYLE).ask()

def get_path_input(prompt, default=""):
    """Get file path input from user."""
    return questionary.path(prompt, default=default, style=BIOS_STYLE).ask()

def get_confirmation(prompt, default=False):
    """Get yes/no confirmation from user."""
    return questionary.confirm(prompt, default=default, style=BIOS_STYLE).ask()
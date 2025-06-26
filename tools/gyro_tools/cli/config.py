from rich.theme import Theme
from questionary import Style

# Custom theme for CLI
CUSTOM_THEME = Theme({
    "info": "bold cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "title": "bold magenta",
    "option": "bold blue",
    "emoji": "bold",
})

# Custom style for questionary with rounded corners
BIOS_STYLE = Style([
    ('qmark', 'fg:#673ab7 bold'),       # Question mark
    ('question', 'bold'),               # Question text
    ('pointer', 'fg:#cc5454 bold'),     # Pointer to selected item
    ('selected', 'fg:#cc5454 bg:#000000'), # Selected item
    ('answer', 'fg:#f44336 bold'),      # Answer text
    ('separator', 'fg:#cc5454'),         # Separator
    ('instruction', ''),                # Instruction text
    ('text', ''),
    ('disabled', 'fg:#858585 italic')
])

# Application metadata
APP_NAME = "GyroSI Baby ML CLI"
APP_SUBTITLE = "Mechanical • Auditable • Encrypted • Interactive"
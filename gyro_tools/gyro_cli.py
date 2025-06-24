import argparse
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.theme import Theme
from rich.progress import Progress, BarColumn, TextColumn
from rich.prompt import Confirm
from core.extension_manager import ExtensionManager
from gyro_curriculum_manager import CURRICULUM_RESOURCES, ingest_resource

# Custom theme for CLI
custom_theme = Theme(
    {
        "info": "bold cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "success": "bold green",
        "title": "bold magenta",
        "option": "bold blue",
        "emoji": "bold",
    }
)
console = Console(theme=custom_theme)


def show_welcome():
    # Clean, markup-free main panel
    console.print(
        Panel(
            Text("👶  GyroSI Baby ML CLI", justify="center"),
            subtitle="Mechanical • Auditable • Encrypted • Interactive",
            style="title",
            padding=(1, 2),
            box=box.DOUBLE,
        )
    )
    # Feature table with color highlights and spacing
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_row(
        "[bold cyan]🧠[/bold cyan]",
        "[option]Knowledge Management[/option]",
        "[info]Export, import, fork, link, info, delete, list[/info]",
    )
    table.add_row(
        "[bold blue]💬[/bold blue]",
        "[option]Session Management[/option]",
        "[info]List, export, import, link, info, delete[/info]",
    )
    table.add_row(
        "[bold yellow]🛡️[/bold yellow]",
        "[option]Integrity Checks[/option]",
        "[info]Check knowledge/session integrity visually[/info]",
    )
    table.add_row(
        "[bold green]⚙️[/bold green]",
        "[option]System Status[/option]",
        "[info]Show health, extensions, operator matrix status[/info]",
    )
    table.add_row(
        "[bold]❓[/bold]",
        "[option]Help[/option]",
        "[info]Show usage and examples for any command[/info]",
    )
    console.print(table)
    # Section divider
    console.print("\n" + "─" * 60 + "\n")
    # Real command examples
    console.print("[bold cyan]Examples:[/bold cyan]")
    console.print(
        "  [green]python gyro_tools/gyro_cli.py knowledge list[/green]         # List all knowledge packages"
    )
    console.print(
        "  [green]python gyro_tools/gyro_cli.py session export --session-id <ID> --output mysession.session.gyro[/green]  # Export a session"
    )
    console.print(
        "  [green]python gyro_tools/gyro_cli.py system health[/green]         # Show system health metrics"
    )
    # Help tip
    console.print(
        "\n[bold]Tip:[/bold] Add [yellow]--help[/yellow] after any command for more options."
    )


# --- Knowledge Management Visuals ---
def knowledge_list(args):
    knowledge_dir = Path("data/knowledge")
    if not knowledge_dir.exists() or not any(knowledge_dir.iterdir()):
        console.print(
            Panel(
                "[info]No knowledge packages found.[/info]", title="🧠 Knowledge List", style="info"
            )
        )
        return
    table = Table(title="[bold magenta]Knowledge Packages[/bold magenta]", box=box.ROUNDED)
    table.add_column("🆔 ID", style="option")
    table.add_column("📄 Meta", style="info")
    table.add_column("🔗 Linked Sessions", style="success")
    for pkg in knowledge_dir.iterdir():
        meta_path = pkg / "knowledge.meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            knowledge_id = meta.get("knowledge_id", pkg.name)
            meta_str = meta.get("description") or meta.get("name") or "(no description)"
            # Count linked sessions
            session_count = 0
            sessions_dir = Path("data/sessions")
            if sessions_dir.exists():
                for sess in sessions_dir.iterdir():
                    link_path = sess / "active_knowledge.link"
                    if link_path.exists():
                        try:
                            with open(link_path) as lf:
                                linked_id = lf.read().strip()
                            if linked_id == knowledge_id:
                                session_count += 1
                        except Exception:
                            continue
            table.add_row(knowledge_id, meta_str, str(session_count))
        except Exception as e:
            table.add_row(pkg.name, f"[error]Error: {e}[/error]", "?")
    console.print(table)


def knowledge_export(args):
    knowledge_id = getattr(args, "knowledge_id", None)
    output = getattr(args, "output", None)
    if not knowledge_id or not output:
        console.print(
            Panel(
                "[error]You must specify --knowledge-id and --output.[/error]",
                title="🧠 Export",
                style="error",
            )
        )
        return
    try:
        mgr = ExtensionManager(knowledge_id=knowledge_id)
        with Progress(
            TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True
        ) as progress:
            progress.add_task(
                description=f"[cyan]Exporting knowledge '{knowledge_id}'...",
                total=100,
                completed=100,
            )
            mgr.export_knowledge(output)
        console.print(
            Panel(
                f"[success]Knowledge '{knowledge_id}' exported to {output}![/success]",
                title="🧠 Export",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[error]Error exporting knowledge: {e}[/error]", title="🧠 Export", style="error"
            )
        )


def knowledge_import(args):
    input_path = getattr(args, "input", None)
    if not input_path:
        console.print(
            Panel("[error]You must specify --input.[/error]", title="🧠 Import", style="error")
        )
        return
    try:
        mgr = ExtensionManager()
        with Progress(
            TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True
        ) as progress:
            progress.add_task(
                description=f"[cyan]Importing knowledge from '{input_path}'...",
                total=100,
                completed=100,
            )
            new_knowledge_id = mgr.import_knowledge(input_path)
        console.print(
            Panel(
                f"[success]Knowledge imported as {new_knowledge_id}![/success]",
                title="🧠 Import",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[error]Error importing knowledge: {e}[/error]", title="🧠 Import", style="error"
            )
        )


def knowledge_fork(args):
    source = getattr(args, "source", None)
    session = getattr(args, "session", None)
    if not source:
        console.print(
            Panel(
                "[error]You must specify --source knowledge UUID to fork.[/error]",
                title="🧬 Fork",
                style="error",
            )
        )
        return
    try:
        mgr = ExtensionManager(knowledge_id=source)
        new_knowledge_id = mgr.fork_knowledge(new_session=bool(session))
        msg = f"Knowledge '{source}' forked as {new_knowledge_id}!"
        if session:
            msg += f"\nSession '{session}' linked to forked knowledge."
        console.print(Panel(f"[success]{msg}[/success]", title="🧬 Fork", style="success"))
    except Exception as e:
        console.print(
            Panel(f"[error]Error forking knowledge: {e}[/error]", title="🧬 Fork", style="error")
        )


def knowledge_link(args):
    session = getattr(args, "session", None)
    knowledge = getattr(args, "knowledge", None)
    if not session or not knowledge:
        console.print(
            Panel(
                "[error]You must specify --session and --knowledge to link.[/error]",
                title="🔗 Link",
                style="error",
            )
        )
        return
    try:
        mgr = ExtensionManager(session_id=session)
        mgr.link_to_knowledge(knowledge)
        console.print(
            Panel(
                f"[success]Session '{session}' linked to knowledge '{knowledge}'![/success]",
                title="🔗 Link",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(f"[error]Error linking session: {e}[/error]", title="🔗 Link", style="error")
        )


def knowledge_info(args):
    knowledge_id = args.knowledge_id
    pkg_dir = Path("data/knowledge") / knowledge_id
    meta_path = pkg_dir / "knowledge.meta.json"
    if not meta_path.exists():
        console.print(
            Panel(
                f"[error]Knowledge package '{knowledge_id}' not found.[/error]",
                title="🧠 Knowledge Info",
                style="error",
            )
        )
        return
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        table = Table(
            title=f"[bold magenta]Knowledge Info: {knowledge_id}[/bold magenta]", box=box.ROUNDED
        )
        table.add_column("Field", style="option")
        table.add_column("Value", style="info")
        for k, v in meta.items():
            table.add_row(str(k), str(v))
        # Count linked sessions
        session_count = 0
        sessions_dir = Path("data/sessions")
        if sessions_dir.exists():
            for sess in sessions_dir.iterdir():
                link_path = sess / "active_knowledge.link"
                if link_path.exists():
                    try:
                        with open(link_path) as lf:
                            linked_id = lf.read().strip()
                        if linked_id == knowledge_id:
                            session_count += 1
                    except Exception:
                        continue
        table.add_row("Linked Sessions", str(session_count))
        console.print(table)
    except Exception as e:
        console.print(
            Panel(
                f"[error]Error reading knowledge info: {e}[/error]",
                title="🧠 Knowledge Info",
                style="error",
            )
        )


def confirm_action(prompt):
    return Confirm.ask(f"[warning]{prompt}[/warning]", default=False)


def knowledge_delete(args):
    knowledge_id = getattr(args, "knowledge_id", None)
    if not knowledge_id:
        console.print(
            Panel(
                "[error]You must specify --knowledge-id to delete.[/error]",
                title="🗑️ Delete",
                style="error",
            )
        )
        return
    pkg_dir = Path("data/knowledge") / knowledge_id
    if not pkg_dir.exists():
        console.print(
            Panel(
                f"[error]Knowledge package '{knowledge_id}' not found.[/error]",
                title="🗑️ Delete",
                style="error",
            )
        )
        return
    if not confirm_action(
        f"Are you sure you want to permanently delete knowledge '{knowledge_id}'? This cannot be undone."
    ):
        console.print(Panel("[info]Delete cancelled.[/info]", title="🗑️ Delete", style="info"))
        return
    try:
        import shutil

        shutil.rmtree(pkg_dir)
        console.print(
            Panel(
                f"[success]Knowledge '{knowledge_id}' deleted![/success]",
                title="🗑️ Delete",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(f"[error]Error deleting knowledge: {e}[/error]", title="🗑️ Delete", style="error")
        )


# --- Session Management Visuals ---
def session_list(args):
    sessions_dir = Path("data/sessions")
    if not sessions_dir.exists() or not any(sessions_dir.iterdir()):
        console.print(
            Panel("[info]No sessions found.[/info]", title="💬 Session List", style="info")
        )
        return
    table = Table(title="[bold blue]Sessions[/bold blue]", box=box.ROUNDED)
    table.add_column("💬 ID", style="option")
    table.add_column("🧠 Knowledge", style="info")
    table.add_column("📅 Created", style="success")
    for sess in sessions_dir.iterdir():
        meta_path = sess / "session.meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            session_id = meta.get("id", sess.name)
            created = meta.get("created") or meta.get("created_at") or "?"
            # Linked knowledge
            link_path = sess / "active_knowledge.link"
            knowledge_id = "?"
            if link_path.exists():
                try:
                    with open(link_path) as lf:
                        knowledge_id = lf.read().strip()
                except Exception:
                    pass
            table.add_row(session_id, knowledge_id, created)
        except Exception as e:
            table.add_row(sess.name, f"[error]Error: {e}[/error]", "?")
    console.print(table)


def session_export(args):
    session_id = getattr(args, "session_id", None)
    output = getattr(args, "output", None)
    if not session_id or not output:
        console.print(
            Panel(
                "[error]You must specify --session-id and --output.[/error]",
                title="💬 Export",
                style="error",
            )
        )
        return
    try:
        mgr = ExtensionManager(session_id=session_id)
        with Progress(
            TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True
        ) as progress:
            progress.add_task(
                description=f"[cyan]Exporting session '{session_id}'...", total=100, completed=100
            )
            mgr.export_session(output)
        console.print(
            Panel(
                f"[success]Session '{session_id}' exported to {output}![/success]",
                title="💬 Export",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(f"[error]Error exporting session: {e}[/error]", title="💬 Export", style="error")
        )


def session_import(args):
    input_path = getattr(args, "input", None)
    if not input_path:
        console.print(
            Panel("[error]You must specify --input.[/error]", title="💬 Import", style="error")
        )
        return
    try:
        mgr = ExtensionManager()
        with Progress(
            TextColumn("[progress.description]{task.description}"), BarColumn(), transient=True
        ) as progress:
            progress.add_task(
                description=f"[cyan]Importing session from '{input_path}'...",
                total=100,
                completed=100,
            )
            new_session_id = mgr.import_session(input_path)
        console.print(
            Panel(
                f"[success]Session imported as {new_session_id}![/success]",
                title="💬 Import",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(f"[error]Error importing session: {e}[/error]", title="💬 Import", style="error")
        )


def session_link(args):
    session = getattr(args, "session", None)
    knowledge = getattr(args, "knowledge", None)
    if not session or not knowledge:
        console.print(
            Panel(
                "[error]You must specify --session and --knowledge to link.[/error]",
                title="🔗 Link",
                style="error",
            )
        )
        return
    try:
        mgr = ExtensionManager(session_id=session)
        mgr.link_to_knowledge(knowledge)
        console.print(
            Panel(
                f"[success]Session '{session}' linked to knowledge '{knowledge}'![/success]",
                title="🔗 Link",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(f"[error]Error linking session: {e}[/error]", title="🔗 Link", style="error")
        )


def session_info(args):
    session_id = args.session_id
    sess_dir = Path("data/sessions") / session_id
    meta_path = sess_dir / "session.meta.json"
    if not meta_path.exists():
        console.print(
            Panel(
                f"[error]Session '{session_id}' not found.[/error]",
                title="💬 Session Info",
                style="error",
            )
        )
        return
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        table = Table(title=f"[bold blue]Session Info: {session_id}[/bold blue]", box=box.ROUNDED)
        table.add_column("Field", style="option")
        table.add_column("Value", style="info")
        for k, v in meta.items():
            table.add_row(str(k), str(v))
        # Linked knowledge
        link_path = sess_dir / "active_knowledge.link"
        knowledge_id = "?"
        if link_path.exists():
            try:
                with open(link_path) as lf:
                    knowledge_id = lf.read().strip()
            except Exception:
                pass
        table.add_row("Linked Knowledge", knowledge_id)
        # Phase (if available)
        phase_path = sess_dir / "phase.bin"
        if phase_path.exists():
            try:
                with open(phase_path, "rb") as pf:
                    phase_bytes = pf.read()
                    if phase_bytes:
                        phase = int.from_bytes(phase_bytes, "little")
                        table.add_row("Phase", str(phase))
            except Exception:
                pass
        console.print(table)
    except Exception as e:
        console.print(
            Panel(
                f"[error]Error reading session info: {e}[/error]",
                title="💬 Session Info",
                style="error",
            )
        )


def session_delete(args):
    session_id = getattr(args, "session_id", None)
    if not session_id:
        console.print(
            Panel(
                "[error]You must specify --session-id to delete.[/error]",
                title="🗑️ Delete",
                style="error",
            )
        )
        return
    sess_dir = Path("data/sessions") / session_id
    if not sess_dir.exists():
        console.print(
            Panel(
                f"[error]Session '{session_id}' not found.[/error]", title="🗑️ Delete", style="error"
            )
        )
        return
    if not confirm_action(
        f"Are you sure you want to permanently delete session '{session_id}'? This cannot be undone."
    ):
        console.print(Panel("[info]Delete cancelled.[/info]", title="🗑️ Delete", style="info"))
        return
    try:
        import shutil

        shutil.rmtree(sess_dir)
        console.print(
            Panel(
                f"[success]Session '{session_id}' deleted![/success]",
                title="🗑️ Delete",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(f"[error]Error deleting session: {e}[/error]", title="🗑️ Delete", style="error")
        )


# --- Integrity Visuals ---
def integrity_check_knowledge(args):
    knowledge_id = getattr(args, "knowledge_id", None)
    if not knowledge_id:
        console.print(
            Panel(
                "[error]You must specify --knowledge-id to check.[/error]",
                title="🛡️ Knowledge Check",
                style="error",
            )
        )
        return
    pkg_dir = Path("data/knowledge") / knowledge_id
    meta_path = pkg_dir / "knowledge.meta.json"
    nav_log_path = pkg_dir / "navigation_log" / "genome.log"
    manifest_path = pkg_dir / "navigation_log" / "manifest.json"
    ext_dir = pkg_dir / "extensions"
    integrity_path = pkg_dir / "integrity.sha256"
    errors = []
    details = []
    if not pkg_dir.exists():
        errors.append(f"Knowledge package '{knowledge_id}' does not exist.")
    if not meta_path.exists():
        errors.append("Missing knowledge.meta.json")
    else:
        details.append("Metadata present.")
    if not nav_log_path.exists():
        errors.append("Missing navigation log (genome.log)")
    else:
        details.append("Navigation log present.")
    if not manifest_path.exists():
        errors.append("Missing navigation log manifest.json")
    else:
        details.append("Navigation log manifest present.")
    if not ext_dir.exists() or not any(ext_dir.iterdir()):
        details.append("No extension pattern files found.")
    else:
        details.append(
            f"Extension pattern files: {[f.name for f in ext_dir.iterdir() if f.is_file()]}"
        )
    if integrity_path.exists():
        details.append("Integrity file present. (Checksum not verified)")
    if errors:
        console.print(
            Panel("\n".join(errors + details), title="🛡️ Knowledge Check: FAILED", style="error")
        )
    else:
        console.print(
            Panel(
                "\n".join(details + ["Knowledge package integrity check PASSED."]),
                title="🛡️ Knowledge Check: PASSED",
                style="success",
            )
        )


def integrity_check_session(args):
    session_id = getattr(args, "session_id", None)
    if not session_id:
        console.print(
            Panel(
                "[error]You must specify --session-id to check.[/error]",
                title="🛡️ Session Check",
                style="error",
            )
        )
        return
    sess_dir = Path("data/sessions") / session_id
    meta_path = sess_dir / "session.meta.json"
    phase_path = sess_dir / "phase.bin"
    events_path = sess_dir / "events.log"
    knowledge_link = sess_dir / "active_knowledge.link"
    errors = []
    details = []
    if not sess_dir.exists():
        errors.append(f"Session '{session_id}' does not exist.")
    if not meta_path.exists():
        errors.append("Missing session.meta.json")
    else:
        details.append("Session metadata present.")
    if not phase_path.exists():
        details.append("Missing phase.bin")
    else:
        details.append("Phase file present.")
    if not events_path.exists():
        details.append("Missing events.log")
    else:
        details.append("Events log present.")
    if not knowledge_link.exists():
        details.append("Missing active_knowledge.link")
    else:
        details.append("Knowledge link present.")
    if errors:
        console.print(
            Panel("\n".join(errors + details), title="🛡️ Session Check: FAILED", style="error")
        )
    else:
        console.print(
            Panel(
                "\n".join(details + ["Session integrity check PASSED."]),
                title="🛡️ Session Check: PASSED",
                style="success",
            )
        )


# --- System Visuals ---
def system_health(args):
    try:
        mgr = ExtensionManager()
        health = mgr.get_system_health()
        table = Table(title="[bold green]System Health[/bold green]", box=box.ROUNDED)
        table.add_column("Metric", style="option")
        table.add_column("Value", style="info")
        for k, v in health.items():
            table.add_row(str(k), str(v))
        console.print(table)
    except Exception as e:
        console.print(
            Panel(
                f"[error]Error retrieving system health: {e}[/error]",
                title="⚙️ System Health",
                style="error",
            )
        )


def system_extensions(args):
    try:
        mgr = ExtensionManager()
        exts = mgr.extensions
        table = Table(title="[bold green]Loaded Extensions[/bold green]", box=box.ROUNDED)
        table.add_column("Name", style="option")
        table.add_column("Version", style="info")
        table.add_column("Footprint", style="success")
        for ext in exts.values():
            try:
                name = (
                    ext.get_extension_name()
                    if hasattr(ext, "get_extension_name")
                    else str(type(ext).__name__)
                )
                version = (
                    ext.get_extension_version() if hasattr(ext, "get_extension_version") else "?"
                )
                footprint = (
                    f"{ext.get_footprint_bytes()} bytes"
                    if hasattr(ext, "get_footprint_bytes")
                    else "?"
                )
                table.add_row(name, version, footprint)
            except Exception as e:
                table.add_row("[error]Error[/error]", f"[error]{e}[/error]", "?")
        console.print(table)
    except Exception as e:
        console.print(
            Panel(
                f"[error]Error listing extensions: {e}[/error]", title="⚙️ Extensions", style="error"
            )
        )


def system_matrix(args):
    try:
        from core.gyro_core import GyroEngine

        engine = GyroEngine()
        # If no exception, matrix is valid
        console.print(
            Panel(
                "[success]Operator matrix: VALID[/success]",
                title="⚙️ Operator Matrix",
                style="success",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[error]Operator matrix: INVALID\n{e}[/error]",
                title="⚙️ Operator Matrix",
                style="error",
            )
        )


def curriculum_list(args):
    resources = [
        {
            "name": "WordNet",
            "type": "Lexical DB",
            "size": "30MB",
            "desc": "Synonyms, definitions, word relations",
        },
        {
            "name": "Wiktionary",
            "type": "Dictionary",
            "size": "1-2GB",
            "desc": "Definitions, etymology, usage",
        },
        {
            "name": "Wikipedia (Simple)",
            "type": "Encyclopedia",
            "size": "800MB",
            "desc": "Simple English, easier to parse",
        },
        {
            "name": "Gutenberg Top 100",
            "type": "Literature",
            "size": "50MB",
            "desc": "Classic books, public domain",
        },
        {
            "name": "UDHR",
            "type": "Legal/Philosophy",
            "size": "<1MB",
            "desc": "Universal Declaration of Human Rights",
        },
        {
            "name": "Tatoeba",
            "type": "Sentence DB",
            "size": "500MB",
            "desc": "Example sentences, translations",
        },
        {
            "name": "News Crawl",
            "type": "News",
            "size": "100MB-1GB",
            "desc": "Recent, open news articles",
        },
        {
            "name": "OpenSubtitles (sample)",
            "type": "Dialogues",
            "size": "100MB",
            "desc": "Conversational, real-world",
        },
        {
            "name": "English Wikibooks",
            "type": "Textbooks",
            "size": "200MB",
            "desc": "How-tos, educational content",
        },
        {
            "name": "English Wikisource",
            "type": "Docs",
            "size": "300MB",
            "desc": "Public domain docs, speeches",
        },
        {
            "name": "British Literary Classics",
            "type": "Literature",
            "size": "50MB",
            "desc": "Austen, Dickens, Wilde: wit, manners",
        },
        {
            "name": "Etiquette & Manners Manuals",
            "type": "Nonfiction",
            "size": "5MB",
            "desc": "Politeness, assertiveness, healthy comm.",
        },
    ]
    table = Table(title="[bold green]Standard Curriculum Resources[/bold green]", box=box.ROUNDED)
    table.add_column("Name", style="option")
    table.add_column("Type", style="info")
    table.add_column("Size", style="success")
    table.add_column("Description", style="")
    for r in resources:
        table.add_row(r["name"], r["type"], r["size"], r["desc"])
    console.print(table)


def curriculum_command(args):
    console = Console()
    if args.list:
        table = Table(title="Available Curriculum Resources")
        table.add_column("Key", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("URL", style="magenta")
        for key, info in CURRICULUM_RESOURCES.items():
            table.add_row(key, info["name"], info["url"])
        console.print(table)
        return
    if args.download:
        key = args.download
        if key not in CURRICULUM_RESOURCES:
            console.print(f"[red]Resource '{key}' not found.[/red]")
            return
        dest_dir = args.dest or "."

        def progress_cb(downloaded, total):
            if total:
                percent = int(downloaded / total * 100)
                console.print(
                    f"[progress] Downloaded {downloaded}/{total} bytes ({percent}%)", end="\r"
                )
            else:
                console.print(f"[progress] Downloaded {downloaded} bytes", end="\r")

        try:
            with Progress() as progress:
                task = progress.add_task(f"Downloading {key}", total=100)

                def rich_cb(downloaded, total):
                    percent = int(downloaded / total * 100) if total else 0
                    progress.update(task, completed=percent)

                ingest_resource(key, dest_dir, progress_cb=rich_cb)
            console.print(f"[green]Successfully ingested {key}.[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="GyroSI Baby ML: Unified CLI (visual, mechanical, auditable)",
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # --- Knowledge ---
    knowledge_parser = subparsers.add_parser("knowledge", help="Knowledge management actions")
    k_sub = knowledge_parser.add_subparsers(dest="kcmd", metavar="<action>")
    k_sub.add_parser("list", help="List all knowledge packages").set_defaults(func=knowledge_list)
    k_export = k_sub.add_parser("export", help="Export a knowledge package")
    k_export.add_argument("--knowledge-id", required=True, help="Knowledge UUID to export")
    k_export.add_argument("--output", required=True, help="Output .gyro file path")
    k_export.set_defaults(func=knowledge_export)
    k_import = k_sub.add_parser("import", help="Import a knowledge package")
    k_import.add_argument("--input", required=True, help="Input .gyro file path")
    k_import.set_defaults(func=knowledge_import)
    k_fork = k_sub.add_parser("fork", help="Fork a knowledge package")
    k_fork.add_argument("--source", required=True, help="Source knowledge UUID to fork")
    k_fork.add_argument("--session", help="Session UUID to link to forked knowledge (optional)")
    k_fork.set_defaults(func=knowledge_fork)
    k_link = k_sub.add_parser("link", help="Link a session to knowledge")
    k_link.add_argument("--session", required=True, help="Session UUID")
    k_link.add_argument("--knowledge", required=True, help="Knowledge UUID")
    k_link.set_defaults(func=knowledge_link)
    k_info = k_sub.add_parser("info", help="Show info for a knowledge package")
    k_info.add_argument("--knowledge-id", required=True, help="Knowledge UUID to show info for")
    k_info.set_defaults(func=knowledge_info)
    k_delete = k_sub.add_parser("delete", help="Delete a knowledge package")
    k_delete.add_argument("--knowledge-id", required=True, help="Knowledge UUID to delete")
    k_delete.set_defaults(func=knowledge_delete)

    # --- Session ---
    session_parser = subparsers.add_parser("session", help="Session management actions")
    s_sub = session_parser.add_subparsers(dest="scmd", metavar="<action>")
    s_sub.add_parser("list", help="List all sessions").set_defaults(func=session_list)
    s_export = s_sub.add_parser("export", help="Export a session")
    s_export.add_argument("--session-id", required=True, help="Session UUID to export")
    s_export.add_argument("--output", required=True, help="Output .session.gyro file path")
    s_export.set_defaults(func=session_export)
    s_import = s_sub.add_parser("import", help="Import a session")
    s_import.add_argument("--input", required=True, help="Input .session.gyro file path")
    s_import.set_defaults(func=session_import)
    s_link = s_sub.add_parser("link", help="Link a session to knowledge")
    s_link.add_argument("--session", required=True, help="Session UUID")
    s_link.add_argument("--knowledge", required=True, help="Knowledge UUID")
    s_link.set_defaults(func=session_link)
    s_info = s_sub.add_parser("info", help="Show info for a session")
    s_info.add_argument("--session-id", required=True, help="Session UUID to show info for")
    s_info.set_defaults(func=session_info)
    s_delete = s_sub.add_parser("delete", help="Delete a session")
    s_delete.add_argument("--session-id", required=True, help="Session UUID to delete")
    s_delete.set_defaults(func=session_delete)

    # --- Integrity ---
    integrity_parser = subparsers.add_parser(
        "integrity", help="Integrity checks for knowledge/session"
    )
    i_sub = integrity_parser.add_subparsers(dest="icmd", metavar="<action>")
    i_know = i_sub.add_parser("check-knowledge", help="Check integrity of a knowledge package")
    i_know.add_argument("--knowledge-id", required=True, help="Knowledge UUID to check")
    i_know.set_defaults(func=integrity_check_knowledge)
    i_sess = i_sub.add_parser("check-session", help="Check integrity of a session")
    i_sess.add_argument("--session-id", required=True, help="Session UUID to check")
    i_sess.set_defaults(func=integrity_check_session)

    # --- System ---
    system_parser = subparsers.add_parser(
        "system", help="System health, extensions, operator matrix status"
    )
    sys_sub = system_parser.add_subparsers(dest="syscmd", metavar="<action>")
    sys_health = sys_sub.add_parser("health", help="Show system health")
    sys_health.set_defaults(func=system_health)
    sys_ext = sys_sub.add_parser("extensions", help="List loaded extensions")
    sys_ext.set_defaults(func=system_extensions)
    sys_matrix = sys_sub.add_parser("matrix", help="Show operator matrix status")
    sys_matrix.set_defaults(func=system_matrix)

    # Curriculum command
    curriculum_parser = subparsers.add_parser(
        "curriculum", help="Manage and ingest curriculum resources"
    )
    curriculum_parser.add_argument("--list", action="store_true", help="List available resources")
    curriculum_parser.add_argument(
        "--download", type=str, help="Download and ingest a resource by key"
    )
    curriculum_parser.add_argument("--dest", type=str, help="Destination directory for downloads")
    curriculum_parser.set_defaults(func=curriculum_command)

    # Help subcommand
    help_parser = subparsers.add_parser("help", help="Show help and usage examples")
    help_parser.set_defaults(func=lambda args: parser.print_help())

    # Parse args
    if len(sys.argv) == 1:
        show_welcome()
        parser.print_help()
        return
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        show_welcome()
        parser.print_help()


if __name__ == "__main__":
    main()

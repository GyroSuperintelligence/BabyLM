#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from rich.table import Table
from rich.text import Text
from rich import box

# Import CLI modules
from gyro_tools.cli.interactive import run_interactive_mode
from gyro_tools.cli.utils.display import console, show_panel
from gyro_tools.cli.config import APP_NAME, APP_SUBTITLE
from gyro_tools.cli.commands import knowledge, session, integrity, system, curriculum


def show_welcome():
    """Show welcome screen with examples."""
    show_panel(Text(f"👶 {APP_NAME}", justify="center"), subtitle=APP_SUBTITLE, style="title")

    # Feature table
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

    # Examples
    console.print("\n" + "─" * 60 + "\n")
    console.print("[bold cyan]Examples:[/bold cyan]")
    console.print(
        "  [green]python gyro_tools/gyro_cli.py[/green]                      # Interactive mode"
    )
    console.print(
        "  [green]python gyro_tools/gyro_cli.py knowledge list[/green]       # List knowledge packages"
    )
    console.print(
        "  [green]python gyro_tools/gyro_cli.py session export --session-id <ID> --output mysession.session.gyro[/green]"
    )
    console.print(
        "  [green]python gyro_tools/gyro_cli.py system health[/green]        # Show system health"
    )
    console.print(
        "\n[bold]Tip:[/bold] Add [yellow]--help[/yellow] after any command for more options."
    )


def main():
    # If no CLI args provided, start interactive mode
    if len(sys.argv) == 1:
        run_interactive_mode()
        return

    parser = argparse.ArgumentParser(
        description=f"{APP_NAME}: Unified CLI (visual, mechanical, auditable)",
        add_help=False,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # --- Knowledge Command ---
    knowledge_parser = subparsers.add_parser("knowledge", help="Knowledge management actions")
    k_sub = knowledge_parser.add_subparsers(dest="kcmd", metavar="<action>")

    k_sub.add_parser("list", help="List all knowledge packages").set_defaults(
        func=lambda args: knowledge.list_packages()
    )

    k_export = k_sub.add_parser("export", help="Export a knowledge package")
    k_export.add_argument("--knowledge-id", required=True, help="Knowledge UUID to export")
    k_export.add_argument("--output", required=True, help="Output .gyro file path")
    k_export.set_defaults(
        func=lambda args: knowledge.export_package(args.knowledge_id, args.output)
    )

    k_import = k_sub.add_parser("import", help="Import a knowledge package")
    k_import.add_argument("--input", required=True, help="Input .gyro file path")
    k_import.set_defaults(func=lambda args: knowledge.import_package(args.input))

    k_fork = k_sub.add_parser("fork", help="Fork a knowledge package")
    k_fork.add_argument("--source", required=True, help="Source knowledge UUID to fork")
    k_fork.add_argument("--session", action="store_true", help="Create and link new session")
    k_fork.set_defaults(func=lambda args: knowledge.fork_package(args.source, args.session))

    k_info = k_sub.add_parser("info", help="Show info for a knowledge package")
    k_info.add_argument("--knowledge-id", required=True, help="Knowledge UUID")
    k_info.set_defaults(func=lambda args: knowledge.show_package_info(args.knowledge_id))

    k_delete = k_sub.add_parser("delete", help="Delete a knowledge package")
    k_delete.add_argument("--knowledge-id", required=True, help="Knowledge UUID to delete")
    k_delete.set_defaults(func=lambda args: knowledge.delete_package(args.knowledge_id))

    # --- Session Command ---
    session_parser = subparsers.add_parser("session", help="Session management actions")
    s_sub = session_parser.add_subparsers(dest="scmd", metavar="<action>")

    s_sub.add_parser("list", help="List all sessions").set_defaults(
        func=lambda args: session.list_sessions()
    )

    s_export = s_sub.add_parser("export", help="Export a session")
    s_export.add_argument("--session-id", required=True, help="Session UUID to export")
    s_export.add_argument("--output", required=True, help="Output .session.gyro file path")
    s_export.set_defaults(func=lambda args: session.export_session(args.session_id, args.output))

    s_import = s_sub.add_parser("import", help="Import a session")
    s_import.add_argument("--input", required=True, help="Input .session.gyro file path")
    s_import.set_defaults(func=lambda args: session.import_session(args.input))

    s_link = s_sub.add_parser("link", help="Link a session to knowledge")
    s_link.add_argument("--session", required=True, help="Session UUID")
    s_link.add_argument("--knowledge", required=True, help="Knowledge UUID")
    s_link.set_defaults(func=lambda args: session.link_to_knowledge(args.session, args.knowledge))

    s_info = s_sub.add_parser("info", help="Show info for a session")
    s_info.add_argument("--session-id", required=True, help="Session UUID")
    s_info.set_defaults(func=lambda args: session.show_session_info(args.session_id))

    s_delete = s_sub.add_parser("delete", help="Delete a session")
    s_delete.add_argument("--session-id", required=True, help="Session UUID to delete")
    s_delete.set_defaults(func=lambda args: session.delete_session(args.session_id))

    # --- Integrity Command ---
    integrity_parser = subparsers.add_parser("integrity", help="Integrity checks")
    i_sub = integrity_parser.add_subparsers(dest="icmd", metavar="<action>")

    i_know = i_sub.add_parser("check-knowledge", help="Check knowledge package integrity")
    i_know.add_argument("--knowledge-id", required=True, help="Knowledge UUID to check")
    i_know.set_defaults(func=lambda args: integrity.check_knowledge(args.knowledge_id))

    i_sess = i_sub.add_parser("check-session", help="Check session integrity")
    i_sess.add_argument("--session-id", required=True, help="Session UUID to check")
    i_sess.set_defaults(func=lambda args: integrity.check_session(args.session_id))

    # --- System Command ---
    system_parser = subparsers.add_parser("system", help="System status and info")
    sys_sub = system_parser.add_subparsers(dest="syscmd", metavar="<action>")

    sys_sub.add_parser("health", help="Show system health").set_defaults(
        func=lambda args: system.show_health()
    )

    sys_sub.add_parser("extensions", help="List loaded extensions").set_defaults(
        func=lambda args: system.show_extensions()
    )

    sys_sub.add_parser("matrix", help="Check operator matrix status").set_defaults(
        func=lambda args: system.check_matrix()
    )

    # --- Curriculum Command ---
    curriculum_parser = subparsers.add_parser("curriculum", help="Manage curriculum resources")
    curriculum_parser.add_argument("--list", action="store_true", help="List available resources")
    curriculum_parser.add_argument("--download", type=str, help="Download resource by key")
    curriculum_parser.add_argument("--dest", type=str, default=".", help="Destination directory")
    curriculum_parser.set_defaults(
        func=lambda args: (
            curriculum.list_resources()
            if args.list
            else (
                curriculum.download_resource(args.download, args.dest)
                if args.download
                else curriculum_parser.print_help()
            )
        )
    )

    # --- Help Command ---
    help_parser = subparsers.add_parser("help", help="Show help and usage examples")
    help_parser.set_defaults(func=lambda args: parser.print_help())

    # Parse arguments
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        show_welcome()
        parser.print_help()


if __name__ == "__main__":
    main()

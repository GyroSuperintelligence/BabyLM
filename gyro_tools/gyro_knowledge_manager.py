import argparse
import sys
from core.extension_manager import ExtensionManager


def export_knowledge(args):
    try:
        mgr = ExtensionManager(knowledge_id=args.knowledge_id)
        mgr.export_knowledge(args.output)
        print(f"Knowledge {args.knowledge_id} exported to {args.output}")
    except Exception as e:
        print(f"Error exporting knowledge: {e}", file=sys.stderr)
        sys.exit(1)


def import_knowledge(args):
    try:
        mgr = ExtensionManager()
        new_knowledge_id = mgr.import_knowledge(args.input)
        print(f"Knowledge imported as {new_knowledge_id}")
        if args.new_session:
            print("(New session created and linked)")
    except Exception as e:
        print(f"Error importing knowledge: {e}", file=sys.stderr)
        sys.exit(1)


def fork_knowledge(args):
    try:
        mgr = ExtensionManager(knowledge_id=args.source)
        new_knowledge_id = mgr.fork_knowledge(new_session=bool(args.session))
        print(f"Knowledge {args.source} forked as {new_knowledge_id}")
        if args.session:
            print(f"Session {args.session} linked to forked knowledge.")
    except Exception as e:
        print(f"Error forking knowledge: {e}", file=sys.stderr)
        sys.exit(1)


def link_session(args):
    try:
        mgr = ExtensionManager(session_id=args.session)
        mgr.link_to_knowledge(args.knowledge)
        print(f"Session {args.session} linked to knowledge {args.knowledge}")
    except Exception as e:
        print(f"Error linking session: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GyroSI Knowledge Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export
    export_parser = subparsers.add_parser(
        "export-knowledge", help="Export a knowledge package to a .gyro bundle"
    )
    export_parser.add_argument("--knowledge-id", required=True, help="Knowledge UUID to export")
    export_parser.add_argument("--output", required=True, help="Output .gyro file path")
    export_parser.set_defaults(func=export_knowledge)

    # Import
    import_parser = subparsers.add_parser(
        "import-knowledge", help="Import a knowledge package from a .gyro bundle"
    )
    import_parser.add_argument("--input", required=True, help="Input .gyro file path")
    import_parser.add_argument(
        "--new-session",
        action="store_true",
        help="Create a new session linked to imported knowledge",
    )
    import_parser.set_defaults(func=import_knowledge)

    # Fork
    fork_parser = subparsers.add_parser(
        "fork-knowledge", help="Fork a knowledge package for new learning path"
    )
    fork_parser.add_argument("--source", required=True, help="Source knowledge UUID to fork")
    fork_parser.add_argument(
        "--session", help="Session UUID to link to forked knowledge (optional)"
    )
    fork_parser.set_defaults(func=fork_knowledge)

    # Link
    link_parser = subparsers.add_parser(
        "link-session", help="Link a session to a knowledge package"
    )
    link_parser.add_argument("--session", required=True, help="Session UUID")
    link_parser.add_argument("--knowledge", required=True, help="Knowledge UUID")
    link_parser.set_defaults(func=link_session)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

import argparse
import sys
from core.extension_manager import ExtensionManager


def export_session(args):
    try:
        mgr = ExtensionManager(session_id=args.session_id)
        mgr.export_session(args.output)
        print(f"Session {args.session_id} exported to {args.output}")
    except Exception as e:
        print(f"Error exporting session: {e}", file=sys.stderr)
        sys.exit(1)


def import_session(args):
    try:
        mgr = ExtensionManager()
        new_session_id = mgr.import_session(args.input)
        print(f"Session imported as {new_session_id}")
    except Exception as e:
        print(f"Error importing session: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GyroSI Session Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export
    export_parser = subparsers.add_parser(
        "export-session", help="Export a session to a .session.gyro bundle"
    )
    export_parser.add_argument("--session-id", required=True, help="Session UUID to export")
    export_parser.add_argument("--output", required=True, help="Output .session.gyro file path")
    export_parser.set_defaults(func=export_session)

    # Import
    import_parser = subparsers.add_parser(
        "import-session", help="Import a session from a .session.gyro bundle"
    )
    import_parser.add_argument("--input", required=True, help="Input .session.gyro file path")
    import_parser.set_defaults(func=import_session)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

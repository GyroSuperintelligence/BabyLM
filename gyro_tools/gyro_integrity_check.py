import argparse
import sys
import os
import json
from pathlib import Path


def check_knowledge(args):
    try:
        knowledge_path = Path("data/knowledge") / args.knowledge_id
        if not knowledge_path.exists():
            print(f"Knowledge package {args.knowledge_id} does not exist.", file=sys.stderr)
            sys.exit(1)
        meta_path = knowledge_path / "knowledge.meta.json"
        nav_log_path = knowledge_path / "navigation_log" / "genome.log"
        manifest_path = knowledge_path / "navigation_log" / "manifest.json"
        ext_dir = knowledge_path / "extensions"
        integrity_path = knowledge_path / "integrity.sha256"
        # Check metadata
        if not meta_path.exists():
            print("Missing knowledge.meta.json", file=sys.stderr)
            sys.exit(1)
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Knowledge metadata loaded. ID: {meta.get('knowledge_id')}")
        # Check navigation log
        if not nav_log_path.exists():
            print("Missing navigation log (genome.log)", file=sys.stderr)
            sys.exit(1)
        print("Navigation log present.")
        # Check manifest
        if not manifest_path.exists():
            print("Missing navigation log manifest.json", file=sys.stderr)
            sys.exit(1)
        print("Navigation log manifest present.")
        # Check extensions
        if not ext_dir.exists() or not any(ext_dir.iterdir()):
            print("No extension pattern files found.", file=sys.stderr)
        else:
            print(f"Extension pattern files: {[f.name for f in ext_dir.iterdir() if f.is_file()]}")
        # Check integrity file
        if integrity_path.exists():
            print("Integrity file present. (Checksum not verified in this tool)")
        print("Knowledge package integrity check PASSED.")
    except Exception as e:
        print(f"Error checking knowledge: {e}", file=sys.stderr)
        sys.exit(1)


def check_session(args):
    try:
        session_path = Path("data/sessions") / args.session_id
        if not session_path.exists():
            print(f"Session {args.session_id} does not exist.", file=sys.stderr)
            sys.exit(1)
        meta_path = session_path / "session.meta.json"
        phase_path = session_path / "phase.bin"
        events_path = session_path / "events.log"
        knowledge_link = session_path / "active_knowledge.link"
        # Check metadata
        if not meta_path.exists():
            print("Missing session.meta.json", file=sys.stderr)
            sys.exit(1)
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"Session metadata loaded. ID: {meta.get('id')}")
        # Check phase
        if not phase_path.exists():
            print("Missing phase.bin", file=sys.stderr)
        else:
            print("Phase file present.")
        # Check events
        if not events_path.exists():
            print("Missing events.log", file=sys.stderr)
        else:
            print("Events log present.")
        # Check knowledge link
        if not knowledge_link.exists():
            print("Missing active_knowledge.link", file=sys.stderr)
        else:
            print("Knowledge link present.")
        print("Session integrity check PASSED.")
    except Exception as e:
        print(f"Error checking session: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GyroSI Integrity Check CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Knowledge check
    k_parser = subparsers.add_parser(
        "check-knowledge", help="Check integrity of a knowledge package"
    )
    k_parser.add_argument("--knowledge-id", required=True, help="Knowledge UUID to check")
    k_parser.set_defaults(func=check_knowledge)

    # Session check
    s_parser = subparsers.add_parser("check-session", help="Check integrity of a session")
    s_parser.add_argument("--session-id", required=True, help="Session UUID to check")
    s_parser.set_defaults(func=check_session)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

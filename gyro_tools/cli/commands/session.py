import json
from pathlib import Path
from core.extension_manager import ExtensionManager
from ..utils.display import (
    console, show_error, show_success, show_info,
    create_table, show_progress
)
from ..utils.prompts import confirm_action

def list_sessions(args=None):
    """List all sessions."""
    sessions_dir = Path("data/sessions")
    if not sessions_dir.exists() or not any(sessions_dir.iterdir()):
        show_info("No sessions found.", "Session List")
        return
    
    rows = []
    for sess in sessions_dir.iterdir():
        meta_path = sess / "session.meta.json"
        if not meta_path.exists():
            continue
        
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            
            session_id = meta.get("id", sess.name)
            created = meta.get("created") or meta.get("created_at") or "?"
            
            # Get linked knowledge
            link_path = sess / "active_knowledge.link"
            knowledge_id = "?"
            if link_path.exists():
                try:
                    with open(link_path) as lf:
                        knowledge_id = lf.read().strip()
                except Exception:
                    pass
            
            rows.append([session_id, knowledge_id, created])
        except Exception as e:
            rows.append([sess.name, f"[error]Error: {e}[/error]", "?"])
    
    table = create_table(
        "Sessions",
        [("💬 ID", "option"), ("🧠 Knowledge", "info"), ("📅 Created", "success")],
        rows
    )
    console.print(table)

def export_session(session_id, output):
    """Export a session."""
    if not session_id or not output:
        show_error("You must specify session ID and output path.", "Export")
        return False
    
    try:
        mgr = ExtensionManager(session_id=session_id)
        show_progress(
            f"Exporting session '{session_id}'",
            mgr.export_session,
            output
        )
        show_success(f"Session '{session_id}' exported to {output}!", "Export")
        return True
    except Exception as e:
        show_error(f"Error exporting session: {e}", "Export")
        return False

def import_session(input_path):
    """Import a session."""
    if not input_path:
        show_error("You must specify input path.", "Import")
        return None
    
    try:
        mgr = ExtensionManager()
        new_session_id = show_progress(
            f"Importing session from '{input_path}'",
            mgr.import_session,
            input_path
        )
        show_success(f"Session imported as {new_session_id}!", "Import")
        return new_session_id
    except Exception as e:
        show_error(f"Error importing session: {e}", "Import")
        return None

def link_to_knowledge(session_id, knowledge_id):
    """Link a session to a knowledge package."""
    if not session_id or not knowledge_id:
        show_error("You must specify both session and knowledge IDs.", "Link")
        return False
    
    try:
        mgr = ExtensionManager(session_id=session_id)
        mgr.link_to_knowledge(knowledge_id)
        show_success(
            f"Session '{session_id}' linked to knowledge '{knowledge_id}'!",
            "Link"
        )
        return True
    except Exception as e:
        show_error(f"Error linking session: {e}", "Link")
        return False

def delete_session(session_id):
    """Delete a session."""
    if not session_id:
        show_error("You must specify session ID to delete.", "Delete")
        return False
    
    sess_dir = Path("data/sessions") / session_id
    if not sess_dir.exists():
        show_error(f"Session '{session_id}' not found.", "Delete")
        return False
    
    if not confirm_action(
        f"Are you sure you want to permanently delete session '{session_id}'? "
        "This cannot be undone."
    ):
        show_info("Delete cancelled.", "Delete")
        return False
    
    try:
        import shutil
        shutil.rmtree(sess_dir)
        show_success(f"Session '{session_id}' deleted!", "Delete")
        return True
    except Exception as e:
        show_error(f"Error deleting session: {e}", "Delete")
        return False

def show_session_info(session_id):
    """Show detailed info for a session."""
    sess_dir = Path("data/sessions") / session_id
    meta_path = sess_dir / "session.meta.json"
    
    if not meta_path.exists():
        show_error(f"Session '{session_id}' not found.", "Session Info")
        return
    
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        
        rows = [[str(k), str(v)] for k, v in meta.items()]
        
        # Add linked knowledge
        link_path = sess_dir / "active_knowledge.link"
        knowledge_id = "None"
        if link_path.exists():
            try:
                with open(link_path) as lf:
                    knowledge_id = lf.read().strip()
            except Exception:
                pass
        rows.append(["Linked Knowledge", knowledge_id])
        
        # Add phase if available
        phase_path = sess_dir / "phase.bin"
        if phase_path.exists():
            try:
                with open(phase_path, "rb") as pf:
                    phase_bytes = pf.read()
                    if phase_bytes:
                        phase = int.from_bytes(phase_bytes, "little")
                        rows.append(["Phase", str(phase)])
            except Exception:
                pass
        
        table = create_table(
            f"Session Info: {session_id}",
            [("Field", "option"), ("Value", "info")],
            rows
        )
        console.print(table)
    except Exception as e:
        show_error(f"Error reading session info: {e}", "Session Info")
import json
from pathlib import Path
from core.extension_manager import ExtensionManager
from ..utils.display import (
    console, show_error, show_success, show_info,
    create_table, show_progress
)
from ..utils.prompts import confirm_action

def list_packages(args=None):
    """List all knowledge packages."""
    knowledge_dir = Path("data/knowledge")
    if not knowledge_dir.exists() or not any(knowledge_dir.iterdir()):
        show_info("No knowledge packages found.", "Knowledge List")
        return
    
    rows = []
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
            session_count = _count_linked_sessions(knowledge_id)
            rows.append([knowledge_id, meta_str, str(session_count)])
        except Exception as e:
            rows.append([pkg.name, f"[error]Error: {e}[/error]", "?"])
    
    table = create_table(
        "Knowledge Packages",
        [("🆔 ID", "option"), ("📄 Meta", "info"), ("🔗 Linked Sessions", "success")],
        rows
    )
    console.print(table)

def export_package(knowledge_id, output):
    """Export a knowledge package."""
    if not knowledge_id or not output:
        show_error("You must specify knowledge ID and output path.", "Export")
        return False
    
    try:
        mgr = ExtensionManager(knowledge_id=knowledge_id)
        show_progress(
            f"Exporting knowledge '{knowledge_id}'",
            mgr.export_knowledge,
            output
        )
        show_success(f"Knowledge '{knowledge_id}' exported to {output}!", "Export")
        return True
    except Exception as e:
        show_error(f"Error exporting knowledge: {e}", "Export")
        return False

def import_package(input_path):
    """Import a knowledge package."""
    if not input_path:
        show_error("You must specify input path.", "Import")
        return None
    
    try:
        mgr = ExtensionManager()
        new_knowledge_id = show_progress(
            f"Importing knowledge from '{input_path}'",
            mgr.import_knowledge,
            input_path
        )
        show_success(f"Knowledge imported as {new_knowledge_id}!", "Import")
        return new_knowledge_id
    except Exception as e:
        show_error(f"Error importing knowledge: {e}", "Import")
        return None

def fork_package(source, link_session=False):
    """Fork a knowledge package."""
    if not source:
        show_error("You must specify source knowledge UUID to fork.", "Fork")
        return None
    
    try:
        mgr = ExtensionManager(knowledge_id=source)
        new_knowledge_id = mgr.fork_knowledge(new_session=link_session)
        
        msg = f"Knowledge '{source}' forked as {new_knowledge_id}!"
        if link_session:
            msg += "\nNew session created and linked to forked knowledge."
        
        show_success(msg, "Fork")
        return new_knowledge_id
    except Exception as e:
        show_error(f"Error forking knowledge: {e}", "Fork")
        return None

def delete_package(knowledge_id):
    """Delete a knowledge package."""
    if not knowledge_id:
        show_error("You must specify knowledge ID to delete.", "Delete")
        return False
    
    pkg_dir = Path("data/knowledge") / knowledge_id
    if not pkg_dir.exists():
        show_error(f"Knowledge package '{knowledge_id}' not found.", "Delete")
        return False
    
    if not confirm_action(
        f"Are you sure you want to permanently delete knowledge '{knowledge_id}'? "
        "This cannot be undone."
    ):
        show_info("Delete cancelled.", "Delete")
        return False
    
    try:
        import shutil
        shutil.rmtree(pkg_dir)
        show_success(f"Knowledge '{knowledge_id}' deleted!", "Delete")
        return True
    except Exception as e:
        show_error(f"Error deleting knowledge: {e}", "Delete")
        return False

def show_package_info(knowledge_id):
    """Show detailed info for a knowledge package."""
    pkg_dir = Path("data/knowledge") / knowledge_id
    meta_path = pkg_dir / "knowledge.meta.json"
    
    if not meta_path.exists():
        show_error(f"Knowledge package '{knowledge_id}' not found.", "Knowledge Info")
        return
    
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        
        rows = [[str(k), str(v)] for k, v in meta.items()]
        rows.append(["Linked Sessions", str(_count_linked_sessions(knowledge_id))])
        
        table = create_table(
            f"Knowledge Info: {knowledge_id}",
            [("Field", "option"), ("Value", "info")],
            rows
        )
        console.print(table)
    except Exception as e:
        show_error(f"Error reading knowledge info: {e}", "Knowledge Info")

def _count_linked_sessions(knowledge_id):
    """Count sessions linked to a knowledge package."""
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
    
    return session_count
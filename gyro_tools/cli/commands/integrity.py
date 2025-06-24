from pathlib import Path
from ..utils.display import show_panel, console

def check_knowledge(knowledge_id):
    """Check integrity of a knowledge package."""
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
        details.append("✓ Metadata present")
    
    if not nav_log_path.exists():
        errors.append("Missing navigation log (genome.log)")
    else:
        details.append("✓ Navigation log present")
    
    if not manifest_path.exists():
        errors.append("Missing navigation log manifest.json")
    else:
        details.append("✓ Navigation log manifest present")
    
    if not ext_dir.exists() or not any(ext_dir.iterdir()):
        details.append("⚠️ No extension pattern files found")
    else:
        ext_files = [f.name for f in ext_dir.iterdir() if f.is_file()]
        details.append(f"✓ Extension files: {', '.join(ext_files)}")
    
    if integrity_path.exists():
        details.append("✓ Integrity file present (checksum not verified)")
    
    if errors:
        content = "\n".join([f"❌ {e}" for e in errors] + details)
        show_panel(content, title="🛡️ Knowledge Check: FAILED", style="error")
    else:
        content = "\n".join(details + ["\n✅ Knowledge package integrity check PASSED"])
        show_panel(content, title="🛡️ Knowledge Check: PASSED", style="success")

def check_session(session_id):
    """Check integrity of a session."""
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
        details.append("✓ Session metadata present")
    
    if not phase_path.exists():
        details.append("⚠️ Missing phase.bin")
    else:
        details.append("✓ Phase file present")
    
    if not events_path.exists():
        details.append("⚠️ Missing events.log")
    else:
        details.append("✓ Events log present")
    
    if not knowledge_link.exists():
        details.append("⚠️ Missing active_knowledge.link")
    else:
        details.append("✓ Knowledge link present")
    
    if errors:
        content = "\n".join([f"❌ {e}" for e in errors] + details)
        show_panel(content, title="🛡️ Session Check: FAILED", style="error")
    else:
        content = "\n".join(details + ["\n✅ Session integrity check PASSED"])
        show_panel(content, title="🛡️ Session Check: PASSED", style="success")
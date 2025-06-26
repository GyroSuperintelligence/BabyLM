from core.g5 import GyroOperations
from core.g1 import GyroEngine
from ..utils.display import console, show_error, create_table, show_panel

def show_health():
    """Show system health metrics."""
    try:
        mgr = GyroOperations()
        health = mgr.get_system_health()
        
        rows = [[str(k), str(v)] for k, v in health.items()]
        
        table = create_table(
            "System Health",
            [("Metric", "option"), ("Value", "info")],
            rows
        )
        console.print(table)
    except Exception as e:
        show_error(f"Error retrieving system health: {e}", "System Health")

def show_extensions():
    """Show loaded extensions."""
    try:
        mgr = GyroOperations()
        exts = mgr.extensions
        
        rows = []
        for ext in exts.values():
            try:
                name = (
                    ext.get_extension_name()
                    if hasattr(ext, "get_extension_name")
                                        else str(type(ext).__name__)
                )
                version = (
                    ext.get_extension_version() 
                    if hasattr(ext, "get_extension_version") 
                    else "?"
                )
                footprint = (
                    f"{ext.get_footprint_bytes()} bytes"
                    if hasattr(ext, "get_footprint_bytes")
                    else "?"
                )
                rows.append([name, version, footprint])
            except Exception as e:
                rows.append(["[error]Error[/error]", f"[error]{e}[/error]", "?"])
        
        table = create_table(
            "Loaded Extensions",
            [("Name", "option"), ("Version", "info"), ("Footprint", "success")],
            rows
        )
        console.print(table)
    except Exception as e:
        show_error(f"Error listing extensions: {e}", "Extensions")

def check_matrix():
    """Check operator matrix status."""
    try:
        engine = GyroEngine()
        # If no exception, matrix is valid
        show_panel(
            "✅ Operator matrix: VALID",
            title="🔢 Operator Matrix",
            style="success"
        )
    except Exception as e:
        show_panel(
            f"❌ Operator matrix: INVALID\n{e}",
            title="🔢 Operator Matrix",
            style="error"
        )
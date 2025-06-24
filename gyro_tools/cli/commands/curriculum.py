from gyro_tools.gyro_curriculum_manager import CURRICULUM_RESOURCES, ingest_resource
from ..utils.display import console, show_error, show_success, create_table
from rich.progress import Progress

def list_resources():
    """List available curriculum resources."""
    rows = []
    for key, info in CURRICULUM_RESOURCES.items():
        rows.append([key, info["name"], info["url"]])
    
    table = create_table(
        "Available Curriculum Resources",
        [("Key", "cyan"), ("Name", "green"), ("URL", "magenta")],
        rows
    )
    console.print(table)

def download_resource(resource_key, dest_dir="."):
    """Download and ingest a curriculum resource."""
    if resource_key not in CURRICULUM_RESOURCES:
        show_error(f"Resource '{resource_key}' not found.", "Curriculum Download")
        return False
    
    try:
        with Progress() as progress:
            task = progress.add_task(f"Downloading {resource_key}", total=100)
            
            def progress_callback(downloaded, total):
                percent = int(downloaded / total * 100) if total else 0
                progress.update(task, completed=percent)
            
            ingest_resource(resource_key, dest_dir, progress_cb=progress_callback)
        
        show_success(f"Successfully ingested {resource_key}.", "Curriculum Download")
        return True
    except Exception as e:
        show_error(f"Error downloading resource: {e}", "Curriculum Download")
        return False
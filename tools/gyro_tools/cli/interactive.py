import questionary
from .config import APP_NAME, APP_SUBTITLE, BIOS_STYLE
from .utils.display import console, show_panel
from .utils.prompts import select_from_list, get_text_input, get_path_input, get_confirmation
from .commands import knowledge, session, integrity, system, curriculum
from rich.text import Text
from rich import box

def run_interactive_mode():
    """Run the CLI in interactive mode."""
    show_panel(
        Text(f"👶 {APP_NAME} 👶", justify="center"),
        subtitle=APP_SUBTITLE,
        style="title"
    )
    
    while True:
        action = select_from_list(
            "Main Menu",
            [
                "🧠 Knowledge Management",
                "💬 Session Management", 
                "🛡️ Integrity Checks",
                "⚙️ System Status",
                "📚 Curriculum",
                questionary.Separator(),
                "Exit"
            ],
            allow_back=False
        )
        
        if action is None or action == "Exit":
            break
        elif action == "🧠 Knowledge Management":
            knowledge_menu()
        elif action == "💬 Session Management":
            session_menu()
        elif action == "🛡️ Integrity Checks":
            integrity_menu()
        elif action == "⚙️ System Status":
            system_menu()
        elif action == "📚 Curriculum":
            curriculum_menu()
    
    console.print("\n[info]Exiting interactive mode. Goodbye![/info]\n")

def knowledge_menu():
    """Interactive menu for knowledge management."""
    while True:
        choice = select_from_list(
            "Knowledge Management",
            [
                "📋 List Packages",
                "ℹ️ Package Info",
                "📤 Export Package",
                "📥 Import Package",
                "🧬 Fork Package",
                "🗑️ Delete Package"
            ]
        )
        
        if choice is None:
            break
        
        if choice == "📋 List Packages":
            knowledge.list_packages()
            input("\nPress Enter to continue...")
        
        elif choice == "ℹ️ Package Info":
            knowledge_id = get_text_input("Enter Knowledge ID:")
            if knowledge_id:
                knowledge.show_package_info(knowledge_id)
                input("\nPress Enter to continue...")
        
        elif choice == "📤 Export Package":
            knowledge_id = get_text_input("Enter Knowledge ID to export:")
            if knowledge_id:
                output = get_path_input("Enter output file path (.gyro):")
                if output:
                    knowledge.export_package(knowledge_id, output)
                    input("\nPress Enter to continue...")
        
        elif choice == "📥 Import Package":
            input_path = get_path_input("Enter package file path to import:")
            if input_path:
                knowledge.import_package(input_path)
                input("\nPress Enter to continue...")
        
        elif choice == "🧬 Fork Package":
            source = get_text_input("Enter source Knowledge ID to fork:")
            if source:
                link_session = get_confirmation("Create and link a new session?")
                knowledge.fork_package(source, link_session)
                input("\nPress Enter to continue...")
        
        elif choice == "🗑️ Delete Package":
            knowledge_id = get_text_input("Enter Knowledge ID to DELETE:")
            if knowledge_id:
                knowledge.delete_package(knowledge_id)
                input("\nPress Enter to continue...")

def session_menu():
    """Interactive menu for session management."""
    while True:
        choice = select_from_list(
            "Session Management",
            [
                "📋 List Sessions",
                "ℹ️ Session Info",
                "📤 Export Session",
                "📥 Import Session",
                "🔗 Link to Knowledge",
                "🗑️ Delete Session"
            ]
        )
        
        if choice is None:
            break
        
        if choice == "📋 List Sessions":
            session.list_sessions()
            input("\nPress Enter to continue...")
        
        elif choice == "ℹ️ Session Info":
            session_id = get_text_input("Enter Session ID:")
            if session_id:
                session.show_session_info(session_id)
                input("\nPress Enter to continue...")
        
        elif choice == "📤 Export Session":
            session_id = get_text_input("Enter Session ID to export:")
            if session_id:
                output = get_path_input("Enter output file path (.session.gyro):")
                if output:
                    session.export_session(session_id, output)
                    input("\nPress Enter to continue...")
        
        elif choice == "📥 Import Session":
            input_path = get_path_input("Enter session file path to import:")
            if input_path:
                session.import_session(input_path)
                input("\nPress Enter to continue...")
        
        elif choice == "🔗 Link to Knowledge":
            session_id = get_text_input("Enter Session ID:")
            if session_id:
                knowledge_id = get_text_input("Enter Knowledge ID to link:")
                if knowledge_id:
                    session.link_to_knowledge(session_id, knowledge_id)
                    input("\nPress Enter to continue...")
        
        elif choice == "🗑️ Delete Session":
            session_id = get_text_input("Enter Session ID to DELETE:")
            if session_id:
                session.delete_session(session_id)
                input("\nPress Enter to continue...")

def integrity_menu():
    """Interactive menu for integrity checks."""
    while True:
        choice = select_from_list(
            "Integrity Checks",
            [
                "🧠 Check Knowledge Package",
                "💬 Check Session"
            ]
        )
        
        if choice is None:
            break
        
        if choice == "🧠 Check Knowledge Package":
            knowledge_id = get_text_input("Enter Knowledge ID to check:")
            if knowledge_id:
                integrity.check_knowledge(knowledge_id)
                input("\nPress Enter to continue...")
        
        elif choice == "💬 Check Session":
            session_id = get_text_input("Enter Session ID to check:")
            if session_id:
                integrity.check_session(session_id)
                input("\nPress Enter to continue...")

def system_menu():
    """Interactive menu for system status."""
    while True:
        choice = select_from_list(
            "System Status",
            [
                "💚 System Health",
                "🔌 Loaded Extensions",
                "🔢 Operator Matrix Status"
            ]
        )
        
        if choice is None:
            break
        
        if choice == "💚 System Health":
            system.show_health()
            input("\nPress Enter to continue...")
        
        elif choice == "🔌 Loaded Extensions":
            system.show_extensions()
            input("\nPress Enter to continue...")
        
        elif choice == "🔢 Operator Matrix Status":
            system.check_matrix()
            input("\nPress Enter to continue...")

def curriculum_menu():
    """Interactive menu for curriculum management."""
    while True:
        choice = select_from_list(
            "Curriculum Management",
            [
                "📚 List Available Resources",
                "⬇️ Download Resource"
            ]
        )
        
        if choice is None:
            break
        
        if choice == "📚 List Available Resources":
            curriculum.list_resources()
            input("\nPress Enter to continue...")
        
        elif choice == "⬇️ Download Resource":
            # First show available resources
            curriculum.list_resources()
            resource_key = get_text_input("\nEnter resource key to download:")
            if resource_key:
                dest_dir = get_path_input("Enter destination directory (default: current):", default=".")
                curriculum.download_resource(resource_key, dest_dir)
                input("\nPress Enter to continue...")
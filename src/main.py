import flet as ft
from frontend.gyro_app import main_app
import yaml

def main():
    # Load config
    # with open("config/gyro_config.yaml") as f:
    #     config = yaml.safe_load(f)
    
    # Start the Flet app
    ft.app(target=main_app)

if __name__ == "__main__":
    main()

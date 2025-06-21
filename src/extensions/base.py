class GyroExtension:
    def get_learning_state(self) -> dict:
        """State that should be exported with knowledge"""
        pass

    def get_session_state(self) -> dict:
        """State that stays with session"""
        pass

    def set_learning_state(self, state: dict):
        """Restore learning state from knowledge package"""
        pass

    def set_session_state(self, state: dict):
        """Restore session state"""
        pass 


class ExtensionInterface:

    def inference(self, prompt: str, *args) -> str:
        """Modify prompt during inference"""
        pass

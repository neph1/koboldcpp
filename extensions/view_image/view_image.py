from extension import ExtensionInterface
from .image_handler import get_prompt_for_image_from_path
import os
import yaml

class view_image(ExtensionInterface):
    
    def __init__(self):
        with open(os.path.realpath(os.path.join(os.path.dirname(__file__), "config.yaml")), "r") as stream:
            try:
                config_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.config = config_file
        self.image_path = os.path.realpath(os.path.join(os.path.dirname(__file__), self.config['IMAGES_PATH']))

    def inference(self, newprompt, genparams, max_context, *args):
        query_string = newprompt
        stop_sequence = genparams.get('stop_sequence', [])

        if stop_sequence:
            query_string = query_string.rsplit(stop_sequence[0], 1)[-1] or query_string
        
        is_image = query_string.find('[image:')

        if is_image == -1:
            return newprompt

        image = query_string.split('[image:')[1].split(']')[0].strip()

        is_valid = image.endswith('.jpg') or image.endswith('.png')

        result = get_prompt_for_image_from_path(image_path=os.path.join(self.image_path, image), mode=self.config['CLIP_MODE'], model_name=self.config['DEFAULT_MODEL'])
        
        result_length = len(result)
        max_new_prompt = max_context-result_length

        newprompt = (newprompt[-(max_new_prompt):]) if len(newprompt) > max_new_prompt else newprompt
        return f'Image contains:[{result}] {newprompt}'

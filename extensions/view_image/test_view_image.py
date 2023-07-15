import image_handler
from image_handler import get_prompt_for_image_from_path

def test_get_prompt_for_image_from_path():
    prompt = get_prompt_for_image_from_path(image_path='images/test.jpg', mode='fast')
    print(f'result: {prompt}')


if __name__ == "__main__":
    test_get_prompt_for_image_from_path()
    

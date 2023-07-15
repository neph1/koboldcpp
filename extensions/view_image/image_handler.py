import csv
import open_clip
import os
import torch
import base64
import cv2
import yaml

from PIL import Image

import clip_interrogator
from clip_interrogator import Config, Interrogator

from pydantic import BaseModel, Field
from io import BytesIO

__version__ = "0.1.6"

ci = None
    
def load(clip_model_name):
    global ci
    if ci is None:
        print(f"Loading CLIP Interrogator {clip_interrogator.__version__}...")
        with open(os.path.realpath(os.path.join(os.path.dirname(__file__), "config.yaml")), "r") as stream:
            try:
                config_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        config = Config(
            cache_path = config_file["CACHE_PATH"],
            clip_model_name=clip_model_name if clip_model_name is not None else config_file['MODEL_NAME'],
        )
        ci = Interrogator(config)

    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        ci.load_clip_model()

def unload():
    global ci
    if ci is not None:
        print("Offloading CLIP Interrogator...")
        ci.blip_model = ci.blip_model.to(devices.cpu)
        ci.clip_model = ci.clip_model.to(devices.cpu)
        ci.blip_offloaded = True
        ci.clip_offloaded = True

def image_analysis(image, clip_model_name):
    load(clip_model_name)

    image = image.convert('RGB')
    image_features = ci.image_to_features(image)

    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)

    medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
    artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
    movement_ranks = {movement: sim for movement, sim in zip(top_movements, ci.similarities(image_features, top_movements))}
    trending_ranks = {trending: sim for trending, sim in zip(top_trendings, ci.similarities(image_features, top_trendings))}
    flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}

    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks

def interrogate(image, mode, caption=None):
    if mode == 'best':
        prompt = ci.interrogate(image, caption=caption)
    elif mode == 'caption':
        prompt = ci.generate_caption(image) if caption is None else caption
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image, caption=caption)
    elif mode == 'fast':
        prompt = ci.interrogate_fast(image, caption=caption)
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image)
    else:
        raise Exception(f"Unknown mode {mode}")
    return prompt

def image_to_prompt(image, mode, clip_model_name):
    try:
        load(clip_model_name)
        image = image.convert('RGB')
        prompt = interrogate(image, mode)
    except torch.cuda.OutOfMemoryError as e:
        prompt = "Ran out of VRAM"
        print(e)
    except RuntimeError as e:
        prompt = f"Exception {type(e)}"
        print(e)

    return prompt


# decode_base64_to_image from modules/api/api.py, could be imported from there
def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e

def encode_image_to_base64(image_path):
    img = cv2.imread(image_path)
    if image_path.endswith('.jpg'):
        encoded_img = cv2.imencode('.jpg', img)
    elif image_path.endswith('.png'):
        encoded_img = cv2.imencode('.png', img)
    return base64.b64encode(encoded_img[1]).decode('utf-8')

class InterrogatorAnalyzeRequest(BaseModel):
    image_path: str = Field(
        default="",
        title="Image path",
        description="Image to load. Must be .jpg or .png",
    )
    clip_model_name: str = Field(
        default="ViT-L-14/openai",
        title="Model",
        description="The interrogate model used. See the models endpoint for a list of available models.",
    )

def get_prompt_for_image_from_path(image_path : str, mode = 'fast', model_name = "ViT-L-14/openai"):
    img = Image.open(image_path)
    return image_to_prompt(img, mode, model_name)

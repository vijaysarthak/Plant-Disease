"""
image_utils.py — turns a photo into the numbers the model expects.

This exact function is used in TWO places:
  * in the notebook, for validation / test / field photos,
  * in the app, for the photo the user uploads.
Because both use the same code, the app sees photos exactly the way the model saw them in testing.

Only needs numpy and Pillow (no PyTorch), so the app stays small.
"""
import numpy as np
from PIL import Image

from config import IMAGE_SIZE, RESIZE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def resize_shorter_side(image, size):
    """Resize so the SHORTER side becomes `size` pixels, keeping the shape (no stretching)."""
    width, height = image.size
    scale = size / min(width, height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    return image.resize((new_width, new_height), Image.BICUBIC)


def center_crop(image, size):
    """Cut a size x size square out of the middle of the image."""
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def prepare_image(image):
    """
    PIL image (any size)  ->  numpy array of shape (3, 224, 224), type float32.

    Steps:
      1. make sure it has 3 colour channels (RGB)
      2. resize the shorter side to 256, then cut out the 224 x 224 centre
      3. pixel values 0..255  ->  0..1
      4. normalise each colour: (value - mean) / std   (the pretrained model was trained this way)
      5. change the order from (height, width, colour) to (colour, height, width) — PyTorch/ONNX order
    """
    image = image.convert("RGB")
    image = resize_shorter_side(image, RESIZE_SIZE)
    image = center_crop(image, IMAGE_SIZE)

    pixels = np.asarray(image, dtype=np.float32) / 255.0      # shape (224, 224, 3)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    pixels = (pixels - mean) / std

    pixels = pixels.transpose(2, 0, 1)                         # shape (3, 224, 224)
    return np.ascontiguousarray(pixels, dtype=np.float32)


def load_image(path):
    """Open an image file and return it as an RGB PIL image."""
    with Image.open(path) as image:
        return image.convert("RGB")

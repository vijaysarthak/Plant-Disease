"""
leaf_dataset.py — gives PyTorch the photos one by one during training.

  * `train_transform`  : random changes for TRAINING photos (crop, flip, rotate, colour),
                         so the model learns the disease and not the exact photo.
  * `eval_transform`   : NO random changes; uses prepare_image() from image_utils.py,
                         the same function the app uses.
  * `LeafDataset`      : turns a list of image rows into (image tensor, label) pairs.

Used by: the notebook only (needs PyTorch).
"""
import torch
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.image_utils import prepare_image

# Random changes for training photos. Every time a photo is loaded it looks a little different.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),   # random part of the photo, resized to 224
    transforms.RandomHorizontalFlip(),                             # mirror left-right (50% of the time)
    transforms.RandomVerticalFlip(),                               # mirror up-down (50% of the time)
    transforms.RandomRotation(20),                                 # rotate up to 20 degrees
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),  # change light / colour
    transforms.ToTensor(),                                         # pixels -> numbers 0..1
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),             # same normalisation as prepare_image
])


def eval_transform(image):
    """Validation / test / field photos: exactly what the app does (see src/image_utils.py)."""
    return torch.from_numpy(prepare_image(image))


class LeafDataset(torch.utils.data.Dataset):
    """A list of rows -> PyTorch dataset. dataset[i] gives (image tensor, label number)."""

    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, int(row["label"])

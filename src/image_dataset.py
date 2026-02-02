import torch
import torchvision.transforms as tf
from torch.utils.data import Dataset
from torchvision.io import read_image

from pathlib import Path


class ImageDataset(Dataset):
    """Dataset for loading ImageNet images"""
    def __init__(self, path_images: str | Path, size: int, device: str, start_index: int = 0) -> None:
        path_images = Path(path_images)

        self.img_list = tuple(path_images.iterdir())[start_index:size + start_index]
        self.trafos = tf.Compose([
            tf.Resize((224, 224)),
        ])

        self.size = len(self.img_list)
        self.device = device


    def __len__(self):
        return self.size


    def __getitem__(self, i) -> torch.Tensor:
        img = read_image(self.img_list[i]) # load image
        img = img / 255 # convert from [0, 255] to [0, 1]

        # cut image down to largest possible square
        c, h, w = img.shape
        new_hw = min(h, w)
        h_start = (h - new_hw) // 2
        w_start = (w - new_hw) // 2
        img = img[:, h_start:h_start+new_hw, w_start:w_start+new_hw]

        if c == 1: # grayscale image -> add color channels
            img = img.repeat(3, 1, 1)

        img = self.trafos(img) # apply transformations

        img = img.to(self.device) # move to correct device

        return img

import torch
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import (
    # added for image augmentation
    ToPILImage,
    RandomCrop,
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    ToTensor,
    # /added for image augmentation
    CenterCrop, 
    ConvertImageDtype, 
    Normalize, 
    Resize
)
from torchvision.transforms.functional import InterpolationMode
import io
import jsonlines
from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import requests
from PIL import Image

# adopted form https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/hybrid_clip/run_hybrid_clip.py
class Transform(torch.nn.Module):
    def __init__(self, image_size, augment_images, augmentation_args):
        super().__init__()
        if augment_images:
            crop_size = int(image_size * 0.8)
            self.transforms = torch.nn.Sequential(
                # image augmentation transforms
                RandomCrop(crop_size),
                ColorJitter(),
                RandomHorizontalFlip(augmentation_args.random_horizontal_flip),
                RandomVerticalFlip(augmentation_args.random_vertical_flip),
                RandomResizedCrop(crop_size, scale=(0.8, 1.2), ratio=(1.0, 1.0)),
                # /image augmentation transforms
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            )
        else:
            self.transforms = torch.nn.Sequential(
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x

class ImageTextDataset(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored.
            The expected format is jsonlines where each line is a json object containing to keys.
            `filename`: The path to the image.
            `captions`: An `array` of captions.
        split: (string): Dataset split name. Is used for parsing jsonl files from `root` folder.
        captions_per_image: (int): number of captions per image to use. Defaults to 5.
        augment_captions: (bool): If true the jsonl files with `textaug_` prefix are selected from root
            folder. 
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        csv_file: str,
        split: str, 
        captions_per_image:int = 5,
        augment_captions:bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(csv_file, transforms, transform, target_transform)
        # self.root = root
        if augment_captions:
            prefix = "textaug_"
        else:
            prefix = ""
        self.df = pd.read_csv(csv_file)
        fps_empty_msg = f"""\
        The `filepaths` is empty. Please make sure that `root` folder contains jsonl files
        named properly: [textaug_]{split}*.jsonl.
        `textaug_` prefix is expected if `augment_captions` is `True`.
        """
        
        self.captions = []
        self.image_paths = []
        for i, row in self.df.iterrows():
            self.image_paths.append(row["url"])
            self.captions.append(row["text_caption"])
    
    def _load_image(self, idx: int):
        response = requests.get(self.image_paths[idx])
        imageStream = io.BytesIO(response.content)
        imageFile = Image.open(imageStream)
        return torchvision.transforms.functional.pil_to_tensor(imageFile)

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)

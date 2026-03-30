import random
from tqdm import tqdm

from PIL import Image
import albumentations as A
import numpy as np

from torch.utils.data import Dataset

from utils import pil_to_tensor


class DIV2KDataset(Dataset):
    def __init__(self,
                 image_paths: list[str],
                 scale: int,
                 patch_size: int,
                 load_immediatly: bool = False,
                 is_test: bool = False,
                 augment: bool = True):

        if not load_immediatly:
            self.images = image_paths
        else:
            self.images = [
                Image.open(path).convert("RGB") for path in tqdm(image_paths)
            ]
        self.scale = scale
        self.patch_size = patch_size
        self.is_test = is_test
        self.augment = augment and not is_test
        self.transforms = A.Compose(
            [
                # A.RandomCrop(self.patch_size, self.path_size, p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],
            additional_targets={"hr_image": "image"},
            is_check_shapes=False
        )

    def __len__(self) -> int:
        return len(self.images)

    def random_crop_pair(self, lr_img, hr_img):
        lr_w, lr_h = lr_img.size
        hr_patch = self.patch_size * self.scale

        x = random.randint(0, lr_w - self.patch_size)
        y = random.randint(0, lr_h - self.patch_size)

        lr_crop = lr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        hr_crop = hr_img.crop(
            (
                x * self.scale,
                y * self.scale,
                x * self.scale + hr_patch,
                y * self.scale + hr_patch,
            )
        )
        return lr_crop, hr_crop

    def augment_pair(self, lr_img, hr_img):
        augmented = self.transforms(
            image=np.array(lr_img),
            hr_image=np.array(hr_img),
        )
        lr_img = Image.fromarray(augmented["image"])
        hr_img = Image.fromarray(augmented["hr_image"])
        return lr_img, hr_img

    def __getitem__(self, idx: int):
        if isinstance(self.images[idx], str):
            hr_img = Image.open(self.images[idx]).convert("RGB")
        else:
            hr_img = self.images[idx]

        w, h = hr_img.size
        w = w - (w % self.scale)
        h = h - (h % self.scale)
        hr_img = hr_img.crop((0, 0, w, h))
        lr_img = hr_img.resize((w // self.scale, h // self.scale), Image.BICUBIC)

        if not self.is_test:
            lr_img, hr_img = self.random_crop_pair(lr_img, hr_img)
            if self.augment:
                lr_img, hr_img = self.augment_pair(lr_img, hr_img)

        lr_tensor = pil_to_tensor(lr_img)
        hr_tensor = pil_to_tensor(hr_img)
        return lr_tensor, hr_tensor


class TestDataset(Dataset):
    def __init__(self,
                 hr_image_paths: list[str],
                 lr_image_paths: list[str],
                 load_immediatly: bool = False):

        if not load_immediatly:
            self.hr_images = hr_image_paths
            self.lr_images = lr_image_paths
        else:
            self.hr_images = [
                Image.open(path).convert("RGB") for path in tqdm(hr_image_paths)
            ]
            self.lr_images = [
                Image.open(path).convert("RGB") for path in tqdm(lr_image_paths)
            ]


    def __len__(self) -> int:
        return len(self.hr_images)

    def __getitem__(self, idx: int):
        if isinstance(self.hr_images[idx], str):
            hr_img = Image.open(self.hr_images[idx]).convert("RGB")
            lr_img = Image.open(self.lr_images[idx]).convert("RGB")
        else:
            hr_img = self.hr_images[idx]
            lr_img = self.lr_images[idx]

        lr_tensor = pil_to_tensor(lr_img)
        hr_tensor = pil_to_tensor(hr_img)
        return lr_tensor, hr_tensor

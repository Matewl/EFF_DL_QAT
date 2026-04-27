import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A
import lightning as L

from utils import pil_to_tensor


def list_images(directory: str | Path) -> list[str]:
    return sorted(str(path) for path in Path(directory).glob("*.png"))


class DIV2KDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        scale: int,
        patch_size: int,
        load_immediatly: bool = False,
        is_test: bool = False,
        augment: bool = True,
    ):
        if not load_immediatly:
            self.images = image_paths
        else:
            self.images = [Image.open(path).convert("RGB") for path in tqdm(image_paths)]

        self.scale = scale
        self.patch_size = patch_size
        self.is_test = is_test
        self.augment = augment and not is_test
        self.transforms = None
        self.transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],
            additional_targets={"hr_image": "image"},
            is_check_shapes=False,
        )

    def __len__(self) -> int:
        return len(self.images)

    def random_crop_pair(self, lr_img: Image.Image, hr_img: Image.Image):
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

    def augment_pair(self, lr_img: Image.Image, hr_img: Image.Image):
        if self.transforms is not None:
            augmented = self.transforms(image=np.array(lr_img), hr_image=np.array(hr_img))
            return Image.fromarray(augmented["image"]), Image.fromarray(augmented["hr_image"])

        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
            hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            rotation = random.choice((Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270))
            lr_img = lr_img.transpose(rotation)
            hr_img = hr_img.transpose(rotation)
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

        return pil_to_tensor(lr_img), pil_to_tensor(hr_img)


class TestDataset(Dataset):
    def __init__(self, hr_image_paths: list[str], lr_image_paths: list[str], load_immediatly: bool = False):
        if not load_immediatly:
            self.hr_images = hr_image_paths
            self.lr_images = lr_image_paths
        else:
            self.hr_images = [Image.open(path).convert("RGB") for path in tqdm(hr_image_paths)]
            self.lr_images = [Image.open(path).convert("RGB") for path in tqdm(lr_image_paths)]

    def __len__(self) -> int:
        return len(self.hr_images)

    def __getitem__(self, idx: int):
        if isinstance(self.hr_images[idx], str):
            hr_img = Image.open(self.hr_images[idx]).convert("RGB")
            lr_img = Image.open(self.lr_images[idx]).convert("RGB")
        else:
            hr_img = self.hr_images[idx]
            lr_img = self.lr_images[idx]

        return pil_to_tensor(lr_img), pil_to_tensor(hr_img)


class ESPCNDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        scale: int = 2,
        patch_size: int = 32,
        batch_size: int = 16,
        num_workers: int = 4,
        load_immediatly: bool = False,
        augment: bool = True,
    ):
        if hasattr(super(), "__init__"):
            super().__init__()
        self.data_root = Path(data_root)
        self.scale = scale
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_immediatly = load_immediatly
        self.augment = augment
        self.test_sets = {}

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            train_paths = list_images(self.data_root / "DIV2K_train_HR")
            val_paths = list_images(self.data_root / "DIV2K_valid_HR")

            self.train_dataset = DIV2KDataset(
                train_paths,
                scale=self.scale,
                patch_size=self.patch_size,
                load_immediatly=self.load_immediatly,
                is_test=False,
                augment=self.augment,
            )
            self.val_dataset = DIV2KDataset(
                val_paths,
                scale=self.scale,
                patch_size=self.patch_size,
                load_immediatly=self.load_immediatly,
                is_test=True,
                augment=False,
            )

        if stage in (None, "test", "fit"):
            self.test_sets = {}
            for name in ("Set5", "Set14"):
                hr_paths = list_images(self.data_root / name / "GTmod12")
                lr_paths = list_images(self.data_root / name / f"LRbicx{self.scale}")
                self.test_sets[name.lower()] = TestDataset(
                    hr_image_paths=hr_paths,
                    lr_image_paths=lr_paths,
                    load_immediatly=self.load_immediatly,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for dataset in self.test_sets.values()
        ]

    @property
    def test_dataset_names(self) -> list[str]:
        return list(self.test_sets.keys())

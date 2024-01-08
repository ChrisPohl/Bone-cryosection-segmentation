from pathlib import Path

import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def get_positions(image, patch_size, stride):
    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    # define valid patches
    rows = [r for r in range(0, h - patch_size, stride)]
    cols = [c for c in range(0, w - patch_size, stride)]

    if not rows[-1] == h:
        rows.append(h - patch_size)
    if not cols[-1] == w:
        cols.append(w - patch_size)

    positions = []
    for c in cols:
        for r in rows:
            sl = np.s_[r : r + patch_size, c : c + patch_size]
            positions.append(sl)

    return positions


def extract_patches(image, positions):
    patches = []
    for sl in positions:
        patches.append(image[sl])

    return patches


def stitch_patches(patches, imshape, positions):
    image = np.zeros(imshape, dtype=patches[0].dtype)
    for patch, sl in zip(patches, positions):
        image[sl] = patch

    return image


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        samples=None,
        mask_suffix="_masks",
        transform=None,
        target_transform=None,
    ):
        self.transform = transform or T.ToTensor()
        self.target_transform = target_transform or T.ToTensor()

        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)

        mask_names = [fp.stem.split("_")[0] for fp in data_dir.iterdir()]
        samples = samples or np.unique(mask_names)

        self.mask_files = natsorted(
            [
                fp
                for fp in data_dir.iterdir()
                if (
                    fp.stem.endswith(mask_suffix) and (fp.stem.split("_")[0] in samples)
                )
            ]
        )

        self.image_files = natsorted(
            [
                fp
                for fp in data_dir.iterdir()
                if (
                    not fp.stem.endswith(mask_suffix)
                    and (fp.stem.split("_")[0] in samples)
                )
            ]
        )

        assert len(self.image_files) == len(
            self.mask_files
        ), f"Number of images and masks does not match. Images: {len(self.image_files)}, masks: {len(self.mask_files)}."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx])
        img = self.transform(img)

        msk = Image.open(self.mask_files[idx])
        msk = self.target_transform(msk)

        return img, msk
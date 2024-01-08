from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from natsort import natsorted
from skimage.filters.thresholding import threshold_otsu
from tifffile import imread, imwrite
from tqdm import tqdm

from unet_classifier.data import get_positions
from unet_classifier.model import LitUNet

transform_val = T.ToTensor()

if __name__ == "__main__":
    # setting up paths to data, model and outputs
    base_dir = Path("")
    data_dir = base_dir / "data" / "test"

    # path to pretrained model
    pretrained_model = base_dir / "trained_models" / "unet.ckpt"
    model = LitUNet.load_from_checkpoint(pretrained_model).eval()

    impaths = natsorted(
        {
            fp
            for fp in data_dir.iterdir()
            if not fp.stem.endswith(("mask", "otsu", "unet"))
        }
    )

    print("Segmenting images.")
    for fp in impaths:
        print("Segmenting", fp.name)

        img = imread(fp)
        positions = get_positions(img, 512, 512 // 2)

        # segmentation unet
        y_unet = torch.zeros(img[:, :, 0].shape, dtype=torch.float32)
        for pos in tqdm(positions):
            ypos, xpos = pos
            ymin, ymax = ypos.start, ypos.stop
            xmin, xmax = xpos.start, xpos.stop

            with torch.no_grad():
                x = transform_val(img[pos]).unsqueeze(0).to("cuda")
                y = model(x).sigmoid().cpu()

            y_unet[pos] = torch.maximum(y_unet[pos], y)

        msk_unet = np.array(y_unet > 0.8)

        # segmentation otsu
        thr = threshold_otsu(img[:, :, 1])
        msk_otsu = img[:, :, 1] >= thr

        imwrite(fp.with_stem(fp.stem + "_unet"), msk_unet)
        imwrite(fp.with_stem(fp.stem + "_otsu"), msk_otsu)

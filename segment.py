from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
from skimage.filters.thresholding import threshold_otsu
from tifffile import imwrite
from tqdm import tqdm

from unet_classifier.model import LitUNet

transform_val = T.ToTensor()

if __name__ == "__main__":
    # setting up paths to data, model and outputs
    base_dir = Path("")
    data_dir = base_dir / "data" / "test"

    # path to pretrained model
    pretrained_model = base_dir / "trained_models" / "unet.ckpt"
    model = LitUNet.load_from_checkpoint(pretrained_model).half()

    impaths = natsorted(
        {
            fp
            for fp in data_dir.iterdir()
            if not fp.stem.endswith(("mask", "otsu", "unet"))
        }
    )

    print("Segmenting images.")
    for fp in tqdm(impaths):
        img = Image.open(fp)
        img_np = np.asarray(img)

        # segmentation unet
        with torch.no_grad():
            x = transform_val(img).unsqueeze(0).to("cuda").half()
            y = model(x)

            msk_unet = y[0, 0].cpu().numpy() > 0.9

        # segmentation otsu
        thr = threshold_otsu(img_np[:, :, 1])
        msk_otsu = img_np[:, :, 1] >= thr

        imwrite(fp.with_stem(fp.stem + "_unet"), msk_unet)
        imwrite(fp.with_stem(fp.stem + "_otsu"), msk_otsu)

from pathlib import Path

import pytorch_lightning as pl
import torchvision.transforms as T
from natsort import natsorted
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tifffile import imread, imwrite
from torch.utils.data import DataLoader

from unet_classifier.data import ImageDataset, get_positions
from unet_classifier.model import LitUNet

if __name__ == "__main__":
    # setting up paths to data, model and outputs
    base_dir = Path("")
    data_dir = base_dir / "data"

    # comment if you don't have writing access to folder
    # model_dir = base_dir / "models"
    # if not model_dir.exists():
    #     model_dir.mkdir()

    # out_dir = base_dir / "results"
    # if not out_dir.exists():
    #     out_dir.mkdir()

    # data processing
    seed = 10  # reproducibility
    pl.seed_everything(seed, workers=True)

    # model params
    # image dimensions
    patchsize = 512  # if OutOfMemoryError decrease to 256, not smaller
    stride = patchsize  # overlap between patches
    batchsize = 8  # at least 2, the higher the better
    num_channels = 3  # 1 for grayscale / fluorescent images
    num_classes = 1  # 1 for binary segmentation 3 for glomeruli

    # training
    epochs = 100  # or more or until loss stops decreasing
    learning_rate = 0.01  # will be automatically decreased at plateau for `decrease_lr_after` epochs
    patience = 50  # if loss does not decrease for that many epochs training is finished
    device = "cuda"  # don't change

    # path to pretrained model
    # patch size of pretrained model must match `patchsize`
    pretrained_model = None  # e.g. base_dir / "models" / "100_glomeruli_256_0.0834.pth"

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    patch_dir = Path("patches")
    patch_dir.mkdir(exist_ok=True)

    # data augmentation for training and validation datasets
    transform_train = T.Compose(
        [
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=(0.75, 1.5), contrast=(0.5, 1), hue=(-0.15, 0.15)
                    ),
                    T.GaussianBlur((9, 9)),
                ]
            ),
            T.ToTensor(),
        ]
    )
    transform_val = T.ToTensor()

    for dir in [train_dir]:
        mask_files = natsorted([fp for fp in dir.glob("*_mask.tif")])
        available_masks = {fp.stem.split("_")[0] for fp in mask_files}
        image_files = natsorted(
            [
                fp
                for fp in dir.glob("*.tif")
                if (fp.stem in available_masks) and (not fp.stem.endswith("_mask"))
            ]
        )

        try:
            for img_id in available_masks:
                fp_img = dir / f"{img_id}.tif"
                fp_msk = dir / f"{img_id}_mask.tif"

                img = imread(fp_img)
                msk = imread(fp_msk)

                pos = get_positions(img, patch_size=patchsize, stride=stride)

                for i, sl in enumerate(pos):
                    outname = f"{img_id}_{i:03}"
                    imwrite(patch_dir / f"{outname}.tif", img[sl])
                    imwrite(patch_dir / f"{outname}_mask.tif", msk[sl])
        except FileExistsError:
            print(f"Patches already exist. To recompute them delete {patch_dir}.")

    train_dir = patch_dir

    sample_names = natsorted({fp.stem.split("_")[0] for fp in train_dir.glob("*.tif")})
    train_names = sample_names[:15]
    val_names = sample_names[15:]

    print("Creating train set from", train_names)
    trainset = ImageDataset(
        train_dir, train_names, mask_suffix="_mask", transform=transform_train
    )
    print("Creating validation set from", val_names)
    valset = ImageDataset(
        train_dir, val_names, mask_suffix="_mask", transform=transform_val
    )

    print("Datasets loaded. Train", len(trainset), "| Val", len(valset))

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
    valloader = DataLoader(valset, batch_size=batchsize, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss", save_last=True, save_top_k=10
    )
    stopping_callback = EarlyStopping(monitor="val/loss", patience=50)

    callbacks = [checkpoint_callback, stopping_callback]
    model = LitUNet(3, 1, lr=learning_rate)

    trainer = pl.Trainer(
        max_epochs=epochs,
        precision="16-mixed",
        default_root_dir="",
        callbacks=callbacks,
    )

    trainer.fit(model, trainloader, valloader)

# Quantitative analysis of trabecular bone tissue cryosections via a fully automated neural network-based approach

![Graphical overview](fig/overview.png)

We created a fully automated approach to segment trabecular tissue by classification of tissue versus background, for a variety of scientific questions. We combined segementation by different approaches with a simple QuPath based cell counting protocol. To measure tissue area and determine cell count within the segmented area. 
![Graphical results](fig/results.png)

In the published paper we used three different approaches. A QuPath based network, a OTSU based thresholding approach and a deep analysis utilizing Unet. We compared their efficiency regarding accuracy with a manually created Ground Truth. This repository includes scripts, data and steps to repeat and utilize created protocols necessary for the presented fully automated trabecular bone segmentation approach. 

## Setup for QuPath segmentation
For QuPath Image classification we included three json files of respective classifiers trained by independently created segmentation ground truths of three seperate investigators.

### Installation
Install QuPath following the instructions provided on [their website](https://qupath.readthedocs.io/en/0.4/index.html).
To recreate and apply pretrained classifiers add the included json data in this repository at `./qupath_classifiers` download them and copy them into the corresponding project folder created by QuPath under `./project/classifiers`. 
### Usage
We included the complete groovy script used for QuPath segmentation and cellcount in this repository. `./qupath_classifiers/cellcount.md`
To run simply copy the script into the Qupath command window or into the QuPath script repository and apply it via the software itself.

## Setup for U-Net and Otsu segmentation

### Download data

We provide our data and model checkpoints at: drive.
Extract the images to `./data` and models to `./trained_models`.

### Installation

For training and inference we use `pytorch`, please install it following the instructions on [their website](https://pytorch.org/get-started/locally/). All other dependencies may be installed using `pip install -r requirements.txt`.

### Usage

#### Segmenting images from the paper

This runs segmentation for images stored in `./data` using a U-Net (with the model checkpoint stored in `./trained_models`), and Otsu thresholding. Masks are stored as `.tif` files in `./results`.

```sh
python run_segmentation.py
```

#### Training the model with custom images

This trains the model using images and masks stored in `./training_data`. For our paper we used following parameters:

```sh
python train_unet.py --learning_rate 0.01 --num_epochs 100
```

The best checkpoint is stored in `./trained_models/best.pth`.

## Citation

```bibtex
@article{pohl2024,
    doi = {},
    author = {},
    journal = {},
    title = {Quantitative analysis of trabecular bone tissue cryosections via a fully automated neural network-based approach},
    year = {2024},
    volume = {},
    pages = {},
    number = {}
}
```

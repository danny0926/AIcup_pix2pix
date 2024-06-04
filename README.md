

# AI cup using PyTorch pix2pix

This method is based on the [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git). Please check this github page for more detail.


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/danny0926/AIcup_pix2pix.git
cd AIcup_pix2pix
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  - For Repl users, please click [![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix).



### pix2pix train/test
- Download a pix2pix dataset (e.g.[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Preprocessing the model:
```bash
### make sure that the origin data is downloaded.
cd dataset
### the customize preprocessing
python preprocess.py --input_dir /path/to/input/dir/ --output_dir /path/to/output/dir/
### combine image data and label data 
python ai_cup_combine.py --fold_A /path/to/fold_A/ --fold_B /path/to/fold_B/
```
- Train a model:
```bash
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction AtoB
```
To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`.

- Test the model:
```bash
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction AtoB
```
- The test results will be saved to a html file here: `./results/facades_pix2pix/test_latest/index.html`. You can find more scripts at `scripts` directory.
- To train and test pix2pix-based colorization models, please add `--model colorization` and `--dataset_mode colorization`. See our training [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization) for more details.

### Using the pretrained model
If you want to use the trained model for the aicup, first download the [trained_model.zip](https://drive.google.com/file/d/1r3qU4AIT9TF2JzlWTjHJmafFTw-Kn3gb/view?usp=sharing) and put under the checkpoint folder


## Acknowledgments
TODO

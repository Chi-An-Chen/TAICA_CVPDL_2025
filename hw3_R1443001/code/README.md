# Homework 1 - Object Detection

## Installation

```bash
conda create -n HW_3 python=3.12 -y
conda activate HW_3

cd src

pip install -q uv
uv pip install -r requirements.txt

# Install pytorch based on your CUDA version ex.12.4
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Dataset
### Data Preparation: Download the MNIST dataset images (without label) from [Google Drive](https://drive.google.com/file/d/1xVCJD6M6sE-tZJYxenLzvuEkSiYXig_F/view?usp=sharing).


## Training
> Start Train model. Create contents(visualize training results) and checkpoints folders.
```bash
cd src
```
```bash
python 001_train.py
```

## Inference
> Start Inference. Create generated_mnist(10,000 images) and diffusion_process.png.
```bash
python 002_test.py
```

## Evaluation
> Calculate FID score with two images folders (Training data / Generated data)
```bash
python -m pytorch_fid dataset/MNIST generated_mnist
```
> Create .npz file -> generated_mnist.npz
```bash
python -m pytorch_fid --save-stats generated_mnist generated_mnist
```
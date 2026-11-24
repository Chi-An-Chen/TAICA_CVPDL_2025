# Homework 1 - Object Detection

## Installation

Start in `hw2_R1443001/`. Resolve the `src/` paths in the data preparation and training examples from this homework directory.

```bash
conda create -n HW_2 python=3.12 -y
conda activate HW_2


pip install -q uv
uv pip install -r requirements.txt

# Install pytorch based on your CUDA version ex.12.4
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Dataset
### Data Preparation: Download the dataset images (training, validation, and test sets) and corresponding labels from [Kaggle](https://www.kaggle.com/competitions/taica-cvpdl-2025-hw-1/overview).

```bash
kaggle original dataset
├── sample_submission.csv
├── test
│   └── img.png
└── train
    ├── img.txt
    └── img.png
```

> Create labels file of train images
```bash
cd src/dataset
python 001_to_yolo.py
```
> Split original train's data into train, val set 
```bash
python 003_copy_paste_v2.py
```
> Split original train's data into train, val set 
```bash
python 004_merge.py
```

## Training
```bash
cd src/model/{model_name}
```
Set the dataset configurations in the `data.yaml` file within each model’s folder.

### For YOLO-Drone
#### Training
```bash
cd yolo_12m_drone
python 001_train.py
```

#### Testing & Evaluation
```bash
python 002_eval.py
```

#### Inference
> Generate prediction.csv for Kaggle competition
```bash
python 003_predict_yolo.py
```
or for SAHI(Slicing Aided Hyper Inference) mode
```bash
python 003_sahi.py
```
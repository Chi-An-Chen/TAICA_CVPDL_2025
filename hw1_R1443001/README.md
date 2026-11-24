# Homework 1 - Object Detection

## Installation

Start in `hw1_R1443001/`. Resolve the `src/` paths in the data preparation and training examples from this homework directory.

```bash
conda create -n HW_1 python=3.12 -y
conda activate HW_1


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
│   └── img
└── train
    ├── gt.txt
    └── img
```

> Create labels file of train images
```bash
cd src/dataset
python 001_to_yolo.py
```
> Split original train's data into train, val set 
```bash
python 002_preprocess.py
```
> Split original train's data into train, val set 
```bash
python 003_choose_for_val.py
```
### Finally prepare your dataset as shown below. 
```bash
.
├── images
│   ├── train
│   │   └── 00x.jpg
│   └── val
│ 
└── labels
    ├── train
    │   └── 00x.txt
    └── val
```

## Training
```bash
cd src/model/{model_name}
```
Set the dataset configurations in the `data.yaml` file within each model’s folder.

### For Hyper-YOLO
#### Training
```bash
python train_dual.py --device 0 --batch 8 --flat-cos-lr --data data_aug.yaml --img 640 --cfg models/detect/yolov9-c-hyper.yaml --name yolov9-c-hyper --hyp data/hyps/hyp.scratch-self.yaml --epochs 50 --save-period 5
```

#### Testing & Evaluation
```bash
python val_dual.py --data data_aug.yaml --img 640 --batch 32 --device 0 --weights runs/train/yolov9-c-hyper/weights/best.pt --save-json --name yolov9_s_hyper_c_640_val --conf-sweep
```

#### Prediction output

`--save-json` in the validation command writes COCO-style prediction JSON. The separate [YOLOv12 prediction script](src/model/yolo_12m/003_predict_yolo.py) is the CSV submission entry for that model.

### For the bundled YOLOv12 scripts
#### Training
```bash
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
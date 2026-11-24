# NIU Final project - Classification of citrus diseases and insect pests

## Catalog of citrus diseases and insect pests
```
"disease_canker"
"disease_greasy_spot"
"disease_sooty_moulds"
"health_murcott"
"health_ponkan"
"pest_aphid"
"pest_phyllocnistis"
"pest_thrips"
```
The [project report](<TAICA_Final Project Report.pdf>) records the experiment. Begin in `taica_final_project/` for installation, then use `src/` for the numbered scripts. Provide the citrus dataset and split files at the paths configured in the scripts.

## Installation

```bash
conda create -n final_project python=3.12 -y
conda activate final_project

pip install -q uv
uv pip install -r requirement.txt

# Install pytorch based on your CUDA version ex.12.4
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

```bash
cd src
```

## 1. Evaluate baseline model
- 根據目前 torchvision 中的 Classification 預訓練模型，選擇每一個模型在 ImageNet-1K 測試中 Acc@1 最高的版本進行評估，作為teacher model (ConvNext Large)。
```bash
python 000_base_model.py
```
### Results of baseline model
|Model |Accuracy |Precision |Recall |F1 Score|Params|
|---|---|---|---|---|---|
|ConvNeXt_Large|0.9797|0.9801|0.9797|0.9795|197.8M|
|EfficientNet_V2_L|0.9661|0.9671|0.9661|0.9657|118.5M|
|ResNeXt101_64x4d|0.9695|0.9706|0.9695|0.9696|83.5M|
|RegNet_Y_128GF|0.9627|0.9643|0.9627|0.9625|644.8M|
|MobileNet_V3_Large|0.9610|0.9635|0.9610|0.9600|5.5M|
|Wide_ResNet101_2|0.9525|0.9552|0.9525|0.9531|126.9M|
|ResNet152|0.9475|0.9550|0.9475|0.9505|60.2M|

*根據目前結果，目前的模型在簡單的finetune後最低皆有95%左右的準確度，因此訂定實驗方向朝輕量化和量化模型前進*

## 2. Fine-tuning Teacher model
- 根據 baseline model 的結果，選擇效能最佳的模型 (ConvNext Large) 的 version 2版本進行微調 (finetune)。
> *註:由於 torchvision 沒有提供 ConvNext V2 的預訓練權重，因此改用 timm 且同時比較 large 和 huge 的性能* 
```bash
python 001_finetune_teacher.py
```

## 3. Knowledge distillation

[002_distillation.py](src/002_distillation.py) loads the ConvNeXt V2 Large teacher from `checkpoints_finetune/best_teacher_model_convnextv2_large.pth` and writes student outputs under `checkpoints_kd/`.

```bash
python 002_distillation.py
```

## 4. Compare training methods
```bash
cd 003_compare_model
# Select one script for the target backbone, for example:
python 003_compare_mobilevit.py
```

## 5. Smmarize results
```bash
cd ..
python 004_compaire_all.py
```

## 6. Transfer model from .pth to .onnx and test the model
```bash
python 005_to_onnx.py
```
```bash
python 006_test_onnx.py
```

## 7. Compare two model using GradCAM
```bash
python 007_gradcam.py
```

## Other utilities

From `src/`, inspect the dataset statistics or the list of small models using the corresponding script:

```bash
python utils/cal_std_mean.py
python see_dataset.py
python ../see_model.py
```

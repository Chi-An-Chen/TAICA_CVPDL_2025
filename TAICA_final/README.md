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
## Installation

```bash
conda create -n final_project python=3.12 -y
conda activate final_project

cd src

pip install -q uv
uv pip install -r requirements.txt

# Install pytorch based on your CUDA version ex.12.4
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## Evaluate baseline model
```bash
python 000_base_model.py
```

## Fine-tuning Teacher model
```bash
python 001_finetune_teacher.py
```

## Quantization
```bash
python 002_quantization.py
```

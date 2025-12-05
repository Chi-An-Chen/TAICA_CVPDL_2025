import timm
import gc
import torch
from tqdm import tqdm

# 獲取所有 timm 模型列表
model_list = timm.list_models()
print(model_list)
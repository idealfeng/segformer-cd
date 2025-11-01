# 检查你的模型
import torch
from models.segformer import build_segformer_distillation

model = build_segformer_distillation()

# 查找所有norm层
for name, module in model.named_modules():
    if 'norm' in name.lower() or 'bn' in name.lower():
        print(f"{name}: {type(module)}")
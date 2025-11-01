import numpy as np
from config import cfg

# 随便选一个文件
img_id = "top_potsdam_2_10_RGB_crop0001"

# 检查Block30特征
feat30_path = cfg.FEATURE_BLOCK30_DIR / f"{img_id}.npz"
feat30_data = np.load(feat30_path)
print("Block30 keys:", feat30_data.files)
print("Block30 shape:", feat30_data['features'].shape)
# 期望输出：(1, 1280, H, W) 或类似

# 检查Encoder特征
feat_enc_path = cfg.FEATURE_ENCODER_DIR / f"{img_id}.npz"
feat_enc_data = np.load(feat_enc_path)
print("Encoder keys:", feat_enc_data.files)
print("Encoder shape:", feat_enc_data['features'].shape)
# 期望输出：(1, 256, H, W) 或类似
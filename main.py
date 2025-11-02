# debug_sam.py (【最终修正版】)
"""调试SAM - 验证GT-guided box是否工作"""
import torch
import numpy as np
from models.sam_wrapper import SAMWrapper
from dataset import build_dataloader

# ✅ 核心修正第一步: 把所有执行代码，都放进一个函数里
def run_debug():
    # 加载模型
    model = SAMWrapper('pretrained_weights/sam_vit_h_4b8939.pth')
    model.eval()

    # 加载数据
    # ✅ 核心修正第二步: 在创建DataLoader时，临时关闭多进程
    test_loader = build_dataloader('test', batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(test_loader))

    images = batch['image'].to(model.device)
    labels = batch['label'].to(model.device) # 假设dataset返回的是二值label

    print("=" * 60)
    print("调试SAM GT-guided box")
    print("=" * 60)

    print(f"\n输入:")
    print(f"  images: {images.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  labels唯一值: {torch.unique(labels)}")

    # 检查前景像素
    fg_mask = (labels == 1)
    fg_ratio = fg_mask.sum().item() / labels.numel() if labels.numel() > 0 else 0
    print(f"\n前景像素:")
    print(f"  比例: {fg_ratio*100:.2f}%")
    print(f"  总数: {fg_mask.sum().item()}")

    # ========== 关键：检查forward是否收到labels ==========
    print("\n" + "=" * 60)
    print("测试1: 不传labels（应该用全图box）")
    print("=" * 60)
    with torch.no_grad():
        output1 = model.forward(images) # 假设你的forward现在可以不接收labels
        pred1 = (torch.sigmoid(output1) > 0.5).sum().item() / output1.numel()
        print(f"预测为前景的比例: {pred1*100:.2f}%")

    print("\n" + "=" * 60)
    print("测试2: 传入labels（应该用GT-guided box）")
    print("=" * 60)
    with torch.no_grad():
        output2 = model.forward(images, labels=labels)
        pred2 = (torch.sigmoid(output2) > 0.5).sum().item() / output2.numel()
        print(f"预测为前景的比例: {pred2*100:.2f}%")

    print("\n" + "=" * 60)
    print("对比:")
    print(f"  不传labels: {pred1*100:.2f}%")
    print(f"  传入labels: {pred2*100:.2f}%")
    print(f"  真实前景:   {fg_ratio*100:.2f}%")
    print("=" * 60)

    if abs(pred1 - pred2) < 0.01:
        print("\n❌ 问题：两种方式结果几乎相同！")
        print("   → labels可能没有正确传递到forward")
        print("   → 或者box计算有问题")
    else:
        print("\n✅ 两种方式结果不同，说明labels传递成功")
        if abs(pred2 - fg_ratio) > 0.1: # 放宽一点阈值
            print(f"\n⚠️  但预测({pred2*100:.1f}%)与真实({fg_ratio*100:.1f}%)差距较大")
            print("   → SAM的box prompt效果可能不理想，或者box提取逻辑需要检查")

# ✅ 核心修正第三步: 用“安全区”来调用你的执行函数
if __name__ == '__main__':
    run_debug()
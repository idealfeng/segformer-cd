# eval_sam.py (修正版)
"""
评估SAM Teacher模型
注意：4060 8GB只能用batch_size=1
python eval_sam.py `
    --checkpoint pretrained_weights/sam_vit_h_4b8939.pth `
    --batch-size 1 `
    --compute-fps `
    --num-warmup 10 `
    --num-measure 100
"""
import argparse
from pathlib import Path

from models.sam_wrapper import SAMWrapper
from dataset import build_dataloader


def main():
    parser = argparse.ArgumentParser(description='评估SAM Teacher模型')
    parser.add_argument('--checkpoint', type=str,
                        default='pretrained_weights/sam_vit_h_4b8939.pth',
                        help='SAM权重路径')
    parser.add_argument('--model-type', type=str, default='vit_h',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM模型类型 (vit_h=632M, vit_l=308M, vit_b=91M)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (4060 8GB只能用1)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='测试样本数（默认全部360张）')
    parser.add_argument('--compute-fps', action='store_true',
                        help='计算FPS（耗时更长）')
    parser.add_argument('--num-warmup', type=int, default=5,
                        help='Warm-up批次数')
    parser.add_argument('--num-measure', type=int, default=50,
                        help='FPS测量批次数')
    args = parser.parse_args()

    # ✅ 显存检查
    if args.batch_size > 1:
        print("⚠️  警告: SAM-ViT-H在4060 8GB上只能用batch_size=1")
        print("   强制设置batch_size=1")
        args.batch_size = 1

    # 检查权重
    if not Path(args.checkpoint).exists():
        print(f"❌ SAM权重不存在: {args.checkpoint}")
        print("\n请从以下地址下载:")
        if args.model_type == 'vit_h':
            print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        elif args.model_type == 'vit_l':
            print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
        else:
            print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return

    # 加载模型
    model = SAMWrapper(args.checkpoint, model_type=args.model_type)

    # 加载数据
    test_loader = build_dataloader('test', batch_size=args.batch_size, shuffle=False)

    # ✅ 限制样本数（可选）
    if args.num_samples:
        print(f"⚠️  只测试前{args.num_samples}张图像")

    # 评估
    print("\n" + "=" * 60)
    print(f"开始评估SAM-{args.model_type.upper()}...")
    print(f"  显存占用预估: ~5.5GB (batch_size=1)")
    print(f"  预计耗时: ~6-12分钟 (360张)")
    print("=" * 60)

    metrics = model.evaluate(
        test_loader,
        compute_fps=args.compute_fps,
        num_warmup=args.num_warmup,
        num_measure=args.num_measure
    )

    # 打印结果
    model.print_metrics(metrics)

    # 保存结果
    save_name = f'metrics_sam_{args.model_type}.json'
    save_path = Path('outputs/eval_results') / save_name
    model.save_metrics(metrics, save_path)

    print("\n✓ SAM评估完成！")
    print(f"✓ 结果已保存: {save_path}")


if __name__ == '__main__':
    main()

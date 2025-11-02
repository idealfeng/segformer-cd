"""
完整训练评估流程 Pipeline - 优化版
自动执行：训练 → 评估 → 可视化

✅ 新增:
  - 支持batch_size配置
  - 支持FPS测量参数
  - Checkpoint存在性检查
  - 训练失败后续处理

使用方法：
    # 完整流程
    python run_pipeline.py --epochs 100

    # 快速测试（小batch）
    python run_pipeline.py --epochs 3 --batch-size 1

    # 精确FPS测量
    python run_pipeline.py --skip-train --compute-fps --num-warmup 20

    # 跳过训练（只评估）
    python run_pipeline.py --skip-train --checkpoint outputs/checkpoints/best.pth

    # 自定义配置
    python run_pipeline.py --epochs 50 --num-vis 20 --batch-size 4

场景1：完整训练（推荐配置）
# 100 epochs完整训练 + 精确评估
python run_pipeline.py --epochs 100 `
                       --batch-size 4 `
                       --compute-fps `
                       --num-warmup 20 `
                       --num-vis 10 `
                       --error-analysis

场景2：快速测试（3 epochs）
# 快速验证流程
python run_pipeline.py --epochs 3 `
                       --batch-size 2 `
                       --num-vis 5


场景3：只评估现有模型
# 评估3 epochs的模型
python run_pipeline.py --skip-train `
                       --checkpoint outputs/checkpoints/best.pth `
                       --batch-size 4 `
                       --compute-fps

场景4：恢复中断的训练
# 从epoch 50继续训练到100
python run_pipeline.py --epochs 100 `
                       --resume outputs/checkpoints/epoch_50.pth


场景5：批量评估多个checkpoint
# 评估epoch 0, 1, 2
for epoch in 0 1 2; do
    python run_pipeline.py --skip-train `
                           --checkpoint outputs/checkpoints/epoch_${epoch}.pth `
                           --batch-size 4 `
                           --skip-vis
done
"""
import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description="", allow_fail=False):
    """
    运行命令并实时输出

    Args:
        cmd: 命令字符串
        description: 描述文字
        allow_fail: 是否允许失败（用于可选步骤）
    """
    print("\n" + "=" * 70)
    if description:
        print(f"📋 {description}")
    print(f"▶  {cmd}")
    print("=" * 70)

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        if allow_fail:
            print(f"\n⚠️  命令失败但继续: {description or cmd}")
            print(f"   返回码: {result.returncode}")
        else:
            print(f"\n❌ 命令失败: {cmd}")
            print(f"   返回码: {result.returncode}")
            sys.exit(1)

    print(f"\n✅ 完成: {description or cmd}")


def check_checkpoint_exists(checkpoint_path):
    """检查checkpoint是否存在"""
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Checkpoint不存在: {checkpoint_path}")
        print("\n可用的checkpoints:")
        ckpt_dir = Path('outputs/checkpoints')
        if ckpt_dir.exists():
            checkpoints = list(ckpt_dir.glob('*.pth'))
            if checkpoints:
                for ckpt in sorted(checkpoints):
                    print(f"  - {ckpt}")
            else:
                print("  （无）")
        else:
            print("  outputs/checkpoints/ 目录不存在")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='自动化训练评估流程 - 优化版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 完整100 epochs训练
  python run_pipeline.py --epochs 100
  
  # 快速3 epochs测试
  python run_pipeline.py --epochs 3 --num-vis 5 --batch-size 2
  
  # 只评估和可视化
  python run_pipeline.py --skip-train --checkpoint outputs/checkpoints/epoch_50.pth
  
  # 包含FPS测试和错误分析（推荐配置）
  python run_pipeline.py --skip-train --compute-fps --batch-size 4 --error-analysis
  
  # 精确FPS测量（更多warm-up）
  python run_pipeline.py --skip-train --compute-fps --num-warmup 20 --num-measure 200
        '''
    )

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--skip-train', action='store_true',
                        help='跳过训练，只做评估和可视化')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')

    # 评估参数
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/checkpoints/best.pth',
                        help='评估用的checkpoint路径 (默认: best.pth)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='评估batch size (默认: 4, 可选1/2/4/8)')
    parser.add_argument('--compute-fps', action='store_true',
                        help='计算FPS（需要warm-up）')
    parser.add_argument('--num-warmup', type=int, default=10,
                        help='FPS warm-up批次数 (默认: 10)')
    parser.add_argument('--num-measure', type=int, default=100,
                        help='FPS测量批次数 (默认: 100)')

    # 可视化参数
    parser.add_argument('--num-vis', type=int, default=10,
                        help='可视化样本数量 (默认: 10)')
    parser.add_argument('--error-analysis', action='store_true',
                        help='生成错误分析图')
    parser.add_argument('--skip-vis', action='store_true',
                        help='跳过可视化（只训练和评估）')

    args = parser.parse_args()

    # 打印配置
    print("\n" + "=" * 70)
    print("🚀 Pipeline 开始")
    print("=" * 70)
    print(f"配置:")
    print(f"  训练: {'跳过' if args.skip_train else f'{args.epochs} epochs'}")
    if args.resume:
        print(f"  恢复: {args.resume}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  评估: Batch={args.batch_size} {'+ FPS' if args.compute_fps else ''}")
    if args.compute_fps:
        print(f"    └─ Warmup={args.num_warmup}, Measure={args.num_measure}")
    print(f"  可视化: {'跳过' if args.skip_vis else f'{args.num_vis}张 ' + ('错误分析' if args.error_analysis else '')}")

    # ============================================================
    # Step 1: 训练
    # ============================================================
    if not args.skip_train:
        train_cmd = f'python train.py --epochs {args.epochs}'
        if args.resume:
            train_cmd += f' --resume {args.resume}'

        run_command(
            train_cmd,
            f'Step 1/3: 训练 {args.epochs} epochs'
        )

        # 训练完成后，更新checkpoint路径为最新的best.pth
        args.checkpoint = 'outputs/checkpoints/best.pth'
    else:
        print("\n" + "=" * 70)
        print("⏭  Step 1/3: 训练 (已跳过)")
        print("=" * 70)

        # ✅ 新增: 检查checkpoint是否存在
        check_checkpoint_exists(args.checkpoint)

    # ============================================================
    # Step 2: 评估
    # ============================================================
    eval_cmd = f'python eval.py --checkpoint {args.checkpoint}'
    eval_cmd += f' --batch-size {args.batch_size}'  # ✅ 新增

    if args.compute_fps:
        eval_cmd += ' --compute-fps'
        eval_cmd += f' --num-warmup {args.num_warmup}'  # ✅ 新增
        eval_cmd += f' --num-measure {args.num_measure}'  # ✅ 新增

    run_command(
        eval_cmd,
        'Step 2/3: 评估模型'
    )

    # ============================================================
    # Step 3: 可视化
    # ============================================================
    if not args.skip_vis:
        vis_cmd = f'python visualize.py --checkpoint {args.checkpoint}'
        vis_cmd += f' --num-samples {args.num_vis}'
        if args.error_analysis:
            vis_cmd += ' --error-analysis'

        run_command(
            vis_cmd,
            'Step 3/3: 生成可视化'
        )
    else:
        print("\n" + "=" * 70)
        print("⏭  Step 3/3: 可视化 (已跳过)")
        print("=" * 70)

    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "=" * 70)
    print("🎉 Pipeline 完成！")
    print("=" * 70)

    # 输出结果位置
    checkpoint_name = Path(args.checkpoint).stem
    print("\n📊 查看结果:")
    print(f"  ├─ 评估指标:  outputs/eval_results/metrics_{checkpoint_name}.json")
    if not args.skip_vis:
        print(f"  ├─ 可视化:    outputs/visualizations/")
        if args.error_analysis:
            print(f"  ├─ 错误分析:  outputs/error_analysis/")
    print(f"  ├─ 训练日志:  outputs/logs/")
    print(f"  └─ Checkpoint: outputs/checkpoints/")

    print("\n💡 快速查看:")
    print(f"  # 查看指标")
    print(f"  cat outputs/eval_results/metrics_{checkpoint_name}.json")
    if not args.skip_vis:
        print(f"\n  # 查看可视化")
        print(f"  ls outputs/visualizations/")
    print(f"\n  # TensorBoard (如果训练过)")
    print(f"  tensorboard --logdir=outputs/logs")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline 被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


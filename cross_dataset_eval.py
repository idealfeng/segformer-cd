# cross_dataset_eval.py

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from unified_dataset import create_dataloaders_unified
from model import SegformerChangeDetection  # 你的模型
from config import cfg


def evaluate_model(model, test_loader, device='cuda'):
    """评估模型"""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            img_a = batch['img_a'].to(device)
            img_b = batch['img_b'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(img_a, img_b)
            preds = (torch.sigmoid(outputs) > 0.5).long()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 计算指标
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)

    return {
        'F1': f1 * 100,
        'IoU': iou * 100,
        'Precision': precision * 100,
        'Recall': recall * 100
    }


def cross_dataset_evaluation():
    """交叉数据集评估"""

    # 数据集配置
    datasets = {
        'LEVIR-CD': {
            'name': 'levir',
            'root': Path('data/LEVIR-CD'),
            'model': 'outputs/levir_only/best.pth'
        },
        'S2Looking': {
            'name': 's2looking',
            'root': Path('data/S2Looking'),
            'model': 'outputs/s2looking_only/best.pth'
        },
        'WHUCD': {
            'name': 'whucd',
            'root': Path('data/Building change detection dataset_add'),
            'model': 'outputs/whucd_only/best.pth'
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 结果存储
    results = {}

    # 对每个训练数据集
    for train_name, train_config in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Loading model trained on: {train_name}")
        print(f"{'=' * 60}")

        # 加载模型
        model = SegformerChangeDetection()
        checkpoint = torch.load(train_config['model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        results[train_name] = {}

        # 在每个测试数据集上评估
        for test_name, test_config in datasets.items():
            print(f"\nTesting on: {test_name}")

            # 创建测试数据加载器
            _, _, test_loader = create_dataloaders_unified(
                dataset_name=test_config['name'],
                data_root=test_config['root'],
                batch_size=8,
                num_workers=4
            )

            # 评估
            metrics = evaluate_model(model, test_loader, device)
            results[train_name][test_name] = metrics

            # 打印结果
            print(f"  F1: {metrics['F1']:.2f}%")
            print(f"  IoU: {metrics['IoU']:.2f}%")

    # 生成结果表格
    print_results_table(results)

    # 保存结果
    save_results(results)


def print_results_table(results):
    """打印结果表格"""
    print("\n" + "=" * 80)
    print("Cross-Dataset Evaluation Results")
    print("=" * 80)

    # F1分数表
    print("\nF1-Score (%):")
    print(f"{'Train\\Test':<15}", end='')
    test_datasets = list(results[list(results.keys())[0]].keys())
    for test_ds in test_datasets:
        print(f"{test_ds:<15}", end='')
    print()
    print("-" * 80)

    for train_ds in results:
        print(f"{train_ds:<15}", end='')
        for test_ds in test_datasets:
            f1 = results[train_ds][test_ds]['F1']
            if train_ds == test_ds:
                print(f"\033[92m{f1:>14.2f}\033[0m", end='')  # 绿色（对角线）
            else:
                print(f"\033[91m{f1:>14.2f}\033[0m", end='')  # 红色（跨数据集）
        print()

    # IoU表
    print("\nIoU (%):")
    print(f"{'Train\\Test':<15}", end='')
    for test_ds in test_datasets:
        print(f"{test_ds:<15}", end='')
    print()
    print("-" * 80)

    for train_ds in results:
        print(f"{train_ds:<15}", end='')
        for test_ds in test_datasets:
            iou = results[train_ds][test_ds]['IoU']
            if train_ds == test_ds:
                print(f"\033[92m{iou:>14.2f}\033[0m", end='')
            else:
                print(f"\033[91m{iou:>14.2f}\033[0m", end='')
        print()

    print("=" * 80)


def save_results(results):
    """保存结果到CSV"""
    # F1结果
    f1_data = []
    for train_ds in results:
        row = {'Train': train_ds}
        for test_ds in results[train_ds]:
            row[test_ds] = f"{results[train_ds][test_ds]['F1']:.2f}"
        f1_data.append(row)

    df_f1 = pd.DataFrame(f1_data)
    df_f1.to_csv('cross_dataset_f1.csv', index=False)

    # IoU结果
    iou_data = []
    for train_ds in results:
        row = {'Train': train_ds}
        for test_ds in results[train_ds]:
            row[test_ds] = f"{results[train_ds][test_ds]['IoU']:.2f}"
        iou_data.append(row)

    df_iou = pd.DataFrame(iou_data)
    df_iou.to_csv('cross_dataset_iou.csv', index=False)

    print("\nResults saved to:")
    print("  - cross_dataset_f1.csv")
    print("  - cross_dataset_iou.csv")


if __name__ == '__main__':
    cross_dataset_evaluation()
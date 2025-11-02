# utils/results_manager.py
"""
实验结果管理器
用于Phase 3多模型对比实验的结果汇总
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


class ResultsManager:
    """
    管理所有模型的评估结果

    功能:
    - 读取JSON格式的评估结果
    - 汇总到Excel表格
    - 生成LaTeX表格代码
    """

    def __init__(self, results_dir='outputs/eval_results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.excel_path = self.results_dir / 'phase3_comparison.xlsx'
        self.latex_path = self.results_dir / 'phase3_comparison.tex'

    def load_metrics(self, json_path):
        """加载单个模型的JSON结果"""
        with open(json_path, 'r') as f:
            return json.load(f)

    def add_model_result(self, model_name, metrics_json_path):
        """
        添加一个模型的结果到汇总表

        Args:
            model_name: 模型名称（如'SAM', 'SegFormer-B1'）
            metrics_json_path: 该模型的metrics JSON文件路径
        """
        metrics = self.load_metrics(metrics_json_path)

        # 提取关键指标
        row = {
            'Model': model_name,
            'mIoU (%)': f"{metrics.get('mIoU', 0) * 100:.2f}",
            'IoU_fg (%)': f"{metrics.get('IoU', 0) * 100:.2f}",
            'IoU_bg (%)': f"{metrics.get('IoU_bg', 0) * 100:.2f}",
            'F1 (%)': f"{metrics.get('F1', 0) * 100:.2f}",
            'Dice (%)': f"{metrics.get('Dice', 0) * 100:.2f}",
            'Precision (%)': f"{metrics.get('Precision', 0) * 100:.2f}",
            'Recall (%)': f"{metrics.get('Recall', 0) * 100:.2f}",
            'Params (M)': f"{metrics.get('params', 0):.2f}",
            'FLOPs (G)': f"{metrics.get('flops', 0):.2f}",
            'FPS': f"{metrics.get('FPS', 0):.2f}" if 'FPS' in metrics else 'N/A',
            'Latency (ms)': f"{metrics.get('Latency_ms', 0):.2f}" if 'Latency_ms' in metrics else 'N/A'
        }

        return row

    def create_comparison_table(self, model_results):
        """
        创建对比表格

        Args:
            model_results: dict, {model_name: metrics_json_path}

        Example:
            model_results = {
                'SAM-ViT-H': 'outputs/eval_results/metrics_sam.json',
                'SegFormer-B1 (Ours)': 'outputs/eval_results/metrics_best.json',
                'MobileSAM': 'outputs/eval_results/metrics_mobilesam.json',
                ...
            }
        """
        rows = []
        for model_name, json_path in model_results.items():
            print(f"处理: {model_name}")
            row = self.add_model_result(model_name, json_path)
            rows.append(row)

        # 创建DataFrame
        df = pd.DataFrame(rows)

        # 保存到Excel
        with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
            # 主表
            df.to_excel(writer, sheet_name='Comparison', index=False)

            # 格式化
            worksheet = writer.sheets['Comparison']

            # 设置列宽
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + i)].width = max_len

            # 冻结首行
            worksheet.freeze_panes = 'A2'

        print(f"\n✓ Excel表格已保存: {self.excel_path}")

        # 生成LaTeX表格
        self._generate_latex_table(df)

        return df

    def _generate_latex_table(self, df):
        """生成LaTeX表格代码"""
        latex_code = []

        latex_code.append("\\begin{table*}[t]")
        latex_code.append("\\centering")
        latex_code.append("\\caption{Comparison of different models on Potsdam test set.}")
        latex_code.append("\\label{tab:comparison}")
        latex_code.append("\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}")
        latex_code.append("\\toprule")

        # 表头
        header = " & ".join(df.columns) + " \\\\"
        latex_code.append(header)
        latex_code.append("\\midrule")

        # 数据行
        for _, row in df.iterrows():
            row_str = " & ".join(str(v) for v in row.values) + " \\\\"
            latex_code.append(row_str)

        latex_code.append("\\bottomrule")
        latex_code.append("\\end{tabular}")
        latex_code.append("\\end{table*}")

        # 保存
        with open(self.latex_path, 'w') as f:
            f.write('\n'.join(latex_code))

        print(f"✓ LaTeX表格已保存: {self.latex_path}")


if __name__ == '__main__':
    """测试结果管理器"""
    print("=" * 60)
    print("测试结果管理器")
    print("=" * 60)

    manager = ResultsManager()

    # 示例：汇总两个模型的结果
    model_results = {
        'SAM-ViT-H (Teacher)': 'outputs/eval_results/metrics_sam.json',
        'SegFormer-B1 (Ours)': 'outputs/eval_results/metrics_best.json',
    }

    # 检查文件是否存在
    for name, path in model_results.items():
        if not Path(path).exists():
            print(f"⚠️  缺少: {name} ({path})")

    # 创建对比表
    # df = manager.create_comparison_table(model_results)
    # print("\n预览:")
    # print(df)
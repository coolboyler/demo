# D6 Forecast Pipeline

本目录是独立可运行的 `D-6` 预测闭环，包含数据集、训练、验证、测试和推理所需文件。

## 目录

- `scripts/`: 完整脚本
- `new/`: 运行依赖的输入数据
- `results/`: 训练、验证、测试、推理输出

## 核心文件

- 数据集构建: `scripts/build_baseline_d6_dataset.py`
- 基线训练: `scripts/train_baseline_d6.py`
- 最优模型训练: `scripts/train_best_d6.py`
- 单日正式预测: `scripts/forecast_d6.py`
- 区间预测: `scripts/forecast_d6_range.py`
- 日常更新并出下一次 D-6: `scripts/update_d6.py`

## 当前数据

- 历史日表: `new/history_daily.csv`
- 历史小时表: `new/history_hourly.csv`
- D-6 数据集: `new/baseline_d6_dataset.csv`

## 当前结果

- 基线模型摘要: `results/baseline_d6_summary.json`
- 最优模型摘要: `results/best_d6_summary.json`
- 月度对比: `results/best_d6_monthly.csv`
- 测试集结果: `results/best_d6_test_daily.csv`
- 正式预测示例: `results/forecast_d6_20260313.csv`

## 常用命令

在本目录下执行:

```bash
python3 scripts/build_baseline_d6_dataset.py
python3 scripts/train_baseline_d6.py
python3 scripts/train_best_d6.py
python3 scripts/forecast_d6.py
python3 scripts/forecast_d6_range.py
python3 scripts/update_d6.py --actual-file new/actual_input_template.csv
```

## 说明

- `train_best_d6.py` 为当前主模型，方法是“假期分流 + 相似日增强”。
- `forecast_d6.py` 默认使用 `results/best_d6_summary.json` 出正式预测。
- `update_d6.py` 会先写入新增实绩，再自动生成下一次正式 `D-6` 预测。

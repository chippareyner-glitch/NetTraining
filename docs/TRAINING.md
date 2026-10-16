# Rest / Impact 训练与部署

## 输入数据

采集与标注界面在完成 Trial 后同步训练数据，也可以手动执行：

```powershell
python run.py export-labels
python run.py inspect
```

训练脚本递归读取 `data_labeled/**/samples_labeled.csv`，优先使用 `voltage_uV`，没有时退回 `raw_value`；`train_use == 0` 的点不用于训练。主要标签列是 `label_id`、`label_name`、`train_use`，时间列是 `sample_index` 和 `time_s`。

## 两条二分类流程

| 流程 | 命令 | 窗口与输出 |
| --- | --- | --- |
| 普通窗口分类 | `python run.py train-binary` | 默认 128 点、步长 32；Impact 点占比至少 20%；`outputs/impact_binary/` |
| Impact 事件检测 | `python run.py train-event` | 256 点原始窗口、步长 32，2 倍降采样为 128 点；`outputs/impact_event_detector/` |

事件流程根据 `impact_start` 定义正例；事件开始后的 256 点为有效响应区间，第 256 至 499 点作为恢复尾段排除。当前脚本训练完成后还会更新 `outputs/impact_binary_best_event/`，即使通过 `--out_dir` 指定了其他目录。

两条流程都按 Session 划分数据集。同一 Session 的窗口不应同时进入训练和测试集合。对比实验由 `python run.py experiments` 执行，并写入 `outputs/impact_binary_experiments/` 和最佳候选目录。

```powershell
python run.py train-binary --help
python run.py train-binary --epochs 20 --out_dir outputs/impact_binary_trial
python run.py --dry-run train-event --epochs 20
```

普通二分类默认启用 3 点中值滤波，以窗口前 16 点中位数作为基线，使用训练集统计量缩放。默认训练 80 轮、batch size 32，early stopping patience 10。事件流程有独立的窗口规则，以其导出的配置为准。

## 产物与部署

主要输出包括：

- `model.keras`、`model_int8.tflite`、`model_data.h`
- `preprocess_config.json`、`quantization_info.json`、`decision_config.json`
- `report.json`、训练曲线、混淆矩阵和阈值报告

部署时使用同一轮训练产生的模型和三个配置文件。二分类头文件中的模型数组名为 `g_model`。阈值在 MCU 推理后的逻辑中应用，不改变模型输出：

```c
if (prob_impact >= impact_threshold) {
    /* Impact */
} else {
    /* Rest */
}
```

当前代码会扫描测试集上的多个阈值并给出推荐配置；该配置的结果不能当作使用独立阈值校准集得到的无偏评估。后续比较模型时应记录 Session 划分和阈值选择口径。

PC 实时监测优先搜索 `outputs/impact_binary_best_event/`，再搜索 `outputs/impact_event_detector/`。普通二分类产物可通过界面手动选择。PC 预处理在 `monitoring/inference.py`，不要仅凭输入长度相同就混用不同流程的配置。

早期三分类见[单独说明](legacy_three_class.md)。本次重构没有重新训练现有模型，也没有重新验证 MCU 推理结果。

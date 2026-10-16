# 工程结构与阅读索引

所有业务代码在 `src/plant_signal/`；根目录的 `run.py` 负责从源码启动，`cli.py` 集中登记任务。包的 `__init__.py` 不自动导入 TensorFlow 或 Streamlit，因此列任务、检查发布文件不需要加载训练环境。

## 模块职责

| 路径（相对于 `src/plant_signal/`） | 职责 | 原根目录文件 |
| --- | --- | --- |
| `acquisition/parser.py` | 采集串口文本解析，返回结构化样本 | `serial_parser.py` |
| `acquisition/serial_reader.py` | 后台采集线程、缓存、串口占用管理 | `serial_reader.py` |
| `acquisition/trials.py` | Session、Trial、临时 CSV 和元数据 | `trial_manager.py` |
| `annotation/windows.py` | 标签配对、区间与窗口生成 | `build_dataset_from_trial_tags.py` |
| `annotation/export.py` | 完成标注后的逐点训练数据导出和 manifest | `export_labeled_trials.py` |
| `ui/annotation.py` | 主界面、Trial 列表和波形标注 | `annotate_trials.py` |
| `ui/collection.py` | 串口/模拟源采集页面 | `collection_page.py` |
| `ui/mcu.py` | MCU 波形与分类结果展示 | `mcu_result_monitor.py` |
| `ui/realtime.py` | PC 实时模型推理界面和接收状态 | `realtime_monitor_app.py` |
| `ui/realtime_v.py` | 开启 V 协议的薄入口，复用上一个模块 | `realtime_monitor_app_v_format.py` |
| `monitoring/protocol.py` | PC 监测协议与 MCU 分类文本解析 | 从实时监测界面提取 |
| `monitoring/inference.py` | 窗口预处理、形状适配、量化和分数解释 | 从实时监测界面原样提取数值函数 |
| `training/inspect.py` | 标注数据列、标签与 Session 统计 | `inspect_labeled_data.py` |
| `training/binary.py` | 普通二分类及共用模型、评估、量化导出函数 | `train_impact_binary.py` |
| `training/event.py` | 事件窗口及恢复尾段排除、事件检测训练 | `train_impact_event_detector.py` |
| `training/experiments.py` | 固定 Session 划分的对比实验 | `run_impact_binary_experiments.py` |
| `signal/periodic.py` | 周期估计、模板和干扰抵消 | `periodic_filter.py` |
| `ui/period.py` | 周期干扰实验网页 | `period_noise_test.py` |
| `reporting/materials.py` | 已有模型结果的汇报图表 | `make_report_ai_materials.py` |

`training/event.py` 和 `training/experiments.py` 都复用 `training/binary.py` 的函数；修改这些函数会影响三条流程。此次结构重构保留了现有训练算法和输出目录。

## 早期三分类

`training/legacy/` 集中保存早期 Rest/Tap/HoldTouch 流程，避免和当前 Rest/Impact 模型混淆。

| 模块 | 旧文件 | 任务 |
| --- | --- | --- |
| `dataset.py`、`model.py` | 同名根目录模块 | 数据和模型库 |
| `build_annotations.py` | `build_dataset_from_annotations.py` | `build-annotations` |
| `demo.py` | `generate_demo_data.py` | `demo` |
| `train.py`、`evaluate.py` | 同名根目录脚本 | `train`、`evaluate` |
| `quantize.py` | `convert_int8.py` | `quantize` |
| `tflite_test.py` | `test_tflite.py` | `test-tflite` |
| `export_c.py` | `export_model_to_c.py` | `export-c` |
| `pipeline.py` | `update_neural_network.py` | `train-legacy` |

## 依赖方向

```text
cli → 独立子进程中的训练任务 / Streamlit 页面
ui.annotation → ui.collection → acquisition.serial_reader → acquisition.parser
ui.annotation → annotation.export → annotation.windows
ui.realtime → monitoring.protocol + monitoring.inference
training.event / training.experiments → training.binary
ui.period → signal.periodic
```

界面依赖计算和文件模块，纯协议与数值模块不反向依赖界面。快捷键 HTML 放在 `ui/components/keyboard_shortcuts/`，作为包资源分发。

## 本地运行目录

`data_raw/`、`annotations/`、`data_labeled/`、`data/`、`outputs/`、`artifacts/` 位于数据根目录，默认是仓库根目录。可设置 `PLANT_SIGNAL_ROOT` 指向其他数据根目录；已安装的非源码包默认使用当前工作目录。

历史根目录源码在本机 `archive/refactor_20261016/` 保存，两个 `submission_*/` 独立交付目录保留原位。它们被排除在 GitHub 源码包外，日常修改统一在 `src/` 中完成。旧的根目录 Python 命令已迁移，请使用 `run.py` 的任务名。

# 植物电信号采集与 CH32 神经网络工具

面向小组协作的 Python 工程：通过串口采集信号，在网页中标注 Trial，训练 Rest/Impact 分类模型并导出 INT8 TFLite / C 数组，最后查看 PC 或 CH32 MCU 的分类结果。

## 先运行起来

推荐从 Python 3.10 环境开始。已有 `tflm_env` 的本机可直接运行；新组员在自己的环境安装：

```powershell
python -m pip install -e ".[all]"
python run.py
python run.py annotate
python run.py mcu
```

本机 PowerShell 入口会优先使用现有 `tflm_env`，新克隆目录没有该环境时使用 PATH 中的 Python：

```powershell
.\run.ps1                  # 列出任务
.\run.ps1 annotate         # 采集和人工标注
.\run.ps1 monitor-v        # PC 模型推理，支持 V ... uV
.\run.ps1 train-event -DryRun
```

安装为可编辑包后也可使用 `plant-signal annotate` 或 `python -m plant_signal annotate`。仅需训练可安装 `.[train]`，仅需网页可安装 `.[ui]`；依赖统一定义在 `pyproject.toml`。

## 代码在哪里

```text
src/plant_signal/
  acquisition/       串口解析、后台采集、Session/Trial 文件管理
  annotation/        标签区间、窗口生成、标注数据导出
  training/          二分类、事件检测、对比实验、数据检查
    legacy/          早期三分类流程
  monitoring/        实时预处理、量化和监测协议解析
  signal/            周期干扰估计与抵消
  ui/                Streamlit 界面及快捷键组件
  reporting/         汇报图表生成
  config.py          共享配置和早期三分类参数
  paths.py           数据根目录解析
  cli.py             统一任务入口
firmware/            MCU 预处理示例
tests/               软件回归测试
scripts/             源码交付包生成工具
docs/                模块索引、训练、协议和发布说明
```

建议阅读顺序：`acquisition/parser.py` → `acquisition/trials.py` → `annotation/windows.py` → `training/event.py` → `monitoring/inference.py` → `ui/annotation.py`。详细职责和旧文件名对照见[代码索引](docs/PROJECT_STRUCTURE.md)。

## 日常流程

```text
串口 / 模拟源
  → 采集与标注界面
  → data_raw/ + annotations/trial_tags/
  → 完成标注后同步 data_labeled/
  → 数据检查、训练与 INT8 导出
  → outputs/ 中的模型及对应预处理配置
  → PC 实时推理 / CH32 MCU 集成
```

```powershell
python run.py annotate
python run.py inspect
python run.py train-event --epochs 80
python run.py monitor-v
```

`inspect` 默认只打印数据统计。采集、导出和训练任务会写入对应的数据/输出目录；`train-event` 会同时更新 `outputs/impact_event_detector/` 与 `outputs/impact_binary_best_event/`。

## 常用任务

| 命令 | 用途 |
| --- | --- |
| `annotate` | 采集、Session/Trial 管理和人工标注 |
| `export-labels` | 重新同步已完成标注的训练数据 |
| `inspect` | 检查标注数据 |
| `train-event` | Impact 事件检测训练；256 点原始窗口降采样至 128 点 |
| `train-binary` | 普通 128 点窗口二分类 |
| `experiments` | 对比窗口、降采样和类别权重等方案 |
| `monitor` / `monitor-v` | PC 加载 TFLite 实时推理；V 版额外支持 `V ... uV` |
| `mcu` | 显示 MCU 发来的波形与 `CLS/LABEL/CONF` 结果 |
| `period` | 周期干扰测量与模板抵消 |
| `report` | 基于已有结果生成汇报图表 |
| `train-legacy` | 完整早期三分类流程 |

其他三分类单步任务由 `python run.py` 列出。参数原样传给任务脚本：

```powershell
python run.py --dry-run train-binary --epochs 20 --out_dir outputs/my_trial
python run.py train-binary --help
.\run.ps1 train-binary -TaskArgs @('--epochs', '20', '--out_dir', 'outputs/my_trial')
```

Python 的 `--dry-run` 放在任务名前；PowerShell 使用 `-DryRun`。任务参数中的相对路径相对于项目数据根目录解析，默认是源码仓库根目录，可通过 `PLANT_SIGNAL_ROOT` 环境变量指定其他数据根目录。

## 数据、模型与协作

真实数据、标注、训练模型和本机环境保留在本地，GitHub 源码包默认不包含这些内容。新克隆目录需要自行采集数据，或由小组另行共享数据与模型。标注界面也支持模拟源。

采集工具当前解析 `CH0 Voltage=... uV`。两个 PC 监测入口兼容更多协议，详见[串口格式](docs/SERIAL_PROTOCOL.md)。普通二分类、事件二分类和早期三分类的预处理不同，部署时应携带同一轮训练的全部配置，详见[训练说明](docs/TRAINING.md)。

提交前运行：

```powershell
python -B -m unittest discover -s tests -t . -v
python scripts/prepare_release.py --check
python scripts/prepare_release.py
```

最后一条命令生成 `dist/plant-signal-source.zip` 及哈希清单，便于直接发给组员或准备 GitHub 仓库。协作方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，发布范围见[GitHub 准备说明](docs/GITHUB.md)，本次检查结果见[验证记录](docs/VALIDATION.md)。

本工程用于实验分析。演示数据仅用于检查软件流程；真实分类效果需要按独立 Session 评估，并通过实际采集和 MCU 部署验证。

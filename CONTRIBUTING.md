# 小组协作说明

先从 [README](README.md) 启动需要的任务，再按[代码索引](docs/PROJECT_STRUCTURE.md)定位对应模块。

## 环境

在自己的 Python 环境安装 `python -m pip install -e ".[all]"`。已有 `tflm_env` 的本机直接复用现有环境；本次重构没有安装依赖或修改该环境。

运行 `python run.py` 查看任务，使用 `python run.py <任务> --help` 查看支持命令行参数的训练/数据脚本。网页参数通过界面设置；Streamlit 参数如 `--server.port 8502` 可由启动入口转发。

## 如何分工

| 工作 | 主要目录 |
| --- | --- |
| 数据采集、串口问题、Trial 保存 | `src/plant_signal/acquisition/` |
| 标签规则、导出数据和窗口划分 | `src/plant_signal/annotation/` |
| 模型、训练、阈值和实验对比 | `src/plant_signal/training/` |
| PC 推理和模型输入/输出预处理 | `src/plant_signal/monitoring/` |
| 页面布局和交互 | `src/plant_signal/ui/` |
| MCU 接入 | `firmware/`、`docs/SERIAL_PROTOCOL.md` 和训练输出配置 |

`ui/realtime.py` 是普通协议与 V 协议监测的共用实现；`ui/realtime_v.py` 只负责开启额外协议。修改共用逻辑时需要检查两个入口。

## 提交习惯

1. 每个修改聚焦一个功能，在独立分支提交，说明实际变化。
2. 改动标注、预处理或协议时，增加能覆盖边界条件的测试。
3. 执行 `python -B -m unittest discover -s tests -t . -v`，并启动涉及的任务检查界面或参数。
4. 执行 `python scripts/prepare_release.py --check`，查看 `git diff` 与 `git status` 后提交。
5. 在 PR 中列明实际验证范围；软件测试与硬件运行分别记录。

实验数据、标注、模型、日志、虚拟环境、历史快照通过独立渠道共享。它们由 `.gitignore` 排除；发布 ZIP 另用明确的源码清单生成。不要把源码仓库整个目录直接压缩后当作公开交付包。

项目暂未替所有成员选择开源许可证；如需授权第三方复用，由小组确认后加入相应许可证。

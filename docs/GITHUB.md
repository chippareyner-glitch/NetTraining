# GitHub 与组员交付

## 本次发布范围

源码仓库包含 `src/`、`tests/`、`docs/`、`firmware/`、`scripts/`、统一启动入口、依赖配置和协作说明。

本机以下内容保留原位，但不进入源码交付包：真实数据与标注、训练模型、实验图表、日志、虚拟环境、旧源码/ZIP 归档、两个独立 `submission_*/` 交付快照以及本机代理设置目录。

不附带真实数据和模型时，新组员可运行标注界面的模拟源，也可运行早期 `demo` 流程检查工具链；复现实验结果需要小组另行提供相应数据、模型、Session 划分和配置。

## 生成交付包

```powershell
python -B -m unittest discover -s tests -t . -v
python scripts/prepare_release.py --check
python scripts/prepare_release.py
```

得到：

```text
dist/plant-signal-source.zip
dist/plant-signal-source.manifest.json
```

脚本只从明确的源码文件/目录清单收集文件，扫描常见凭据模式及本机用户/项目路径，并在打包后核对 ZIP 内每个文件的 SHA-256。检查输出只列文件、行号和模式类型，不打印可能的凭据值。

这是针对所选源码的静态检查，不代表已经检查所有可能的个人信息，也不检查被排除的本地数据或旧档案。上传前仍需查看实际文件清单和差异。

## 上传源码

发布脚本只生成本地源码包，不会自动创建远程仓库、提交或推送。上传时使用经过检查的源码包或其解压目录；不要把本机完整工作目录直接作为上传包。

上传已有仓库时保留原 Git 历史，在独立工作分支审查更新。许可证由小组另行确认，本次没有代为选择。

## 本机重构记录

整理前快照及校验记录位于被忽略的 `archive/`。本次重构前原文件保存于 `archive/refactor_20261016/before_refactor.zip` 与 `flat_layout/`。数据、标注、现有模型与独立交付目录未迁移；它们与快照记录逐文件核对。最终本地验证结果记录在该归档目录内，不包含进对外源码包。

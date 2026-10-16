# 早期三分类流程

此流程保留 Rest/Tap/HoldTouch 三分类，源码在 `src/plant_signal/training/legacy/`，模型输出到 `artifacts/`。当前 Rest/Impact 训练说明见 [TRAINING.md](TRAINING.md)。

## 演示流程

以下命令会生成演示数据并更新三分类模型产物：

```powershell
python run.py demo
python run.py train
python run.py evaluate
python run.py quantize
python run.py test-tflite
python run.py export-c
```

演示 CSV 位于 `data/plant_demo.csv`，仅用于检查训练、量化和导出流程。输入格式为每行一个 128 点窗口：`session_id,label,p0,...,p127`，类别为 `0=Rest`、`1=Tap`、`2=HoldTouch`。

## 真实窗口数据

```powershell
python run.py build-tags
python run.py train --csv data/plant_real_dataset.csv
```

早期 CSV 标注可通过 `build-annotations` 生成窗口。`train-legacy` 会串联已完成标注导出、窗口生成、训练、评估、INT8 转换和 C 导出，并更新现有三分类产物。

预处理为每窗口减均值，再除以训练集计算的固定 scale；配置写入 `artifacts/preprocess.json`。导出的 C 文件为 `plant_model_data.cc/.h`。`firmware/mcu_preprocess_example.c` 对应这条早期流程，需要替换为同轮模型的实际 scale 与量化参数。

早期 `config.py` 默认采样率为 500 Hz，实时监测默认估计值为 220 Hz。使用时需要匹配采集条件和模型配置。

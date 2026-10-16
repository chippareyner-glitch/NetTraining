# 串口格式

串口号由使用者选择或填写，不在源码中固定为某台电脑的端口。默认波特率为 115200。

## 采集与标注

`acquisition/parser.py` 当前接受的采样文本：

```text
CH0 Voltage=886218.562500 uV
```

数值保存到 `voltage_uV`。采集过程中先写 `trial_*.partial.csv`，完成保存后写为正式 Trial CSV。Session/Trial 文件由 `acquisition/trials.py` 管理。

标注和导出能读取部分旧 CSV 的 `current_uA` 或 `adc_raw` 列，这不表示当前采集解析器接受旧串口文本。

## 监测界面

| 入口 | 支持的内容 |
| --- | --- |
| `python run.py mcu` | `CH0 Voltage=... uV` 和 MCU 分类行 |
| `python run.py monitor` | `A0=...uA`、`CH0 Voltage=... uV` 和 MCU 分类行 |
| `python run.py monitor-v` | 上述内容，以及 `V     12345 uV` / `V      -123 uV` |

MCU 分类行示例：

```text
CLS=0,LABEL=REST,CONF=81
CLS=1,LABEL=IMPACT,CONF=127
```

`CONF` 按 MCU 原始整数展示，不能直接假设它是百分比。`mcu` 入口显示 MCU 已计算的分类；两个 `monitor` 入口另可加载 PC 端 TFLite 模型进行推理。

监测文本解析位于 `monitoring/protocol.py`，不依赖 Streamlit 或串口硬件。普通协议与 V 协议共享实现，额外的 V 格式仅在 `monitor-v` 中开启。

# 第 0 阶段运行说明

在仓库根目录执行：

```bash
PYTHONPATH=src python3 -m thermotaxis validate \
  --config configs/phase0_reference.json

PYTHONPATH=src python3 -m thermotaxis run \
  --config configs/phase0_reference.json \
  --output-root results/phase0
```

第二个命令生成由配置摘要命名的目录，其中包括：

- `config.resolved.json`：完整解析后的模型配置；
- `manifest.json`：Git、Python、平台和生成时间；
- `result.json`：固定位置的本地传感器探针结果与独立时钟计数。

该探针只验证第 0 阶段的实验契约和可复现性。它没有运动和控制器，不能作为趋温结果引用。

运行新核心测试：

```bash
PYTHONPATH=src python3 -m pytest -q tests
```

运行合并后的旧平台测试：

```bash
python3 -m pytest -q 程序/test_core.py
```

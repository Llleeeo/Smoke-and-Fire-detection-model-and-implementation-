# YOLOv5 数据修复包

这个目录用于修复原始导出中的数据泄漏、重复图片、错误数据路径和伪装成 JPEG 的 HEIC 文件。原始目录不会被修改。

## 目录

- `dataset/`：重新按原始图片来源分组后的训练、验证和测试数据。
- `audit/`：清洗摘要、入选/移除清单、验证输出和数据质量报告。
- `scripts/`：可重复运行的清洗和验证程序。
- `original_model_reference/`：同伴提供的旧模型与训练结果；指标来自旧划分，只作参考。
- `yolov5/`：与旧权重中记录的 Git 提交一致的 Ultralytics YOLOv5 源码。

GitHub 版本不会提交数据集图片/标签、重复模型权重、虚拟环境和训练输出。运行 `setup_environment.sh` 时会自动下载固定提交 `3fb11111c6a8088fbc91430a1f99d207c16f0620` 的 YOLOv5；本机生成的数据仍保留在原目录中。

## 重要说明

旧的 `best.pt` 可以继续用于试跑推理，但不能代表清洗后数据上的最终模型。要得到可信指标，必须使用 `dataset/data.yaml` 重新训练，并在新的 `test` 分组上评估。

## 验证

```bash
python3 scripts/verify_dataset.py dataset
```

## 训练

先建立 Python 环境：

```bash
./setup_environment.sh
source .venv/bin/activate
```

再运行：

```bash
./train_clean.sh
```

仓库根目录存在旧的 `best.pt` 时，默认从它继续微调；否则从 `yolov5s.pt` 开始。也可以设置 `INITIAL_WEIGHTS=/path/to/weights.pt`。可在命令末尾添加 YOLOv5 参数，例如 `--device 0 --batch 32`。

## 推理

```bash
./infer_clean.sh /path/to/image-or-video
```

默认使用同伴提供的旧权重。重新训练后可设置：

```bash
WEIGHTS=/path/to/new/best.pt ./infer_clean.sh /path/to/image-or-video
```

## 独立测试

用旧权重查看清洗测试集上的基线，或把 `WEIGHTS` 换成重新训练后的权重：

```bash
./evaluate_clean.sh
WEIGHTS=/path/to/new/best.pt ./evaluate_clean.sh
```

## 清洗规则

- 先以 `.rf.` 前的文件名识别同源图片，再用 64 位感知哈希（汉明距离不超过 4）合并不同文件名的连拍或近重复画面。
- 按标签类别组合分层，将感知相似组稳定地分为 80%/10%/10%。
- 同一来源不会跨 train/valid/test。
- 每个感知相似组只保留一张标注最完整的代表图；YOLOv5 训练时再进行在线增强。
- 空标签图片作为刻意设计的负样本保留；抽样检查显示其主要是笔、铅笔等香烟相似物。
- HEIC 内容但扩展名为 `.jpg`/`.JPG` 的文件转换为真正的 JPEG。

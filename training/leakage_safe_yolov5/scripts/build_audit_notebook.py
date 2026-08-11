#!/usr/bin/env python3
"""Create the reader-facing, executable data-quality audit notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    summary = json.loads((package_root / "audit" / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((package_root / "audit" / "verification.json").read_text(encoding="utf-8"))
    output = summary["output"]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# YOLOv5 数据质量修复审计\n\n"
            "## tl;dr\n\n"
            f"- 清洗后共有 **{output['train']['images']:,} / {output['valid']['images']:,} / {output['test']['images']:,}** 张训练、验证和测试图像。\n"
            "- 三组之间的原始来源重叠和逐字节图像重复均为 **0**。\n"
            f"- 共移除 {summary['checks']['removed_records']:,} 个重复或同来源多余版本，并把 "
            f"{sum(output[s].get('heic_converted', 0) for s in ('train', 'valid', 'test'))} 个伪装成 JPEG 的 HEIC 文件转换为标准 JPEG。\n"
            "- 旧模型指标仍只能作参考；可信的模型结论需要在清洗数据上重新训练。"
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "目标是修复跨分组泄漏、重复图片、错误图片编码和不可直接运行的数据配置。"
            "拆分单位是 Roboflow 文件名中 `.rf.` 之前的原始来源名称；按来源所含类别组合分层，"
            "稳定分为 80%/10%/10%。训练集保留不同增强版本，验证和测试每个来源只保留一张代表图。\n\n"
            "### Key Assumptions\n\n"
            "- `.rf.` 之前的名称代表同一张原始图片。抽样图像支持这一假设。\n"
            "- 空标签图是用于抑制误报的负样本。抽样中主要为笔、铅笔等香烟相似物。\n"
            "- 旧训练结果没有在清洗后的测试集上重新计算。"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n加载清洗摘要和独立验证输出。"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n\n"
            "package_root = Path.cwd()\n"
            "summary = json.loads((package_root / 'audit' / 'summary.json').read_text(encoding='utf-8'))\n"
            "verification = json.loads((package_root / 'audit' / 'verification.json').read_text(encoding='utf-8'))\n"
            "summary['generated_at'], verification['passed']"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n先比较清洗前后数据量，再核对泄漏和标签完整性。"),
        nbf.v4.new_code_cell(
            "rows = []\n"
            "for split in ('train', 'valid', 'test'):\n"
            "    before = summary['input']['split_counts'][split]\n"
            "    after = summary['output'][split]\n"
            "    rows.append({\n"
            "        'split': split,\n"
            "        'before_images': before['images'],\n"
            "        'after_images': after['images'],\n"
            "        'after_sources': after['sources'],\n"
            "        'after_annotations': after['annotations'],\n"
            "        'after_empty_labels': after['empty_labels'],\n"
            "    })\n"
            "pd.DataFrame(rows)"
        ),
        nbf.v4.new_code_cell(
            "pd.DataFrame([\n"
            "    {'split_pair': pair, **counts}\n"
            "    for pair, counts in verification['overlaps'].items()\n"
            "]), verification['failures']"
        ),
        nbf.v4.new_code_cell(
            "pd.DataFrame([\n"
            "    {\n"
            "        'split': split,\n"
            "        'cigarette': values.get('class_0', 0),\n"
            "        'flame': values.get('class_1', 0),\n"
            "        'smoke': values.get('class_2', 0),\n"
            "        'boxes': values.get('box', 0),\n"
            "        'polygons': values.get('polygon', 0),\n"
            "    }\n"
            "    for split, values in summary['output'].items()\n"
            "])"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- 数据层面的关键问题已经修复：配对完整、坐标合法、图片可解码、跨分组来源和哈希均隔离。\n"
            "- 评估集按原始来源计数，而不是按增强图片计数，因此规模变小但可信度更高。\n"
            "- 空标签负样本被保留；若标注规范并非有意加入相似物负样本，应再由标注负责人确认。\n"
            "- 下一步应重新训练并单独运行 `val.py --task test`，旧的 mAP 不应继续作为最终结论。"
        ),
    ]
    notebook_path = package_root / "audit" / "data_quality_audit.ipynb"
    nbf.write(notebook, notebook_path)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(package_root)}})
    client.execute()
    nbf.write(notebook, notebook_path)
    print(notebook_path)


if __name__ == "__main__":
    main()

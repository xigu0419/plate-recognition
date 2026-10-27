# 多场景车牌识别系统

基于 **Python 3.9 + OpenCV + PaddleOCR** 的图片车牌识别程序，带 Tkinter 图形界面，也可命令行批量识别。项目包含一套在 CCPD 数据集上可复现的评测流程。

## 下载

Windows 免安装版（内置 PaddleOCR 模型，解压后双击 `车牌识别工具.exe`）：
[Releases · v3.0](https://github.com/xigu0419/plate-recognition/releases/tag/v3.0)

## 功能

- 单张图片的车牌定位与识别，输出车牌号、置信度、是否合法
- 图形界面：选择图片 → 开始识别 → 查看标注图与结果
- 命令行：`python plate_recognition.py 图片.jpg`
- 中文路径兼容、自动切换 Python 环境、错误日志记录
- 评测脚本：CCPD 6000 张真实街景图片，支持定位/识别分阶段统计

## 效果（CCPD test 500 张）

| 指标 | 结果 |
|---|---|
| 端到端整牌正确率 | 99.0%（495 / 500） |
| 定位命中率（IoU ≥ 0.5） | 100% |
| 字符正确率 | 99.8% |
| 单张识别耗时 | 约 133 ms（不含模型加载） |

关键改进（详见 `docs/实验报告_数据集评测.md`）：

- 定位由 HSV 颜色阈值改为 PaddleOCR 文字检测，IoU ≥ 0.5 命中率 20.7% → 100%
- 已裁好的车牌条跳过检测模型、直连识别模型，正确率 95.4% → 100%，耗时 1625 ms → 133 ms
- 修复倾斜车牌被过滤条件误杀；同一车牌重复上报率 7.3% → 0%

## 目录结构

| 路径 | 说明 |
|---|---|
| `plate_recognition.py` | 主程序（GUI + CLI + 识别流程） |
| `models/` | PaddleOCR 检测 / 识别 / 方向分类推理模型（离线可用） |
| `docs/` | 使用说明与数据集评测报告 |
| `evaluate_ccpd.py` | 在 CCPD 上评测整牌正确率、定位命中率、字符正确率 |
| `prepare_ccpd.py` / `check_dataset.py` | 数据集解析与划分检查 |
| `summary_report.py` | 由评测结果生成 Markdown 报告 |
| `crop_plate.py` | 车牌区域裁剪工具 |
| `build_share.py` / `PlateApp.spec` | PyInstaller 打包脚本 |
| `requirements.txt` | 依赖版本 |

## 快速开始

环境建议 Python 3.9（与 `paddlepaddle==2.6.2` 匹配）：

```bash
pip install -r requirements.txt
python plate_recognition.py            # 打开图形界面
python plate_recognition.py 图片.jpg    # 命令行识别
```

Windows 下也可以直接双击 `启动程序.bat`；把图片拖到 `识别车牌.bat` 上可快速识别。

## 数据与隐私

- 仓库**不包含** CCPD 数据集、测试图片和任何车牌照片；评测脚本默认读取项目根目录下的 `ccpd_data/`。
- `models/` 中的 OCR 模型来自 PaddleOCR（Apache-2.0），随仓库提供以便离线运行。
- 打包生成的 `build/`、`dist/` 体积较大，未上传；需要分享版可按 `build_share.py` 自行打包。

## 说明

个人学习项目，代码按“能跑、可复现”为目标组织，未做工程化重构。使用前请把 `docs/` 中示例命令里的本地路径替换成你自己的路径。

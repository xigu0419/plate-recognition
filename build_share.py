# -*- coding: utf-8 -*-
"""
一键打包脚本 —— 生成可分发的「车牌识别工具」文件夹
====================================================
运行环境: conda 环境 plate (Python 3.9 + PaddleOCR)

用法:
    conda activate plate
    cd /d D:\\CPP\\plate_recognition
    python build_share.py

产物:
    dist\\PlateApp\\                     打包原始输出
    dist\\车牌识别工具_分享版\\           可直接发给别人的分享包
        ├─ 车牌识别工具.exe
        ├─ _internal\\                 运行依赖 + 内置OCR模型
        ├─ 示例图片\\
        ├─ 使用说明.txt
        └─ 双击启动.bat

注意: 若 PaddleOCR 环境路径不同, 请修改下面的 CONDA_ENV 变量。

—— 打包踩过的坑(重要, 勿删) ——
1. _ctypes 加载失败
   原因: conda 版 Python 的 _ctypes.pyd 依赖 Library\\bin\\ffi.dll,
        PyInstaller 不会自动收集。
   解决: 把 Library\\bin 下所有 DLL 用 --add-binary 全部打包。

2. imghdr / imgaug / lmdb 等模块缺失
   原因: paddleocr 内部使用 `import tools.xxx` / `from ppocr.xxx import`
        这类顶层导入, PyInstaller 无法静态分析, 导致其依赖全部漏收集。
   解决: 用 --collect-all / --hidden-import 显式补齐(见下方列表)。

3. importlib.metadata.PackageNotFoundError: imageio
   原因: 打包后缺少包的 dist-info 元数据。
   解决: --copy-metadata, 同时程序内做了运行时兜底补丁。

4. 中文路径下模型打不开
   原因: Paddle 的 C++ 推理接口不支持非 ASCII 路径。
   解决: 程序启动时自动把模型复制到临时 ASCII 目录(见 ascii_safe_models)。
"""
import os
import shutil
import subprocess
import sys

# ============ 配置 ============
CONDA_ENV = r"C:\Users\19500\.conda\envs\plate"      # conda 环境路径
PY = os.path.join(CONDA_ENV, "python.exe")
LIB_BIN = os.path.join(CONDA_ENV, "Library", "bin")  # conda 的 C 库目录
PROJECT = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "PlateApp"

# 需要显式补齐的第三方依赖
HIDDEN_IMPORTS = [
    "imghdr", "imgaug", "lmdb", "scipy", "six", "requests", "tqdm", "visualdl",
    "openpyxl", "pdf2docx", "premailer", "attrdict", "fire", "flask", "babel",
    "fontTools", "matplotlib", "imageio", "shapely", "pyclipper", "rapidfuzz",
    "skimage", "yaml", "lxml", "bs4", "cssutils", "docx", "fitz", "contourpy",
    "cycler", "kiwisolver", "dateutil", "pytz", "pandas", "networkx", "opt_einsum",
    "decorator", "psutil", "termcolor", "tifffile", "lazy_loader", "more_itertools",
    "importlib_metadata", "zipp", "Crypto", "crc32c", "rarfile",
]
COPY_METADATA = ["imageio", "imgaug", "paddleocr", "scipy"]
COLLECT_ALL = ["paddleocr", "paddle", "imgaug", "imageio", "lmdb",
               "pyclipper", "shapely"]

# 必须排除的模块 —— 否则分享包会白白胖一大圈
#
# 背景: 训练检测模型时往 conda 的 plate 环境里装过 PyTorch(用于跑实验),
# PyInstaller 的依赖分析会把它一起收进来, 光 torch 目录就有 311 MB,
# 分享包从 705 MB 直接涨到 1028 MB。
# 但本程序运行时**完全不需要** torch: 识别用的是 PaddleOCR, 定位用的是
# PaddleOCR 自带的检测模型。所以这里显式排除, 把体积压回去。
EXCLUDE_MODULES = [
    "torch", "torchvision", "torchaudio", "torchtext",
    "sympy", "mpmath", "fsspec", "filelock",
    "ultralytics", "tqdm.notebook",
]

SHARE_DIR_NAME = "车牌识别工具_分享版"


def run_build():
    if not os.path.exists(PY):
        print(f"[错误] 找不到 Python: {PY}")
        print("       请修改脚本顶部的 CONDA_ENV 变量")
        return False

    if not os.path.isdir(os.path.join(PROJECT, "models")):
        print("[错误] 缺少 models 目录(内置OCR模型)")
        print("       请把 PaddleOCR 的模型放到 models/ 下, 结构:")
        print("       models/det/ch/ch_PP-OCRv4_det_infer/")
        print("       models/rec/ch/ch_PP-OCRv4_rec_infer/")
        print("       models/cls/ch_ppocr_mobile_v2.0_cls_infer/")
        return False

    cmd = [PY, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--onedir", "--windowed", "--name", APP_NAME]

    # 关键1: 打包 Library\bin 下全部 DLL(修复 _ctypes)
    n_dll = 0
    if os.path.isdir(LIB_BIN):
        for f in os.listdir(LIB_BIN):
            if f.lower().endswith(".dll"):
                cmd += ["--add-binary", os.path.join(LIB_BIN, f) + ";."]
                n_dll += 1
    print(f"[信息] 将打包 {n_dll} 个 DLL")

    # 关键2: 内置 OCR 模型
    cmd += ["--add-data", "models;models"]

    # 关键3: 补齐隐藏导入
    for m in HIDDEN_IMPORTS:
        cmd += ["--hidden-import", m]
    for m in COPY_METADATA:
        cmd += ["--copy-metadata", m]
    for m in COLLECT_ALL:
        cmd += ["--collect-all", m]
    cmd += ["--collect-submodules", "skimage"]
    cmd += ["--hidden-import", "PIL._tkinter_finder"]

    # 关键4: 排除训练用的大包(torch 等), 否则体积白白涨 300+ MB
    for m in EXCLUDE_MODULES:
        cmd += ["--exclude-module", m]

    cmd.append("plate_recognition.py")

    print("[信息] 开始打包(约需 1-3 分钟)...")
    r = subprocess.run(cmd, cwd=PROJECT)
    if r.returncode != 0:
        print("[警告] PyInstaller 返回非 0, 请检查输出")

    if not os.path.exists(os.path.join(PROJECT, "dist", APP_NAME)):
        print("[错误] 打包失败, 未生成 dist 目录")
        return False
    return True


def make_share():
    """生成分享版文件夹"""
    src = os.path.join(PROJECT, "dist", APP_NAME)
    share = os.path.join(PROJECT, "dist", SHARE_DIR_NAME)

    print("[信息] 生成分享版文件夹 ...")
    if os.path.isdir(share):
        shutil.rmtree(share)
    shutil.copytree(src, share)

    # exe 改中文名
    old = os.path.join(share, APP_NAME + ".exe")
    new = os.path.join(share, "车牌识别工具.exe")
    if os.path.exists(old):
        os.rename(old, new)

    # 清理日志
    lg = os.path.join(share, "错误日志.txt")
    if os.path.exists(lg):
        os.remove(lg)

    # 示例图片
    img_dir = os.path.join(share, "示例图片")
    os.makedirs(img_dir, exist_ok=True)
    sample = os.path.join(PROJECT, "sample_images", "OIP-C.jpg")
    if os.path.exists(sample):
        shutil.copy2(sample, os.path.join(img_dir, "示例车牌.jpg"))

    # 启动器
    # 注意: 必须用 UTF-8 写, 不能用 ascii —— 文件名"车牌识别工具.exe"是中文,
    # 用 ascii 会在写文件时直接抛 UnicodeEncodeError。
    # 配合第 2 行的 chcp 65001, cmd 才能正确解析中文文件名。
    bat = os.path.join(share, "双击启动.bat")
    with open(bat, "w", encoding="utf-8") as f:
        f.write("@echo off\r\nchcp 65001 >nul\r\ncd /d \"%~dp0\"\r\n")
        f.write("start \"\" \"%~dp0车牌识别工具.exe\"\r\nexit\r\n")

    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(share) for f in fs) / 1024 / 1024
    print("=" * 50)
    print("✅ 打包完成")
    print(f"   分享包: {share}")
    print(f"   体积  : {size:.0f} MB")
    print("   把该文件夹整体压缩后发给别人即可(对方无需装 Python)")
    print("=" * 50)
    return True


def main():
    print("智能车牌识别系统 —— 打包工具")
    print("=" * 50)
    if run_build():
        make_share()


if __name__ == "__main__":
    main()

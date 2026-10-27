# -*- coding: utf-8 -*-
"""
批量评测脚本 —— 测试车牌识别的准确率
======================================
用法:
    1. 把测试图片放到 test_images/ 文件夹
    2. 运行: python evaluate.py

文件名约定(可选, 但强烈建议):
    <车牌号>_<场景说明>.jpg
    例如:  鲁Q67858_正面特写.jpg
           京A12345_整车前脸.jpg
    文件名开头的车牌号会被当作"正确答案", 用来自动统计准确率。

输出:
    - 每张图的识别结果、耗时、是否正确
    - 总体准确率统计
    - 标注图保存到 test_images/结果/ 下
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plate_recognition as pr

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(HERE, "test_images")
OUT_DIR = os.path.join(TEST_DIR, "结果")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 从文件名开头提取车牌号作为标准答案
PLATE_IN_NAME = re.compile(r"^([\u4e00-\u9fa5][A-Za-z][A-Za-z0-9]{5,6})")


def expected_from_name(fname):
    m = PLATE_IN_NAME.match(os.path.basename(fname))
    if m:
        return pr.clean_text(m.group(1)).upper()
    return None


def main():
    if not os.path.isdir(TEST_DIR):
        print("[错误] 找不到 test_images 文件夹")
        return

    files = sorted(f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith(EXTS))
    if not files:
        print("[提示] test_images 文件夹里还没有图片")
        print("       请把测试用的车牌照片放进去, 然后再运行本脚本")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    if not pr._PADDLE_AVAILABLE:
        print("[错误] OCR 引擎不可用, 请用 plate 环境的 Python 运行")
        print("       或双击 启动程序.bat 的方式运行")
        return

    print("=" * 72)
    print("  车牌识别批量评测")
    print("=" * 72)
    print("测试图片: %d 张, 目录: %s" % (len(files), TEST_DIR))
    print()

    total = 0
    correct = 0
    no_label = 0
    total_time = 0.0

    for f in files:
        path = os.path.join(TEST_DIR, f)
        exp = expected_from_name(f)

        t0 = time.time()
        try:
            plates, ann = pr.recognize_all(path)
        except Exception as e:
            print("[异常] %-34s -> %s" % (f, e))
            continue
        dt = time.time() - t0
        total_time += dt
        total += 1

        got = plates[0]["text"] if plates else ""
        all_texts = " | ".join(p["text"] for p in plates) if plates else "(无)"

        # 保存标注图
        try:
            out = os.path.join(OUT_DIR, os.path.splitext(f)[0] + "_结果.jpg")
            pr.imwrite_any(out, ann)
        except Exception:
            pass

        if exp:
            ok = (got == exp)
            if ok:
                correct += 1
            mark = "✅ 正确" if ok else "❌ 错误"
            print("%-32s %s   识别=%-12s 期望=%-12s %.1fs"
                  % (f, mark, got or "(空)", exp, dt))
            if len(plates) > 1:
                print("%34s 多个: %s" % ("", all_texts))
        else:
            no_label += 1
            print("%-32s ⚪ 无标注  识别=%-12s  %.1fs" % (f, all_texts, dt))

    print()
    print("=" * 72)
    print("  统计结果")
    print("=" * 72)
    print("  共测试      : %d 张" % total)
    if total - no_label > 0:
        acc = correct / (total - no_label) * 100
        print("  有标注      : %d 张" % (total - no_label))
        print("  识别正确    : %d 张" % correct)
        print("  ✅ 准确率   : %.1f%%" % acc)
    if no_label:
        print("  未标注      : %d 张(需人工核对)" % no_label)
    if total:
        print("  平均耗时    : %.2f 秒/张" % (total_time / total))
    print("  标注图目录  : %s" % OUT_DIR)
    print("=" * 72)


if __name__ == "__main__":
    main()

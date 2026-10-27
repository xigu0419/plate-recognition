# -*- coding: utf-8 -*-
"""
evaluate_ccpd.py —— 用 CCPD 数据集评测车牌识别管线。

跑法(必须用装了 PaddleOCR 的 plate 环境):
  python evaluate_ccpd.py --split test --mode gt-crop   # 只测"识别",用真值角点矫正
  python evaluate_ccpd.py --split test --mode full      # 测完整管线(定位+识别)
  python evaluate_ccpd.py --split test --limit 30       # 只跑 30 张

两种模式的区别:
  gt-crop  用数据集给的四角点精准矫正后送 OCR  => 衡量"识别能力"上限
  full     完全走程序自己的颜色定位 + 矫正 + 识别 => 衡量真实端到端效果

两个模式跑出来的差距,就说明瓶颈在"定位"还是在"识别"。

输出:
  ccpd_data/eval_<split>_<mode>.jsonl   逐条结果(可断点续跑)
  ccpd_data/eval_<split>_<mode>.md      Markdown 报告
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import plate_recognition as pr          # noqa: E402

DATA_DIR = os.path.join(HERE, "ccpd_data")
IMG_DIR = os.path.join(DATA_DIR, "images")


# ---------------------------------------------------------------- 基础工具
def edit_distance(a, b):
    """Levenshtein 编辑距离"""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,                 # 删除
                         cur[j - 1] + 1,               # 插入
                         prev[j - 1] + (ca != cb))     # 替换
        prev = cur
    return prev[-1]


def charset_sim(gt, pred):
    """字符级正确率(1 - 编辑距离/长度)"""
    if not gt:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - edit_distance(gt, pred) / float(len(gt)))


def norm(s):
    """去空白、统一大写"""
    return "".join((s or "").split()).upper()


def warp_gt(img, corners):
    """按真值四角点做透视矫正(gt-crop 模式用)"""
    pts = np.array(corners, dtype="float32")
    tl, tr, br, bl = pts
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if w < 8 or h < 4:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype="float32")
    m = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- 单张评测
def run_one(rec, mode):
    path = os.path.join(IMG_DIR, rec["file"])
    img = pr.imread_any(path)
    if img is None:
        return None

    gt = norm(rec["plate"])
    t0 = time.time()

    if mode == "gt-crop":
        roi = warp_gt(img, rec["corners"]) if rec.get("corners") else None
        if roi is None:
            return {"file": rec["file"], "gt": gt, "pred": "", "conf": 0.0,
                    "exact": False, "charsim": 0.0, "localized": True,
                    "n_cands": 0, "ms": 0, "blur": rec.get("blur"),
                    "brightness": rec.get("brightness"), "error": "矫正失败"}
        text, conf = pr.recognize_text(roi)
        pred = norm(text)
        return {"file": rec["file"], "gt": gt, "pred": pred, "conf": conf,
                "exact": (pred == gt), "charsim": charset_sim(gt, pred),
                "localized": True, "n_cands": 1,
                "ms": int((time.time() - t0) * 1000),
                "blur": rec.get("blur"), "brightness": rec.get("brightness")}

    # ---- full: 完整管线 ----
    # 先单独量一次"定位命中":有没有候选框中心落在真值车牌框内
    localized = False
    n_cands = 0
    try:
        cands = pr.detect_plate_candidates(img)
        n_cands = len(cands)
        bb = rec.get("bbox")
        if bb:
            x1, y1, x2, y2 = bb
            for c in cands:
                cx, cy = c["center"]
                if x1 - 12 <= cx <= x2 + 12 and y1 - 12 <= cy <= y2 + 12:
                    localized = True
                    break
    except Exception:
        pass

    plates, _ = pr.recognize_all(path)
    pred = norm(plates[0]["text"]) if plates else ""
    conf = plates[0]["conf"] if plates else 0.0
    return {"file": rec["file"], "gt": gt, "pred": pred, "conf": conf,
            "exact": (pred == gt), "charsim": charset_sim(gt, pred),
            "localized": localized, "n_cands": n_cands,
            "n_plates": len(plates),
            "ms": int((time.time() - t0) * 1000),
            "blur": rec.get("blur"), "brightness": rec.get("brightness")}


# ---------------------------------------------------------------- 分箱统计
def bin_of(val, edges, names):
    if val is None:
        return "未知"
    for e, nm in zip(edges, names):
        if val <= e:
            return nm
    return names[-1]


BLUR_BINS = ([30, 100, 200], ["清晰", "一般", "模糊", "极糊"])
BRIGHT_BINS = ([50, 100, 150], ["很暗", "偏暗", "正常", "偏亮"])


def report_bins(results, key, bins, title):
    edges, names = bins
    groups = {}
    for r in results:
        nm = bin_of(r.get(key), edges, names)
        g = groups.setdefault(nm, [0, 0, 0.0])
        g[0] += 1
        g[1] += 1 if r["exact"] else 0
        g[2] += r["charsim"]
    print("\n按%s分箱:" % title)
    for nm in names:
        if nm not in groups:
            continue
        total, ok, cs = groups[nm]
        print("  %-4s : %4d 张   整牌正确 %5.1f%%   字符正确 %5.1f%%" % (
            nm, total, 100.0 * ok / total, 100.0 * cs / total))
    return groups


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--mode", default="gt-crop", choices=["full", "gt-crop"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张")
    ap.add_argument("--labels", default=os.path.join(DATA_DIR, "labels.jsonl"))
    args = ap.parse_args()

    recs = []
    with open(args.labels, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if args.split != "all":
        recs = [r for r in recs if r.get("split") == args.split]
    if args.limit:
        recs = recs[:args.limit]

    # 只评测已经下载好的图片
    have = [r for r in recs if os.path.exists(os.path.join(IMG_DIR, r["file"]))]
    missing = len(recs) - len(have)
    print("待评测 %d 张 (%s / %s),其中 %d 张图片还没下载完,先跳过"
          % (len(have), args.split, args.mode, missing))
    if not have:
        print("没有可评测的图片,先等下载完成")
        return

    out_path = os.path.join(DATA_DIR, "eval_%s_%s.jsonl" % (args.split, args.mode))
    done = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["file"]] = r
                except Exception:
                    pass
    print("已有缓存结果 %d 条(会自动续跑)" % len(done))

    results, t0 = [], time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, rec in enumerate(have):
            if rec["file"] in done:
                results.append(done[rec["file"]])
                continue
            r = run_one(rec, args.mode)
            if r is None:
                print("  读图失败: %s" % rec["file"])
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(r)
            if (i + 1) % 20 == 0:
                acc = 100.0 * sum(1 for x in results if x["exact"]) / len(results)
                el = time.time() - t0
                eta = el / (i + 1) * (len(have) - i - 1) / 60.0
                print("  %d/%d  当前正确率 %.1f%%  已用 %.0fs  预计还需 %.1f 分钟"
                      % (i + 1, len(have), acc, el, eta), flush=True)

    n = len(results)
    exact = sum(1 for r in results if r["exact"])
    loc = sum(1 for r in results if r["localized"])
    chars = sum(r["charsim"] for r in results) / max(n, 1)
    avg_ms = sum(r["ms"] for r in results) / max(n, 1)

    print("\n" + "=" * 58)
    print(" 评测结果   split=%s   mode=%s" % (args.split, args.mode))
    print("=" * 58)
    print(" 样本数        : %d" % n)
    print(" 整牌正确率    : %.1f%%   (%d/%d)" % (100.0 * exact / n, exact, n))
    print(" 定位命中率    : %.1f%%   (%d/%d)" % (100.0 * loc / n, loc, n))
    print(" 字符正确率    : %.1f%%" % (100.0 * chars))
    print(" 平均耗时      : %.0f ms/张" % avg_ms)

    gb = report_bins(results, "blur", BLUR_BINS, "清晰度(blur 越小越清晰)")
    gd = report_bins(results, "brightness", BRIGHT_BINS, "亮度")

    fails = [r for r in results if not r["exact"]]
    fails.sort(key=lambda x: x["charsim"])

    md = os.path.join(DATA_DIR, "eval_%s_%s.md" % (args.split, args.mode))
    with open(md, "w", encoding="utf-8") as f:
        f.write("# CCPD 车牌识别评测报告\n\n")
        f.write("- 数据: CCPD-YOLO `%s` split,模式 `%s`\n" % (args.split, args.mode))
        f.write("- 样本数: %d\n" % n)
        f.write("- **整牌正确率: %.1f%%** (%d/%d)\n" % (100.0 * exact / n, exact, n))
        f.write("- 定位命中率: %.1f%% (%d/%d)\n" % (100.0 * loc / n, loc, n))
        f.write("- 字符正确率: %.1f%%\n" % (100.0 * chars))
        f.write("- 平均耗时: %.0f ms/张\n\n" % avg_ms)
        f.write("## 失败样例(按相似度升序,前 60)\n\n")
        f.write("| 真值 | 识别结果 | 置信度 | 清晰度 | 亮度 |\n|---|---|---|---|---|\n")
        for r in fails[:60]:
            f.write("| %s | %s | %.2f | %s | %s |\n" % (
                r["gt"], r["pred"] or "(空)", r["conf"],
                r.get("blur"), r.get("brightness")))
    print("\n报告已写入: %s" % md)

    print("\n最严重的失败样例(前 15):")
    for r in fails[:15]:
        print("  真值 %s  ->  识别 %-8s 相似度 %.2f  blur=%-4s 亮度=%-4s"
              % (r["gt"], r["pred"] or "(空)", r["charsim"],
                 r.get("blur"), r.get("brightness")))


if __name__ == "__main__":
    main()

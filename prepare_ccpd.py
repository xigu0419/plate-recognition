# -*- coding: utf-8 -*-
"""
prepare_ccpd.py —— 把 CCPD-YOLO(pose) 数据集的 ndjson 落地成本地可用的数据集。

这个 ndjson 每行是一张图,带:
  file         CCPD 文件名 —— 里面直接编码了车牌真值、倾斜角、亮度、模糊度
  url          图片下载地址(带签名,会过期,要尽早下载)
  width/height 原图尺寸
  split        train / val / test
  annotations.pose  [[cls, x1,y1,v1, x2,y2,v2, x3,y3,v3, x4,y4,v4]]  归一化坐标
                   4 个关键点 = 车牌左上/右上/右下/左下 四个角

产出:
  ccpd_data/images/*.jpg       下载下来的原图
  ccpd_data/labels.jsonl       每行一条:真值车牌号 / 外接框 / 四角点 / split / 质量指标
  ccpd_data/ccpd.yaml          YOLO 训练用的数据集描述(给以后训练检测器用)

用法:
  python prepare_ccpd.py                      # 全部下载
  python prepare_ccpd.py --limit 16           # 只下 16 张(冒烟测试)
  python prepare_ccpd.py --split test         # 只下 test
  python prepare_ccpd.py --labels-only        # 不下载,只重新生成 labels.jsonl
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------------
# CCPD 文件名里的车牌号编码表(官方定义,顺序不能改)
# ----------------------------------------------------------------------------
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
             "新", "警", "学", "O"]
# 字母表故意跳过 I 和 O(容易和 1/0 混淆),最后补一个 O
ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789") + ["O"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NDJSON = os.path.join(
    r"D:\CPP\.dsh-drop\session-edd17531-0418-493c-bb1a-a18a3ecee725",
    "12669f693ac0-ccpd-yolo.ndjson")
DEFAULT_OUT = os.path.join(HERE, "ccpd_data")


def decode_plate(num_str):
    """把 '0_0_8_25_24_26_32' 这种索引串还原成 '皖AJ1028'。"""
    try:
        idx = [int(x) for x in num_str.split("_")]
    except ValueError:
        return None
    if len(idx) < 7:
        return None
    if not (0 <= idx[0] < len(PROVINCES)):
        return None
    chars = [PROVINCES[idx[0]]]
    for k in idx[1:]:
        if not (0 <= k < len(ADS)):
            return None
        chars.append(ADS[k])
    return "".join(chars)


def parse_ccpd_name(fn):
    """
    CCPD 文件名格式:
      <面积比>-<水平倾斜>_<垂直倾斜>-<x1&y1_x2&y2>-<四角点>-<车牌编码>-<亮度>-<模糊度>.jpg
    返回 dict,解析失败返回 None。
    """
    stem = fn.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) < 7:
        return None
    area, tilt, bbox, corners, number, brightness, blur = parts[:7]

    out = {"area_ratio": area, "plate": decode_plate(number)}

    # 倾斜角 90_93 -> 水平 90 度(基本摆正),垂直 93 度
    try:
        hz, vt = tilt.split("_")
        out["tilt_h"], out["tilt_v"] = int(hz), int(vt)
    except ValueError:
        out["tilt_h"] = out["tilt_v"] = None

    # 外接框 254&456_478&527 -> x1,y1,x2,y2
    try:
        p1, p2 = bbox.split("_")
        x1, y1 = p1.split("&")
        x2, y2 = p2.split("&")
        out["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
    except ValueError:
        out["bbox"] = None

    # 四角点:CCPD 顺序是 右下 右上 左上 左下
    try:
        pts = []
        for quad in corners.split("_"):
            x, y = quad.split("&")
            pts.append([int(x), int(y)])
        if len(pts) == 4:
            # 统一重排成 左上 右上 右下 左下,方便后面的透视矫正
            rb, rt, lt, lb = pts
            out["corners"] = [lt, rt, rb, lb]
        else:
            out["corners"] = None
    except ValueError:
        out["corners"] = None

    try:
        out["brightness"] = int(brightness)
        out["blur"] = int(blur)
    except ValueError:
        out["brightness"] = out["blur"] = None

    return out


def load_ndjson(path):
    header, images = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "dataset":
                header = obj
            elif obj.get("type") == "image":
                images.append(obj)
    return header, images


def pose_to_boxes(pose, w, h):
    """
    解析 YOLO pose 标注。

    每行是: cls, cx, cy, bw, bh, (x, y, v) * 4   —— 全部归一化到 0~1
    其中 4 个关键点就是车牌的四个角。

    注意:角点在标注里的先后顺序不固定(这份数据是 右下/左下/左上/右上),
    所以这里不按位置硬取,而是按 x/y 几何排序重排成 左上/右上/右下/左下,
    这样只要车牌倾斜不超过 45 度就一定正确。
    """
    out = []
    for row in (pose or []):
        if len(row) < 17:
            continue
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        kp = row[5:17]
        pts = []
        for i in range(0, 12, 3):
            if kp[i + 2] <= 0:          # 可见性为 0 的点丢弃
                continue
            pts.append([kp[i] * w, kp[i + 1] * h])
        if len(pts) != 4:
            continue
        by_x = sorted(pts, key=lambda p: p[0])
        left = sorted(by_x[:2], key=lambda p: p[1])      # 左边两点按 y 排
        right = sorted(by_x[2:], key=lambda p: p[1])     # 右边两点按 y 排
        tl, bl = left
        tr, br = right
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append({
            "corners": [tl, tr, br, bl],
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "bbox_xywh": [cx * w, cy * h, bw * w, bh * h],
        })
    return out


def fetch(url, dst, retries=3):
    tmp = dst + ".part"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = r.read()
            if len(data) < 256 or data[:2] != b"\xff\xd8":
                raise IOError("不是有效 JPEG (%d 字节)" % len(data))
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return len(data)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", default=DEFAULT_NDJSON)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张,0=全部")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--labels-only", action="store_true")
    args = ap.parse_args()

    header, images = load_ndjson(args.ndjson)
    if header:
        print("数据集: %s  类别: %s  关键点: %s" % (
            header.get("name"), header.get("class_names"), header.get("kpt_shape")))
    print("ndjson 中图片记录: %d" % len(images))

    if args.split != "all":
        images = [im for im in images if im.get("split") == args.split]
    if args.limit:
        images = images[:args.limit]

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    # ---- 1. 生成标注(纯解析,不依赖图片) ----
    records = []
    bad_name = 0
    for im in images:
        fn = im.get("file") or ""
        meta = parse_ccpd_name(fn) or {}
        if not meta.get("plate"):
            bad_name += 1
        w, h = im.get("width"), im.get("height")
        pose = (im.get("annotations") or {}).get("pose")
        boxes = pose_to_boxes(pose, w, h) if (pose and w and h) else []
        rec = {
            "file": fn,
            "path": os.path.join("images", fn),
            "url": im.get("url"),
            "split": im.get("split"),
            "width": w, "height": h,
            "plate": meta.get("plate"),
            "bbox": (boxes[0]["bbox"] if boxes else meta.get("bbox")),
            "bbox_xywh": (boxes[0]["bbox_xywh"] if boxes else None),
            "corners": (boxes[0]["corners"] if boxes else meta.get("corners")),
            "tilt_h": meta.get("tilt_h"), "tilt_v": meta.get("tilt_v"),
            "brightness": meta.get("brightness"), "blur": meta.get("blur"),
        }
        records.append(rec)

    labels_path = os.path.join(args.out, "labels.jsonl")
    with open(labels_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("标注已写入: %s  (%d 条, 文件名解析失败 %d 条)" % (labels_path, len(records), bad_name))

    # ---- 2. 下载图片 ----
    if not args.labels_only:
        todo = [r for r in records
                if r["url"] and not os.path.exists(os.path.join(img_dir, r["file"]))]
        print("需要下载: %d 张 (已有 %d 张)" % (len(todo), len(records) - len(todo)))
        if todo:
            t0 = time.time()
            done, fail = 0, []
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(fetch, r["url"], os.path.join(img_dir, r["file"])): r
                        for r in todo}
                for fut in as_completed(futs):
                    r = futs[fut]
                    try:
                        fut.result()
                        done += 1
                    except Exception as e:
                        fail.append((r["file"], str(e)[:80]))
                    if done % 200 == 0 and done:
                        sp = done / max(time.time() - t0, 1e-6)
                        print("  已下载 %d/%d  (%.1f 张/秒)" % (done, len(todo), sp), flush=True)
            print("下载完成: 成功 %d, 失败 %d, 耗时 %.1f 秒"
                  % (done, len(fail), time.time() - t0))
            for fn, err in fail[:10]:
                print("   失败: %s -> %s" % (fn, err))
            if fail:
                with open(os.path.join(args.out, "failed.txt"), "w", encoding="utf-8") as f:
                    for fn, err in fail:
                        f.write("%s\t%s\n" % (fn, err))

    # ---- 3. 顺便写一份 YOLO 数据集描述,以后训练检测器要用 ----
    yaml_path = os.path.join(args.out, "ccpd.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# CCPD-YOLO 数据集描述(自动生成)\n")
        f.write("path: %s\n" % args.out.replace("\\", "/"))
        f.write("train: images\nval: images\n")
        f.write("names:\n  0: license_plate\n")
    print("YOLO 描述已写入: %s" % yaml_path)


if __name__ == "__main__":
    main()

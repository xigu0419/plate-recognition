# -*- coding: utf-8 -*-
"""
check_dataset.py —— 校验 CCPD 标注解析是否正确、数据集分布如何。

关键校验:从文件名解析出的外接框,应与 annotations.pose 推出的角点外接框一致。
两者能对上,就说明车牌号解码和四个角点都取对了。
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "ccpd_data", "labels.jsonl")


def main():
    recs = []
    with open(LABELS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    print("总记录数: %d" % len(recs))

    # ---- 车牌号解码情况 ----
    ok_plate = [r for r in recs if r.get("plate")]
    print("车牌号解码成功: %d / %d" % (len(ok_plate), len(recs)))

    lens = Counter(len(r["plate"]) for r in ok_plate)
    print("车牌长度分布: %s" % dict(sorted(lens.items())))

    # 省份字分布(只打索引,避免控制台编码问题)
    prov = Counter(r["plate"][0] for r in ok_plate)
    print("省份字种类数: %d" % len(prov))
    for ch, c in prov.most_common(8):
        print("   %s(U+%04X): %d" % ("?", ord(ch), c))

    # 第二位必须是字母(民用牌规则)
    bad2 = [r for r in ok_plate if not r["plate"][1].isalpha()]
    print("第二位非字母(可疑): %d" % len(bad2))
    for r in bad2[:5]:
        print("   %s" % r["plate"])

    # ---- 角点 / 外接框一致性 ----
    paired = [r for r in recs if r.get("bbox") and r.get("corners")]
    print("同时有文件名框和角点的记录: %d" % len(paired))
    worst = 0.0
    worst_rec = None
    big = 0
    for r in paired:
        x1, y1, x2, y2 = r["bbox"]
        cx = [p[0] for p in r["corners"]]
        cy = [p[1] for p in r["corners"]]
        d = max(abs(min(cx) - x1), abs(min(cy) - y1),
                abs(max(cx) - x2), abs(max(cy) - y2))
        if d > worst:
            worst, worst_rec = d, r
        if d > 2.0:
            big += 1
    print("角点外接框 vs 文件名框 最大偏差: %.3f 像素" % worst)
    print("偏差 > 2 像素的记录数: %d  (越接近 0 说明解析越准)" % big)

    # ---- 图像尺寸范围 ----
    wh = Counter((r.get("width"), r.get("height")) for r in recs)
    print("分辨率分布: %s" % dict(wh.most_common(5)))

    # ---- split ----
    print("split 分布: %s" % dict(Counter(r.get("split") for r in recs)))

    # ---- 难度指标 ----
    def stat(key):
        vals = sorted(r[key] for r in recs if r.get(key) is not None)
        if not vals:
            return "无数据"
        n = len(vals)
        return "min=%d p25=%d 中位=%d p75=%d max=%d" % (
            vals[0], vals[n // 4], vals[n // 2], vals[3 * n // 4], vals[-1])

    print("亮度(brightness): %s" % stat("brightness"))
    print("模糊度(blur, 越小越糊): %s" % stat("blur"))
    print("水平倾斜(tilt_h, 90=正): %s" % stat("tilt_h"))
    print("垂直倾斜(tilt_v, 90=正): %s" % stat("tilt_v"))

    # 倾斜样本占比(偏离 90 度超过 5 度算明显倾斜)
    tilted = sum(1 for r in recs
                 if r.get("tilt_h") is not None
                 and (abs(r["tilt_h"] - 90) > 5 or abs(r["tilt_v"] - 90) > 5))
    print("明显倾斜的样本: %d (%.1f%%)" % (tilted, 100.0 * tilted / max(len(recs), 1)))

    # ---- 同名车图片是否重复(同一辆车多帧) ----
    # CCPD 文件名前 6 位是区域编号,同一辆车往往共享前缀
    prefix = Counter(r["file"][:6] for r in recs)
    print("区域前缀种类数: %d, 最大同类 %d 张" % (len(prefix), prefix.most_common(1)[0][1]))


if __name__ == "__main__":
    main()

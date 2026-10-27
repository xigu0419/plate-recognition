# -*- coding: utf-8 -*-
"""
智能车牌识别系统 v3.0
========================================
基于 OpenCV + PaddleOCR 的图片车牌识别程序

用法:
    python plate_recognition.py            打开图形界面
    python plate_recognition.py 图片.jpg    命令行识别
"""
import os
import sys
import re
import time
import traceback
import threading
import subprocess

import cv2
import numpy as np

# ---------- 程序根目录 ----------
def app_dir():
    """获取程序所在目录(exe 或 .py 所在目录)"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = app_dir()
LOG_PATH = os.path.join(APP_DIR, "错误日志.txt")

def log(msg):
    """写日志到文件"""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass

# ---------- 兼容补丁: 打包环境可能缺少个别标准库模块 ----------
def _install_shims():
    """
    PaddleOCR 内部会 import 一些较老的标准库模块。
    这些模块在 Python 3.13+ 已被移除, PyInstaller 打包时也可能漏收集,
    这里做兜底处理, 避免程序直接无法启动。
    """
    import types

    # imghdr: 用于判断图片格式, 3.13 已移除
    try:
        import imghdr  # noqa: F401
    except Exception:
        m = types.ModuleType("imghdr")
        m.what = lambda file, h=None: None
        sys.modules["imghdr"] = m
        log("已启用 imghdr 兼容补丁")

    # importlib.metadata.version: 部分库(如 imageio)启动时会查询自身版本,
    # 打包后缺少 dist-info 元数据会抛 PackageNotFoundError, 这里兜底。
    try:
        import importlib.metadata as _md
        if not getattr(_md, "_dsh_patched", False):
            _orig_version = _md.version

            def _safe_version(name):
                try:
                    return _orig_version(name)
                except Exception:
                    return "0.0.0"

            _md.version = _safe_version
            _md._dsh_patched = True
    except Exception:
        pass

_install_shims()

# ---------- 导入 OCR(带详细错误诊断) ----------
_PADDLE_AVAILABLE = False
_PADDLE_ERR = ""
PaddleOCR = None

try:
    import paddle
    _PADDLE_CORE_OK = True
except Exception:
    _PADDLE_CORE_OK = False
    _PADDLE_ERR = "导入 paddle 失败:\n" + traceback.format_exc()
    log("导入 paddle 失败:\n" + traceback.format_exc())

if _PADDLE_CORE_OK:
    try:
        from paddleocr import PaddleOCR
        _PADDLE_AVAILABLE = True
    except Exception:
        _PADDLE_ERR = "导入 paddleocr 失败:\n" + traceback.format_exc()
        log("导入 paddleocr 失败:\n" + traceback.format_exc())

# ---------- 图片读写(兼容中文路径) ----------
def imread_any(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img

def imwrite_any(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp"):
        ext = ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("图片编码失败")
    buf.tofile(path)

def clean_text(t):
    t = str(t).replace(" ", "").replace("·", "").replace("•", "")
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", t)

# ---------- 模型目录探测(支持内置模型, 离线可用) ----------
def ascii_safe_models(root):
    """
    重要: Paddle 的 C++ 推理接口无法打开含非 ASCII(如中文)字符的路径。
    若程序被放在中文目录下(例如 "车牌识别工具" 文件夹、桌面),
    则先把模型复制到临时 ASCII 目录, 再从那里加载。
    """
    try:
        root.encode("ascii")
        return root                      # 路径本身是 ASCII, 无需处理
    except UnicodeEncodeError:
        pass

    import tempfile
    import shutil
    import hashlib

    # 依次尝试几个 ASCII 临时目录
    candidates = [
        tempfile.gettempdir(),
        os.path.join(os.environ.get("SystemDrive", "C:") + "\\", "PlateOCRModels"),
        "C:\\PlateOCRModels",
    ]
    base = None
    for b in candidates:
        try:
            b.encode("ascii")
        except UnicodeEncodeError:
            continue
        base = b
        break
    if base is None:
        return root

    tag = hashlib.md5(root.encode("utf-8")).hexdigest()[:10]
    dst = os.path.join(base, "plate_models_" + tag)
    marker = os.path.join(dst, ".complete")

    if not os.path.exists(marker):
        try:
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(root, dst)
            with open(marker, "w", encoding="utf-8") as f:
                f.write("ok")
            log("模型已复制到 ASCII 路径: " + dst)
        except Exception:
            log("模型复制失败:\n" + traceback.format_exc())
            return root
    return dst

def find_bundled_models():
    """
    查找内置模型目录, 找到返回 dict, 否则 None
    支持结构: <root>/det/ch/ch_PP-OCRv4_det_infer 等
    """
    roots = [
        os.path.join(APP_DIR, "models"),
        os.path.join(APP_DIR, "_internal", "models"),
        os.path.join(APP_DIR, "paddleocr_models"),
    ]
    for root in roots:
        det = os.path.join(root, "det", "ch", "ch_PP-OCRv4_det_infer")
        rec = os.path.join(root, "rec", "ch", "ch_PP-OCRv4_rec_infer")
        cls = os.path.join(root, "cls", "ch_ppocr_mobile_v2.0_cls_infer")
        if os.path.isdir(det) and os.path.isdir(rec):
            # 处理中文路径问题
            safe = ascii_safe_models(root)
            return {
                "det": os.path.join(safe, "det", "ch", "ch_PP-OCRv4_det_infer"),
                "rec": os.path.join(safe, "rec", "ch", "ch_PP-OCRv4_rec_infer"),
                "cls": (os.path.join(safe, "cls", "ch_ppocr_mobile_v2.0_cls_infer")
                        if os.path.isdir(cls) else None),
                "root": safe,
            }
    return None

# ---------- OCR 引擎(懒加载) ----------
_engine = None
_engine_lock = threading.Lock()

def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            if not _PADDLE_AVAILABLE:
                raise RuntimeError("OCR 引擎不可用:\n" + _PADDLE_ERR)
            kwargs = dict(use_angle_cls=True, lang="ch", show_log=False, use_gpu=False)
            m = find_bundled_models()
            if m:
                kwargs["det_model_dir"] = m["det"]
                kwargs["rec_model_dir"] = m["rec"]
                if m["cls"]:
                    kwargs["cls_model_dir"] = m["cls"]
                log(f"使用内置模型: {m['root']}")
            _engine = PaddleOCR(**kwargs)
        return _engine

def parse_result(res):
    """解析 PaddleOCR 输出 -> (文本, 平均置信度)"""
    texts, confs = [], []
    if not res:
        return None, 0.0
    for block in res:
        if not block:
            continue
        for item in block:
            try:
                txt = clean_text(item[1][0])
                conf = float(item[1][1])
                if txt:
                    texts.append(txt)
                    confs.append(conf)
            except Exception:
                continue
    if not texts:
        return None, 0.0
    return "".join(texts), sum(confs) / len(confs)


# ---------- 车牌格式校验(利用车牌规则提高准确率) ----------
PROVINCES = set("京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新")
# 车牌字母中不使用 I 和 O(易与 1、0 混淆)
PLATE_LETTERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
PLATE_ALNUM = PLATE_LETTERS | set("0123456789")

def plate_score(text):
    """
    给识别结果打分, 越像合法车牌分越高:
      首位是省份简称 +40 (否则 -30)
      长度 7 位 +30 / 8 位新能源 +25
      第二位是字母 +15
      其余位是字母数字 +2/位
    返回 (分数, 是否合法车牌)
    """
    if not text:
        return -999, False

    score = 0
    n = len(text)

    if text[0] in PROVINCES:
        score += 40
    else:
        score -= 30

    valid = False
    if n == 7:
        score += 30
        valid = text[0] in PROVINCES
    elif n == 8:
        score += 25
        valid = text[0] in PROVINCES
    else:
        score -= abs(n - 7) * 8

    if n >= 2 and text[1] in PLATE_LETTERS:
        score += 15

    for ch in text[2:]:
        if ch in PLATE_ALNUM:
            score += 2
        elif ch in PROVINCES or not ch.isalnum():
            score -= 6

    return score, valid

def ocr_once(engine, img_bgr):
    """完整流程(检测+方向分类+识别)。注意: PaddleOCR 期望 BGR 顺序的 numpy 数组"""
    text, conf = parse_result(engine.ocr(img_bgr, cls=True))
    return text or "", conf or 0.0

def rec_direct(engine, imgs):
    """
    直接调用"识别模型", 跳过检测模型 —— 这是识别准确率的关键。

    原因: 车牌区域本来就是**单行文本**。若再走一遍检测流程, 检测模型会把
    这个窄长条重新切分, 实测会把省份汉字切成单独的框然后丢掉, 于是
      皖AX188Y -> X188 ,  皖AY5E19 -> 皖AY5E , 皖AKV067 -> None
    直接喂给识别模型则整条一起解码, 上述错例全部修正, 置信度也从"丢字后
    的勉强值"回到 0.999 以上。

    imgs: BGR numpy 数组列表(可含多个放大尺度)
    返回 [(text, conf), ...]; 接口不可用时返回 None(由调用方回退)
    """
    rec = getattr(engine, "text_recognizer", None)
    if rec is None:
        return None
    try:
        out = rec(imgs)          # 一次调用处理全部尺度, 比逐张快很多
    except Exception:
        log("直接识别失败:\n" + traceback.format_exc())
        return None
    # PaddleOCR 2.x 返回 (结果列表, 耗时列表)
    res = out[0] if (isinstance(out, tuple) and len(out) == 2) else out
    if not isinstance(res, (list, tuple)) or len(res) != len(imgs):
        return None
    outs = []
    for item in res:
        try:
            outs.append((clean_text(item[0]), float(item[1])))
        except Exception:
            outs.append(("", 0.0))
    return outs

def vote_best(results):
    """
    多个 (text, conf) 结果投票, 综合评分 = 车牌格式分*10 + 票数*20 + 置信度*5
    这样"符合车牌规则"的结果会优先, 再考虑多尺度一致性和置信度。
    """
    votes = {}
    for t, c in results:
        if not t:
            continue
        info = votes.setdefault(t, {"count": 0, "best_conf": 0.0})
        info["count"] += 1
        info["best_conf"] = max(info["best_conf"], c)
    if not votes:
        return None, 0.0
    best_text, best_conf, best_total = None, 0.0, -1e9
    for t, info in votes.items():
        fmt, _ = plate_score(t)
        total = fmt * 10 + info["count"] * 20 + info["best_conf"] * 5
        if total > best_total:
            best_total, best_text, best_conf = total, t, info["best_conf"]
    return best_text, best_conf

def recognize_text(roi_bgr):
    """
    识别已经裁好的车牌条(多尺度 + 投票)。
    主路径是直接调用识别模型; 只有当结果不像合法车牌时,
    才回退到完整流程(检测+识别)补充, 兼顾少数裁切不准的情况。
    返回 (text, conf)
    """
    engine = get_engine()
    h = roi_bgr.shape[0]

    # 依车牌高度自适应选择放大倍数
    scales = [1.0]
    for s in (2.0, 3.0, 4.0, 6.0):
        if 60 <= h * s <= 700:
            scales.append(s)
    if len(scales) == 1:      # 极小图, 强制放大
        scales = [2.0, 4.0, 6.0]
    scales = scales[:4]

    imgs = [roi_bgr if s == 1.0 else cv2.resize(
        roi_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        for s in scales]

    results = []
    direct = rec_direct(engine, imgs)
    if direct:
        results.extend(direct)

    best_text, best_conf = vote_best(results)

    # 直接识别的结果不像车牌 → 用完整流程兜底再投一次票
    fmt, _ = plate_score(best_text or "")
    if fmt < 45:
        for im in imgs:
            t, c = ocr_once(engine, im)
            if t:
                results.append((t, c))
        best_text, best_conf = vote_best(results)

    return best_text, best_conf

# ---------- 车牌定位(支持倾斜 / 多颜色 / 多车牌) ----------
# 各颜色车牌在 HSV 下的范围。蓝牌/绿牌是白字彩底, 黄牌是黑字黄底
PLATE_COLOR_RANGES = [
    ((95, 70, 50), (135, 255, 255)),    # 蓝牌
    ((35, 40, 50), (95, 255, 255)),     # 绿牌(新能源)
    ((10, 60, 90), (45, 255, 255)),     # 黄牌
]

def _order_points(pts):
    """四点排序为: 左上, 右上, 右下, 左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def warp_quad(img, quad, pad=0.0):
    """
    按任意四点做透视变换, 把倾斜/斜拍的车牌矫正成水平矩形。
    quad: 4 个角点, 顺序不限(内部自动排序为 左上/右上/右下/左下)
    pad : 向外扩边的比例。文字检测框往往比车牌外框略紧, 会切掉边上的字
          (尤其是首位省份汉字), 留一点边能救回来。
    返回矫正后的车牌图, 失败返回 None
    """
    try:
        box = _order_points(np.asarray(quad, dtype="float32").reshape(4, 2))
    except Exception:
        return None
    if pad > 0:
        c = box.mean(axis=0)
        box = (box - c) * (1.0 + pad) + c
    (tl, tr, br, bl) = box
    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    w, h = int(round(w)), int(round(h))
    if w < 8 or h < 4:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    # 若矫正后是竖着的(车牌被旋转近 90°), 转成横向
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def deskew_plate(img, rect):
    """
    用 minAreaRect 的四个角点做透视变换(颜色定位那条路用)。
    rect: cv2.minAreaRect 的返回值
    """
    try:
        box = cv2.boxPoints(rect)
    except Exception:
        return None
    return warp_quad(img, box)

def detect_color_candidates(img, max_candidates=6):
    """
    【颜色定位】按 HSV 颜色阈值找候选区域(支持倾斜、支持多个车牌)。

    注意: 这条路在真实照片里并不可靠 —— 在 CCPD 数据集 500 张测试图上,
    它只有 20.7% 能框到车牌(67% 完全框不到), 因为色偏/暗光会让蓝色跑出阈值。
    现在它只作为"文字检测定位"的补充(车牌特写时仍然有用)。
    返回列表, 每项 {"quad": 四点, "rect": minAreaRect, "score": 评分, ...}
    """
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = None
    for lo, hi in PLATE_COLOR_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        return []

    # 形态学核大小随图像尺寸自适应:
    # 整车照里车牌较小用小核, 车牌特写用大核把字符连起来
    k = int(min(h_img, w_img) * 0.015)
    k = max(3, min(k, 25))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                             (k * 3 + 1, k + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 面积下限放低, 保证整车照片里的小车牌也能检出
    min_area = max(150, int(w_img * h_img * 0.0005))

    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(c)          # 关键: 支持旋转, 不是正矩形
        rw, rh = rect[1]
        if rw < 8 or rh < 4:
            continue
        long_side, short_side = max(rw, rh), min(rw, rh)
        if short_side <= 0:
            continue
        ratio = long_side / short_side
        # 单层车牌约 3.1, 双层黄牌约 2.0, 放宽到 1.2~8.0
        if not (1.2 <= ratio <= 8.0):
            continue
        # 填充度: 车牌应为实心矩形
        fill = area / (rw * rh + 1e-6)
        if fill < 0.35:
            continue

        ratio_score = 1.0 - min(abs(ratio - 3.1) / 3.1, 1.0)
        score = (area / (w_img * h_img)) * 2.2 + ratio_score * 0.8 + fill * 0.6
        quad = cv2.boxPoints(rect)
        cands.append({
            "rect": rect, "quad": quad, "source": "color",
            "score": score, "ratio": ratio, "area": area,
            "center": (int(rect[0][0]), int(rect[0][1])),
        })

    cands.sort(key=lambda d: -d["score"])
    return cands[:max_candidates]

# ---------- 车牌定位(主力: 文字检测) ----------
def _quad_bounds(quad):
    """四点的轴对齐外接框 (x1, y1, x2, y2)"""
    p = np.asarray(quad, dtype="float32").reshape(-1, 2)
    return [float(p[:, 0].min()), float(p[:, 1].min()),
            float(p[:, 0].max()), float(p[:, 1].max())]

def _iou_xyxy(a, b):
    """两个轴对齐框的交并比 (IoU)"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0

def _same_plate_region(a, b, iou_th=0.30, dist_th=18):
    """
    判断两个候选框是不是"同一块车牌的区域"。

    为什么不能只看中心点距离: 同一块车牌会同时产生"整块车牌框"和
    "把字符合并成行的框", 两者大小不同, 中心点可以错开 40 像素以上。
    实测 500 张测试图里有 23 张(4.6%)因此把同一块车牌显示了两三次。
    所以这里改成"框明显重叠 或 中心点很近"都算同一块。
    """
    if a.get("quad") is not None and b.get("quad") is not None:
        if _iou_xyxy(_quad_bounds(a["quad"]),
                     _quad_bounds(b["quad"])) >= iou_th:
            return True
    return (abs(a["center"][0] - b["center"][0]) < dist_th and
            abs(a["center"][1] - b["center"][1]) < dist_th)

def _quad_sides(quad):
    """
    四边形的两组对边长度 -> (长边, 短边)。

    注意: 判断"像不像车牌"必须用**真实边长**, 不能用轴对齐外接框。
    车牌被旋转 25~32 度后, 它的轴对齐外接框会变成近似正方形
    (实测 皖AMP033 那类倾斜样本的外接框宽高比只有 1.09~1.30),
    用外接框算就会把正确的车牌框当成"形状不对"而丢掉。
    实测有 4 张失败样本就是这么被误滤的。
    """
    p = np.asarray(quad, dtype="float32").reshape(4, 2)
    s = [float(np.linalg.norm(p[i] - p[(i + 1) % 4])) for i in range(4)]
    a, b = max(s[0], s[2]), max(s[1], s[3])
    return max(a, b), max(1e-6, min(a, b))

def _quad_ratio(quad):
    """四边形真实长宽比(长边/短边)"""
    lo, sh = _quad_sides(quad)
    return lo / sh

def _merge_boxes_to_lines(boxes, max_gap_ratio=1.2):
    """
    把文字检测得到的零散文本框, 按"同一行 + x 方向相邻"聚合成文字行。

    车牌会被检测成 7 个字符框, 或者整块 1 个框; 两种都要能处理,
    所以这里只是"能合就合", 合并后仍然把原始框也留作候选。
    """
    items = []
    for b in boxes:
        x1, y1, x2, y2 = _quad_bounds(b)
        items.append({
            "quad": np.asarray(b, dtype="float32").reshape(4, 2),
            "box": [x1, y1, x2, y2],
            "cy": (y1 + y2) / 2.0,
            "h": max(1.0, y2 - y1),
        })
    if not items:
        return []

    used = [False] * len(items)
    lines = []
    for i, it in enumerate(items):
        if used[i]:
            continue
        cur = list(it["box"])
        members = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            ch = max(1.0, cur[3] - cur[1])
            ccy = (cur[1] + cur[3]) / 2.0
            for j, jt in enumerate(items):
                if used[j]:
                    continue
                # 两框中心 y 要足够接近, 高度量级也要相近(同一行文字)
                if abs(jt["cy"] - ccy) > 0.6 * max(ch, jt["h"]):
                    continue
                if not (0.4 <= jt["h"] / ch <= 2.5):
                    continue
                # x 方向相邻(允许一点点间隙)
                gap = max(cur[0] - jt["box"][2], jt["box"][0] - cur[2])
                if gap > max_gap_ratio * max(ch, jt["h"]):
                    continue
                cur = [min(cur[0], jt["box"][0]), min(cur[1], jt["box"][1]),
                       max(cur[2], jt["box"][2]), max(cur[3], jt["box"][3])]
                members.append(j)
                used[j] = True
                changed = True
        lines.append({"box": cur, "members": members})

    out = []
    for ln in lines:
        x1, y1, x2, y2 = ln["box"]
        out.append({
            # 合并后的行用轴对齐四点表示(已经是水平的, 不需要再矫正)
            "quad": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                             dtype="float32"),
            "box": ln["box"],
            "n_boxes": len(ln["members"]),
        })
    return out

def _line_plate_score(quad, n_boxes, w_img, h_img):
    """
    给一个文字行打"像不像车牌"的分:
      宽高比接近单层车牌 3.14 或双层车牌 2.0  -> 加分
      由 7 个字符框合并而来(或本身就是整块) -> 加分
      面积占整图比例合理                       -> 加分
    """
    bw, bh = _quad_sides(quad)     # 用真实边长, 旋转过的车牌也算得准
    ratio = bw / bh

    # 宽高比: 单层蓝牌约 3.14, 双层黄牌约 2.0
    best = min(abs(ratio - 3.14), abs(ratio - 2.0))
    ratio_score = max(0.0, 1.0 - best / 1.6)

    # 字符框数量: 车牌 7 个字符(新能源 8); 整块检测成 1 个框也接受
    if n_boxes == 1:
        char_score = 0.7
    else:
        char_score = max(0.0, 1.0 - abs(n_boxes - 7) / 4.0)

    # 面积占比: 车牌一般占整图 0.3% ~ 12%
    frac = (bw * bh) / max(1.0, w_img * h_img)
    if 0.003 <= frac <= 0.12:
        size_score = 1.0
    elif frac < 0.003:
        size_score = max(0.0, frac / 0.003)
    else:
        size_score = max(0.0, 1.0 - (frac - 0.12) / 0.4)

    return ratio_score * 1.2 + char_score * 1.0 + size_score * 0.6

def detect_text_candidates(img, max_candidates=8):
    """
    【文字检测定位】用 PaddleOCR 自带的文字检测模型(DBNet)找车牌候选。

    为什么用它当主力: 这个模型是在海量真实场景文字上训练过的, 对色偏、暗光、
    模糊远比 HSV 颜色阈值鲁棒。在 CCPD 数据集 500 张测试图上实测:
        颜色阈值  : 只有 20.7% 能框到车牌, 67% 完全框不到
        文字检测  : 100% 能框到车牌, 99.8% 的 IoU>=0.7, 中位 IoU 0.91
    而且检测模型本来就已经随程序打包, 不增加任何依赖和体积。
    """
    engine = get_engine()
    det = getattr(engine, "text_detector", None)
    if det is None:
        return []
    try:
        boxes, _ = det(img)
    except Exception:
        log("文字检测失败:\n" + traceback.format_exc())
        return []
    if boxes is None or len(boxes) == 0:
        return []

    h_img, w_img = img.shape[:2]
    cands = []

    # (1) 原始检测框: 车牌也可能被整块检出
    for b in boxes:
        q = np.asarray(b, dtype="float32").reshape(4, 2)
        x1, y1, x2, y2 = _quad_bounds(q)
        bw, bh = _quad_sides(q)          # 真实边长, 旋转车牌也不算错
        ratio = bw / bh
        if not (1.4 <= ratio <= 9.0):
            continue
        cands.append({"quad": q, "source": "text",
                      "n_boxes": 1, "score": _line_plate_score(q, 1, w_img, h_img),
                      "ratio": ratio, "area": bw * bh,
                      "center": (int((x1 + x2) / 2), int((y1 + y2) / 2))})

    # (2) 合并成行后的框: 车牌被拆成 7 个字符框时靠这个
    for ln in _merge_boxes_to_lines(boxes):
        q = ln["quad"]
        x1, y1, x2, y2 = _quad_bounds(q)
        bw, bh = _quad_sides(q)
        ratio = bw / bh
        if not (1.4 <= ratio <= 9.0):
            continue
        cands.append({"quad": q, "source": "text",
                      "n_boxes": ln["n_boxes"],
                      "score": _line_plate_score(q, ln["n_boxes"], w_img, h_img),
                      "ratio": ratio, "area": bw * bh,
                      "center": (int((x1 + x2) / 2), int((y1 + y2) / 2))})

    cands.sort(key=lambda d: -d["score"])
    return cands[:max_candidates]

def detect_plate_candidates(img, max_candidates=6):
    """
    车牌定位总入口: 两路互补。

      1) 文字检测(主力) —— 对色偏/暗光/模糊鲁棒, 实测 100% 命中
      2) HSV 颜色(补充) —— 车牌特写、或文字检测漏检时有用

    返回按得分降序的候选列表, 每项含:
      quad   四个角点(用于透视矫正)
      rect   minAreaRect(兼容旧接口)
      score  像不像车牌的评分
      source "text" / "color"
    """
    cands = detect_text_candidates(img, max_candidates=max_candidates * 2)
    try:
        cands += detect_color_candidates(img, max_candidates=max_candidates)
    except Exception:
        log("颜色定位失败:\n" + traceback.format_exc())

    # 去掉位置几乎重合的重复候选(保留更可靠的那一路)
    #
    # 重要: 排序必须先按"来源"分主次, 再看得分。
    # 原因: 两路候选的分数不是一个量纲 ——
    #   文字检测候选按"宽高比+字符数+面积"算, 大约 2.3 分;
    #   颜色候选按"面积占整图比例"算, 大片蓝色区域能到 3.0 分。
    # 如果只按分数排, 颜色候选会把文字候选整个挤出前几名,
    # 而实测文字检测的定位命中率(100%)远高于颜色(20.7%)。
    def rank_key(d):
        return (0 if d.get("source") == "text" else 1, -d.get("score", 0.0))

    dedup = []
    for c in sorted(cands, key=rank_key):
        if c.get("rect") is None:
            q = np.asarray(c["quad"], dtype="float32").reshape(4, 2)
            c["rect"] = cv2.minAreaRect(q.astype("float32"))
        # 同一块车牌可能同时被"整块检测框"和"合并成行的框"命中,
        # 两者中心点能错开很多, 所以按"框重叠 或 中心很近"去重
        if not any(_same_plate_region(c, d) for d in dedup):
            dedup.append(c)
    return dedup[:max_candidates]

def recognize_candidate(img, cand):
    """
    对一个候选车牌区域做透视矫正 + 识别。

    两个细节都是实测出来的:
      1) 优先用四点(quad)矫正, 比 minAreaRect 更贴合倾斜车牌
      2) 检测框比车牌外框略紧时会切掉边上的字(最常见是首位省份汉字,
         例如 皖AMP033 被读成 AMP033)。所以第一次认不出合法车牌时,
         向外留边再认一次, 靠"车牌合法性打分"把完整结果选出来。
      3) 黄牌是黑字黄底, 正向认不出时尝试反色。
    返回 (text, conf)
    """
    quad = cand.get("quad")
    warps = []
    if quad is not None:
        for pad in (0.0, 0.08, 0.16):
            w = warp_quad(img, quad, pad=pad)
            if w is not None and w.size:
                warps.append(w)
    if not warps and cand.get("rect") is not None:
        w = deskew_plate(img, cand["rect"])
        if w is not None and w.size:
            warps.append(w)
    if not warps:
        return None, 0.0

    # 第一遍: 按检测框原样识别 —— 绝大多数情况一次就中
    text, conf = recognize_text(warps[0])
    fmt, valid = plate_score(text or "")
    if valid:
        return text, conf
    results = [(text, conf)] if text else []

    t = (text or "").upper()
    # 什么时候值得"向外留边"重试:
    #   结果看着就是车牌, 只是比正常的 7 位少了 1~2 位 —— 说明检测框偏紧,
    #   把边上的字切掉了。两种典型情况:
    #     a) 首位省份汉字被切掉 → 6 位纯 ASCII 字母数字(皖AMP033 读成 AMP033)
    #     b) 中间某位被切掉     → 6 位且以省份汉字开头(皖ATL123 读成 皖AL123)
    #   其它短结果("中"、"L"、"广"、"信")是噪声候选, 留边也救不回来,
    #   不满足条件就不重试 —— 这个判断把整条链路的耗时压下来一大截。
    looks_like_plate = (len(t) in (5, 6) and
                        (t[0] in PROVINCES or (t.isascii() and t.isalnum())))
    if looks_like_plate and len(warps) > 1:
        for warp in warps[1:]:
            t2, c2 = recognize_text(warp)
            if t2:
                results.append((t2, c2))
                if plate_score(t2)[1]:
                    break
    # 完全不像车牌 → 试反色(黄牌是黑字黄底, 反色后可能有救)
    elif fmt < 45:
        try:
            t2, c2 = recognize_text(cv2.bitwise_not(warps[0]))
            if t2:
                results.append((t2, c2))
        except Exception:
            pass

    best, conf = vote_best(results)
    return best, conf

def ocr_fullimage_best(img):
    """
    整图 OCR 兜底: 让 PaddleOCR 自己在图里找文字,
    再挑选最像车牌的文本块(避免把车标、广告字混进来)。
    返回 (text, conf, box) 或 (None, 0.0, None)
    """
    engine = get_engine()
    try:
        res = engine.ocr(img, cls=True)
    except Exception:
        log("整图OCR失败:\n" + traceback.format_exc())
        return None, 0.0, None
    if not res:
        return None, 0.0, None

    best = None
    for block in res:
        if not block:
            continue
        for item in block:
            try:
                box = item[0]
                txt = clean_text(item[1][0])
                conf = float(item[1][1])
            except Exception:
                continue
            if not txt:
                continue
            fmt, _ = plate_score(txt)
            total = fmt * 10 + conf * 5
            if best is None or total > best[3]:
                best = (txt, conf, box, total)
    if best is None:
        return None, 0.0, None
    return best[0], best[1], best[2]

def draw_plate(ann, p):
    """在图上画出车牌位置(支持旋转框)和识别结果"""
    color = (0, 210, 0) if p.get("valid") else (0, 165, 255)
    pts = None
    if p.get("rect") is not None:
        pts = cv2.boxPoints(p["rect"]).astype(int)
    elif p.get("box") is not None:
        pts = np.array(p["box"], dtype=int)
    if pts is None:
        return
    cv2.polylines(ann, [pts], True, color, 3)
    tx = int(pts[:, 0].min())
    ty = int(pts[:, 1].min())
    label = "{} {:.2f}".format(p["text"], p["conf"])
    cv2.putText(ann, label, (tx, max(22, ty - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# 至少给多少个候选做 OCR。
# 车牌往往就是得分最高的那个候选, 但多车牌时第二块牌可能排在第 2~3 位,
# 所以先看够 3 个, 再决定要不要提前收工。
MAX_OCR_CANDIDATES = 3

def recognize_all(path, max_plates=5):
    """
    识别图片中的车牌(支持多个、支持倾斜)。
    返回 (plates, annotated)
      plates: [{"text","conf","rect","box","valid","score"}, ...]
    """
    img = imread_any(path)
    ann = img.copy()

    cands = detect_plate_candidates(img)
    plates = []      # 通过车牌格式校验的结果 —— 只有当它们是"车牌"才显示
    guesses = []     # 没过校验、但勉强读出来的结果 —— 只在找不到合法车牌时兜底

    for idx, cand in enumerate(cands):
        text, conf = recognize_candidate(img, cand)
        if not text:
            continue
        fmt, valid = plate_score(text)
        item = {
            "text": text, "conf": conf, "rect": cand["rect"], "box": None,
            "quad": cand.get("quad"),
            "valid": valid, "score": fmt * 10 + conf * 5,
            "center": cand["center"],
        }
        # 关键: 只有"看起来就是车牌"的结果才当一块车牌显示。
        # 否则车标、店铺招牌、车身文字都会被当成车牌报出来 ——
        # 实测会多报出 AOKANG / 沿街旺铺 / SANTAN / 百姓 / 02025 这类噪声,
        # 以及同一块车牌被读坏的第二遍(皖A6J968 旁边又多一个 A6J968)。
        if valid:
            plates.append(item)
        elif fmt >= 10 and conf >= 0.80:
            guesses.append(item)
        # 提前收工: 候选是按"像不像车牌"排序的, 车牌通常就在第一个。
        # 已经拿到合法且高置信的结果后, 没必要再把剩下的候选都跑一遍 OCR
        # (每个候选要做多尺度识别 + 留边重试, 是整条链路里最慢的一步)。
        # 至少看 3 个候选, 以便多车牌时不会漏掉第二块牌。
        if (idx + 1 >= MAX_OCR_CANDIDATES and
                any(p["conf"] >= 0.90 for p in plates)):
            break

    # 一块合法车牌都没找到(比如车牌极糊、倾斜很大) → 至少给一个最像的结果
    if not plates:
        plates = guesses

    # 去重: 同一块车牌只保留最可信的那一次识别
    #   ① 车牌号完全相同 -> 必然是同一块牌(一个号码只可能出现在一块牌上)
    #   ② 框明显重叠       -> 也是同一块牌, 只是两次读数略有出入
    #      (例如同一块牌一次读成 皖A92870、一次读成 皖A9287D)
    #   ③ 中心点很近       -> 兜底
    # 单靠中心点距离是不够的, 原因见 _same_plate_region 的注释。
    plates.sort(key=lambda p: -p["score"])
    dedup = []
    for p in plates:
        dup = False
        for q in dedup:
            if p["text"] == q["text"] or _same_plate_region(p, q, dist_th=40):
                dup = True
                break
        if not dup:
            dedup.append(p)

    # 颜色定位失败 → 用 PaddleOCR 在自己的检测结果里挑最像车牌的文字
    if not dedup:
        t, c, box = ocr_fullimage_best(img)
        if t:
            fmt, valid = plate_score(t)
            dedup.append({
                "text": t, "conf": c, "rect": None, "box": box,
                "valid": valid, "score": fmt * 10 + c * 5,
                "center": (0, 0),
            })

    dedup = dedup[:max_plates]
    for p in dedup:
        draw_plate(ann, p)
    return dedup, ann

def recognize(path):
    """识别一张图片(兼容旧接口, 返回最佳车牌)"""
    plates, ann = recognize_all(path)
    if not plates:
        return None, 0.0, None, ann
    top = plates[0]
    return top["text"], top["conf"], top.get("rect"), ann

# ---------- 命令行模式 ----------
def cli_mode(path):
    print("=" * 46)
    print("  智能车牌识别系统 v3.0")
    print("=" * 46)
    print("[图片]", path)
    if not _PADDLE_AVAILABLE:
        print("[错误] OCR 引擎不可用")
        print(_PADDLE_ERR)
        return
    try:
        plates, ann = recognize_all(path)
        if plates:
            print("[识别到 %d 个车牌]" % len(plates))
            for i, p in enumerate(plates, 1):
                flag = "合法" if p["valid"] else "存疑"
                print("  %d) %-12s 置信度 %.1f%%  [%s]" %
                      (i, p["text"], p["conf"] * 100, flag))
            summary = " | ".join(p["text"] for p in plates)
            log("识别成功: %s -> %s" % (os.path.basename(path), summary))
        else:
            print("[结果] 未能识别出车牌号")
            log("识别无结果: " + os.path.basename(path))
        out = os.path.splitext(path)[0] + "_result.jpg"
        imwrite_any(out, ann)
        print("[标注图]", out)
    except Exception as e:
        print("[错误]", e)
        log("识别失败:\n" + traceback.format_exc())

# ---------- 图形界面 ----------
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    try:
        from PIL import Image, ImageTk
        HAVE_PIL = True
    except Exception:
        HAVE_PIL = False

    C_BG, C_CARD, C_ACC = "#0f1117", "#1a1d28", "#3aa6ff"
    C_OK, C_WARN, C_TXT, C_DIM = "#37d67a", "#ffb648", "#e8ecf3", "#8a93a6"
    FONT = "Microsoft YaHei UI"

    root = tk.Tk()
    root.title("智能车牌识别系统 v3.0")
    root.geometry("1120x730")
    root.minsize(940, 620)
    root.configure(bg=C_BG)

    state = {"path": None, "img": None, "busy": False, "result": "", "photo": None}

    # ---- 顶部 ----
    top = tk.Frame(root, bg=C_BG)
    top.pack(fill="x", padx=20, pady=(16, 8))
    tk.Label(top, text="🚗 智能车牌识别系统", font=(FONT, 20, "bold"),
             fg=C_TXT, bg=C_BG).pack(side="left")
    tk.Label(top, text="OpenCV + PaddleOCR", font=(FONT, 10),
             fg=C_DIM, bg=C_BG).pack(side="left", padx=12, pady=(8, 0))

    main = tk.Frame(root, bg=C_BG)
    main.pack(fill="both", expand=True, padx=20, pady=6)

    # ---- 左侧预览 ----
    left = tk.Frame(main, bg=C_BG)
    left.pack(side="left", fill="both", expand=True)
    canvas = tk.Canvas(left, bg=C_CARD, highlightthickness=1,
                       highlightbackground="#2c3245", bd=0)
    canvas.pack(fill="both", expand=True)
    canvas.create_text(300, 250, text="请选择车牌图片",
                       font=(FONT, 13), fill=C_DIM)

    # ---- 右侧面板 ----
    right = tk.Frame(main, bg=C_BG, width=330)
    right.pack(side="right", fill="y", padx=(18, 0))

    def mkbtn(text, cmd, bg, state_="normal"):
        b = tk.Button(right, text=text, command=cmd, font=(FONT, 12, "bold"),
                      bg=bg, fg="#0b1220", activebackground=bg, activeforeground="#0b1220",
                      relief="flat", bd=0, pady=10, cursor="hand2", state=state_)
        b.pack(fill="x", pady=5)
        return b

    def show_img(img):
        if not HAVE_PIL or img is None:
            return
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 50:
            cw, ch = 640, 560
        h, w = img.shape[:2]
        s = min(cw / w, ch / h, 1.0)
        disp = cv2.cvtColor(cv2.resize(img, (int(w * s), int(h * s))),
                            cv2.COLOR_BGR2RGB)
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(disp))
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=state["photo"])

    def choose():
        p = filedialog.askopenfilename(
            title="选择车牌图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")])
        if p:
            load(p)

    def load(p):
        try:
            img = imread_any(p)
        except Exception as e:
            messagebox.showerror("错误", str(e))
            return
        state["path"], state["img"] = p, img
        res_lbl.config(text="——", fg=C_OK)
        conf_lbl.config(text="")
        stat_lbl.config(text="已加载: " + os.path.basename(p))
        show_img(img)

    def go():
        if state["busy"] or not state["path"]:
            return
        # OCR 引擎不可用时, 直接给出明确提示
        if not _PADDLE_AVAILABLE:
            messagebox.showerror(
                "OCR 引擎不可用",
                "没有找到 PaddleOCR 识别引擎。\n\n"
                "原因: 当前 Python 不是装有 PaddleOCR 的环境。\n"
                f"当前: {sys.executable}\n\n"
                "解决办法:\n"
                "  双击本目录下的「启动程序.bat」运行即可\n\n"
                f"或用命令行:\n  \"{CONDA_PY}\" plate_recognition.py"
            )
            return
        state["busy"] = True
        run_btn.config(state="disabled")
        stat_lbl.config(text="正在识别中, 请稍候…(首次运行较慢)")
        threading.Thread(target=work, daemon=True).start()

    def work():
        try:
            plates, ann = recognize_all(state["path"])
            try:
                imwrite_any(os.path.splitext(state["path"])[0] + "_result.jpg", ann)
            except Exception:
                pass
            root.after(0, lambda: done(plates, ann))
        except Exception as e:
            log("GUI识别失败:\n" + traceback.format_exc())
            root.after(0, lambda: fail(e))

    def done(plates, ann):
        state["busy"] = False
        run_btn.config(state="normal")
        if plates:
            # 多个车牌时全部展示
            text = "  ".join(p["text"] for p in plates)
            state["result"] = text
            best = plates[0]
            res_lbl.config(text=text, fg=C_OK)
            if len(plates) > 1:
                conf_lbl.config(text="共识别到 %d 个车牌 (首个置信度 %.1f%%)"
                                     % (len(plates), best["conf"] * 100))
            else:
                conf_lbl.config(text="置信度: {:.1%}".format(best["conf"]))
            show_img(ann)
            stat_lbl.config(text="识别成功: " + text)
            hist.insert(0, "{}  {}".format(time.strftime("%H:%M:%S"), text))
        else:
            state["result"] = ""
            res_lbl.config(text="未识别出车牌", fg=C_WARN)
            conf_lbl.config(text="建议换更清晰的照片")
            stat_lbl.config(text="未能识别")

    def fail(e):
        state["busy"] = False
        run_btn.config(state="normal")
        stat_lbl.config(text="出错: " + str(e))

    def copy_res():
        if state["result"]:
            try:
                root.clipboard_clear()
                root.clipboard_append(state["result"])
                stat_lbl.config(text="已复制: " + state["result"])
            except Exception:
                pass

    tk.Label(right, text="车牌号", font=(FONT, 13, "bold"), fg=C_TXT, bg=C_BG).pack(anchor="w", pady=(4, 2))
    res_lbl = tk.Label(right, text="——", font=("Consolas", 26, "bold"),
                       fg=C_OK, bg=C_CARD, pady=12, wraplength=300)
    res_lbl.pack(fill="x")
    conf_lbl = tk.Label(right, text="", font=(FONT, 10), fg=C_DIM, bg=C_BG)
    conf_lbl.pack(anchor="w", pady=(2, 10))

    mkbtn("📂   选择图片", choose, C_ACC)
    run_btn = mkbtn("🔍   开始识别", go, C_OK)
    mkbtn("📋   复制结果", copy_res, "#7a5ce0")

    tk.Label(right, text="识别记录", font=(FONT, 12, "bold"), fg=C_TXT, bg=C_BG).pack(anchor="w", pady=(14, 4))
    hist = tk.Listbox(right, bg=C_CARD, fg=C_TXT, font=("Consolas", 10),
                      highlightthickness=0, bd=0, height=7)
    hist.pack(fill="both", expand=True)

    stat_lbl = tk.Label(root, text="就绪", font=(FONT, 9), fg=C_DIM, bg=C_BG, anchor="w")
    stat_lbl.pack(side="bottom", fill="x", padx=20, pady=(0, 10))

    if not _PADDLE_AVAILABLE:
        stat_lbl.config(
            text="⚠ 未检测到 OCR 引擎 —— 请双击本目录的「启动程序.bat」运行",
            fg=C_WARN)
        res_lbl.config(text="OCR不可用", fg=C_WARN)
        conf_lbl.config(text="当前 Python 不是装有 PaddleOCR 的环境")

    # 命令行传入图片则自动加载
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(300, lambda: load(sys.argv[1]))

    canvas.bind("<Configure>", lambda e: show_img(state["img"]))
    root.mainloop()

# ---------- 入口 ----------
# 装有 PaddleOCR 的 conda 环境(系统自带的 Python 3.14 装不了 PaddleOCR)
CONDA_PY = r"C:\Users\19500\.conda\envs\plate\python.exe"

def try_relaunch():
    """
    如果当前是用系统 Python(缺 paddle)启动的, 自动改用 plate 环境重新启动。
    这样双击 .py 文件也能正常工作。
    返回 True 表示已经切换(调用方应直接退出)。
    """
    # 打包版无需处理
    if getattr(sys, "frozen", False):
        return False
    # 已经重启过一次, 避免死循环
    if os.environ.get("PLATE_RELAUNCHED") == "1":
        return False
    # 目标解释器不存在
    if not os.path.isfile(CONDA_PY):
        return False
    # 当前已经是目标解释器
    try:
        if os.path.normcase(os.path.abspath(sys.executable)) == \
           os.path.normcase(os.path.abspath(CONDA_PY)):
            return False
    except Exception:
        pass
    # 当前环境其实有 paddle, 不用切换
    if _PADDLE_AVAILABLE:
        return False

    env = dict(os.environ)
    env["PLATE_RELAUNCHED"] = "1"
    try:
        log(f"当前解释器缺少 paddle, 自动切换到: {CONDA_PY}")
        subprocess.Popen([CONDA_PY, os.path.abspath(__file__)] + sys.argv[1:], env=env)
        return True
    except Exception:
        log("自动切换失败:\n" + traceback.format_exc())
        return False

def show_ocr_error_dialog():
    """OCR 不可用时给出明确的解决办法"""
    import tkinter as tk
    from tkinter import messagebox
    r = tk.Tk()
    r.withdraw()
    messagebox.showerror(
        "OCR 引擎不可用",
        "没有找到 PaddleOCR 识别引擎。\n\n"
        "原因: 你使用的 Python 不是装有 PaddleOCR 的那个环境。\n"
        f"当前 Python:\n{sys.executable}\n\n"
        "解决办法(任选其一):\n"
        "  1. 双击本目录下的「启动程序.bat」即可\n"
        f"  2. 用命令行运行:\n     \"{CONDA_PY}\" plate_recognition.py\n\n"
        "详细错误见同目录「错误日志.txt」"
    )
    r.destroy()

def main():
    # 用错 Python 时自动切换到正确环境
    if try_relaunch():
        return

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args and os.path.isfile(args[0]):
        cli_mode(args[0])
    else:
        run_gui()

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
辅助工具：手动截取车牌区域
==========================
当自动定位车牌失败时，可以用这个工具手动框选车牌，
程序会帮你把车牌区域单独截取并保存。

用法：
    python crop_plate.py            # 处理 sample_images 里的图片
    python crop_plate.py 图片路径
"""

import os
import sys
import cv2


def select_plate(img_path):
    """让用户用鼠标框选车牌区域"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[错误] 无法读取图片：{img_path}")
        return

    # 如果图片太大，先缩小方便操作
    max_w = 1200
    if img.shape[1] > max_w:
        scale = max_w / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"[信息] 图片较大，已缩放为 {img.shape[1]}x{img.shape[0]}")

    # 记录鼠标框选区域
    refPt = []
    cropping = False

    def on_mouse(event, x, y, flags, param):
        nonlocal cropping, refPt
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:
            refPt.append((x, y))
            cropping = False
            # 画出矩形
            cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", img)
            print(f"[选择] 框选了: {refPt[0]} -> {refPt[1]}")

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_mouse)
    print("鼠标左键拖拽框选车牌区域，选好后按回车确认，按 ESC 取消。")
    cv2.imshow("image", img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 回车确认
            break
        elif key == 27:  # ESC 取消
            print("[信息] 已取消")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    if len(refPt) == 2:
        x1, y1 = refPt[0]
        x2, y2 = refPt[1]
        # 规范化坐标
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 裁剪
        crop = img[y1:y2, x1:x2]

        # 保存裁剪结果
        save_path = os.path.splitext(img_path)[0] + "_plate.jpg"
        cv2.imwrite(save_path, crop)
        print(f"[保存] 车牌裁剪图已保存到: {save_path}")
        print(f"[裁剪尺寸] {crop.shape[1]}x{crop.shape[0]}")

        # 显示裁剪结果
        cv2.imshow("Cropped Plate", crop)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[错误] 没有完成框选")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
        if os.path.isdir(sample_dir):
            imgs = [f for f in os.listdir(sample_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if imgs:
                img_path = os.path.join(sample_dir, imgs[0])
                print(f"[信息] 未指定图片，默认使用: {img_path}")
            else:
                print("[错误] sample_images 文件夹里没有图片！")
                sys.exit(1)
        else:
            print("[错误] 请指定图片路径，例如：python crop_plate.py car.jpg")
            sys.exit(1)

    select_plate(img_path)

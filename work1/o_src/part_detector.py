#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#输入：图像，输出：（名称）
"""
同型号零件复检工具（传统图像处理，无深度学习）

用法:
    1) 标定模板（只需做一次，用确认合格的标准件图片）:
       python part_detector.py calibrate 标准件.jpg
       -> 生成 part_template.npz

    2) 复检新图片（零件任意位置、任意角度、远近不同都可以）:
       python part_detector.py check 新图1.jpg [新图2.jpg ...]
       -> 自动判断: 是不是同一个零件 + 尺寸/孔位是否合格

原理（工业检测里的"金模板 + 几何引导验证"思路）:
    1. 分割: 灰度 -> 模糊 -> 反二值化 -> 形态学 -> 最大外轮廓
    2. 认零件: Hu 矩 matchShapes 形状匹配（平移/旋转/缩放不变）
    3. 验尺寸: 比对缩放/旋转不变的相对特征（圆度、密实度、长宽比等）
    4. 验孔位: 标定时把每个孔记在"零件局部坐标系"（相对质心、随角度归一化），
       复检时按零件实测位置/角度/大小预测每个孔应在哪，在预测点做小范围
       霍夫圆 + 亮度验证 => 孔在不在、孔径对不对一目了然。
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# ---------------------- 参数 ----------------------
THRESH = 120               # 二值化阈值（黑零件浅背景）
SHAPE_MATCH_MAX = 0.15     # matchShapes 距离 < 此值 => 同一零件
HOLE_POS_TOL = 0.18        # 预测孔位容差（外接圆半径的倍数）
TEMPLATE_FILE = Path(__file__).parent / "part_template.npz"

TOLERANCES = {             # 相对特征公差（与模板值的相对偏差）
    "circularity": 0.15,   # 圆度 4*pi*A/P^2
    "solidity": 0.05,      # 密实度 轮廓面积/凸包面积
    "aspect": 0.15,        # 最小外接矩形长宽比
    "extent": 0.10,        # 轮廓面积/最小外接矩形面积
    "hole_ratio": 0.30,    # 单孔半径 / 零件外接圆半径 的偏差
}


# ---------------------- 基础分割 ----------------------
def segment(img):
    """返回零件外轮廓（最大外层轮廓）与二值图，找不到返回 (None, binary)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, THRESH, 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, binary
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1000:
        return None, binary
    return cnt, binary


def part_frame(cnt):
    """零件的位姿: 质心、最小外接矩形角度、外接圆半径。"""
    M = cv2.moments(cnt)
    centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    rect = cv2.minAreaRect(cnt)
    angle = np.deg2rad(rect[2])          # 零件姿态角
    (_, _), enc_r = cv2.minEnclosingCircle(cnt)
    return centroid, angle, enc_r


def to_local(pos, centroid, angle, enc_r):
    """世界坐标 -> 零件局部坐标（随零件旋转/缩放归一化）。"""
    ca, sa = np.cos(angle), np.sin(angle)
    R = np.array([[ca, sa], [-sa, ca]])
    return R @ ((np.asarray(pos, float) - centroid) / enc_r)


def to_world(local, centroid, angle, enc_r):
    """零件局部坐标 -> 世界坐标。"""
    ca, sa = np.cos(angle), np.sin(angle)
    R = np.array([[ca, -sa], [sa, ca]])
    return centroid + enc_r * (R @ np.asarray(local, float))


# ---------------------- 孔检测 ----------------------
def hough_circles_roi(gray, center, exp_r):
    """在预测孔位附近的小 ROI 内跑霍夫圆，返回最佳圆 (cx,cy,r) 或 None。"""
    h, w = gray.shape[:2]
    pad = int(exp_r * 2.0)
    x0, y0 = int(center[0]) - pad, int(center[1]) - pad
    x1, y1 = int(center[0]) + pad, int(center[1]) + pad
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, w), min(y1, h)
    roi = gray[y0c:y1c, x0c:x1c]
    if roi.size == 0:
        return None
    blur = cv2.medianBlur(roi, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=max(int(exp_r), 5),
        param1=100, param2=8,
        minRadius=max(4, int(exp_r * 0.7)), maxRadius=int(exp_r * 1.4),
    )
    if circles is None:
        return None
    best, best_d = None, 1e9
    for c in circles[0]:
        cx, cy, r = c[0] + x0c, c[1] + y0c, c[2]
        d = np.hypot(cx - center[0], cy - center[1])
        if d < best_d and d <= exp_r:      # 圆心离预测点不能超过一个孔径
            best, best_d = (int(cx), int(cy), int(r)), d
    return best


def global_hough_holes(gray, part_mask, enc_r):
    """全局霍夫圆找候选孔（标定用）。p2=15 在清晰图上足够，
    再用孔径占比下限过滤噪声圆。"""
    blur = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    holes = []
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=int(enc_r * 0.2),
        param1=100, param2=15,
        minRadius=max(5, int(enc_r * 0.04)), maxRadius=int(enc_r * 0.45),
    )
    if circles is not None:
        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            if not (0 <= cx < w and 0 <= cy < h and part_mask[cy, cx] == 255):
                continue
            if r / enc_r < 0.05:            # 过小的圆视为噪声
                continue
            if any(np.hypot(cx - e[0], cy - e[1]) < 0.8 * max(r, e[2])
                   for e in holes):
                continue
            holes.append((cx, cy, r))
    holes.sort(key=lambda x: x[2] * x[2], reverse=True)
    return holes


# ---------------------- 特征提取 ----------------------
def extract_features(img):
    """分割 + 位姿 + 缩放/旋转不变特征。失败返回 None。"""
    cnt, binary = segment(img)
    if cnt is None:
        return None
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    centroid, angle, enc_r = part_frame(cnt)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    hull_area = cv2.contourArea(cv2.convexHull(cnt))

    return {
        "contour": cnt,
        "binary": binary,
        "area": area,
        "perimeter": perimeter,
        "centroid": centroid,
        "angle": angle,
        "enc_radius": enc_r,
        "circularity": 4 * np.pi * area / max(perimeter ** 2, 1e-6),
        "solidity": area / max(hull_area, 1e-6),
        "aspect": max(rw, rh) / max(min(rw, rh), 1e-6),
        "extent": area / max(rw * rh, 1e-6),
    }


# ---------------------- 标定 ----------------------
def cmd_calibrate(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图像: {img_path}")
        return 1
    feat = extract_features(img)
    if feat is None:
        print("未能从图中分割出零件，请确认图片是黑零件浅背景")
        return 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    part_mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(part_mask, [feat["contour"]], -1, 255, -1)

    holes = global_hough_holes(gray, part_mask, feat["enc_radius"])
    hole_locals = []
    for hx, hy, r in holes:
        local = to_local((hx, hy), feat["centroid"], feat["angle"], feat["enc_radius"])
        hole_locals.append([local[0], local[1], r / feat["enc_radius"]])

    np.savez(
        TEMPLATE_FILE,
        contour=feat["contour"],
        enc_radius=feat["enc_radius"],
        circularity=feat["circularity"],
        solidity=feat["solidity"],
        aspect=feat["aspect"],
        extent=feat["extent"],
        hole_locals=np.array(hole_locals),
    )
    print("=" * 52)
    print(f"模板已标定并保存: {TEMPLATE_FILE}")
    print(f"  面积: {feat['area']:.0f} px^2 | 外接圆半径: {feat['enc_radius']:.0f} px")
    print(f"  圆度: {feat['circularity']:.4f} | 密实度: {feat['solidity']:.4f}")
    print(f"  长宽比: {feat['aspect']:.4f} | 矩形填充率: {feat['extent']:.4f}")
    print(f"  孔数: {len(hole_locals)}")
    for i, (lx, ly, rr) in enumerate(hole_locals, 1):
        print(f"    孔{i}: 局部坐标({lx:.3f},{ly:.3f}), 孔径占比 {rr:.3f}")
    print("=" * 52)
    return 0


# ---------------------- 复检 ----------------------
def verify_holes(img, feat, tpl_holes):
    """几何引导验证: 按零件位姿预测每个模板孔的位置, 在预测点做小范围霍夫圆 +
    前景比例验证(孔在二值图中会和零件合并成前景)。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = feat["binary"]
    h, w = gray.shape[:2]

    results = []
    for lx, ly, rr in tpl_holes:
        exp_r = rr * feat["enc_radius"]
        center = to_world((lx, ly), feat["centroid"], feat["angle"], feat["enc_radius"])
        cx, cy = int(round(center[0])), int(round(center[1]))
        # 1) 局部霍夫圆
        found = hough_circles_roi(gray, (cx, cy), exp_r) if (
            0 <= cx < w and 0 <= cy < h) else None
        if found is not None:
            fx, fy, fr = found
            # 验证孔在二值图中确实与零件前景相连(比例≥0.4)
            cm = np.zeros(gray.shape, np.uint8)
            cv2.circle(cm, (fx, fy), int(fr * 0.55), 255, -1)
            fg = cv2.countNonZero(cv2.bitwise_and(binary, cm))
            total = cv2.countNonZero(cm)
            if total > 0 and fg / total >= 0.4:
                results.append(((lx, ly, rr), "OK",
                                {"center": (fx, fy), "radius": fr}))
                continue
        results.append(((lx, ly, rr), "MISSING", {"center": (cx, cy),
                                                  "radius": int(exp_r)}))
    return results


def cmd_check(img_path, out_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图像: {img_path}")
        return 1
    if not TEMPLATE_FILE.exists():
        print(f"找不到模板 {TEMPLATE_FILE}，请先用 calibrate 命令标定")
        return 1

    tpl = np.load(TEMPLATE_FILE, allow_pickle=True)
    tpl_contour = tpl["contour"]
    feat = extract_features(img)
    if feat is None:
        print(f"[{img_path}] 未检测到零件 -> NG")
        return 1

    # 1) 形状匹配（Hu 矩, 平移/旋转/缩放不变）
    match_dist = cv2.matchShapes(tpl_contour, feat["contour"],
                                 cv2.CONTOURS_MATCH_I1, 0)
    is_same_part = match_dist < SHAPE_MATCH_MAX

    # 2) 相对特征比对
    checks = []

    def add(name, val, ref, tol):
        ok = abs(val - ref) / max(abs(ref), 1e-6) <= tol
        checks.append((name, val, ref, ok))

    add("圆度", feat["circularity"], float(tpl["circularity"]), TOLERANCES["circularity"])
    add("密实度", feat["solidity"], float(tpl["solidity"]), TOLERANCES["solidity"])
    add("长宽比", feat["aspect"], float(tpl["aspect"]), TOLERANCES["aspect"])
    add("矩形填充率", feat["extent"], float(tpl["extent"]), TOLERANCES["extent"])

    # 3) 几何引导孔位验证（独立于形状匹配, 始终执行以便完整诊断）
    tpl_holes = [tuple(h) for h in tpl["hole_locals"]]
    hole_results = verify_holes(img, feat, tpl_holes)
    holes_ok = all(s == "OK" for _, s, _ in hole_results)
    checks.append(("孔位/孔数", f"{sum(1 for _,s,_ in hole_results if s=='OK')}/"
                  f"{len(tpl_holes)} 验证通过", f"{len(tpl_holes)} 个孔",
                  holes_ok and is_same_part))

    # 4) 孔径检查
    for (lx, ly, rr), status, info in hole_results:
        if status != "OK":
            continue
        ratio = info["radius"] / feat["enc_radius"]
        ok = abs(ratio - rr) / max(rr, 1e-6) <= TOLERANCES["hole_ratio"]
        checks.append((f"孔径@({lx:.2f},{ly:.2f})", round(ratio, 3),
                       round(float(rr), 3), ok))

    all_ok = is_same_part and all(c[3] for c in checks)
    status = "PASS" if all_ok else "NG"

    # 可视化
    vis = img.copy()
    cv2.drawContours(vis, [feat["contour"]], -1, (0, 255, 0), 2)
    for (lx, ly, rr), st, info in hole_results:
        color = (0, 200, 0) if st == "OK" else (0, 0, 255)
        cv2.circle(vis, info["center"], info["radius"], color, 2)
        cv2.circle(vis, info["center"], 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 200, 0) if all_ok else (0, 0, 255)
    cv2.putText(vis, f"{status}  shapeDist={match_dist:.4f}", (20, 40),
                font, 1.0, color, 2)
    cv2.putText(vis, f"Part matched: {'YES' if is_same_part else 'NO'}",
                (20, 75), font, 0.7, (0, 0, 0), 2)

    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"check_{Path(img_path).stem}.jpg"
    cv2.imwrite(str(out_path), vis)

    # 报告
    print("=" * 52)
    print(f"复检: {img_path}")
    print(f"  形状匹配距离: {match_dist:.4f} (阈值 {SHAPE_MATCH_MAX}) "
          f"-> {'同一零件' if is_same_part else '不是该零件'}")
    for name_, val, ref, ok in checks:
        flag = "PASS" if ok else "FAIL"
        print(f"  {name_}: 实测={val} 模板={ref} -> {flag}")
    print(f"  总判定: {status}")
    print(f"  结果图: {out_path}")
    print("=" * 52)
    return 0 if all_ok else 2


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd, paths = sys.argv[1], sys.argv[2:]
    out_dir = Path(__file__).parent / "output"
    rc = 0
    for p in paths:
        if cmd == "calibrate":
            rc = cmd_calibrate(p)
            break                      # 模板只用第一张标定
        elif cmd == "check":
            rc |= cmd_check(p, out_dir)
        else:
            print(__doc__)
            return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

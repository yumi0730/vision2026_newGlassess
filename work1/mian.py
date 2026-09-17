import cv2
import numpy as np
import math


# =========================================================
# 图片路径存放位置
# =========================================================
IMAGE_PATH = r"E:\Windows_Desktop\work1\12.jpg"
# =========================================================
# ① 红色圆点检测 """检测图片中的红色圆形物体。然后返回：[(x, y, radius), ...]"""
# =========================================================
def detect_red_circle(image):
    # 转换到 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 中分成两个区间
    mask1 = cv2.inRange(
        hsv,
        np.array([0, 43, 46]),
        np.array([10, 255, 255])
    )
    mask2 = cv2.inRange(
        hsv,
        np.array([156, 43, 46]),
        np.array([180, 255, 255])
    )
    # 合并两个红色区域
    mask = cv2.bitwise_or(mask1, mask2)
    # 去除小噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # 查找红色轮廓
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    red_points = []

    for contour in contours:
        # 面积太小，认为是噪声
        area = cv2.contourArea(contour)
        if area < 30:
            
            continue
        # 获取外接矩形
        x, y, w, h = cv2.boundingRect(contour)

        if w == 0 or h == 0:
            continue

        # 红圆点的宽高应该比较接近
        ratio = w / float(h)

        if ratio < 0.75 or ratio > 1.25:
            continue

        # 计算圆形度
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * math.pi * area /
            (perimeter * perimeter)
        )

        # 圆形度越接近 1，越像圆
        if circularity < 0.70:
            continue

        # 获取最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # 限制圆点大小
        if radius < 5 or radius > 200:
            continue

        red_points.append(
            (int(cx), int(cy), int(radius))
        )

    return red_points


# =========================================================
# ② 判断黑色轮廓是否可靠
# =========================================================
def valid_black_contour(contour):
    """
    判断一个黑色轮廓是否有可能是黑线。

    主要作用：
    排除黑色小杂物和形状异常的干扰物。
    """

    # -----------------------------------------------------
    # 面积过滤
    # -----------------------------------------------------
    area = cv2.contourArea(contour)

    if area < 80:
        return False

    # -----------------------------------------------------
    # 外接矩形
    # -----------------------------------------------------
    x, y, w, h = cv2.boundingRect(contour)

    if w < 8 or h < 8:
        return False

    # -----------------------------------------------------
    # 计算轮廓填充率
    #
    # 面积 / 外接矩形面积
    #
    # 太低说明轮廓比较零散，可能是噪声。
    # -----------------------------------------------------
    fill_ratio = area / float(w * h)

    if fill_ratio < 0.15:
        return False

    # -----------------------------------------------------
    # 计算轮廓中心
    # -----------------------------------------------------
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return False

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # 外接矩形中心
    rect_cx = x + w / 2
    rect_cy = y + h / 2

    # 轮廓中心与矩形中心的距离
    error = math.sqrt(
        (cx - rect_cx) ** 2 +
        (cy - rect_cy) ** 2
    )

    # 两个中心相差太大，认为轮廓形状不可靠
    if error > max(w, h) * 0.35:
        return False

    return True


# =========================================================
# ③ 获取黑色轮廓中心点
# =========================================================
def get_center(contour, y_offset):
    """
    计算轮廓中心点。

    y_offset：
        因为我们检测的是 ROI，
        所以需要把 ROI 坐标转换回原图坐标。
    """

    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None

    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"]) + y_offset

    return x, y


# =========================================================
# ④ 黑线检测
# =========================================================
def detect_black_line(image):
    """
    检测黑线，并返回黑线中心点。

    图片被分成4个区域：
    
        ┌──────────────┐
        │     第1层     │
        ├──────────────┤
        │     第2层     │
        ├──────────────┤
        │     第3层     │
        ├──────────────┤
        │     第4层     │
        └──────────────┘

    每一层寻找一个最可能属于黑线的轮廓。
    """

    height, width = image.shape[:2]

    # -----------------------------------------------------
    # 转灰度
    # -----------------------------------------------------
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # 黑色区域变成白色
    # -----------------------------------------------------
    _, binary = cv2.threshold(
        gray,
        80,
        255,
        cv2.THRESH_BINARY_INV
    )

    # -----------------------------------------------------
    # 去除小噪声
    # -----------------------------------------------------
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    centers = []

    # 每层高度
    slice_height = height // 4

    # =====================================================
    # 逐层检测
    # =====================================================
    for i in range(4):

        y1 = i * slice_height

        # 最后一层直接到图片底部
        y2 = height if i == 3 else (i + 1) * slice_height

        roi = binary[y1:y2, :]

        # 查找当前区域中的黑色轮廓
        contours, _ = cv2.findContours(
            roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        # =================================================
        # 筛选候选轮廓
        # =================================================
        for contour in contours:

            if not valid_black_contour(contour):
                continue

            center = get_center(
                contour,
                y1
            )

            if center is None:
                continue

            cx, cy = center

            # 轮廓面积
            area = cv2.contourArea(contour)

            # -------------------------------------------------
            # 基础评分 = 面积
            # -------------------------------------------------
            score = area

            # -------------------------------------------------
            # 如果已经找到上一层中心点，
            # 优先选择与上一层连续的轮廓。
            #
            # 这是防止黑色干扰物抢走中心点的关键。
            # -------------------------------------------------
            if centers:

                previous_x = centers[-1][0]

                distance = abs(
                    cx - previous_x
                )

                # 距离越近，说明越可能属于同一条黑线
                if distance < width * 0.15:

                    score *= (
                        1 +
                        (1 -
                         distance /
                         (width * 0.15)) * 1.5
                    )

                # 跳得特别远，大幅降低优先级
                elif distance > width * 0.25:

                    score *= 0.1

            candidates.append(
                (score, contour, center)
            )

        # 当前层没有可靠轮廓
        if not candidates:
            continue

        # 选择评分最高的轮廓
        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        _, contour, center = candidates[0]

        cx, cy = center

        # -----------------------------------------------------
        # 再次检查中心点是否突然跳跃
        # -----------------------------------------------------
        if centers:

            previous_x = centers[-1][0]

            # 最大允许横向跳跃
            if abs(cx - previous_x) > width * 0.30:

                # 尝试从其他候选中寻找连续的轮廓
                found = False

                for _, other_contour, other_center in candidates[1:]:

                    other_x = other_center[0]

                    if abs(other_x - previous_x) <= width * 0.30:

                        contour = other_contour
                        cx, cy = other_center

                        found = True
                        break

                # 没找到可靠轮廓
                if not found:
                    continue

        # -----------------------------------------------------
        # 保存中心点
        # -----------------------------------------------------
        centers.append(
            (cx, cy)
        )

        # -----------------------------------------------------
        # 只画中心点
        #
        # 不画黑线轮廓
        # 不画连接线
        # -----------------------------------------------------
        cv2.circle(
            image,
            (cx, cy),
            6,
            (0, 255, 0),
            -1
        )

    return centers


# =========================================================
# ⑤ 根据黑线中心点判断方向
# =========================================================
def judge_line(centers):
    """
    根据所有黑线中心点的整体趋势判断：

        直行
        左转
        右转

    注意：
    这里不再根据“黑线距离图片中心有多远”判断。

    因此即使整条直线偏左或偏右，
    只要它本身是直的，仍然判断为直行。
    """

    # 少于两个点无法计算方向
    if len(centers) < 2:
        return "直行"

    # 按照 Y 坐标从上到下排序
    centers = sorted(
        centers,
        key=lambda p: p[1]
    )

    # -----------------------------------------------------
    # 提取 X、Y 坐标
    # -----------------------------------------------------
    x = np.array(
        [p[0] for p in centers],
        dtype=np.float32
    )

    y = np.array(
        [p[1] for p in centers],
        dtype=np.float32
    )

    # =====================================================
    # 两个点：直接计算角度
    # =====================================================
    if len(centers) == 2:

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        if abs(dy) < 1:
            return "直行"

        angle = math.degrees(
            math.atan2(
                dx,
                abs(dy)
            )
        )

    # =====================================================
    # 3个或4个点：
    # 用所有点拟合一条整体直线。
    #
    # 这样可以降低单个中心点错误带来的影响。
    # =====================================================
    else:

        slope, _ = np.polyfit(
            y,
            x,
            1
        )

        angle = math.degrees(
            math.atan(slope)
        )

    # -----------------------------------------------------
    # 输出角度，方便调试
    # -----------------------------------------------------
    print(
        "黑线中心点：",
        centers
    )

    print(
        "整体角度：",
        round(angle, 2),
        "°"
    )

    # -----------------------------------------------------
    # 小于15°认为是直行
    # -----------------------------------------------------
    if abs(angle) <= 15:
        return "直行"

    # -----------------------------------------------------
    # 保留原来的方向定义
    # -----------------------------------------------------
    if angle > 0:
        return "左转 变慢"

    return "右转 变慢"


# =========================================================
# ⑥ 主程序
# =========================================================

# ---------------------------------------------------------
# 读取图片
# ---------------------------------------------------------
image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    print(
        "无法读取图片：",
        IMAGE_PATH
    )

    exit()


# ---------------------------------------------------------
# 检测黑线
# ---------------------------------------------------------
centers = detect_black_line(
    image
)


# ---------------------------------------------------------
# 判断方向
# ---------------------------------------------------------
direction = judge_line(
    centers
)


# ---------------------------------------------------------
# 检测红圆点
# ---------------------------------------------------------
red_points = detect_red_circle(
    image
)


# ---------------------------------------------------------
# 在图片上标记红圆点
# ---------------------------------------------------------
for x, y, radius in red_points:

    # 红点中心
    cv2.circle(
        image,
        (x, y),
        5,
        (255, 0, 0),
        -1
    )

    # 红点外接圆
    cv2.circle(
        image,
        (x, y),
        radius,
        (0, 255, 0),
        2
    )

    # RED文字
    cv2.putText(
        image,
        "RED",
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# =========================================================
# 组合最终结果
# =========================================================
result = direction

if red_points:
    result += "  红点"


# =========================================================
# 控制台输出
# =========================================================
print("--------------------------------")
print("识别结果：", result)
print("黑线中心点数量：", len(centers))
print("红圆点数量：", len(red_points))
print("--------------------------------")


# =========================================================
# 在图片上显示识别结果
# =========================================================
cv2.putText(
    image,
    result,
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (0, 0, 255),
    3
)


# =========================================================
# 显示图片
# =========================================================
cv2.imshow(
    "Line Detection",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()

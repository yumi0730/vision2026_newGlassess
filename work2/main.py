import cv2
import numpy as np
import os
import json
import sys
import math


# ============================================================
#                  参数设置
# ============================================================

# 模板保存文件夹
TEMPLATE_DIR = "templates"

# 检测结果保存文件夹
RESULT_DIR = "results"

# ORB最大特征点数量
ORB_FEATURES = 1000

# 最小零件面积
MIN_AREA = 300

# 最大零件面积
MAX_AREA = 1000000

# 最低识别分数
MATCH_THRESHOLD = 45


# 创建文件夹
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
#                  ORB
# ============================================================

orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES
)


# ============================================================
#                  图像预处理
# ============================================================

def preprocess(image):
    """
    图像预处理。

    主要作用：
    1. 降噪
    2. 增强零件边缘
    """

    # 转灰度
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 高斯滤波
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return gray


# ============================================================
#                  自动提取零件Mask
# ============================================================

def create_mask(image):
    """
    自动提取零件。

    兼容：
    绿色桌面
    白色背景
    浅色背景

    返回：
        mask
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # 方法1：绿色背景
    # --------------------------------------------------------

    lower_green = np.array(
        [30, 25, 20]
    )

    upper_green = np.array(
        [100, 255, 255]
    )

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # 非绿色区域
    mask_green = cv2.bitwise_not(
        green_mask
    )

    # --------------------------------------------------------
    # 方法2：灰度阈值
    #
    # 主要针对白色背景
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # OTSU自动阈值
    _, mask_dark = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 判断哪一种Mask更合理
    # --------------------------------------------------------

    def calculate_score(mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return 0

        largest = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            largest
        )

        image_area = (
            image.shape[0] *
            image.shape[1]
        )

        ratio = area / image_area

        # 太小或者太大都不合理
        if ratio < 0.005:
            return 0

        if ratio > 0.8:
            return 0

        return area

    score_green = calculate_score(
        mask_green
    )

    score_dark = calculate_score(
        mask_dark
    )

    # --------------------------------------------------------
    # 选择较合理的Mask
    # --------------------------------------------------------

    if score_green > score_dark:

        mask = mask_green

    else:

        mask = mask_dark

    # --------------------------------------------------------
    # 形态学处理
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # 去掉小噪声
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # 填补小缺口
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask


# ============================================================
#                  获取最大零件轮廓
# ============================================================

def get_main_contour(mask):
    """
    获取面积最大的轮廓。
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        valid_contours.append(
            contour
        )

    if len(valid_contours) == 0:

        return None

    # 面积最大的轮廓
    contour = max(
        valid_contours,
        key=cv2.contourArea
    )

    return contour


# ============================================================
#                  Hu矩
# ============================================================

def calculate_hu(contour):

    moments = cv2.moments(
        contour
    )

    hu = cv2.HuMoments(
        moments
    )

    hu = hu.flatten()

    result = []

    for value in hu:

        if value == 0:

            value = 1e-10

        value = (
            -np.sign(value) *
            np.log10(abs(value))
        )

        result.append(
            float(value)
        )

    return np.array(result)


# ============================================================
#                  ORB特征
# ============================================================

def calculate_orb(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = (
        orb.detectAndCompute(
            gray,
            None
        )
    )

    return keypoints, descriptors


# ============================================================
#                  几何特征
# ============================================================

def calculate_geometry(contour):

    # 面积
    area = cv2.contourArea(
        contour
    )

    # 周长
    perimeter = cv2.arcLength(
        contour,
        True
    )

    # 外接矩形
    x, y, w, h = cv2.boundingRect(
        contour
    )

    # 长宽比
    aspect_ratio = (
        w / float(h)
        if h != 0
        else 0
    )

    # 圆度
    #
    # 越接近1越圆
    #
    if perimeter > 0:

        circularity = (
            4 *
            math.pi *
            area /
            (perimeter * perimeter)
        )

    else:

        circularity = 0

    # 最小外接矩形
    rect = cv2.minAreaRect(
        contour
    )

    (_, _), (rw, rh), angle = rect

    if rw < rh:

        angle = angle + 90

    return {
        "area": float(area),
        "width": int(w),
        "height": int(h),
        "aspect_ratio": float(
            aspect_ratio
        ),
        "circularity": float(
            circularity
        ),
        "angle": float(angle)
    }


# ============================================================
#                  提取所有特征
# ============================================================

def extract_features(
    image,
    contour
):

    hu = calculate_hu(
        contour
    )

    keypoints, descriptors = (
        calculate_orb(
            image
        )
    )

    geometry = calculate_geometry(
        contour
    )

    return {
        "hu": hu,
        "descriptors": descriptors,
        "geometry": geometry
    }


# ============================================================
#                  裁剪零件
# ============================================================

def crop_part(
    image,
    contour,
    padding=20
):

    x, y, w, h = cv2.boundingRect(
        contour
    )

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image.shape[1],
        x + w + padding
    )

    y2 = min(
        image.shape[0],
        y + h + padding
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    return crop


# ============================================================
#                  生成模板
# ============================================================

def create_template(
    image_path,
    part_name
):

    print()
    print("=" * 60)
    print("开始生成模板")
    print("=" * 60)

    # --------------------------------------------------------
    # 读取图片
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "错误：无法读取图片！"
        )

        return False

    print(
        "图片：",
        image_path
    )

    # --------------------------------------------------------
    # 创建Mask
    # --------------------------------------------------------

    mask = create_mask(
        image
    )

    # --------------------------------------------------------
    # 找零件
    # --------------------------------------------------------

    contour = get_main_contour(
        mask
    )

    if contour is None:

        print()
        print(
            "错误：没有检测到零件！"
        )

        print(
            "请确保图片中只有一个零件，"
        )

        print(
            "并且零件与背景存在明显区别。"
        )

        return False

    # --------------------------------------------------------
    # 检查是不是检测到了多个物体
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_count = 0

    for c in contours:

        area = cv2.contourArea(c)

        if (
            MIN_AREA <
            area <
            MAX_AREA
        ):

            valid_count += 1

    if valid_count > 1:

        print()
        print(
            "警告：图片中可能存在多个物体。"
        )

        print(
            "检测到：",
            valid_count,
            "个物体"
        )

        print(
            "程序将使用面积最大的物体作为模板。"
        )

    # --------------------------------------------------------
    # 裁剪零件
    # --------------------------------------------------------

    crop = crop_part(
        image,
        contour
    )

    # --------------------------------------------------------
    # 重新提取裁剪图的轮廓
    # --------------------------------------------------------

    crop_mask = create_mask(
        crop
    )

    crop_contour = get_main_contour(
        crop_mask
    )

    if crop_contour is None:

        print(
            "模板轮廓提取失败！"
        )

        return False

    # --------------------------------------------------------
    # 提取特征
    # --------------------------------------------------------

    features = extract_features(
        crop,
        crop_contour
    )

    # --------------------------------------------------------
    # 创建模板目录
    # --------------------------------------------------------

    template_folder = os.path.join(
        TEMPLATE_DIR,
        part_name
    )

    os.makedirs(
        template_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 保存模板图片
    # --------------------------------------------------------

    template_image_path = os.path.join(
        template_folder,
        "template.png"
    )

    cv2.imwrite(
        template_image_path,
        crop
    )

    # --------------------------------------------------------
    # 保存ORB描述子
    # --------------------------------------------------------

    descriptor_path = os.path.join(
        template_folder,
        "descriptors.npy"
    )

    if features["descriptors"] is not None:

        np.save(
            descriptor_path,
            features["descriptors"]
        )

    # --------------------------------------------------------
    # 保存特征参数
    # --------------------------------------------------------

    geometry = features[
        "geometry"
    ]

    feature_data = {

        "name": part_name,

        "hu": features[
            "hu"
        ].tolist(),

        "area": geometry[
            "area"
        ],

        "width": geometry[
            "width"
        ],

        "height": geometry[
            "height"
        ],

        "aspect_ratio": geometry[
            "aspect_ratio"
        ],

        "circularity": geometry[
            "circularity"
        ],

        "angle": geometry[
            "angle"
        ]
    }

    json_path = os.path.join(
        template_folder,
        "features.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_data,
            f,
            ensure_ascii=False,
            indent=4
        )

    # --------------------------------------------------------
    # 保存Mask
    # --------------------------------------------------------

    mask_path = os.path.join(
        template_folder,
        "mask.png"
    )

    cv2.imwrite(
        mask_path,
        crop_mask
    )

    # --------------------------------------------------------
    # 打印结果
    # --------------------------------------------------------

    print()
    print("模板生成成功！")
    print()
    print(
        "零件名称：",
        part_name
    )

    print(
        "模板图片：",
        template_image_path
    )

    print(
        "特征文件：",
        json_path
    )

    print(
        "ORB文件：",
        descriptor_path
    )

    print(
        "零件宽度：",
        geometry["width"]
    )

    print(
        "零件高度：",
        geometry["height"]
    )

    print(
        "长宽比：",
        round(
            geometry["aspect_ratio"],
            3
        )
    )

    print(
        "圆度：",
        round(
            geometry["circularity"],
            3
        )
    )

    print("=" * 60)

    # --------------------------------------------------------
    # 显示模板
    # --------------------------------------------------------

    show_image = crop.copy()

    cv2.imshow(
        "Generated Template",
        show_image
    )

    print()
    print(
        "按任意键关闭模板预览..."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True


# ============================================================
#                  加载模板
# ============================================================

def load_templates():

    templates = []

    if not os.path.exists(
        TEMPLATE_DIR
    ):

        return templates

    # --------------------------------------------------------
    # 遍历模板目录
    # --------------------------------------------------------

    for name in os.listdir(
        TEMPLATE_DIR
    ):

        folder = os.path.join(
            TEMPLATE_DIR,
            name
        )

        if not os.path.isdir(folder):

            continue

        json_path = os.path.join(
            folder,
            "features.json"
        )

        descriptor_path = os.path.join(
            folder,
            "descriptors.npy"
        )

        image_path = os.path.join(
            folder,
            "template.png"
        )

        if not os.path.exists(
            json_path
        ):

            continue

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            descriptors = None

            if os.path.exists(
                descriptor_path
            ):

                descriptors = np.load(
                    descriptor_path
                )

            templates.append({

                "name": data["name"],

                "hu": np.array(
                    data["hu"]
                ),

                "aspect_ratio":
                    data[
                        "aspect_ratio"
                    ],

                "circularity":
                    data[
                        "circularity"
                    ],

                "descriptors":
                    descriptors,

                "image":
                    image_path
            })

        except Exception as e:

            print(
                "模板读取失败：",
                folder,
                e
            )

    return templates


# ============================================================
#                  Hu矩匹配
# ============================================================

def compare_hu(
    hu1,
    hu2
):

    distance = np.linalg.norm(
        hu1 - hu2
    )

    score = (
        100 *
        math.exp(
            -distance / 3
        )
    )

    return min(
        100,
        score
    )


# ============================================================
#                  几何特征匹配
# ============================================================

def compare_geometry(
    current,
    template
):

    # --------------------------------------------------------
    # 长宽比
    # --------------------------------------------------------

    ratio1 = current[
        "aspect_ratio"
    ]

    ratio2 = template[
        "aspect_ratio"
    ]

    ratio_difference = abs(
        ratio1 - ratio2
    )

    ratio_score = max(
        0,
        100 -
        ratio_difference * 100
    )

    # --------------------------------------------------------
    # 圆度
    # --------------------------------------------------------

    circularity1 = current[
        "circularity"
    ]

    circularity2 = template[
        "circularity"
    ]

    circularity_difference = abs(
        circularity1 -
        circularity2
    )

    circularity_score = max(
        0,
        100 -
        circularity_difference * 100
    )

    return (
        ratio_score * 0.5 +
        circularity_score * 0.5
    )


# ============================================================
#                  ORB匹配
# ============================================================

def compare_orb(
    descriptors1,
    descriptors2
):

    if descriptors1 is None:
        return 0

    if descriptors2 is None:
        return 0

    if len(descriptors1) < 2:
        return 0

    if len(descriptors2) < 2:
        return 0

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    try:

        matches = matcher.knnMatch(
            descriptors1,
            descriptors2,
            k=2
        )

    except Exception:

        return 0

    good_matches = []

    for pair in matches:

        if len(pair) != 2:

            continue

        m, n = pair

        if (
            m.distance <
            0.75 * n.distance
        ):

            good_matches.append(m)

    # 根据好匹配数量计算分数
    score = min(
        100,
        len(good_matches) * 5
    )

    return score


# ============================================================
#                  识别一个零件
# ============================================================

def recognize_part(
    image,
    contour,
    templates
):

    # --------------------------------------------------------
    # 当前零件裁剪
    # --------------------------------------------------------

    crop = crop_part(
        image,
        contour
    )

    # --------------------------------------------------------
    # 当前零件Mask
    # --------------------------------------------------------

    mask = create_mask(
        crop
    )

    current_contour = get_main_contour(
        mask
    )

    if current_contour is None:

        return "未知零件", 0

    # --------------------------------------------------------
    # 当前零件特征
    # --------------------------------------------------------

    hu = calculate_hu(
        current_contour
    )

    geometry = calculate_geometry(
        current_contour
    )

    _, descriptors = calculate_orb(
        crop
    )

    best_name = "未知零件"
    best_score = 0

    # --------------------------------------------------------
    # 与每一个模板比较
    # --------------------------------------------------------

    for template in templates:

        # Hu
        hu_score = compare_hu(
            hu,
            template["hu"]
        )

        # 几何
        geometry_score = (
            compare_geometry(
                geometry,
                template
            )
        )

        # ORB
        orb_score = compare_orb(
            descriptors,
            template["descriptors"]
        )

        # ----------------------------------------------------
        # 综合评分
        #
        # Hu矩       40%
        # ORB        35%
        # 几何特征   25%
        # ----------------------------------------------------

        total_score = (
            hu_score * 0.40 +
            orb_score * 0.35 +
            geometry_score * 0.25
        )

        if total_score > best_score:

            best_score = total_score
            best_name = template[
                "name"
            ]

    # --------------------------------------------------------
    # 判断未知零件
    # --------------------------------------------------------

    if best_score < MATCH_THRESHOLD:

        best_name = "未知零件"

    return (
        best_name,
        best_score
    )


# ============================================================
#                  图片检测
# ============================================================

def detect_image(
    image_path
):

    print()
    print("=" * 60)
    print("开始零件检测")
    print("=" * 60)

    # --------------------------------------------------------
    # 读取图片
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "无法读取图片！"
        )

        return

    # --------------------------------------------------------
    # 加载模板
    # --------------------------------------------------------

    templates = load_templates()

    print(
        "模板数量：",
        len(templates)
    )

    if len(templates) == 0:

        print(
            "模板库为空！"
        )

        return

    print()
    print("当前模板：")

    for template in templates:

        print(
            "  ",
            template["name"]
        )

    # --------------------------------------------------------
    # 找零件
    # --------------------------------------------------------

    parts, mask = find_all_parts(
        image
    )

    print()
    print(
        "检测到零件：",
        len(parts)
    )

    result = image.copy()

    detection_data = []

    # --------------------------------------------------------
    # 一个一个识别
    # --------------------------------------------------------

    for i, contour in enumerate(parts):

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # 识别
        name, score = recognize_part(
            image,
            contour,
            templates
        )

        # 中心点
        center_x = x + w // 2
        center_y = y + h // 2

        # ----------------------------------------------------
        # 输出
        # ----------------------------------------------------

        print(
            "ID {:02d} | {:15s} | "
            "分数 {:.1f}% | "
            "中心 ({}, {})".format(
                i + 1,
                name,
                score,
                center_x,
                center_y
            )
        )

        # ----------------------------------------------------
        # 绘制框
        # ----------------------------------------------------

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # 绘制中心点
        # ----------------------------------------------------

        cv2.circle(
            result,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # 显示名称
        # ----------------------------------------------------

        text = "{} {:.1f}%".format(
            name,
            score
        )

        cv2.putText(
            result,
            text,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # 保存数据
        # ----------------------------------------------------

        detection_data.append({

            "id": i + 1,

            "name": name,

            "score": round(
                float(score),
                2
            ),

            "x": int(x),

            "y": int(y),

            "width": int(w),

            "height": int(h),

            "center_x": int(
                center_x
            ),

            "center_y": int(
                center_y
            )
        })

    # --------------------------------------------------------
    # 保存结果图片
    # --------------------------------------------------------

    filename = os.path.basename(
        image_path
    )

    result_path = os.path.join(
        RESULT_DIR,
        "result_" + filename
    )

    cv2.imwrite(
        result_path,
        result
    )

    # --------------------------------------------------------
    # 保存JSON
    # --------------------------------------------------------

    json_name = os.path.splitext(
        filename
    )[0] + ".json"

    json_path = os.path.join(
        RESULT_DIR,
        json_name
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            detection_data,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()
    print(
        "检测结果图片：",
        result_path
    )

    print(
        "检测结果数据：",
        json_path
    )

    # --------------------------------------------------------
    # 显示结果
    # --------------------------------------------------------

    cv2.imshow(
        "Part Detection Result",
        result
    )

    print()
    print(
        "按任意键退出..."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
#                  找到所有零件
# ============================================================

def find_all_parts(image):

    mask = create_mask(
        image
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    parts = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < 10 or h < 10:
            continue

        parts.append(
            contour
        )

    # 按从左到右排序
    parts.sort(
        key=lambda c:
        cv2.boundingRect(c)[0]
    )

    return parts, mask


# ============================================================
#                  批量生成模板
# ============================================================

def batch_create_templates():

    print()
    print("=" * 60)
    print("批量生成模板")
    print("=" * 60)

    folder = input(
        "请输入单个零件图片文件夹："
    ).strip()

    if not os.path.isdir(folder):

        print(
            "文件夹不存在！"
        )

        return

    files = os.listdir(
        folder
    )

    count = 0

    for filename in files:

        ext = os.path.splitext(
            filename
        )[1].lower()

        if ext not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]:

            continue

        image_path = os.path.join(
            folder,
            filename
        )

        # 文件名作为零件名称
        part_name = os.path.splitext(
            filename
        )[0]

        print()
        print(
            "正在处理：",
            filename
        )

        success = create_template(
            image_path,
            part_name
        )

        if success:

            count += 1

    print()
    print(
        "批量生成完成！"
    )

    print(
        "成功生成：",
        count,
        "个模板"
    )


# ============================================================
#                  主程序
# ============================================================

def main():

    while True:

        print()
        print("=" * 60)
        print("        传统算法零件识别系统")
        print("=" * 60)

        print()
        print("1. 一张图片生成一个模板")
        print("2. 批量生成模板")
        print("3. 检测一张零件图片")
        print("4. 查看模板库")
        print("Q. 退出")
        print()

        choice = input(
            "请选择："
        ).strip().lower()

        # ----------------------------------------------------
        # 单张图片生成模板
        # ----------------------------------------------------

        if choice == "1":

            image_path = input(
                "请输入单个零件图片路径："
            ).strip()

            part_name = input(
                "请输入零件名称："
            ).strip()

            if part_name == "":

                print(
                    "零件名称不能为空！"
                )

                continue

            create_template(
                image_path,
                part_name
            )

        # ----------------------------------------------------
        # 批量生成
        # ----------------------------------------------------

        elif choice == "2":

            batch_create_templates()

        # ----------------------------------------------------
        # 图片检测
        # ----------------------------------------------------

        elif choice == "3":

            image_path = input(
                "请输入待检测图片路径："
            ).strip()

            detect_image(
                image_path
            )

        # ----------------------------------------------------
        # 查看模板
        # ----------------------------------------------------

        elif choice == "4":

            templates = load_templates()

            print()
            print(
                "当前模板数量：",
                len(templates)
            )

            for template in templates:

                print(
                    "  -",
                    template["name"]
                )

        # ----------------------------------------------------
        # 退出
        # ----------------------------------------------------

        elif choice == "q":

            print(
                "程序退出。"
            )

            break

        else:

            print(
                "输入错误，请重新选择。"
            )


# ============================================================
#                  程序入口
# ============================================================

if __name__ == "__main__":

    main()
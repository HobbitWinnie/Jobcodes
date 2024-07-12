import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class BoundingBox:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

def get_bounding_boxes_from_mask(mask, classes):
    """将分割掩码转换为每个类别的边界框注释"""
    bounding_boxes = []

    # 遍历每个类别
    for cls in classes:
        # 创建二进制掩码，掩码中该类别为前景，其它为背景
        binary_mask = np.uint8(mask == cls)

        # 查找该类别二进制掩码中的轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历该类别的所有轮廓
        for contour in contours:
            # 获取每个轮廓的边界框坐标
            x, y, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(label=cls, xmin=x, ymin=y, xmax=x + w, ymax=y + h)
            bounding_boxes.append(bbox)

    return bounding_boxes

def mask_to_box_annotation(mask_path, classes):
    """转换单个掩码图像中的边界框注释"""
    # 读取分割掩码图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 获取边界框注释
    bounding_boxes = get_bounding_boxes_from_mask(mask, classes)
    return bounding_boxes

# 示例用法
if __name__ == "__main__":
    mask_path = 'path/to/your/mask_image.png'
    classes = [1, 2, 3]  # 定义目标类别

    bboxes = mask_to_box_annotation(mask_path, classes)

    # 输出生成的边界框注释
    for bbox in bboxes:
        print(f"Label: {bbox.label}, Bounding Box: ({bbox.xmin}, {bbox.ymin}, {bbox.xmax}, {bbox.ymax})")
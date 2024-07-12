import random
from PIL import Image, ImageDraw

@dataclass
class BoundingBox:
    label: str
    x_center: float
    y_center: float
    width: float
    height: float
    image_width: int
    image_height: int

class BoxToCaption:
    def __init__(self, bboxes, image_size):
        self.bboxes = bboxes
        self.image_width, self.image_height = image_size

    def is_center(self, bbox):
        """判断目标是否在图像中心"""
        center_x, center_y = self.image_width / 2, self.image_height / 2
        x_min = center_x - self.image_width * 0.1
        x_max = center_x + self.image_width * 0.1
        y_min = center_y - self.image_height * 0.1
        y_max = center_y + self.image_height * 0.1

        return x_min <= bbox.x_center <= x_max and y_min <= bbox.y_center <= y_max

    def generate_captions(self):
        """生成五个描述"""
        center_captions = []
        non_center_captions = []

        for bbox in self.bboxes:
            if self.is_center(bbox):
                center_captions.append(f"A {bbox.label} is in the center of the image.")
            else:
                non_center_captions.append(f"A {bbox.label} is not in the center of the image.")

        all_captions = {
            'center': center_captions,
            'non_center': non_center_captions,
        }

        label_count = {}
        for bbox in self.bboxes:
            label_count[bbox.label] = label_count.get(bbox.label, 0) + 1

        random_captions = []
        for _ in range(3):
            label = random.choice(list(label_count.keys()))
            count = label_count[label]
            if count > 10:
                random_captions.append(f"There are many {label}s in the image.")
            else:
                random_captions.append(f"There are {count} {label}(s) in the image.")

        all_captions['random'] = random_captions
        return all_captions

# Example usage
if __name__ == "__main__":
    # 假设边界框注释如下
    bboxes = [
        BoundingBox(label="cat", x_center=0.5, y_center=0.5, width=0.2, height=0.2, image_width=640, image_height=480),
        BoundingBox(label="dog", x_center=0.2, y_center=0.2, width=0.3, height=0.3, image_width=640, image_height=480),
        BoundingBox(label="bird", x_center=0.7, y_center=0.7, width=0.1, height=0.1, image_width=640, image_height=480),
        # 添加更多注释...
    ]

    image_size = (640, 480)  # 假设图像尺寸

    b2c = BoxToCaption(bboxes, image_size)
    captions = b2c.generate_captions()

    # 输出生成的描述
    print("Center Captions:")
    for caption in captions['center']:
        print(caption)

    print("\nNon-Center Captions:")
    for caption in captions['non_center']:
        print(caption)

    print("\nRandom Captions:")
    for caption in captions['random']:
        print(caption)
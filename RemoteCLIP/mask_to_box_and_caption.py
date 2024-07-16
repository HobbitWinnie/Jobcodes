import cv2  
import os  
import numpy as np  
import random  

class MaskToBoxAndCaption:  
    def __init__(self, mask_path, rgb_to_class):  
        self.mask_path = mask_path  
        self.rgb_to_class = rgb_to_class  
        self.bboxes = []  
        self.categories = []  
        self.bbox_dict = {}  

    def load_mask(self):  
        # 读取掩码图像并转换为 RGB 格式  
        self.mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)  
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2RGB)  

    def extract_bboxes(self):  
        # 获取掩码中的所有唯一颜色（即类别标签）  
        unique_colors = np.unique(self.mask.reshape(-1, self.mask.shape[2]), axis=0)  
        
        # 将唯一颜色转化成元组  
        unique_colors = [tuple(color) for color in unique_colors]  
        
        # 对每一个唯一颜色进行处理  
        for color in unique_colors:  
            if color in self.rgb_to_class:  
                # 创建一个二值掩码图，仅保留当前颜色的像素  
                binary_mask = cv2.inRange(self.mask, np.array(color), np.array(color))  
                
                # 查找所有轮廓  
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
                
                for contour in contours:  
                    x, y, w, h = cv2.boundingRect(contour)  
                    xmin, ymin, xmax, ymax = x, y, x + w, y + h  
                
                    # 将当前类的边界框信息添加到列表中  
                    self.bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])  # 转换成 [x, y, width, height] 格式  
                    self.categories.append(self.rgb_to_class[color])  # 使用类别名称  

    def visualize_bboxes(self, output_path):  
        # 可视化边界框  
        output_image = self.mask.copy()  
        for bbox in self.bboxes:  
            xmin, ymin, width, height = bbox  
            xmax, ymax = xmin + width, ymin + height  
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  
        cv2.imwrite(output_path, output_image)  

    def generate_captions(self):  
        captions = []  
        image_shape = self.mask.shape  
        categories = self.categories  

        def is_center(bbox, image_shape):  
            # 获取图像的中心坐标  
            img_center_x, img_center_y = image_shape[1] / 2, image_shape[0] / 2  
            center_x, center_y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2  
            
            # 判断目标是否位于图像的中心区域  
            return (0.25 * img_center_x < center_x < 1.75 * img_center_x) and (0.25 * img_center_y < center_y < 1.75 * img_center_y)  

        # 按照目标位置生成前两个描述  
        center_objects = {categories[i] for i, bbox in enumerate(self.bboxes) if is_center(bbox, image_shape)}  
        non_center_objects = {categories[i] for i, bbox in enumerate(self.bboxes) if not is_center(bbox, image_shape)}  

        if center_objects:  
            captions.append(f'The object(s) in the center of the image: {", ".join(center_objects)}')  
        else:  
            captions.append('There are no objects in the center of the image.')  

        if non_center_objects:  
            captions.append(f'The object(s) not located in the center: {", ".join(non_center_objects)}')  
        else:  
            captions.append('All objects are located in the center of the image.')  

        unique_objects = list(set(categories))  
        for _ in range(3):  
            num_objects = random.randint(1, len(unique_objects))  
            selected_objects = random.choices(unique_objects, k=num_objects)  
            first_object = selected_objects[0]  
            count = categories.count(first_object)  
            caption = f'There are {"many" if count > 10 else count} {first_object}(s) in the image.'  

            if caption not in captions:  
                captions.append(caption)  

        return captions  


# 使用示例  
if __name__ == "__main__":  
    # iSAID RGB to class name mapping  
    rgb_to_class_mapping = {  
        (0, 0, 63): "ship",  
        (0, 63, 63): "store tank",  
        (0, 63, 0): "baseball diamond",  
        (0, 63, 127): "tennis court",  
        (0, 63, 191): "basketball court",  
        (0, 63, 255): "ground track field",  
        (0, 127, 63): "bridge",  
        (0, 127, 127): "large vehicle",  
        (0, 0, 127): "small vehicle",  
        (0, 0, 191): "helicopter",  
        (0, 0, 255): "swimming pool",  
        (0, 191, 127): "roundabout",  
        (0, 127, 191): "soccer ball field",  
        (0, 127, 255): "plane",  
        (0, 100, 155): "harbor"  
    }  

    ROOT_DIR = '/home/nw/Codes/RemoteCLIP'  
    DATASET_DIR = os.path.join(ROOT_DIR, 'Datasets/Segmentation-4/iSAID')  
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')  
    os.makedirs(OUTPUT_DIR, exist_ok=True)  

    image_path = os.path.join(DATASET_DIR, 'val/Semantic_masks/images/P0130_instance_color_RGB.png')  
    if not os.path.exists(image_path):  
        print(f"Error: The image path '{image_path}' does not exist.")  
    else:  
        mask_to_box_and_caption = MaskToBoxAndCaption(image_path, rgb_to_class_mapping)  
        mask_to_box_and_caption.load_mask()  
        mask_to_box_and_caption.extract_bboxes()  

        print("Generated Bboxes:")  
        print(mask_to_box_and_caption.bboxes)  
        print("Generated Categories:")  
        print(mask_to_box_and_caption.categories)  

        output_image_path = os.path.join(OUTPUT_DIR, 'P0003_instance_color_RGB_bbox.png')  
        mask_to_box_and_caption.visualize_bboxes(output_image_path)  
        print(f"Bbox visualization saved to {output_image_path}")  

        captions = mask_to_box_and_caption.generate_captions()  
        for caption in captions:  
            print(caption)
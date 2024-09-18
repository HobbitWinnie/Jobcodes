import os  
import random  
from PIL import Image  

def random_slice_image(input_dir, output_dir, slice_size=256, num_slices=16):  
    # 确保输出目录存在  
    os.makedirs(output_dir, exist_ok=True)  
    
    # 遍历输入目录中的所有文件  
    for filename in os.listdir(input_dir):  
        if filename.endswith('.png'):  
            img_path = os.path.join(input_dir, filename)  
            img = Image.open(img_path)  
            
            # 获取图像的尺寸  
            width, height = img.size  
            
            # 确保图像尺寸足够大  
            if width < slice_size or height < slice_size:  
                print(f"Image {filename} is too small for slicing.")  
                continue  
            
            # 生成指定数量的随机切片  
            for n in range(num_slices):  
                left = random.randint(0, width - slice_size)  
                upper = random.randint(0, height - slice_size)  
                right = left + slice_size  
                lower = upper + slice_size  
                
                # 裁剪图像  
                cropped_img = img.crop((left, upper, right, lower))  
                
                # 保存裁剪后的图像  
                cropped_filename = f"{os.path.splitext(filename)[0]}_slice_{n}.png"  
                cropped_img.save(os.path.join(output_dir, cropped_filename))  

# 示例用法  
input_directory = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel/image_tianji'  # 输入目录路径  
output_directory = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_512_dataset/image'  # 输出目录路径  
random_slice_image(input_directory, output_directory, slice_size=512, num_slices=6)
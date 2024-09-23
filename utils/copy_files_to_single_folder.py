import os  
import shutil  
from tqdm import tqdm  

def copy_files_to_single_folder(root_folder_path, target_folder_path):  
    # 确保目标目录存在  
    os.makedirs(target_folder_path, exist_ok=True)  

    # 遍历根目录中的每个子文件夹  
    for subfolder in tqdm(os.listdir(root_folder_path), desc="复制进度"):  
        subfolder_path = os.path.join(root_folder_path, subfolder)  
        if not os.path.isdir(subfolder_path):  
            continue  

        # 遍历子文件夹中的每个文件  
        for root, _, files in os.walk(subfolder_path):  
            for file in files:  
                # 构建源文件路径和目标文件路径  
                source_file_path = os.path.join(root, file)  
                target_file_path = os.path.join(target_folder_path, file)  

                # 如果目标文件已存在，可以选择跳过或覆盖  
                if os.path.exists(target_file_path):  
                    print(f"文件已存在，跳过：{target_file_path}")  
                    continue  

                # 复制文件到目标文件夹  
                shutil.copy2(source_file_path, target_file_path)  
                print(f"已复制：{source_file_path} 到 {target_file_path}")  

# 示例用法  
root_image_folder_path = '/home/Dataset/nw/GF2_Data/MultiLabel_dataset/data'  
target_folder_path = '/home/Dataset/nw/GF2_Data/MultiLabel_dataset/Images'  
copy_files_to_single_folder(root_image_folder_path, target_folder_path)
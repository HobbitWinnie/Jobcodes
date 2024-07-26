# import os  

# def rename_files_in_subfolders(base_path):  
#     for root, dirs, files in os.walk(base_path, topdown=False):  
#         # 检查是否是子文件夹  
#         if os.path.abspath(root) != os.path.abspath(base_path):  
#             folder_name = os.path.basename(root)  
#             for file_name in files:  
#                 old_file_path = os.path.join(root, file_name)  
#                 new_file_name = f"{folder_name}_{file_name}"  
#                 new_file_path = os.path.join(root, new_file_name)  
#                 os.rename(old_file_path, new_file_path)  
#                 print(f"Renamed: {old_file_path} -> {new_file_path}")  

# # 指定你的基本路径  
# base_path = '/home/nw/Data/Output/LABELS_16_CLASSES'  
# rename_files_in_subfolders(base_path)

import os  
import shutil  

def rename_and_move_files(base_path, target_path):  
    # 如果目标路径不存在，创建它  
    if not os.path.exists(target_path):  
        os.makedirs(target_path)  
    
    # 遍历所有子文件夹和文件  
    for root, dirs, files in os.walk(base_path, topdown=False):  
        # 检查是否是子文件夹  
        if os.path.abspath(root) != os.path.abspath(base_path):  
            folder_name = os.path.basename(root)  
            for file_name in files:  
                old_file_path = os.path.join(root, file_name)  
                new_file_name = f"{folder_name}_{file_name}"  
                new_file_path = os.path.join(target_path, new_file_name)  
                
                # 移动并重命名文件  
                shutil.move(old_file_path, new_file_path)  
                print(f"Moved and Renamed: {old_file_path} -> {new_file_path}")  

# 指定你的基本路径和目标路径  
base_path = '/home/nw/Data/Output/LABELS_8_CLASSES'  
target_path = '/home/nw/Data/Output/LABELS_8_CLASSES'  
rename_and_move_files(base_path, target_path)
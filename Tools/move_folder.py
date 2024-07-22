import os  
import shutil  


def copy_files_flat(src, dst):  
    """复制源目录中的所有文件到目标目录, 并扁平化"""  
    if not os.path.exists(dst):  
        os.makedirs(dst)  
    for root, _, files in os.walk(src):  
        for file_name in files:  
            src_file = os.path.join(root, file_name)  
            dst_file = os.path.join(dst, file_name)  

            # 处理重名文件的情况  
            if os.path.exists(dst_file):  
                count = 1  
                base, extension = os.path.splitext(file_name)  
                new_file_name = f"{base}_{count}{extension}"  
                dst_file = os.path.join(dst, new_file_name)  
                while os.path.exists(dst_file):  
                    count += 1  
                    new_file_name = f"{base}_{count}{extension}"  
                    dst_file = os.path.join(dst, new_file_name)  

            print(f"Copying {src_file} to {dst_file}")  
            shutil.copy2(src_file, dst_file)  


def copy_folders(source_directory, target_directory, folder_names):  
    """  
    :param source_directory: str, 源文件夹的路径  
    :param target_directory: str, 目标文件夹的路径  
    :param folder_names: list, 需要复制的目录名称列表  
    """  
    if not os.path.exists(target_directory):  
        os.makedirs(target_directory)  

    for root, dirs, files in os.walk(source_directory):  
        for folder_name in folder_names:  
            for dir_name in dirs:  
                if dir_name == folder_name:  
                    folder_path = os.path.join(root, dir_name)  
                    if os.path.isdir(folder_path):  # 确保这是一个目录  
                        # 保留源文件夹的二级目录名称作为新的文件夹名称  
                        relative_path = os.path.relpath(folder_path, source_directory)  
                        path_parts = relative_path.split(os.sep)  
                        if len(path_parts) > 1:  
                            # 取第二级目录名称  
                            new_folder_name = path_parts[1]  
                        else:  
                            # 只有一级目录的情况  
                            new_folder_name = path_parts[0]  

                        new_path = os.path.join(target_directory, new_folder_name)  
                        
                        # 如果目标路径已存在相同名称的文件夹，进行重命名  
                        if os.path.exists(new_path):  
                            count = 1  
                            while os.path.exists(new_path + f"_{count}"):  
                                count += 1  
                            new_path = new_path + f"_{count}"  

                        print(f"Copying files from {folder_path} to {new_path}")  
                        copy_files_flat(folder_path, new_path)  


# 示例参数  
source_directory = '/mnt/d/nw/Datasets/million-AID/train'  
target_directory = '/mnt/d/nw/Datasets/million-AID-16-classes'  

classes8=[
    "agriculture_land","commercial_land","industrial_land",
    "public_service_land","residential_land",
    "transportation_land","unutilized_land","water_area"
    ]

classes16=[
    "arable_land","grassland","woodland","commercial_area",
    "factory_area","mining_area","power_station","sports_land",
    "detached_house","airport_area","highway_area","port_area",
    "railway_area","bare_land","lake","river"]

# 运行脚本  
copy_folders(source_directory, target_directory, classes16)
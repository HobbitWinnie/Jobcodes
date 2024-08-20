import os  

# 定义文件和目录路径  
original_txt_file_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel/classify.txt' 
cleaned_txt_file_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel/classify_clean.txt'  # 清理后保存的文件路径  
deleted_records_file_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel/deleted_records.txt'  # 删除的记录保存的文件路径  

image_path = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel/image_tianji'  # 替换为目录B的路径  

# 读取目录B中的所有文件名  
files_in_directory_b = set(os.path.splitext(f)[0] for f in os.listdir(image_path))  

# 使用生成器避免大量占用内存  
def process_lines(lines, files_in_directory_b):  
    valid_lines = []  
    deleted_lines = []  
    for line in lines:  
        # 提取文件名（假设文件名在空格前，并去掉它的扩展名）  
        filename = line.split(maxsplit=1)[0]  

        # 检查去掉后缀名后的文件是否存在于目录B中  
        if filename in files_in_directory_b:  
            valid_lines.append(line)  
        else:  
            deleted_lines.append(line)  

    return valid_lines, deleted_lines  

# 打开并读取原始txt文件  
with open(original_txt_file_path, 'r', encoding='utf-8') as file:  
    lines = file.readlines()  

# 处理行  
valid_lines, deleted_lines = process_lines(lines, files_in_directory_b)  

# 将有效的行写入清理后的新txt文件  
with open(cleaned_txt_file_path, 'w', encoding='utf-8') as file:  
    file.writelines(valid_lines)  

# 将删除的行写入另一个新txt文件  
with open(deleted_records_file_path, 'w', encoding='utf-8') as file:  
    file.writelines(deleted_lines)  

print("数据清洗完成。")  
print(f"有效记录已保存至: {cleaned_txt_file_path}")  
print(f"删除记录已保存至: {deleted_records_file_path}")  
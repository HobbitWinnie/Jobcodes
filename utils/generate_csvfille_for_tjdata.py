import csv  
import os  
import pandas as pd  
from sklearn.model_selection import train_test_split  

# 标签字典及其顺序  
label_mapping = {  
    "A_1": "林地",  
    "A_2": "草地",  
    "A_3": "耕地",  
    "B_1": "商业区",  
    "C_1": "工业区",  
    "C_2": "矿区",  
    "C_3": "发电站",  
    "E_1": "住宅区",  
    "F_1": "机场",  
    "F_2": "道路",  
    "F_3": "港口区",  
    "F_4": "桥梁",  
    "F_5": "火车站",  
    "F_6": "高架桥",  
    "G_1": "裸地",  
    "H_1": "开阔水域",  
    "H_2": "小水体"  
}  

# 输入 TXT 文件名  
DATA_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel'
txt_file_name = "classify.txt"  
input_txt_file = os.path.join(DATA_DIR, txt_file_name)

# 输出 CSV 文件名  
output_csv_name = "multilabel_all.csv"  
output_csv_file = os.path.join(DATA_DIR, output_csv_name)

# 标签列表，用于顺序标记  
label_keys = list(label_mapping.keys())  

# Step 1: 读取 TXT 文件并生成 CSV  
with open(input_txt_file, 'r', encoding='utf-8') as txt_file, open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:  
    csv_writer = csv.writer(csv_file)  
    # 写入 CSV 表头  
    header = ["文件名"] + list(label_mapping.values())  
    csv_writer.writerow(header)  

    for line in txt_file:  
        parts = line.strip().split()  
        if len(parts) != 2:  
            continue  

        filename, tags = parts  
        # 初始化标签为0  
        tag_values = [0] * len(label_keys)  

        # 检查文件的标签并设置为1  
        for tag in tags.split(','):  
            if tag in label_keys:  
                tag_index = label_keys.index(tag)  
                tag_values[tag_index] = 1  

        # 写入 CSV  
        csv_writer.writerow([filename] + tag_values)  

print("CSV file has been created successfully.")

# Step 2: 将 CSV 文件随机分割为训练集和测试集  
# 输出 CSV 文件名  
train_csv_file = "multilabel_train.csv"  
test_csv_file = "multilabel_test.csv"  

# 读取原始 CSV 文件  
data = pd.read_csv(os.path.join(DATA_DIR, output_csv_file))

# 使用 train_test_split 将数据分割为训练集和测试集  
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  

# 将分割的数据写入新的 CSV 文件  
train_data.to_csv(os.path.join(DATA_DIR, train_csv_file), index=False)  
test_data.to_csv(os.path.join(DATA_DIR, test_csv_file), index=False)  

print("Data has been split into train and test datasets successfully.")

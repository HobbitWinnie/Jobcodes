import pandas as pd  
from sklearn.model_selection import train_test_split  
import os  


# 原始 CSV 文件名  
DATA_DIR = '/home/Dataset/nw/Multilabel-Datasets/TIANJI_multilabel'

original_csv_file = "multilabel_all.csv"  

# 输出 CSV 文件名  
train_csv_file = "train_0.4.csv"  
test_csv_file = "test_0.6.csv"  

# 读取原始 CSV 文件  
data = pd.read_csv(os.path.join(DATA_DIR,original_csv_file))

# 使用 train_test_split 将数据分割为训练集和测试集  
train_data, test_data = train_test_split(data, test_size=0.6, random_state=42)  

# 将分割的数据写入新的 CSV 文件  
train_data.to_csv(os.path.join(DATA_DIR,train_csv_file), index=False)  
test_data.to_csv(os.path.join(DATA_DIR,test_csv_file), index=False)  

print("Data has been split into train and test datasets successfully.")
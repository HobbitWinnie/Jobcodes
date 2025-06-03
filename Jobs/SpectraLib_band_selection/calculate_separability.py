import numpy as np
import pandas as pd

df = pd.read_csv('/home/Dataset/nw/spectraLib/lib_csv/灌木.csv')  # 假设类别在subclass列

feature_cols = [col for col in df.columns if col.startswith('ref_')]
grouped = df.groupby('subclass')

# 计算每个类别的均值
mean_mat = grouped[feature_cols].mean()

# 计算总类间均值差
sep_scores = np.zeros(len(feature_cols))
for i in range(len(mean_mat)):
    for j in range(i+1, len(mean_mat)):
        diff = mean_mat.iloc[i] - mean_mat.iloc[j]
        sep_scores += diff.abs().values
sep_scores /= (len(mean_mat) * (len(mean_mat)-1) / 2)

# 按分离度降序排列波段
sep_df = pd.DataFrame({'feature': feature_cols, 'separability': sep_scores})
sep_df = sep_df.sort_values('separability', ascending=False)

print(sep_df.head(20))
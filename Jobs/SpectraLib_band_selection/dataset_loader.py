import os
import glob
import pandas as pd

def load_spectral_library(root_dir, file_ext=('.csv', '.txt')):
    """
    自动递归加载整个光谱库目录。
    返回：X(样本,波段), y(标签)、meta字典(一级类、二级类、原文件名等)
    """
    spectrum_list = []
    label_list = []
    meta_list = []

    for class1 in sorted(os.listdir(root_dir)):
        level1_path = os.path.join(root_dir, class1)
        if not os.path.isdir(level1_path): continue
        for class2 in sorted(os.listdir(level1_path)):
            level2_path = os.path.join(level1_path, class2)
            if not os.path.isdir(level2_path): continue
            for fname in glob.glob(os.path.join(level2_path, '*')):
                # 只加载常见的光谱文本文件
                if not fname.lower().endswith(file_ext): continue
                try:
                    # 假设文件第一行是波长，后面行是数值
                    df = pd.read_csv(fname, comment='#', delimiter=',', header=0)
                    # 若只有单行，直接转numpy；若多行（偶见），取均值
                    arr = df.values.mean(axis=0) if len(df) > 1 else df.values.flatten()
                    spectrum_list.append(arr)
                    label_list.append({'class1': class1, 'class2': class2})
                    meta_list.append({'file': fname, 'class1': class1, 'class2': class2})
                except Exception as e:
                    print(f"读取失败:{fname}, 原因:{e}")

    X = pd.DataFrame(spectrum_list)   # 行:样本，列:波段位置
    y = label_list
    meta = meta_list
    return X, y, meta
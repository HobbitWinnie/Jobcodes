import os
import glob
import pandas as pd

def load_spectral_library(root_dir, file_ext=('.csv', '.txt')):
    """
    递归加载整个光谱库目录，适应指定的文件目录结构和txt格式。
    返回: X(样本, 波段), y(标签字典列表), meta(原文件名等)。
    """
    spectrum_list = []
    label_list = []
    meta_list = []
    wavelength_set = set()

    # 第一次遍历：收集所有文件的波长信息，保证X列顺序统一
    wave_dict = {}
    for class1 in sorted(os.listdir(root_dir)):
        level1_path = os.path.join(root_dir, class1)
        if not os.path.isdir(level1_path): continue
        for class2 in sorted(os.listdir(level1_path)):
            level2_path = os.path.join(level1_path, class2)
            if not os.path.isdir(level2_path): continue

            for fname in glob.glob(os.path.join(level2_path, '*.txt')):
                if not fname.lower().endswith(file_ext): continue
                try:
                    df = pd.read_csv(fname, comment='#', sep=r'\s+', header=None)
                    wave = df.iloc[:,0].values
                    wavelength_set.update(wave)
                    wave_dict[fname] = wave
                except Exception as e:
                    print(f"（预扫描）读取失败:{fname}, 原因:{e}")

    sorted_wavelengths = sorted(wavelength_set)  # 所有出现过的波长，排序

    # 第二次遍历：生成统一波长表的样本
    for class1 in sorted(os.listdir(root_dir)):
        level1_path = os.path.join(root_dir, class1)
        if not os.path.isdir(level1_path): continue
        for class2 in sorted(os.listdir(level1_path)):
            level2_path = os.path.join(level1_path, class2)
            if not os.path.isdir(level2_path): continue

            for fname in glob.glob(os.path.join(level2_path, '*')):
                if not fname.lower().endswith(file_ext): continue
                try:
                    df = pd.read_csv(fname, comment='#', sep=r'\s+', header=None)
                    this_wave = df.iloc[:,0].values
                    this_value = df.iloc[:,1].values
                    
                    # 用Series自动对齐到全波段
                    s = pd.Series(data=this_value, index=this_wave)
                    s = s.reindex(sorted_wavelengths)  # 缺的自动补NaN
                    spectrum_list.append(s.values)
                    
                    # 合并标签（推荐下划线拼接，如果符号需要更换可改）
                    merged_label = f"{class1}中的{class2}"             
                    # 标签列表信息     
                    label_list.append({
                        'class1': class1, 
                        'class2': class2, 
                        'merged_label': merged_label
                    })
                    meta_list.append({'file': fname})

                except Exception as e:
                    print(f"读取失败:{fname}, 原因:{e}")

    X = pd.DataFrame(spectrum_list, columns=sorted_wavelengths)   # 行:样本，列:统一波段
    y = label_list
    meta = {'sample_info': meta_list, 'wavelength': sorted_wavelengths}
    return X, y, meta
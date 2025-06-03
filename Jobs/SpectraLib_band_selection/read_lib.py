import os
import pandas as pd

def get_relative_paths(folder, ext='.txt'):
    """递归获取所有txt文件的相对路径"""
    result = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(ext):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder)
                result.append(rel_path)
    return result

def read_spectrum_file(file_path, landclass, subclass, sample_name):
    """读取单个光谱文件，并返回行为series的样本"""
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', header=None)
        df.columns = ['wavelength', 'reflectance']
        df = df[(df['wavelength'] >= 400) & (df['wavelength'] <= 2500)]
        refl_row = df.set_index('wavelength').T
        refl_row['landclass'] = landclass
        refl_row['subclass'] = subclass
        refl_row['sample_name'] = sample_name
        return refl_row
    except Exception as e:
        print(f'文件 {file_path} 读取失败，原因：{e}')
        return None

def main():
    folder = '/home/Dataset/nw/spectraLib/unify/'
    output_folder = '/home/Dataset/nw/spectraLib/lib_csv'
    os.makedirs(output_folder, exist_ok=True)

    relative_paths = get_relative_paths(folder)
    all_samples = []
    
    for rel_path in relative_paths:
        file_path = os.path.join(folder, rel_path)
        parts = rel_path.split('/')
        landclass = parts[0]
        subclass = parts[1] if len(parts) > 2 else ''
        sample_name = parts[-1].replace('.txt', '')
        res = read_spectrum_file(file_path, landclass, subclass, sample_name)
        if res is not None:
            all_samples.append(res)
    
    # 合成大表
    spectra_table = pd.concat(all_samples, ignore_index=True)

    # 列顺序与命名整理
    wave_cols = [c for c in spectra_table.columns if isinstance(c, (int, float)) or str(c).isdigit()]
    wave_cols = sorted(wave_cols, key=lambda x: int(float(x)))
    rename_dict = {col: f"ref_{int(float(col))}" for col in wave_cols}
    spectra_table = spectra_table.rename(columns=rename_dict)
    ordered_cols = ['landclass', 'subclass', 'sample_name'] + [rename_dict[c] for c in wave_cols]
    spectra_table = spectra_table.reindex(columns=ordered_cols)

    # 分组输出
    for landclass, group in spectra_table.groupby('landclass'):
        out_path = os.path.join(output_folder, f'{landclass}.csv')
        group.to_csv(out_path, index=False)
        print(f'已输出：{out_path}')

if __name__ == '__main__':
    main()
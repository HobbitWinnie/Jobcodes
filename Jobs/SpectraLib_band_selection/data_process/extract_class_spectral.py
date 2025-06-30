import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def read_valid_txt_in_folder(folder):
    txt_files = [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
    data_list = []
    file_names = []
    for f in txt_files:
        arr = np.loadtxt(os.path.join(folder, f))
        data_list.append(arr)
        file_names.append(f)
    return data_list, file_names

def preprocess_curve(refl):
    refl = np.where((refl < 0) | (refl > 1), np.nan, refl)
    ok = ~np.isnan(refl)
    valid_count = ok.sum()
    if valid_count > 6:
        max_win = valid_count if valid_count % 2 == 1 else valid_count - 1
        window = min(11, max_win)
        smth = np.copy(refl)
        smth[ok] = savgol_filter(refl[ok], window_length=window, polyorder=3, mode='interp')
        refl = smth
    return refl

def process_one_folder(folder, output_dir):
    data_list, file_names = read_valid_txt_in_folder(folder)
    if len(data_list) < 1:
        print(f"{folder} ：没有光谱文件，跳过。")
        return

    wl0 = data_list[0][:,0]
    for arr in data_list:
        if not np.allclose(arr[:,0], wl0, equal_nan=True):
            print(f"{folder} ：有txt波长不匹配，全部跳过。")
            return

    # 只保留反射率部分，预处理
    reflectances = np.stack([preprocess_curve(arr[:,1]) for arr in data_list])

    # 保证输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    if len(reflectances) == 1:
        mean_curve = reflectances[0]
        print(f"{folder}：仅1条光谱，直接输出 trend_mean.txt")
    else:
        mean_curve = np.nanmean(reflectances, axis=0)
        print(f"{folder}：{len(reflectances)} 条光谱，输出均值曲线到 trend_mean.txt")

    # ---- 保存文本 ----
    np.savetxt(os.path.join(output_dir, "trend_mean.txt"),
               np.column_stack([wl0, mean_curve]),
               fmt="%.6f", header="Wavelength\tMeanTrend")
    print(f"已保存: {os.path.join(output_dir, 'trend_mean.txt')}")

    # ---- 绘图 ----
    plt.figure(figsize=(11,6))

    # 绘制全部光谱
    if len(reflectances) == 1:
        plt.plot(wl0,  np.ma.masked_invalid(reflectances[0]), lw=1.5, alpha=0.8, label='Single Spectrum')
    else:
        for rc in reflectances:
            plt.plot(wl0, np.ma.masked_invalid(rc), lw=1, alpha=0.25)
        plt.plot(wl0, np.ma.masked_invalid(mean_curve), 'r-', lw=2, label='Mean Trend')

    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(output_dir, "trend_summary.png")
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"已保存主趋势图：{png_path}")

def scan_and_process_all_leaf_folders(root_dir, out_root):
    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:  # 只有最底层文件夹才处理
            # 构造目标输出目录：out_root/相对输入目录/
            rel_path = os.path.relpath(dirpath, root_dir)
            out_dir = os.path.join(out_root, rel_path)
            process_one_folder(dirpath, out_dir)

if __name__ == "__main__":
    folder_path = '/home/Dataset/nw/spectraLib/unify'
    out_path = '/home/Dataset/nw/spectraLib/unify_v2'
    scan_and_process_all_leaf_folders(folder_path, out_path) 
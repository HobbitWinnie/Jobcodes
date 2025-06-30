import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def regrid_linear(wl, y, wl_new):
    f = interp1d(wl, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f(wl_new)

def regrid_bin_average(wl, y, wl_new, step=10):
    wl_half = step / 2
    y_new = np.zeros_like(wl_new)
    for i, w in enumerate(wl_new):
        inds = (wl >= w - wl_half) & (wl < w + wl_half)
        if np.any(inds):
            y_new[i] = np.nanmean(y[inds])
        else:
            y_new[i] = np.nan
    return y_new

def plot_all(wl, y, wl_new, lin, binavg, out_png):
    plt.figure(figsize=(11,6))
    plt.plot(wl, y, label='Original Mean', c='black', lw=1)
    plt.plot(wl_new, lin, label='Linear Interp', c='green', lw=1)
    plt.plot(wl_new, binavg, label='Bin Avg (10nm)', c='red', lw=1)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.title("Comparison of Resampling Methods (10nm grid)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def process_trend_mean(trend_mean_path):
    try:
        arr = np.loadtxt(trend_mean_path, skiprows=1) 
        wl, y = arr[:,0], arr[:,1]
    except Exception as e:
        print(f"读取 {trend_mean_path} 失败：{e}")
        return

    wl_new = np.arange(np.ceil(wl.min()), np.floor(wl.max())+1, 10)

    res_lin   = regrid_linear(wl, y, wl_new)
    res_bin   = regrid_bin_average(wl, y, wl_new, step=10)

    base_dir = os.path.dirname(trend_mean_path)
    np.savetxt(os.path.join(base_dir, "trend_mean_10nm_linear.txt"),
               np.column_stack([wl_new, res_lin]),
               fmt="%.6f", header="Wavelength\tMeanTrend_Linear")
    np.savetxt(os.path.join(base_dir, "trend_mean_10nm_binavg.txt"),
               np.column_stack([wl_new, res_bin]),
               fmt="%.6f", header="Wavelength\tMeanTrend_BinAvg")

    plot_png_path = os.path.join(base_dir, "trend_mean_compare_10nm.png")
    plot_all(wl, y, wl_new, res_lin, res_bin, plot_png_path)
    print(f"已保存重采样文件和对比图：{plot_png_path}")

def scan_and_process_all_leaf_folders(root_dir):
    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:  # 只处理叶子文件夹
            trend_mean_path = os.path.join(dirpath, "trend_mean.txt")
            if os.path.exists(trend_mean_path):
                print(f"处理：{trend_mean_path}")
                process_trend_mean(trend_mean_path)
            else:
                print(f"{dirpath} 无 trend_mean.txt，跳过。")

if __name__ == "__main__":
    root_dir = '/home/Dataset/nw/spectraLib/unify_v2'
    scan_and_process_all_leaf_folders(root_dir)
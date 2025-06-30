import os
import numpy as np
import matplotlib.pyplot as plt


# def is_leaf_folder(folder):
#     # 判断该文件夹是否为“叶子”文件夹（没有子文件夹）
#     for entry in os.scandir(folder):
#         if entry.is_dir():
#             return False
#     return True

def plot_spectra_in_leaf_folder(folder, out_png="all_spectra.png"):
    txt_files = [f for f in os.listdir(folder) if f.lower().endswith(".txt")]
    if not txt_files:
        return

    wave = None
    plt.figure(figsize=(10, 6))
    for fname in txt_files:
        file_path = os.path.join(folder, fname)
        try:
            data = np.loadtxt(file_path)
        except Exception:
            print(f"无法读取 {file_path}，已跳过。")
            continue
        if wave is None:
            wave = data[:, 0]
        elif not np.allclose(wave, data[:, 0]):
            print(f"{fname} 的波长列与首文件不一致，已跳过。")
            continue
        refl = data[:, 1]
        plt.plot(wave, refl, label=os.path.splitext(fname)[0])

    if wave is not None and len(txt_files) > 0:
        plt.xlabel("Wavelength")
        plt.ylabel("Reflentance")
        plt.legend(fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(folder, out_png)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"已保存: {out_path}")

def scan_and_plot_all_leaf_folders(root_dir, out_png="spectral_plot.png"):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:  # 没有子文件夹，说明是最底层“叶子”
            plot_spectra_in_leaf_folder(dirpath, out_png=out_png)


if __name__ == "__main__":
    folder_path = '/home/Dataset/nw/spectraLib/unify_v3'
    scan_and_plot_all_leaf_folders(folder_path)
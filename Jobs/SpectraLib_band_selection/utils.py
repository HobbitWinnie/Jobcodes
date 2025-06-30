import itertools
import matplotlib.pyplot as plt
import numpy as np


def get_combinations(n_bands, comb_dim):
    """
    生成所有波段组合
    """
    return list(itertools.combinations(range(n_bands), comb_dim))

def get_label_indices(labels, target_list):
    """
    获得目标标签在样本集中的下标
    """
    return [i for i, l in enumerate(labels) if l in target_list]

def pretty_print_clip_match(match_result):
    """
    格式化输出clip_match的结果。
    输入:  
        match_result: dict, 形如
            {'水稻': [('耕地中的水稻', 0.91), ...], ...}
    """
    for query, label_score_list in match_result.items():
        print(f"{query}:")
        for label, score in label_score_list:
            print(f"    {label}    {score:.4f}")
        print()  # 空行分隔每个query

def plot_selected_bands(
        X_mean_np,
        y_mean,
        band_idxs,
        band_idxs2=None,
        band_names=None,
        save_path=None,
        dpi=150,
        method_desc="Band Selection"
    ):
    n_band = X_mean_np.shape[1]
    x_axis = np.arange(n_band) if band_names is None else band_names

    plt.figure(figsize=(14, 7))
    for i, y in enumerate(y_mean):
        plt.plot(x_axis, X_mean_np[i], label=f"{y}")

    # 主选法（红）
    if band_idxs is not None:
        for j, idx in enumerate(band_idxs):
            x_val = idx if band_names is None else band_names[idx]
            plt.axvline(x=x_val, color='r', linestyle='--', alpha=0.4,
                        label=f"Selected: {method_desc}" if j == 0 else "")
            plt.scatter([x_val]*X_mean_np.shape[0],
                        X_mean_np[:,idx], color='r', s=30, zorder=10,
                        marker="o", alpha=0.6)
    # 可以传入第二法（绿），不强制
    if band_idxs2 is not None:
        for j, idx in enumerate(band_idxs2):
            x_val = idx if band_names is None else band_names[idx]
            plt.axvline(x=x_val, color='g', linestyle='--', alpha=0.3,
                        label=f"Selected: Second" if j == 0 else "")
            plt.scatter([x_val]*X_mean_np.shape[0],
                        X_mean_np[:,idx], color='g', s=30, zorder=10,
                        marker="x", alpha=0.6)

    plt.legend(loc="best", fontsize=10)
    plt.xlabel('Band' if band_names is None else 'Wavelength')
    plt.ylabel('Mean Value')
    plt.title(f'Mean Spectrum ({method_desc})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"图像已保存至: {save_path}")
        plt.close()
    else:
        plt.show()
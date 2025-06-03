import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from scipy.stats import f_oneway


# ========== 参数设置 ========== #
csv_path = '/home/Dataset/nw/spectraLib/lib_csv/草地.csv'          # 输入光谱库表（需包含分类标签和各波段）
class_col = 'subclass'                # 分类目标列，可改为'landclass'等
topN = 10                             # 输出前N个优选波段
score_csv = '/home/nw/Codes/Jobs/SpectraLib_band_selection/test/band_fisher_scores.csv'  # 分离度结果保存
plot_path = '/home/nw/Codes/Jobs/SpectraLib_band_selection/test/fisher_score_curve.png'  # 曲线图路径，None为不保存
anova_score_csv = '/home/nw/Codes/Jobs/SpectraLib_band_selection/test/band_anova_scores.csv'

# ========== 1. 读取和清洗数据 ========== #
df = pd.read_csv(csv_path)
id_cols = ['landclass', 'subclass', 'sample_name']
feature_cols = [c for c in df.columns if c.startswith('ref_')]

# 去除全为NaN或0的波段
feature_cols = [col for col in feature_cols if not (df[col].isna().all() or (df[col]==0).all())]

# ========== 2. 多类别Fisher分离度计算 ========== #
fisher_scores = []
for band in feature_cols:
    stats = df.groupby(class_col)[band].agg(['mean', 'var', 'count'])
    gmean = df[band].mean()
    numer = np.sum(stats['count'] * (stats['mean'] - gmean) ** 2)
    denom = np.sum(stats['count'] * stats['var'])
    score = numer / (denom + 1e-10)
    fisher_scores.append((band, score))

fisher_df = pd.DataFrame(fisher_scores, columns=['band', 'fisher_score'])
fisher_df['wavelength'] = fisher_df['band'].str.replace('ref_', '').astype(int)
fisher_df = fisher_df.sort_values('fisher_score', ascending=False)

# ========== 3. 输出Top波段和分数表 ========== #
print('结论：分离度最佳的波段：')
print(fisher_df.head(topN)[['band', 'wavelength', 'fisher_score']])

fisher_df.to_csv(score_csv, index=False)
print(f"全部分离度结果已保存至：{score_csv}")

# ========== 4. 分离度随波长曲线，可选 ========== #
if plot_path:
    fisher_df_sorted = fisher_df.sort_values('wavelength')
    plt.figure(figsize=(10,4))
    plt.plot(fisher_df_sorted['wavelength'], fisher_df_sorted['fisher_score'], c='C0')
    plt.xlabel('波长 (nm)')
    plt.ylabel('Fisher分离度')
    plt.title(f'{class_col} 多类别Fisher分离度')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"分离度曲线已保存至：{plot_path}")

# ========== 5. 额外：ANOVA F值辅助对比（可选） ========== #
anova_scores = []
for band in feature_cols:
    vals = df[[band, class_col]].dropna()
    if vals[class_col].nunique() < 2:
        # 该波段仅有1类（无可比性），跳过
        continue
    # 按类别分组，剔除该波段缺失样本
    arrays = [vals[vals[class_col]==c][band].values for c in vals[class_col].unique()]
    # 至少每类有两个样本才能做方差分析
    if all(len(arr) >= 2 for arr in arrays):
        f_val, _ = f_oneway(*arrays)
        anova_scores.append((band, f_val))
    else:
        continue

anova_df = pd.DataFrame(anova_scores, columns=['band', 'anova_fscore'])
anova_df['wavelength'] = anova_df['band'].str.replace('ref_', '').astype(int)
anova_df = anova_df.sort_values('anova_fscore', ascending=False)
anova_df.to_csv(anova_score_csv, index=False)
print("已按波段单独跳过NaN完成ANOVA F检验，结果见 band_anova_scores.csv")
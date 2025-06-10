import pandas as pd
import os

from config import cfg
from dataset_loader import load_spectral_library
from selector.selectors import BandSelector
from selector.selectors_auto import rfe_band_select, rf_band_select, lsvc_band_select
from selector.vae_selector import train_vae, vae_band_ranking
from selector.transformer_selector import train_transformer, transformer_band_ranking
from evaluator import composite_score
from clip_semantic import CNCLIPSemanticMatcher, EVACLIPSemanticMatcher, Text2VecSemanticMatcher, EnsembleSemanticMatcher
from utils import pretty_print_clip_match
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


def plot_mean_spectra(X_mean, y_mean, fname='class_means.png'):
    """
    X_mean: numpy数组，形状 [类别数, 特征数]
    y_mean: 类别标签，可以为 str/int/label 列表
    fname: 保存文件名
    """
    plt.figure(figsize=(10, 6))
    x_axis = np.arange(X_mean.shape[1])  # 横坐标为波段索引或波长值，假如有就传（可自定义）

    for idx, (label, row) in enumerate(zip(y_mean, X_mean)):
        plt.plot(x_axis, row, label=str(label), linewidth=2)

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其它中文字体
    # plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("波段索引")
    plt.ylabel("均值")
    plt.title("不同类别的均值光谱曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def run_selectors(X, y, topk, device):
    results = {}

    # # 1. 暴力穷举组合
    # band_selector = BandSelector(band_count=X.shape[1], comb_dim=topk)
    # top_combos = band_selector.recommend(X, y, top_n=1)
    # results['Exhaustive'] = list(top_combos[0][0]) if top_combos else []
    # print("\n暴力穷举选择完成")

    #  VAE
    vae_model = train_vae(
        X, latent_dim=cfg.VAE_LATENT_DIM, epochs=cfg.VAE_EPOCHS,
        batch_size=cfg.VAE_BATCH_SIZE, device=device)
    vae_ranked, _ = vae_band_ranking(vae_model, X, device=device)
    results['VAE'] = list(vae_ranked[:topk])
    print("\n VAE选择完成")

    # Transformer
    transformer_model = train_transformer(
        X, embed_dim=cfg.TRANS_EMBED_DIM, epochs=cfg.TRANS_EPOCHS,
        batch_size=cfg.TRANS_BATCH_SIZE, device=device
    )
    transformer_ranked, _ = transformer_band_ranking(transformer_model, X, device=device)
    results['Transformer'] = list(transformer_ranked[:topk])
    print("\n Transformer选择完成")

    # RFE
    idx_rfe, _ = rfe_band_select(X, y, n_features=topk)
    results['RFE'] = list(idx_rfe)
    print("\n RFE选择完成")

    # 随机森林
    idx_rf, _ = rf_band_select(X, y, n_features=topk)
    results['RandomForest'] = list(idx_rf)
    print("\n 随机森林选择完成")

    # # Linear SVC
    # idx_lsvc, _ = lsvc_band_select(X, y, n_features=topk)
    # results['LinearSVC'] = list(idx_lsvc)
    # print("\n Linear SVC选择完成")

    return results

def report_result(method_band_dict, X, y):
    report = []
    for method, bands in method_band_dict.items():
        X_sub = X[:, bands]
        score, metrics = composite_score(X_sub, y, weights=cfg.METRIC_WEIGHTS)
        report.append({
            'Method': method,
            'BandIdx': bands,
            'CompositeScore': score,
            **metrics
        })
    return pd.DataFrame(report)


if __name__ == "__main__":
    # 1. 显示与保存当前参数 
    cfg.show()
    cfg.save()

    # 2. 加载数据
    X, y_info, meta = load_spectral_library(cfg.DATA_ROOT)
    labels = [item['merged_label'] for item in y_info]          #还有class1，class2，merged_label
    label_set = sorted(list(set(labels)))

    # 3. 利用CLIP语义匹配目标类别  
    matcher_1 = EVACLIPSemanticMatcher(device=cfg.DEVICE)
    matcher_2 = Text2VecSemanticMatcher(device=cfg.DEVICE)
    matcher = EnsembleSemanticMatcher(matcher_1, matcher_2, mode='mean')
    clip_match = matcher.match(cfg.DEFAULT_TARGET_LIST, label_set, topk=cfg.CLIP_TOPK)
    print("=== 融合结果（集成）===")
    pretty_print_clip_match(clip_match)

    resolved_label_set = []
    for q in cfg.DEFAULT_TARGET_LIST:
        resolved_label_set += [l for l, score in clip_match.get(q, [])]
    select_indices = [i for i, l in enumerate(labels) if l in resolved_label_set]
    X_sel = X.iloc[select_indices].values
    y_sel = [labels[i] for i in select_indices]

    print("CLIP 匹配到的类别:", clip_match)
    print(f"筛选后样本数: {len(y_sel)}, 类别: {set(y_sel)}")

    # 4. 按 label 分组，对每一列特征求均值
    df = pd.DataFrame(X_sel)
    df['label'] = y_sel

    # dropna保证 NaN 不参与均值计算（不丢类别）
    X_mean = df.groupby('label').mean(numeric_only=True)      # shape: n_class × n_feat
    y_mean = X_mean.index.values         
    X_mean_np = X_mean.values

    # 保存均值样本特征图
    mean_spectra_path = os.path.join(cfg.RESULT_DIR, f"band_selection_report_mean_spectra.png")
    plot_mean_spectra(X_mean_np, y_mean, mean_spectra_path)

    # 5. 执行多种selector选波段
    le = LabelEncoder()
    y_mean_numeric = le.fit_transform(y_mean)   # 得到数字编码，例如[0, 1, 2, ...]
    results_dict = run_selectors(X_sel, y_mean_numeric, topk=cfg.BAND_COMB_DIM, device=cfg.DEVICE)

    print("\n各方法推荐的波段索引：")
    for method, bands in results_dict.items():
        print(f"{method}: {bands}")

    # 6. 汇总评价
    eval_df = report_result(results_dict, X_mean_np, y_mean_numeric)
    print("\n综合评价分数对比 (可直接导出为论文表)：\n", eval_df)

    # 7. 输出到csv
    result_path = os.path.join(cfg.RESULT_DIR, f"band_selection_report_{'_'.join(cfg.DEFAULT_TARGET_LIST)}.csv")
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    eval_df.to_csv(result_path, index=False)
    print(f"\n实验结果已保存至：{result_path}")
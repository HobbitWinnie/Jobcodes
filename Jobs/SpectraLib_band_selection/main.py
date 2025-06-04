import pandas as pd
import os

from config import cfg
from dataset_loader import load_spectral_library
from selector.selectors import BandSelector
from selector.selectors_auto import rfe_band_select, rf_band_select, lsvc_band_select
from selector.vae_selector import train_vae, vae_band_ranking
from selector.transformer_selector import train_transformer, transformer_band_ranking
from evaluator import composite_score
from clip_semantic import CLIPSemanticMatcher


def run_selectors(X, y, topk, device):
    results = {}

    # 1. 暴力穷举组合
    band_selector = BandSelector(band_count=X.shape[1], comb_dim=topk)
    top_combos = band_selector.recommend(X, y, top_n=1)
    results['Exhaustive'] = list(top_combos[0][0]) if top_combos else []

    # 2. RFE
    idx_rfe, _ = rfe_band_select(X, y, n_features=topk)
    results['RFE'] = list(idx_rfe)

    # 3. 随机森林
    idx_rf, _ = rf_band_select(X, y, n_features=topk)
    results['RandomForest'] = list(idx_rf)

    # 4. Linear SVC
    idx_lsvc, _ = lsvc_band_select(X, y, n_features=topk)
    results['LinearSVC'] = list(idx_lsvc)

    # 5. VAE
    vae_model = train_vae(
        X, latent_dim=cfg.VAE_LATENT_DIM, epochs=cfg.VAE_EPOCHS,
        batch_size=cfg.VAE_BATCH_SIZE, device=device)
    vae_ranked, _ = vae_band_ranking(vae_model, X, device=device)
    results['VAE'] = list(vae_ranked[:topk])

    # 6. Transformer
    transformer_model = train_transformer(
        X, embed_dim=cfg.TRANS_EMBED_DIM, epochs=cfg.TRANS_EPOCHS,
        batch_size=cfg.TRANS_BATCH_SIZE, device=device
    )
    transformer_ranked, _ = transformer_band_ranking(transformer_model, X, device=device)
    results['Transformer'] = list(transformer_ranked[:topk])

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
    class2_labels = [item['class2'] for item in y_info]

    # 3. 利用CLIP语义匹配目标类别
    matcher = CLIPSemanticMatcher(device=cfg.DEVICE)
    label_set = sorted(list(set(class2_labels)))
    clip_match = matcher.match(cfg.DEFAULT_TARGET_LIST, label_set, topk=cfg.CLIP_TOPK, threshold=cfg.CLIP_THRESH)
    resolved_label_set = []
    for q in cfg.DEFAULT_TARGET_LIST:
        resolved_label_set += [l for l, score in clip_match.get(q, [])]
    select_indices = [i for i, l in enumerate(class2_labels) if l in resolved_label_set]
    X_sel = X.iloc[select_indices].values
    y_sel = [class2_labels[i] for i in select_indices]

    print("CLIP 匹配到的类别:", clip_match)
    print(f"筛选后样本数: {len(y_sel)}, 类别: {set(y_sel)}")

    # 4. 执行多种selector选波段
    results_dict = run_selectors(X_sel, y_sel, topk=cfg.BAND_COMB_DIM, device=cfg.DEVICE)

    print("\n各方法推荐的波段索引：")
    for method, bands in results_dict.items():
        print(f"{method}: {bands}")

    # 5. 汇总评价
    eval_df = report_result(results_dict, X_sel, y_sel)
    print("\n综合评价分数对比 (可直接导出为论文表)：\n", eval_df)

    # 6. 输出到csv
    result_path = os.path.join(cfg.RESULT_DIR, f"band_selection_report_{'_'.join(cfg.DEFAULT_TARGET_LIST)}.csv")
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    eval_df.to_csv(result_path, index=False)
    print(f"\n实验结果已保存至：{result_path}")
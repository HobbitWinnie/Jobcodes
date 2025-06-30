import sys
sys.path.append('/home/nw/Codes/Jobs/SpectraLib_band_selection')

import os
import pandas as pd
import numpy as np
from config import cfg
from utils import pretty_print_clip_match, plot_selected_bands
from dataset_loader import load_spectral_library
from selector.single_band_select import (
    variance_band_select, 
    range_band_select, 
    report_result_simple, 
)
from selector.combined_select import greedy_band_combination
from semantic_matcher.Ensemble_semantic_matcher import (
    EVACLIPSemanticMatcher, 
    Text2VecSemanticMatcher, 
    EnsembleSemanticMatcher
)

def prepare_data():
    """
    数据加载和标签摘取函数
    读取原始高光谱库，得到X（样本×波段），
    labels（每个样本的合并标签），
    label_set（全体类别名集合，无重复），
    meta（包含wavelength等元数据信息）
    """
    # 配置显示和保存
    cfg.show()
    cfg.save()
    # 加载主数据
    X, y_info, meta = load_spectral_library(cfg.DATA_ROOT)
    labels = [item['merged_label'] for item in y_info]
    label_set = sorted(list(set(labels)))
    return X, labels, label_set, meta

def semantic_label_matching(label_set):
    """
    用CLIP+Text2Vec做目标类别的语义匹配，获得目标对应的标签集合。
    返回：被选中的最终标签列表
    """
    matcher_1 = EVACLIPSemanticMatcher(device=cfg.DEVICE)
    matcher_2 = Text2VecSemanticMatcher(device=cfg.DEVICE)
    matcher = EnsembleSemanticMatcher(matcher_1, matcher_2, mode='mean')
    clip_match = matcher.match(cfg.DEFAULT_TARGET_LIST, label_set, topk=cfg.CLIP_TOPK)
    print("=== 融合结果（集成）===")
    pretty_print_clip_match(clip_match)
    # 根据最终目标清单筛选需要分析的类别（resolved_label_set）
    resolved_label_set = []
    for q in cfg.DEFAULT_TARGET_LIST:
        resolved_label_set += [l for l, score in clip_match.get(q, [])]
    return resolved_label_set

def calc_group_mean(X, labels, resolved_label_set):
    """
    按照resolved_label_set筛选数据，做类别分组均值，得到每类均值的谱图，以及类别顺序
    X: 原始高光谱样本（可为DataFrame或np.array）
    labels: 每条样本对应的归一标签
    resolved_label_set: 需要分析的目标类别
    返回：X_mean_np（每类均值，shape=(类数, 波段数)），y_mean（类别名称数组）
    """
    select_indices = [i for i, l in enumerate(labels) if l in resolved_label_set]
    X_sel = X.iloc[select_indices].values if isinstance(X, pd.DataFrame) else X[select_indices]
    y_sel = [labels[i] for i in select_indices]
    # 转DataFrame增加分组灵活性
    df = pd.DataFrame(X_sel)
    df['label'] = y_sel
    X_mean = df.groupby('label').mean(numeric_only=True)
    y_mean = X_mean.index.values
    X_mean_np = X_mean.values
    return X_mean_np, y_mean

def single_band_selection(X_mean_np, TOPK=20):
    """
    单波段初筛：按方差法/极差法分别取TOPK波段
    返回字典形式，便于后续整合及扩展其它方法
    """
    selected_bands = {}
    var_idx, var_score = variance_band_select(X_mean_np, TOPK)  # 方差法选
    rng_idx, rng_score = range_band_select(X_mean_np, TOPK)     # 极差法选
    selected_bands['Variance'] = (var_idx, var_score)
    selected_bands['Range'] = (rng_idx, rng_score)
    return selected_bands

def combo_band_selection(X_mean_np, y_mean, selected_bands, COMBO_NUM=7):
    """
    多波段组合优化：在已预筛 band_pool 中，用贪心法找组合区分力最强的COMBO_NUM个波段
    band_pool会自动整合var/range法的波段索引并去重
    优选结果以新的dict key添加到selected_bands
    """
    var_band_idxs, _ = selected_bands['Variance']
    range_band_idxs, _ = selected_bands['Range']
    band_pool = np.unique(np.concatenate([var_band_idxs, range_band_idxs]))
    best_combo, best_score = greedy_band_combination(
        X_mean_np, y_mean, band_pool=band_pool, n_select=COMBO_NUM
    )
    selected_bands[f"GreedyCombo_{COMBO_NUM}"] = (np.array(best_combo), [best_score] * len(best_combo))
    return selected_bands

def save_and_plot_all(selected_bands, X_mean_np, y_mean, meta):
    """
    输出各方法结果表，并对每种方法画对应的selected bands，最终结果保存到csv
    """
    # 生成简易报表
    report_df = report_result_simple(selected_bands, band_names=meta['wavelength'])
    print(report_df)
    # 循环画图保存
    for method in selected_bands:
        save_path = os.path.join(cfg.RESULT_DIR, f"selected_bands_{method}.png")
        band_idxs, _ = selected_bands[method]
        plot_selected_bands(
            X_mean_np,
            y_mean,
            band_idxs,
            band_names=meta['wavelength'],
            save_path=save_path,
            dpi=300,
            method_desc=method
        )
    # 保存最终报告为CSV
    result_path = os.path.join(cfg.RESULT_DIR, "all_band_selection_report.csv")
    report_df.to_csv(result_path, index=False)
    print(f"\n全部方法结果已保存至：{result_path}")

def main():
    """
    主控流程。逐步串联数据加载、语义匹配、分组统计、波段选择与组合、可视化与导出
    """
    # 1. 加载原始库和标签
    X, labels, label_set, meta = prepare_data()
    # 2. 语义智能匹配目标类别
    resolved_label_set = semantic_label_matching(label_set)
    # 3. 每类谱均值准备
    X_mean_np, y_mean = calc_group_mean(X, labels, resolved_label_set)
    # 4. 单波段选取（方差法/极差法）
    TOPK = 20       # 每种方法预选波段个数
    COMBO_NUM = 7   # 组合法最后输出波段个数
    selected_bands = single_band_selection(X_mean_np, TOPK)
    # 5. 自动组合波段优选
    selected_bands = combo_band_selection(X_mean_np, y_mean, selected_bands, COMBO_NUM)
    # 6. 输出报表和图
    save_and_plot_all(selected_bands, X_mean_np, y_mean, meta)

if __name__ == "__main__":
    main()
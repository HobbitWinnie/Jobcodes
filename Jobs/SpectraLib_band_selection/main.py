import sys
sys.path.append('/home/nw/Codes/Jobs/SpectraLib_band_selection')

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from config import cfg
from utils import pretty_print_clip_match, plot_selected_bands
from dataset_loader import load_spectral_library
from selector.single_band_select import (
    variance_band_select, 
    range_band_select, 
    report_result_simple, 
)
from selector.combined_select import (
    greedy_band_selection,
    sffs_band_selection
)
from semantic_matcher.Ensemble_semantic_matcher import (
    EVACLIPSemanticMatcher, 
    Text2VecSemanticMatcher, 
    EnsembleSemanticMatcher,
    CNCLIPSemanticMatcher
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
    matcher_3 = CNCLIPSemanticMatcher(device=cfg.DEVICE)
    matcher = EnsembleSemanticMatcher(matcher_1, matcher_2, mode='mean')

    # 各模型匹配结果
    evaclip_result = matcher_1.match(cfg.DEFAULT_TARGET_LIST, label_set, topk=cfg.CLIP_TOPK)
    text2vec_result = matcher_2.match(cfg.DEFAULT_TARGET_LIST, label_set, topk=cfg.CLIP_TOPK)
    cnclip_result = matcher_3.match(cfg.DEFAULT_TARGET_LIST, label_set, topk=cfg.CLIP_TOPK)
    ensemble_result = matcher.match(cfg.DEFAULT_TARGET_LIST, label_set, topk=cfg.CLIP_TOPK)

    # 打印集成结果
    print("=== 融合结果（集成）===")
    pretty_print_clip_match(ensemble_result)

    # 构造保存DataFrame
    records = []
    for query in cfg.DEFAULT_TARGET_LIST:
        # 取四个模型top1结果
        evaclip_top1 = evaclip_result.get(query, [("", None)])[0]
        text2vec_top1 = text2vec_result.get(query, [("", None)])[0]
        cnclip_top1 = cnclip_result.get(query, [("", None)])[0]
        ensemble_top1 = ensemble_result.get(query, [("", None)])[0]

        def format_result(top1):
            label, score = top1
            return f"{label} ({round(score, 4)})" if label else ""

        result_row = {
            "label": query,
            "cnclip_result": format_result(cnclip_top1),
            "evaclip_result": format_result(evaclip_top1),
            "text2vec_result": format_result(text2vec_top1),
            "ensemble_result": format_result(ensemble_top1),
        }
        records.append(result_row)

    df = pd.DataFrame(records)
    result_path = os.path.join(cfg.RESULT_DIR, "semantic_label_match_results.csv")
    df.to_csv(result_path, index=False, encoding='utf-8-sig')

    # 根据最终目标清单筛选需要分析的类别（resolved_label_set）
    resolved_label_set = []
    for q in cfg.DEFAULT_TARGET_LIST:
        resolved_label_set += [l for l, score in ensemble_result.get(q, [])]
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
    单波段初筛：统一筛选有效波段，再分别用方差法和极差法筛选TOPK
    返回字典形式，便于整合和扩展
    """
    # 筛选非nan波段
    variances = np.nanvar(X_mean_np, axis=0)
    ranges = np.nanmax(X_mean_np, axis=0) - np.nanmin(X_mean_np, axis=0)
    valid = (~np.isnan(variances)) & (~np.isnan(ranges))
    valid_indices = np.where(valid)[0]
    
    selected_bands = {}
    var_idx, var_score = variance_band_select(variances, valid_indices, TOPK)
    rng_idx, rng_score = range_band_select(ranges, valid_indices, TOPK)
    selected_bands['Variance'] = (var_idx, var_score)
    selected_bands['Range'] = (rng_idx, rng_score)
    return selected_bands

def combo_band_selection(X_mean_np, y_mean, selected_bands, COMBO_NUM=7, EPOCH=100):
    """
    多波段组合优化：在已预筛 band_pool 中，用贪心法找组合区分力最强的COMBO_NUM个波段
    band_pool会自动整合var/range法的波段索引并去重
    优选结果以新的dict key添加到selected_bands
    """
    var_band_idxs, _ = selected_bands['Variance']
    range_band_idxs, _ = selected_bands['Range']
    band_pool = np.unique(np.concatenate([var_band_idxs, range_band_idxs]))
    print(f"\n候选波段有：{band_pool}")

    # 贪心选择
    best_combo, best_score = greedy_band_selection(
        X_mean_np, y_mean, band_pool=band_pool, n_select=COMBO_NUM, n_trials=EPOCH
    )
    selected_bands[f"GreedyCombo_{COMBO_NUM}"] = (np.array(best_combo), [best_score] * len(best_combo))

    # 浮动选择
    best_combo_sffs, best_score_sffs = sffs_band_selection(
        X_mean_np, y_mean, band_pool=band_pool, n_select=COMBO_NUM, n_trials=EPOCH
    )
    selected_bands[f"SffsCombo_{COMBO_NUM}"] = (np.array(best_combo_sffs), [best_score_sffs] * len(best_combo_sffs))

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


def evaluate_band_discriminability(X_mean, y_mean, band_idxs, score_func=None):
    """
    用类别均值样本在选定波段下评估类别区分能力
    X_mean: shape=(类别数, 波段数), 每行一个类别均值
    y_mean: 类别名
    band_idxs: 波段索引，用于挑选特定波段分析判别力
    score_func: 可传入自定义的分组判别力函数
    返回:
        - 类别间距离矩阵
        - （可选）group_discriminability得分
    """
    X_sel = X_mean[:, band_idxs]
    # 1. 欧氏距离矩阵
    dist_matrix = cdist(X_sel, X_sel)
    # 2. 可选的分组判别力
    if score_func is not None:
        discri_score = score_func(X_sel, y_mean)
    else:
        discri_score = None
    return dist_matrix, discri_score


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
    TOPK = 0       # 每种方法预选波段个数
    COMBO_NUM = 0   # 组合法最后输出波段个数
    selected_bands = single_band_selection(X_mean_np, TOPK)
   
    # 5. 自动组合波段优选
    EPOCH = 2000
    selected_bands = combo_band_selection(X_mean_np, y_mean, selected_bands, COMBO_NUM, EPOCH)
    
    # # 6. 新增：验证greedy/sffs等组合的分类性能
    # for method in selected_bands:
    #     band_idxs, score_list = selected_bands[method]
    #     if isinstance(score_list, (list, np.ndarray)) and len(score_list) == 1:
    #         best_score = score_list[0]
    #     elif isinstance(score_list, (list, np.ndarray)):
    #         best_score = score_list[-1]  # 若track了每一步得分，取最后一步
    #     else:
    #         best_score = score_list

    #     print(f"\n方法[{method}] 组合的最终判别力分数（best_score）: {best_score:.4f}")

    #     # 可选辅助：类别均值间的距离矩阵，仅供分析
    #     X_sel = X_mean_np[:, band_idxs]
    #     print(f"方法[{method}]得到的类别特征矩阵:\n{X_sel}")

    #     from scipy.spatial.distance import cdist
    #     dist_matrix = cdist(X_sel, X_sel)
    #     print(f"方法[{method}]类别均值间距离矩阵:\n{np.round(dist_matrix,3)}")

    # 7. 输出报表和图
    save_and_plot_all(selected_bands, X_mean_np, y_mean, meta)

if __name__ == "__main__":
    main()
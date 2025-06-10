import numpy as np
import itertools

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
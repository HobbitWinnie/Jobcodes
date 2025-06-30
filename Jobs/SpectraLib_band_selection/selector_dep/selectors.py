from utils import get_combinations
from evaluator import composite_score
import numpy as np

class BandSelector:
    def __init__(self, band_count, comb_dim, alpha=1.0, beta=1.0):
        self.band_count = band_count
        self.comb_dim = comb_dim
        self.alpha = alpha
        self.beta = beta

    def recommend(self, X, y, top_n=5, weights=(0.5, 0.5, 0.0)):
        combos = get_combinations(self.band_count, self.comb_dim)
        results = []
        for bands in combos:
            x_sel = X[:, bands]
            score, metrics = composite_score(x_sel, y, weights=weights)
            results.append((bands, score, metrics))
        best = sorted(results, key=lambda x: -x[1])[:top_n]
        return best
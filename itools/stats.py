"""Metrics and paired significance tests shared by both experiments."""
import numpy as np
from scipy import stats


def accuracy(correct):
    return float(np.mean(correct))


def ece(correct, conf, n_bins=10):
    """Expected calibration error with equal-width bins on [0, 1] (bins are [a, b), the last one closed)."""
    correct, conf = np.asarray(correct, float), np.clip(np.asarray(conf, float), 0.0, 1.0)
    idx = np.minimum(np.searchsorted(np.linspace(0, 1, n_bins + 1), conf, side='right') - 1, n_bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).sum() for b in range(n_bins) if (idx == b).any()) / len(conf))


def mcnemar_exact(before, after):
    """Two-sided exact McNemar test on paired booleans. Returns (n_improved, n_worsened, p)."""
    before, after = np.asarray(before, bool), np.asarray(after, bool)
    better, worse = int(np.sum(~before & after)), int(np.sum(before & ~after))
    n = better + worse
    return better, worse, (float(stats.binomtest(better, n, 0.5).pvalue) if n else float('nan'))


def paired_bootstrap(metric, n, n_boot=10000, seed=42):
    """Bootstrap the paired difference `metric(idx)` over resampled indices. Returns dict(diff, ci_low, ci_high, p)."""
    rng = np.random.default_rng(seed)
    diffs = np.array([metric(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {'diff': float(metric(np.arange(n))), 'ci_low': float(lo), 'ci_high': float(hi), 'p': float(min(p, 1.0))}


def compare_calibration(correct_a, conf_a, correct_b, conf_b, n_boot=10000, n_bins=10):
    """Paired comparison of A vs B on the same items: accuracy diff and ECE diff with bootstrap CIs, plus exact McNemar on accuracy."""
    ca, cb = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    fa, fb = np.asarray(conf_a, float), np.asarray(conf_b, float)
    n = len(ca)
    return {'n': n,
            'accuracy': paired_bootstrap(lambda i: accuracy(ca[i]) - accuracy(cb[i]), n, n_boot),
            'ece': paired_bootstrap(lambda i: ece(ca[i], fa[i], n_bins) - ece(cb[i], fb[i], n_bins), n, n_boot),
            'mcnemar': mcnemar_exact(cb, ca)}

import numpy as np

from txt2onto.evaluate import best_f1_threshold, calibration_bins, precision_target_thresholds, threshold_metrics


def test_threshold_metrics_counts():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    m = threshold_metrics(y, p, 0.5)
    assert m["confusion"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_best_f1_threshold_separable():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.7, 0.9])
    assert threshold_metrics(y, p, best_f1_threshold(y, p))["f1"] == 1.0


def test_precision_target_thresholds_respect_target():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.5).astype(int)
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.2, 5000), 0, 1)
    t = precision_target_thresholds(y, p, 0.98, min_support=100)
    assert y[p >= t["tau_pos"]].mean() >= 0.98
    assert (1 - y[p <= t["tau_neg"]]).mean() >= 0.98
    assert 0 < t["coverage_pos"] < 1 and 0 < t["coverage_neg"] < 1


def test_calibration_bins_cover_all_rows():
    p = np.linspace(0, 1, 101)
    assert sum(b["n"] for b in calibration_bins((p > 0.5).astype(int), p)) == 101

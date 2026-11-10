"""
Tests for src/conformal.py — run them from the project folder with:   pytest -q
Each test checks one small promise of the code with numbers we can work out by hand.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import conformal


def test_softmax_rows_add_up_to_one():
    probs = conformal.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], 1 / 3)             # equal logits -> equal probabilities


def test_conformal_quantile_picks_the_right_score():
    scores = np.arange(1, 20) / 20                   # 19 scores: 0.05, 0.10, ..., 0.95
    # ceil((19 + 1) * 0.9) = 18 -> the 18th smallest score = 0.90
    assert conformal.conformal_quantile(scores, alpha=0.1) == 0.90
    assert conformal.conformal_quantile(scores[:5], alpha=0.1) == float("inf")   # too few scores


def test_group_probabilities_add_class_probabilities():
    probs = np.array([[0.5, 0.3, 0.2]])
    class_groups = [0, 1, 1]                         # class 0 -> group 0, classes 1 and 2 -> group 1
    assert np.allclose(conformal.group_probabilities(probs, class_groups, 2), [[0.5, 0.5]])


def test_coverage_is_about_ninety_percent():
    # Random "model": calibration and test come from the same distribution, so coverage must be >= 0.9
    rng = np.random.default_rng(0)
    class_groups = [0, 0, 1, 1, 2]

    def fake_data(n):
        labels = rng.integers(0, 5, n)
        logits = rng.normal(size=(n, 5))
        logits[np.arange(n), labels] += 2.0          # the true class gets a higher score
        return conformal.softmax(logits), labels

    cal_probs, cal_labels = fake_data(3000)
    test_probs, test_labels = fake_data(3000)
    thresholds = conformal.calibrate_group_thresholds(cal_probs, cal_labels, class_groups, 3, alpha=0.1)
    sets = conformal.prediction_sets(test_probs, class_groups, 3, thresholds)
    true_groups = np.array(class_groups)[test_labels]
    coverage = sets[np.arange(len(test_labels)), true_groups].mean()
    assert 0.88 <= coverage <= 0.93


def test_decision_rule():
    assert conformal.decide([1]) == ("advise", 1)
    assert conformal.decide([1, 3]) == ("refer", None)
    assert conformal.decide([]) == ("unknown", None)
    assert conformal.decide([1], is_unfamiliar=True) == ("unknown", None)


def test_temperature_is_found_back():
    # logits made with temperature 2 -> fitting should find a T close to 2
    rng = np.random.default_rng(1)
    true_logits = rng.normal(size=(4000, 4)) * 2
    labels = np.array([rng.choice(4, p=p) for p in conformal.softmax(true_logits)])
    T = conformal.fit_temperature(true_logits * 2, labels)    # logits made twice too big
    assert 1.7 < T < 2.3

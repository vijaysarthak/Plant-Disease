"""
metrics.py — numbers that tell us how good and how SAFE the system is.

  coverage     : how often the TRUE group is inside the prediction set (target: >= 0.90)
  set_size     : average number of groups in the set (smaller = more useful)
  advice_rate  : how often the set has exactly one group, so advice is given
  IAR          : Inappropriate-Action Rate = wrong advice / all advice given   (our new safety metric)
  ECE          : Expected Calibration Error = gap between confidence and real accuracy (0 = perfect)

Used by: the notebook.
"""
import numpy as np


def advisory_metrics(sets, labels, class_groups, group_codes):
    """sets: True/False table (photos x groups). labels: true class number of every photo."""
    class_groups = np.asarray(class_groups)
    true_groups = class_groups[labels]
    set_size = sets.sum(axis=1)
    advised = set_size == 1                        # exactly one group -> advice is given
    advised_group = sets.argmax(axis=1)            # (only meaningful when advised)
    wrong_advice = advised & (advised_group != true_groups)

    result = {
        "coverage": float(sets[np.arange(len(labels)), true_groups].mean()),
        "set_size": float(set_size.mean()),
        "advice_rate": float(advised.mean()),
        "IAR": float(wrong_advice.sum() / max(advised.sum(), 1)),
        "wrong_advice_all": float(wrong_advice.mean()),
    }
    # coverage of every group separately (NaN if there is no photo of that group)
    for g, code in enumerate(group_codes):
        photos_of_group = true_groups == g
        if photos_of_group.any():
            result[f"coverage_{code}"] = float(sets[photos_of_group, g].mean())
        else:
            result[f"coverage_{code}"] = float("nan")
    return result


def expected_calibration_error(probs, labels, number_of_bins=15):
    """Put photos in bins by confidence; compare each bin's confidence with its accuracy."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bin_edges = np.linspace(0, 1, number_of_bins + 1)
    ece = 0.0
    for b in range(number_of_bins):
        in_bin = (confidence > bin_edges[b]) & (confidence <= bin_edges[b + 1])
        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)

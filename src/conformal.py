"""
conformal.py — the "brain" of the project: turns model scores into a SAFE decision.

Words used here:
  logits       : the raw scores the network outputs, one per class (any number, can be negative)
  probability  : softmax(logits / T) — numbers between 0 and 1 that add up to 1
  temperature T: one number that makes the probabilities honest (fitted on validation photos)
  group probability: probability of a GROUP = sum of the probabilities of its classes
  score        : s(x, g) = 1 - group probability.  Small score = group is likely.
  threshold q_g: one cut-off per group, learned from calibration photos
  prediction set: all groups whose score is <= their threshold

The promise (conformal prediction): if calibration photos and new photos look alike, the TRUE group
is inside the prediction set at least 90% of the time (because ALPHA = 0.10), for EVERY group.

Decision rule (our method):
  * the set has exactly 1 group   -> ADVISE that group's treatment
  * the set has 2 or more groups  -> REFER the farmer to an expert (KVK)
  * the set is empty, or the photo looks unfamiliar (high energy) -> UNKNOWN

Only needs numpy, so the same file is used by the notebook AND by the app.
"""
import math

import numpy as np


# ----------------------------------------------------------------------------
# 1. From logits to probabilities
# ----------------------------------------------------------------------------
def softmax(logits, temperature=1.0):
    """logits (photos x classes) -> probabilities (photos x classes), each row adds up to 1."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)    # subtract the biggest value: avoids overflow
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def negative_log_likelihood(logits, labels, temperature):
    """Average of -log(probability of the correct class). Lower = better calibrated."""
    probs = softmax(logits, temperature)
    correct_probs = probs[np.arange(len(labels)), labels]
    return float(np.mean(-np.log(correct_probs + 1e-12)))


def fit_temperature(logits, labels):
    """Try many temperatures and keep the one with the lowest negative log-likelihood.
    (A simple 'grid search': no clever maths needed.)"""
    best_temperature = 1.0
    best_loss = negative_log_likelihood(logits, labels, 1.0)
    for temperature in np.arange(0.05, 5.0, 0.01):
        loss = negative_log_likelihood(logits, labels, temperature)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


# ----------------------------------------------------------------------------
# 2. Our score: one value per GROUP
# ----------------------------------------------------------------------------
def group_probabilities(probs, class_groups, num_groups):
    """probs (photos x 38 classes) -> group probabilities (photos x 6 groups)."""
    class_groups = np.asarray(class_groups)
    result = np.zeros((len(probs), num_groups))
    for g in range(num_groups):
        result[:, g] = probs[:, class_groups == g].sum(axis=1)
    return result


def group_scores(probs, class_groups, num_groups):
    """s(x, g) = 1 - group probability.  Small score = this group is likely."""
    return 1.0 - group_probabilities(probs, class_groups, num_groups)


# ----------------------------------------------------------------------------
# 3. Calibration: learn one threshold per group
# ----------------------------------------------------------------------------
def conformal_quantile(scores, alpha):
    """The ceil((n + 1) * (1 - alpha))-th smallest score.
    This small '+1' is what makes the 90% promise exact. Returns infinity if there are too few scores."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return math.inf
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


def calibrate_group_thresholds(probs, labels, class_groups, num_groups, alpha, fallback=None):
    """
    'Mondrian' calibration = a separate threshold for every group, using only the calibration
    photos whose TRUE group is that group. So rare groups get their own 90% promise too.

    If a group has too few calibration photos and `fallback` thresholds are given,
    that group keeps its fallback threshold.
    """
    class_groups = np.asarray(class_groups)
    true_groups = class_groups[labels]
    scores = group_scores(probs, class_groups, num_groups)
    score_of_true_group = scores[np.arange(len(labels)), true_groups]

    minimum_photos = math.ceil(1 / alpha) - 1          # with fewer photos the quantile is infinite
    thresholds = np.zeros(num_groups)
    for g in range(num_groups):
        scores_of_group = score_of_true_group[true_groups == g]
        if len(scores_of_group) < minimum_photos and fallback is not None:
            thresholds[g] = fallback[g]
        else:
            thresholds[g] = conformal_quantile(scores_of_group, alpha)
    return thresholds


# ----------------------------------------------------------------------------
# 4. Prediction sets and the decision
# ----------------------------------------------------------------------------
def prediction_sets(probs, class_groups, num_groups, thresholds):
    """True/False table (photos x groups): is group g in the prediction set of photo i?"""
    scores = group_scores(probs, class_groups, num_groups)
    return scores <= np.asarray(thresholds)


def energy_score(logits, temperature=1.0):
    """Energy = -T * log(sum(exp(logits / T))).
    LOW for photos that look like the training photos, HIGH for unfamiliar photos."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    biggest = scaled.max(axis=1, keepdims=True)
    log_sum_exp = biggest[:, 0] + np.log(np.exp(scaled - biggest).sum(axis=1))
    return -temperature * log_sum_exp


def decide(groups_in_set, is_unfamiliar=False):
    """
    groups_in_set: list of group numbers in the prediction set of ONE photo.
    Returns ("advise", group) or ("refer", None) or ("unknown", None).
    """
    if is_unfamiliar or len(groups_in_set) == 0:
        return "unknown", None
    if len(groups_in_set) == 1:
        return "advise", groups_in_set[0]
    return "refer", None


# ----------------------------------------------------------------------------
# 5. Baselines we compare against (used in the notebook only)
# ----------------------------------------------------------------------------
def top1_sets(probs, class_groups, num_groups):
    """Normal classifier: always exactly the group of the most likely class."""
    class_groups = np.asarray(class_groups)
    sets = np.zeros((len(probs), num_groups), dtype=bool)
    best_class = probs.argmax(axis=1)
    sets[np.arange(len(probs)), class_groups[best_class]] = True
    return sets


def aps_scores(probs):
    """APS (standard conformal method on CLASSES): score of a class = total probability of all
    classes ranked at or above it."""
    scores = np.zeros_like(probs)
    for i in range(len(probs)):
        order = np.argsort(-probs[i])                    # most likely class first
        scores[i][order] = np.cumsum(probs[i][order])
    return scores


def calibrate_aps(probs, labels, alpha):
    scores = aps_scores(probs)
    return conformal_quantile(scores[np.arange(len(labels)), labels], alpha)


def aps_group_sets(probs, class_groups, num_groups, threshold):
    """APS set of classes -> set of groups (a group is in if any of its classes is in)."""
    class_groups = np.asarray(class_groups)
    class_sets = aps_scores(probs) <= threshold
    for i in range(len(probs)):
        if not class_sets[i].any():                      # never return an empty set
            class_sets[i, probs[i].argmax()] = True
    sets = np.zeros((len(probs), num_groups), dtype=bool)
    for g in range(num_groups):
        sets[:, g] = class_sets[:, class_groups == g].any(axis=1)
    return sets

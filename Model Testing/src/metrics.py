"""
Step 3b - Evaluation metrics, written by hand (PyTorch tensors only, no scikit-learn).

For a binary task with a "positive" class (wet = 1, or cough = 1):

                         predicted 0 (dry)   predicted 1 (wet)
    true 0 (dry)              TN                  FP
    true 1 (wet)              FN                  TP

    accuracy   = (TP + TN) / all
    precision  = TP / (TP + FP)      "when it says wet, how often is it right?"
    recall     = TP / (TP + FN)      "of all wet coughs, how many did it catch?"  (= sensitivity)
    specificity= TN / (TN + FP)      "of all dry coughs, how many did it call dry?"
    F1         = 2 * P * R / (P + R) balance of precision and recall
    balanced accuracy = (recall + specificity) / 2    fair when classes are unequal (77% dry)
    ROC-AUC    = chance a random wet cough gets a higher wet-score than a random dry cough
"""

import torch


def confusion_matrix(y_true, y_pred, n_classes=2):
    """cm[i, j] = number of samples with true class i predicted as class j."""
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)
    idx = y_true * n_classes + y_pred
    return torch.bincount(idx, minlength=n_classes * n_classes).view(n_classes, n_classes)


def _div(a, b):
    return float(a) / float(b) if b else 0.0


def per_class(cm):
    """Precision, recall and F1 for every class, treating that class as 'positive'."""
    out = []
    for k in range(cm.shape[0]):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp
        p, r = _div(tp, tp + fp), _div(tp, tp + fn)
        out.append({"precision": p, "recall": r, "f1": _div(2 * p * r, p + r),
                    "support": int(cm[k, :].sum().item())})
    return out


def roc_auc(y_true, scores):
    """
    Area under the ROC curve via ranks (Mann-Whitney U):
    AUC = P(score of a random positive > score of a random negative), ties count half.
    """
    y = torch.as_tensor(y_true, dtype=torch.long)
    s = torch.as_tensor(scores, dtype=torch.float64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    s_sorted = s[order]
    ranks = torch.empty_like(s)
    i, n = 0, len(s)
    while i < n:                                  # average rank for tied scores
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = ranks[y == 1].sum().item()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_report(y_true, prob_pos, threshold=0.5, names=("dry", "wet")):
    """All metrics for a binary task. prob_pos = model probability of class 1."""
    prob_pos = torch.as_tensor(prob_pos, dtype=torch.float32)
    y_pred = (prob_pos >= threshold).long()
    cm = confusion_matrix(y_true, y_pred, 2)
    tn, fp, fn, tp = cm[0, 0].item(), cm[0, 1].item(), cm[1, 0].item(), cm[1, 1].item()
    pc = per_class(cm)
    recall, spec = pc[1]["recall"], pc[0]["recall"]
    return {
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "accuracy": _div(tp + tn, cm.sum().item()),
        "precision": pc[1]["precision"],
        "recall": recall,
        "specificity": spec,
        "f1": pc[1]["f1"],
        "macro_f1": (pc[0]["f1"] + pc[1]["f1"]) / 2,
        "balanced_accuracy": (recall + spec) / 2,
        "roc_auc": roc_auc(y_true, prob_pos),
        "names": list(names),
    }


def best_threshold(y_true, prob_pos, metric="f1"):
    """Pick the decision threshold that maximises a metric. Use on VALIDATION data only."""
    best_t, best_v = 0.5, -1.0
    for t in torch.linspace(0.05, 0.95, 91).tolist():
        v = binary_report(y_true, prob_pos, t)[metric]
        if v > best_v:
            best_t, best_v = t, v
    return best_t, best_v


def format_report(r):
    a, b = r["names"]
    cm = r["confusion_matrix"]
    w = max(len(a), len(b), 6)
    lines = [
        f"threshold {r['threshold']:.2f}",
        f"{'':>{w + 7}}pred {a:>{w}}  pred {b:>{w}}",
        f"true {a:>{w}}  {cm[0][0]:>{w + 5}}  {cm[0][1]:>{w + 5}}",
        f"true {b:>{w}}  {cm[1][0]:>{w + 5}}  {cm[1][1]:>{w + 5}}",
        f"accuracy {r['accuracy']:.3f} | balanced acc {r['balanced_accuracy']:.3f} | "
        f"ROC-AUC {r['roc_auc']:.3f}",
        f"{b}: precision {r['precision']:.3f}  recall {r['recall']:.3f}  F1 {r['f1']:.3f} | "
        f"{a} recall (specificity) {r['specificity']:.3f} | macro F1 {r['macro_f1']:.3f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.6, 0.2, 0.4, 0.8, 0.7, 0.4, 0.9]
    print(format_report(binary_report(y, p)))

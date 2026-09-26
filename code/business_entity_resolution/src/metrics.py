def f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def macro_f05(predictions, truth):
    """
    predictions/truth: dict[source1_id, set[target_ids]]
    Computes per-S1 F0.5 and averages, including singletons.
    """
    scores = []
    for s1_id, true_set in truth.items():
        pred_set = predictions.get(s1_id, set())
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        if tp + fp == 0:
            precision = 1.0 if tp == 0 else 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 1.0
        else:
            recall = tp / (tp + fn)

        scores.append(f05(precision, recall))
    return sum(scores) / len(scores) if scores else 0.0

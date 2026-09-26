import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from .schema import validate_labels, InputError
from .pipeline import classify


def binary_metrics(y, predictions, scores=None):
    y = validate_labels(y)
    p = validate_labels(predictions, len(y))
    tn, fp, fn, tp = map(int, confusion_matrix(y, p, labels=[0, 1]).ravel())
    def ratio(a, b):
        return a / b if b else None
    result = {'rows': len(y), 'accuracy': (tp + tn) / len(y),
              'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
              'f1': ratio(2 * tp, 2 * tp + fp + fn), 'fpr': ratio(fp, fp + tn),
              'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != y.shape or not np.isfinite(scores).all():
            raise InputError('Metrik hesabı için geçersiz skorlar.')
        result['roc_auc'] = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None
        result['average_precision'] = float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else None
    return result


def evaluate_scores(y, scores, threshold):
    return {'threshold': float(threshold), **binary_metrics(y, classify(scores, threshold), scores)}


def calibrate_threshold(normal_scores, quantile=.95):
    values = np.asarray(normal_scores, dtype=float)
    if not 0 < quantile < 1 or values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise InputError('Eşik kalibrasyonu için geçerli normal skorlar ve yüzdelik gerekli.')
    return float(np.quantile(values, quantile, method='higher'))

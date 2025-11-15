import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def custom_scorer(y_true, y_pred):
    # tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    auc = roc_auc_score(y_true, y_pred)

    #ap = average_precision_score(y_true, y_pred)

    g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
    return {'f1': f1, 'auc': auc, 'g_mean': g_mean}
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss as sk_hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def to_array(v):
    if isinstance(v, pd.DataFrame):
        return v.values.copy()
    if isinstance(v, pd.Series):
        return v.values.copy()
    if isinstance(v, np.ndarray):
        return v.copy()
    return np.array(v)


def label_accuracy(y_true, y_pred):
    yt = to_array(y_true)
    yp = to_array(y_pred)
    return float((yt == yp).mean())


def calc_metrics(y_true, y_pred):
    yt = to_array(y_true)
    yp = to_array(y_pred)
    met = {
        "exact_acc": accuracy_score(yt, yp),
        "label_acc": label_accuracy(yt, yp),
        "f1_micro": f1_score(yt, yp, average="micro", zero_division=0),
        "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
        "recall_micro": recall_score(yt, yp, average="micro", zero_division=0),
        "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
        "precision_micro": precision_score(yt, yp, average="micro", zero_division=0),
        "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
        "hamming_loss": sk_hamming_loss(yt, yp),
    }
    return {k: float(v) for k, v in met.items()}


def print_metrics(met):
    keys = [
        "exact_acc",
        "label_acc",
        "f1_micro",
        "f1_macro",
        "recall_micro",
        "recall_macro",
        "precision_micro",
        "precision_macro",
        "hamming_loss",
    ]
    for key in keys:
        print(f"{key}: {met[key]:.4f}")

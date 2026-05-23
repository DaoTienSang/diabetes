import numpy as np
import pandas as pd

from src.train.metrics import calc_metrics, label_accuracy, print_metrics


def sample_arr():
    y_true = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
        ]
    )
    y_pred = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
        ]
    )
    return y_true, y_pred


def metric_keys():
    return {
        "exact_acc",
        "label_acc",
        "f1_micro",
        "f1_macro",
        "recall_micro",
        "recall_macro",
        "precision_micro",
        "precision_macro",
        "hamming_loss",
    }


def test_calc_metrics_keys():
    y_true, y_pred = sample_arr()
    met = calc_metrics(y_true, y_pred)
    assert set(met) == metric_keys()


def test_metrics_range():
    y_true, y_pred = sample_arr()
    met = calc_metrics(y_true, y_pred)
    for val in met.values():
        assert 0 <= val <= 1


def test_label_accuracy_value():
    y_true = [[1, 0, 1], [0, 1, 0]]
    y_pred = [[1, 0, 0], [0, 1, 1]]
    assert label_accuracy(y_true, y_pred) == 4 / 6


def test_exact_acc_strict():
    y_true = [[1, 0, 1], [0, 1, 0]]
    y_pred = [[1, 0, 0], [0, 1, 0]]
    met = calc_metrics(y_true, y_pred)
    assert met["exact_acc"] == 0.5
    assert met["exact_acc"] != met["label_acc"]


def test_dataframe_input():
    cols = ["nep", "neu", "ret", "cv", "per_vas"]
    y_true, y_pred = sample_arr()
    yt = pd.DataFrame(y_true, columns=cols)
    yp = pd.DataFrame(y_pred, columns=cols)
    met = calc_metrics(yt, yp)
    assert "f1_micro" in met


def test_list_input():
    y_true = [[1, 0, 1], [0, 1, 0]]
    y_pred = [[1, 0, 0], [0, 1, 1]]
    met = calc_metrics(y_true, y_pred)
    assert "label_acc" in met


def test_all_zero_case():
    y_true = np.zeros((4, 5), dtype=int)
    y_pred = np.zeros((4, 5), dtype=int)
    met = calc_metrics(y_true, y_pred)
    assert met["hamming_loss"] == 0
    assert met["exact_acc"] == 1
    assert met["label_acc"] == 1


def test_all_wrong_case():
    y_true = np.ones((4, 5), dtype=int)
    y_pred = np.zeros((4, 5), dtype=int)
    met = calc_metrics(y_true, y_pred)
    assert met["label_acc"] == 0
    assert met["hamming_loss"] == 1


def test_print_metrics():
    y_true, y_pred = sample_arr()
    met = calc_metrics(y_true, y_pred)
    print_metrics(met)

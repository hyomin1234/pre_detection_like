from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    tnr: float
    f1: float
    bal_acc: float
    fp: int
    fn: int
    tp: int
    tn: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "tnr": self.tnr,
            "f1": self.f1,
            "bal_acc": self.bal_acc,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "tn": self.tn,
        }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> BinaryMetrics:
    y_true = y_true.astype(np.int64)
    y_pred = (y_prob >= thr).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    bal_acc = 0.5 * (recall + tnr)

    return BinaryMetrics(
        precision=precision,
        recall=recall,
        tnr=tnr,
        f1=f1,
        bal_acc=bal_acc,
        fp=fp,
        fn=fn,
        tp=tp,
        tn=tn,
    )


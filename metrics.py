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
    auroc: float
    auprc: float
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
            "auroc": self.auroc,
            "auprc": self.auprc,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "tn": self.tn,
        }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _binary_clf_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_prob, kind="mergesort")
    y_true = y_true[order]
    y_prob = y_prob[order]

    distinct_idx = np.where(np.diff(y_prob))[0]
    threshold_idx = np.r_[distinct_idx, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idx].astype(np.float64)
    fps = (1 + threshold_idx - tps).astype(np.float64)
    return fps, tps


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    fps, tps = _binary_clf_curve(y_true=y_true, y_prob=y_prob)
    fpr = np.r_[0.0, fps / neg, 1.0]
    tpr = np.r_[0.0, tps / pos, 1.0]
    return float(np.trapezoid(tpr, fpr))


def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = int((y_true == 1).sum())
    if pos == 0:
        return float("nan")

    fps, tps = _binary_clf_curve(y_true=y_true, y_prob=y_prob)
    precision = np.divide(
        tps,
        tps + fps,
        out=np.ones_like(tps, dtype=np.float64),
        where=(tps + fps) > 0,
    )
    recall = tps / pos
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


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
    auroc = compute_auroc(y_true=y_true, y_prob=y_prob)
    auprc = compute_auprc(y_true=y_true, y_prob=y_prob)

    return BinaryMetrics(
        precision=precision,
        recall=recall,
        tnr=tnr,
        f1=f1,
        bal_acc=bal_acc,
        auroc=auroc,
        auprc=auprc,
        fp=fp,
        fn=fn,
        tp=tp,
        tn=tn,
    )

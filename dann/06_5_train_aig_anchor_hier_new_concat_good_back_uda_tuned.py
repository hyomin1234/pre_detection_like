#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_5_train_aig_anchor_hier.py

목적:
- 03_8_prepare_aig_anchor_bundle_dataset.py 가 만든 anchor-level hierarchical AIG dataset을 학습합니다.
- 샘플 1개 = anchor 1개 이며, 각 샘플 내부에는 여러 combinational island AIG graph가 포함됩니다.
- 각 island를 GNN으로 인코딩하고, anchor 내부에서 pooling(mean/max)하여 최종 anchor-level 분류를 수행합니다.

핵심 구조:
anchor sample
  -> island_0 graph -> island encoder -> emb_0
  -> island_1 graph -> island encoder -> emb_1
  -> ...
  -> mean/max pooling across islands
  -> concat(anchor_num_feat embedding)
  -> classifier
"""

import os
import re
import json
import gzip
import math
import pickle
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
except Exception as e:
    raise RuntimeError("torch_geometric is required.") from e


# ----------------------------
# Utility
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_pickle_gz(path: str):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    recall = tp / max(tp + fn, 1e-8)
    spec = tn / max(tn + fp, 1e-8)
    prec = tp / max(tp + fp, 1e-8)
    f1 = 2.0 * prec * recall / max(prec + recall, 1e-8)
    bal_acc = 0.5 * (recall + spec)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1e-8)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "balanced_acc": bal_acc,
        "acc": acc,
    }


def grouped_family_split(samples: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if not samples:
        return [], []
    families = sorted(set(str(s.get("source_design", "unknown")) for s in samples))
    rng = random.Random(seed)
    rng.shuffle(families)
    if len(families) <= 1:
        return stratified_split(samples, val_ratio, seed)
    n_val = max(1, int(round(len(families) * val_ratio)))
    n_val = min(n_val, len(families) - 1)
    val_fams = set(families[:n_val])
    train = [s for s in samples if str(s.get("source_design", "unknown")) not in val_fams]
    val = [s for s in samples if str(s.get("source_design", "unknown")) in val_fams]
    if not train or not val:
        return stratified_split(samples, val_ratio, seed)
    return train, val


def stratified_split(samples: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    pos = [s for s in samples if int(s.get("graph_label", 0)) == 1]
    neg = [s for s in samples if int(s.get("graph_label", 0)) == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_one(lst: List[dict]) -> Tuple[List[dict], List[dict]]:
        if len(lst) <= 1:
            return lst[:], []
        n_val = max(1, int(round(len(lst) * val_ratio)))
        n_val = min(n_val, len(lst) - 1)
        return lst[n_val:], lst[:n_val]

    tr_p, va_p = split_one(pos)
    tr_n, va_n = split_one(neg)
    train = tr_p + tr_n
    val = va_p + va_n
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ----------------------------
# Dataset / Collate
# ----------------------------

@dataclass
class AnchorAIGSample:
    sample_id: str
    source_design: str
    label: int
    fold_id: int
    anchor_num_feat: List[float]
    islands: List[dict]
    meta_graph: dict


class AnchorAIGDataset(Dataset):
    def __init__(self, samples: List[dict]):
        self.samples = [
            AnchorAIGSample(
                sample_id=str(s["sample_id"]),
                source_design=str(s.get("source_design", "unknown")),
                label=int(s.get("graph_label", 0)),
                fold_id=int(s.get("fold_id", -1)),
                anchor_num_feat=[float(v) for v in s.get("anchor_num_feat", [])],
                islands=list(s.get("islands", [])),
                meta_graph=dict(s.get("meta_graph", {"nodes": [], "edges": []})),
            )
            for s in samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> AnchorAIGSample:
        return self.samples[idx]

def get_island_graph_feat(isl: dict, mode: str) -> List[float]:
    num_feat = [float(v) for v in isl.get("island_num_feat", [])]
    func_feat = [float(v) for v in isl.get("island_func_feat", [])]

    if mode == "num_only":
        return num_feat
    elif mode == "func_only":
        return func_feat
    elif mode == "concat":
        return num_feat + func_feat
    else:
        raise ValueError(f"Unknown island feature mode: {mode}")

def collate_anchor_aig(batch: List[AnchorAIGSample], vocab=None, island_feat_mode: str = "num_only") -> dict:
    anchor_num_feat = torch.tensor([s.anchor_num_feat for s in batch], dtype=torch.float)
    labels = torch.tensor([s.label for s in batch], dtype=torch.long)
    sample_ids = [s.sample_id for s in batch]
    source_designs = [s.source_design for s in batch]

    island_graphs: List[Data] = []
    island_global_idx: List[int] = [] # index in the overall list of islands in this batch

    meta_graphs: List[Data] = []

    current_island_global_offset = 0
    for anchor_idx, s in enumerate(batch):
        # 1. Process AIG Islands
        isl_to_global_idx = {}
        for isl in s.islands:
            x = torch.tensor(isl["x"], dtype=torch.float)
            edge_index = torch.tensor(isl["edge_index"], dtype=torch.long)
            graph_feat_vals = get_island_graph_feat(isl, island_feat_mode)
            graph_feat = torch.tensor([graph_feat_vals], dtype=torch.float)

            g = Data(
                x=x,
                edge_index=edge_index,
                graph_feat=graph_feat,
                num_nodes=x.size(0),
            )
            island_graphs.append(g)
            isl_id = int(isl["island_id"])
            isl_to_global_idx[isl_id] = current_island_global_offset
            current_island_global_offset += 1

        # 2. Process Meta-Graph
        m_nodes = s.meta_graph.get("nodes", [])
        m_edges = s.meta_graph.get("edges", [])
        
        # Meta-node ID는 이미 "SEQ_X" / "ISL_Y" 형식이므로 n["id"] 그대로 사용
        m_node_to_idx = {n["id"]: i for i, n in enumerate(m_nodes)}
        
        m_x_type = [] # 0 for Island, 1 for SEQ
        m_x_val = []  # Global Island Index OR Cell Type Index
        
        for n in m_nodes:
            if n["type"] == "island":
                m_x_type.append(0)
                # island id는 "ISL_0" 형식이므로, 숫자만 추출해 isl_to_global_idx 매핑
                raw_id = n["id"]
                isl_num = int(str(raw_id).replace("ISL_", "")) if str(raw_id).startswith("ISL_") else raw_id
                m_x_val.append(isl_to_global_idx.get(isl_num, -1))
            else:
                m_x_type.append(1)
                # Map cell type using vocab if available
                cell_type = n.get("details", {}).get("raw_type", "DFF")
                val = 10 # Default DFF in standard_radar_vocab
                if vocab and "raw_cell_vocab" in vocab:
                    val = vocab["raw_cell_vocab"].get(cell_type, vocab["raw_cell_vocab"].get("<UNK>", 0))
                m_x_val.append(val)
        
        m_edge_index = []
        for u, v in m_edges:
            if u in m_node_to_idx and v in m_node_to_idx:
                m_edge_index.append([m_node_to_idx[u], m_node_to_idx[v]])
        
        if not m_edge_index:
            m_edge_index_t = torch.empty((2, 0), dtype=torch.long)
        else:
            m_edge_index_t = torch.tensor(m_edge_index, dtype=torch.long).t().contiguous()

        # Identify which node in the meta-graph is the "Anchor"
        # The anchor is usually the SEQ node that matches parent_sample_id's anchor part
        anchor_node_idx = 0
        anchor_name_part = s.sample_id.split("::")[-1] if "::" in s.sample_id else s.sample_id
        for i, n in enumerate(m_nodes):
            if n["type"] == "seq" and anchor_name_part in str(n.get("name", "")):
                anchor_node_idx = i
                break

        mg = Data(
            x_type=torch.tensor(m_x_type, dtype=torch.long),
            x_val=torch.tensor(m_x_val, dtype=torch.long),
            edge_index=m_edge_index_t,
            num_nodes=len(m_nodes),
            anchor_node_idx=torch.tensor([anchor_node_idx], dtype=torch.long)
        )
        meta_graphs.append(mg)

    if island_graphs:
        island_batch = Batch.from_data_list(island_graphs)
    else:
        island_batch = None

    meta_batch = Batch.from_data_list(meta_graphs)

    return {
        "anchor_num_feat": anchor_num_feat,
        "labels": labels,
        "sample_ids": sample_ids,
        "source_designs": source_designs,
        "island_batch": island_batch,
        "meta_batch": meta_batch,
    }


# ----------------------------
# Model
# ----------------------------

class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def apply_grl(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradientReversal.apply(x, alpha)


class IslandEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, island_num_feat_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + island_num_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        h = F.relu(self.input_proj(batch.x))
        for conv in self.convs:
            h = conv(h, batch.edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        pooled = torch.cat([
            global_mean_pool(h, batch.batch),
            global_max_pool(h, batch.batch),
        ], dim=-1)
        graph_feat = batch.graph_feat.view(pooled.size(0), -1)
        return self.out(torch.cat([pooled, graph_feat], dim=-1))


class MetaGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class AnchorAIGHierModel(nn.Module):
    def __init__(
        self,
        anchor_num_feat_dim: int,
        node_feat_dim: int = 6,
        island_num_feat_dim: int = 9,
        hidden_dim: int = 128,
        island_num_layers: int = 3,
        meta_num_layers: int = 2,
        dropout: float = 0.2,
        num_cell_types: int = 64,
        use_uda: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_uda = use_uda
        self.island_encoder = IslandEncoder(
            node_feat_dim=node_feat_dim,
            island_num_feat_dim=island_num_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=island_num_layers,
            dropout=dropout,
        )
        self.seq_embedding = nn.Embedding(num_cell_types, hidden_dim)
        
        self.meta_gnn = MetaGNN(
            hidden_dim=hidden_dim,
            num_layers=meta_num_layers,
            dropout=dropout,
        )

        self.anchor_meta_encoder = nn.Sequential(
            nn.Linear(anchor_num_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            # hidden_dim * 3: anchor_node_emb + window_global_pool + anchor_num_emb
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_dict: dict, return_domain_logits: bool = False, grl_alpha: float = 1.0):
        anchor_num_feat = batch_dict["anchor_num_feat"]
        batch_size = anchor_num_feat.size(0)
        device = anchor_num_feat.device

        island_batch = batch_dict["island_batch"]
        meta_batch = batch_dict["meta_batch"].to(device)

        # 1. Encode Islands
        if island_batch is not None:
            island_batch = island_batch.to(device)
            island_emb = self.island_encoder(island_batch)
        else:
            island_emb = torch.empty((0, self.hidden_dim), device=device)

        # 2. Build Meta-Graph Node Features
        # x_type: 0 for Island, 1 for SEQ
        # x_val: Island Global Index OR Cell Type Index
        meta_x = torch.zeros((meta_batch.num_nodes, self.hidden_dim), device=device)
        
        is_island = (meta_batch.x_type == 0)
        is_seq = (meta_batch.x_type == 1)
        
        if is_island.any():
            isl_indices = meta_batch.x_val[is_island]
            # Valid island indices (not -1)
            valid_mask = (isl_indices != -1)
            if valid_mask.any():
                meta_x[torch.where(is_island)[0][valid_mask]] = island_emb[isl_indices[valid_mask]]
        
        if is_seq.any():
            seq_indices = meta_batch.x_val[is_seq]
            meta_x[is_seq] = self.seq_embedding(seq_indices)

        # 3. Meta GNN
        meta_h = self.meta_gnn(meta_x, meta_batch.edge_index)

        # 4. Extract Anchor Node Embedding (local context)
        # anchor_node_idx는 sample별 local index이므로 batch offset 더해야 함
        ptr = meta_batch.ptr
        anchor_global_indices = ptr[:-1] + meta_batch.anchor_node_idx
        anchor_emb = meta_h[anchor_global_indices]

        # 5. Extract Window-Level Global Embedding (전체 subgraph context)
        # meta_graph 전체 노드의 mean pool -> anchor window 전역 구조 요약
        window_global_emb = global_mean_pool(meta_h, meta_batch.batch)

        # 6. Fusion & Classifier
        anchor_num_emb = self.anchor_meta_encoder(anchor_num_feat)
        fused = torch.cat([anchor_emb, window_global_emb, anchor_num_emb], dim=-1)
        cls_logits = self.classifier(fused)

        if return_domain_logits:
            if not self.use_uda:
                raise RuntimeError("Domain logits requested while use_uda=False.")
            rev_fused = apply_grl(fused, grl_alpha)
            domain_logits = self.domain_classifier(rev_fused)
            return cls_logits, domain_logits

        return cls_logits


# ----------------------------
# Train / Eval
# ----------------------------

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    out["anchor_num_feat"] = batch["anchor_num_feat"].to(device)
    out["labels"] = batch["labels"].to(device)
    if batch["island_batch"] is not None:
        out["island_batch"] = batch["island_batch"].to(device)
    if batch["meta_batch"] is not None:
        out["meta_batch"] = batch["meta_batch"].to(device)
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Tuple[dict, List[dict], List[dict]]:
    model.eval()
    tp = tn = fp = fn = 0
    probs_out = []
    missed = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        probs = F.softmax(logits, dim=-1)[:, 1]
        preds = (probs >= threshold).long()
        ys = batch["labels"].view(-1).long()

        for i in range(len(ys)):
            y = int(ys[i].item())
            p = int(preds[i].item())
            prob = float(probs[i].item())
            sid = str(batch["sample_ids"][i])
            fam = str(batch["source_designs"][i])
            n_islands = 0
            # batch 단위 리스트라 직접 수집
            probs_out.append({
                "sample_id": sid,
                "family": fam,
                "label": y,
                "prob": prob,
                "pred": p,
            })
            if y == 1 and p == 1:
                tp += 1
            elif y == 0 and p == 0:
                tn += 1
            elif y == 0 and p == 1:
                fp += 1
                missed.append({"sample_id": sid, "family": fam, "label": y, "pred": p, "prob": prob})
            else:
                fn += 1
                missed.append({"sample_id": sid, "family": fam, "label": y, "pred": p, "prob": prob})

    return compute_metrics(tp, tn, fp, fn), probs_out, missed


def _metrics_from_prob_records(prob_records: List[dict], threshold: float) -> dict:
    tp = tn = fp = fn = 0
    for r in prob_records:
        y = int(r["label"])
        p = 1 if float(r["prob"]) >= threshold else 0
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 0:
            tn += 1
        elif y == 0 and p == 1:
            fp += 1
        else:
            fn += 1
    return compute_metrics(tp, tn, fp, fn)


def select_threshold_from_val(
    prob_records: List[dict],
    metric: str = "f1",
    min_precision: float = 0.0,
    grid_start: float = 0.05,
    grid_end: float = 0.95,
    grid_step: float = 0.01,
) -> Tuple[float, dict]:
    if not prob_records:
        return 0.5, {}
    best_t = 0.5
    best_m = _metrics_from_prob_records(prob_records, best_t)
    best_score = float(best_m.get(metric, 0.0))
    t = grid_start
    while t <= grid_end + 1e-12:
        m = _metrics_from_prob_records(prob_records, t)
        if float(m.get("precision", 0.0)) >= min_precision:
            score = float(m.get(metric, 0.0))
            if score > best_score:
                best_score = score
                best_t = float(t)
                best_m = m
        t += grid_step
    return best_t, best_m


def train_one_run(
    train_samples: List[dict],
    val_samples: List[dict],
    test_samples: List[dict],
    outdir: str,
    device: torch.device,
    args,
    target_unlabeled_samples: Optional[List[dict]] = None,
) -> dict:
    from functools import partial
    collate_fn = partial(
        collate_anchor_aig,
        vocab=getattr(args, "vocab", None),
        island_feat_mode=args.island_feat_mode,
    )

    train_ds = AnchorAIGDataset(train_samples)
    val_ds = AnchorAIGDataset(val_samples)
    test_ds = AnchorAIGDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    use_uda = bool(getattr(args, "use_uda", False))
    uda_weight = float(getattr(args, "uda_weight", 1.0))
    uda_warmup_epochs = int(getattr(args, "uda_warmup_epochs", 0))
    min_epochs = int(getattr(args, "min_epochs", 1))
    stop_on_perfect_val = bool(getattr(args, "stop_on_perfect_val", False))
    tune_threshold_on_val = bool(getattr(args, "tune_threshold_on_val", False))
    threshold_metric = str(getattr(args, "threshold_metric", "f1"))
    threshold_min_precision = float(getattr(args, "threshold_min_precision", 0.0))
    threshold_grid_start = float(getattr(args, "threshold_grid_start", 0.05))
    threshold_grid_end = float(getattr(args, "threshold_grid_end", 0.95))
    threshold_grid_step = float(getattr(args, "threshold_grid_step", 0.01))
    eval_threshold = float(getattr(args, "eval_threshold", 0.5))
    target_loader = None
    target_iter = None
    n_target_unlabeled = 0
    if use_uda:
        target_unlabeled_samples = target_unlabeled_samples or []
        n_target_unlabeled = len(target_unlabeled_samples)
        if n_target_unlabeled == 0:
            print("[!] UDA requested but no target unlabeled samples were provided. Disabling UDA for this run.")
            use_uda = False
        else:
            target_ds = AnchorAIGDataset(target_unlabeled_samples)
            target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            target_iter = iter(target_loader)
            print(f"[*] UDA enabled: target unlabeled pool = {n_target_unlabeled}")

    anchor_num_feat_dim = len(train_samples[0].get("anchor_num_feat", [])) if train_samples else len(val_samples[0].get("anchor_num_feat", []))
    
    graph_feat_dim = 0
    first_sample = train_samples[0] if train_samples else (val_samples[0] if val_samples else None)

    if first_sample and first_sample.get("islands"):
        first_island = first_sample["islands"][0]
        num_dim = len(first_island.get("island_num_feat", []))
        func_dim = len(first_island.get("island_func_feat", []))

        if args.island_feat_mode == "num_only":
            graph_feat_dim = num_dim
        elif args.island_feat_mode == "func_only":
            graph_feat_dim = func_dim
        elif args.island_feat_mode == "concat":
            graph_feat_dim = num_dim + func_dim
        else:
            raise ValueError(f"Unknown island feature mode: {args.island_feat_mode}")

    model = AnchorAIGHierModel(
        anchor_num_feat_dim=anchor_num_feat_dim,
        island_num_feat_dim=graph_feat_dim,
        hidden_dim=args.hidden_dim,
        island_num_layers=args.num_layers,
        dropout=args.dropout,
        use_uda=use_uda,
    ).to(device)

    n_pos = sum(int(s.get("graph_label", 0)) == 1 for s in train_samples)
    n_neg = len(train_samples) - n_pos
    pos_weight = max(1.0, float(n_neg) / max(float(n_pos), 1.0)) if args.ht_weight <= 0 else float(args.ht_weight)
    class_weight = torch.tensor([1.0, pos_weight], dtype=torch.float, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    criterion_domain = nn.BCEWithLogitsLoss() if use_uda else None

    history = []
    best_val_bal_acc = -1.0
    best_val_metrics = None
    patience_left = args.early_stop_patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        total_domain_loss = 0.0
        total_domain_acc = 0.0
        domain_steps = 0
        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch)
            loss_cls = criterion(logits, batch["labels"].view(-1).long())
            loss = loss_cls

            if use_uda and criterion_domain is not None and target_loader is not None and epoch > uda_warmup_epochs:
                progress_num = (epoch - 1) * max(len(train_loader), 1) + step
                progress_den = max(args.epochs * max(len(train_loader), 1) - 1, 1)
                p = float(progress_num) / float(progress_den)
                alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

                _, src_domain_logits = model(batch, return_domain_logits=True, grl_alpha=alpha)
                src_targets = torch.zeros(src_domain_logits.size(0), device=device)
                loss_domain_src = criterion_domain(src_domain_logits.view(-1), src_targets)

                try:
                    tgt_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    tgt_batch = next(target_iter)
                tgt_batch = move_batch_to_device(tgt_batch, device)
                _, tgt_domain_logits = model(tgt_batch, return_domain_logits=True, grl_alpha=alpha)
                tgt_targets = torch.ones(tgt_domain_logits.size(0), device=device)
                loss_domain_tgt = criterion_domain(tgt_domain_logits.view(-1), tgt_targets)

                loss_domain = 0.5 * (loss_domain_src + loss_domain_tgt)
                loss = loss + uda_weight * loss_domain

                with torch.no_grad():
                    src_pred = (torch.sigmoid(src_domain_logits.view(-1)) >= 0.5).float()
                    tgt_pred = (torch.sigmoid(tgt_domain_logits.view(-1)) >= 0.5).float()
                    src_acc = float((src_pred == 0.0).float().mean().item()) if src_pred.numel() else 0.0
                    tgt_acc = float((tgt_pred == 1.0).float().mean().item()) if tgt_pred.numel() else 0.0
                    domain_acc = 0.5 * (src_acc + tgt_acc)

                total_domain_loss += float(loss_domain.item())
                total_domain_acc += domain_acc
                domain_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss_cls.item()) * batch["labels"].size(0)
            total_graphs += batch["labels"].size(0)

        train_loss = total_loss / max(total_graphs, 1)
        val_metrics, val_probs, val_missed = evaluate(model, val_loader, device, threshold=0.5)
        row = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics}
        if use_uda and domain_steps > 0:
            row["train_domain_loss"] = total_domain_loss / float(domain_steps)
            row["train_domain_acc"] = total_domain_acc / float(domain_steps)
        history.append(row)
        if use_uda and domain_steps > 0:
            print(
                f"[Epoch {epoch:03d}] loss={train_loss:.6f} "
                f"dom_loss={row['train_domain_loss']:.6f} dom_acc={row['train_domain_acc']:.6f} "
                f"val_bal_acc={val_metrics['balanced_acc']:.6f} val_f1={val_metrics['f1']:.6f}"
            )
        else:
            print(f"[Epoch {epoch:03d}] loss={train_loss:.6f} val_bal_acc={val_metrics['balanced_acc']:.6f} val_f1={val_metrics['f1']:.6f}")

        is_strict_improve = val_metrics["balanced_acc"] > best_val_bal_acc
        is_tie_or_improve = val_metrics["balanced_acc"] >= best_val_bal_acc

        if is_tie_or_improve:
            best_val_bal_acc = val_metrics["balanced_acc"]
            best_val_metrics = dict(val_metrics)
            best_val_metrics["epoch"] = epoch
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pt"))
            with open(os.path.join(outdir, "best_val_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(best_val_metrics, f, indent=2)
            with open(os.path.join(outdir, "val_probabilities.json"), "w", encoding="utf-8") as f:
                json.dump({"num_samples": len(val_probs), "probabilities": val_probs}, f, indent=2)
            with open(os.path.join(outdir, "best_val_missed_samples.json"), "w", encoding="utf-8") as f:
                json.dump(val_missed, f, indent=2)
            with open(os.path.join(outdir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            
            # Optional early exit on perfect source-val score (disabled by default)
            if stop_on_perfect_val and epoch >= min_epochs and best_val_bal_acc >= 1.0:
                print(f"[*] Perfect balanced accuracy (1.0) reached at epoch {epoch}. Stopping early.")
                break

            if is_strict_improve and epoch >= min_epochs:
                patience_left = args.early_stop_patience
            elif epoch >= min_epochs:
                patience_left -= 1
        else:
            if epoch >= min_epochs:
                patience_left -= 1
            if args.early_stop_patience > 0 and epoch >= min_epochs and patience_left <= 0:
                print("[*] Early stopping triggered.")
                break

    if best_val_metrics is None:
        raise RuntimeError("No best validation checkpoint selected.")

    best_model = AnchorAIGHierModel(
        anchor_num_feat_dim=anchor_num_feat_dim,
        island_num_feat_dim=graph_feat_dim,
        hidden_dim=args.hidden_dim,
        island_num_layers=args.num_layers,
        dropout=args.dropout,
        use_uda=use_uda,
    ).to(device)
    best_model.load_state_dict(torch.load(os.path.join(outdir, "best_model.pt"), map_location=device))
    if tune_threshold_on_val:
        _, best_val_probs, _ = evaluate(best_model, val_loader, device, threshold=0.5)
        eval_threshold, tuned_val_metrics = select_threshold_from_val(
            best_val_probs,
            metric=threshold_metric,
            min_precision=threshold_min_precision,
            grid_start=threshold_grid_start,
            grid_end=threshold_grid_end,
            grid_step=threshold_grid_step,
        )
        print(f"[*] Tuned threshold on val: {eval_threshold:.4f} ({threshold_metric}={tuned_val_metrics.get(threshold_metric, 0.0):.6f}, precision={tuned_val_metrics.get('precision', 0.0):.6f})")
    else:
        tuned_val_metrics = {}

    test_metrics, test_probs, test_missed = evaluate(best_model, test_loader, device, threshold=eval_threshold)

    final = {
        "best_val": best_val_metrics,
        "final_test": test_metrics,
        "seed": args.seed,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "anchor_num_feat_dim": anchor_num_feat_dim,
        "node_feat_dim": 6,
        "island_feat_mode": args.island_feat_mode,
        "graph_feat_dim": graph_feat_dim,
        "auto_pos_weight": pos_weight,
        "val_split_mode": args.val_split_mode,
        "use_uda": use_uda,
        "uda_weight": uda_weight,
        "n_target_unlabeled": n_target_unlabeled,
        "uda_warmup_epochs": uda_warmup_epochs,
        "min_epochs": min_epochs,
        "stop_on_perfect_val": stop_on_perfect_val,
        "eval_threshold": eval_threshold,
        "tune_threshold_on_val": tune_threshold_on_val,
        "threshold_metric": threshold_metric,
        "threshold_min_precision": threshold_min_precision,
        "tuned_val_metrics": tuned_val_metrics,
    }
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(outdir, "test_probabilities.json"), "w", encoding="utf-8") as f:
        json.dump({"num_samples": len(test_probs), "probabilities": test_probs}, f, indent=2)
    with open(os.path.join(outdir, "test_missed_samples.json"), "w", encoding="utf-8") as f:
        json.dump(test_missed, f, indent=2)
    if not os.path.exists(os.path.join(outdir, "history.json")):
        with open(os.path.join(outdir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    return final


# ----------------------------
# Main
# ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dataset", default="./03_9_output_asap7_aig_semantic_norm/asap7_aig_semantic_dataset.pkl.gz")
    ap.add_argument("--test-dataset", default="./03_9_output_saed14_aig_semantic_norm/saed14_aig_semantic_dataset.pkl.gz")
    ap.add_argument("--outdir", default="./06_5_output_aig_norm_semantic_as")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-5)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ht-weight", type=float, default=-1.0)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--val-split-mode", choices=["grouped_family", "stratified_sample"], default="grouped_family")
    ap.add_argument("--test-fold", type=int, default=-1, help="single dataset mode: -1 means all folds")
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--standard-vocab", default="./standard_radar_vocab.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-uda", action="store_true", help="Enable GRL-based domain-adversarial training.")
    ap.add_argument("--uda-weight", type=float, default=1.0, help="Weight for domain loss when UDA is enabled.")
    ap.add_argument("--uda-warmup-epochs", type=int, default=2, help="Run source classification only for first N epochs.")
    ap.add_argument("--target-unlabeled-dataset", default="", help="Optional unlabeled target dataset path (pkl.gz).")
    ap.add_argument("--run-uda-ablation", action="store_true", help="Run no-UDA and with-UDA on identical splits.")
    ap.add_argument("--ignore-fold-id", action="store_true", help="Ignore fold_id in dataset and use val_ratio split only.")
    ap.add_argument("--min-epochs", type=int, default=10, help="Minimum epochs before early stopping can trigger.")
    ap.add_argument("--stop-on-perfect-val", action="store_true", help="Stop early when source-val balanced_acc reaches 1.0.")
    ap.add_argument("--eval-threshold", type=float, default=0.5, help="Fixed decision threshold for evaluation.")
    ap.add_argument("--tune-threshold-on-val", action="store_true", help="Select threshold on source validation set before test.")
    ap.add_argument("--threshold-metric", choices=["f1", "balanced_acc", "recall", "precision"], default="f1")
    ap.add_argument("--threshold-min-precision", type=float, default=0.995, help="Precision floor during threshold search.")
    ap.add_argument("--threshold-grid-start", type=float, default=0.05)
    ap.add_argument("--threshold-grid-end", type=float, default=0.95)
    ap.add_argument("--threshold-grid-step", type=float, default=0.01)
    ap.add_argument(
        "--island-feat-mode",
        choices=["num_only", "func_only", "concat"],
        default="concat",
        help="island graph-level feature mode",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    if os.path.exists(args.standard_vocab):
        with open(args.standard_vocab, "r", encoding="utf-8") as f:
            args.vocab = json.load(f)
        print(f"[*] Loaded vocab from {args.standard_vocab}")
    else:
        print(f"[!] Warning: standard-vocab not found at {args.standard_vocab}")

    train_payload = load_pickle_gz(args.train_dataset)
    train_samples = list(train_payload.get("samples", []))

    explicit_target_unlabeled = None
    if args.target_unlabeled_dataset:
        tgt_payload = load_pickle_gz(args.target_unlabeled_dataset)
        explicit_target_unlabeled = list(tgt_payload.get("samples", []))
        print(f"[*] Loaded explicit target unlabeled dataset: {len(explicit_target_unlabeled)} samples")

    if not args.test_dataset:
        args.test_dataset = args.train_dataset

    test_paths = [p.strip() for p in args.test_dataset.split(",") if p.strip()]
    first_test_payload = load_pickle_gz(test_paths[0])
    first_test_samples = list(first_test_payload.get("samples", []))
    same_dataset_mode = (len(test_paths) == 1 and os.path.abspath(args.train_dataset) == os.path.abspath(test_paths[0]))

    run_settings = [("no_uda", False), ("with_uda", True)] if args.run_uda_ablation else [("with_uda" if args.use_uda else "no_uda", bool(args.use_uda))]
    print(f"[*] Run settings: {run_settings}")

    all_fold_results = []

    def run_variants(tr_samples, va_samples, te_samples, fold_out, fold_id, test_lib):
        for setting_name, setting_use_uda in run_settings:
            run_out = os.path.join(fold_out, setting_name) if len(run_settings) > 1 else fold_out
            os.makedirs(run_out, exist_ok=True)
            run_args = argparse.Namespace(**vars(args))
            run_args.use_uda = setting_use_uda
            target_pool = explicit_target_unlabeled if explicit_target_unlabeled is not None else te_samples
            metrics = train_one_run(
                tr_samples,
                va_samples,
                te_samples,
                run_out,
                device,
                run_args,
                target_unlabeled_samples=(target_pool if setting_use_uda else None),
            )
            metrics["fold"] = fold_id
            metrics["setting"] = setting_name
            metrics["use_uda"] = setting_use_uda
            metrics["test_lib"] = test_lib
            metrics["test_families"] = sorted(set(str(s.get("source_design", "unknown")) for s in te_samples))
            all_fold_results.append(metrics)
            print(f"[*] Fold {fold_id} [{setting_name}] Result: {metrics['final_test']}")

    if same_dataset_mode:
        folds_available = sorted(set(int(s.get("fold_id", -1)) for s in train_samples))
        if args.ignore_fold_id:
            folds_available = [-1]
            print("[*] ignore-fold-id enabled: forcing no-fold split mode.")
        if len(folds_available) == 1 and folds_available[0] == -1:
            run_folds = [0]
        else:
            run_folds = folds_available if args.test_fold == -1 else [args.test_fold]

        for f_id in run_folds:
            fold_out = os.path.join(args.outdir, f"fold_{f_id}")
            os.makedirs(fold_out, exist_ok=True)
            print(f"\n{'='*20} Running Fold {f_id} {'='*20}")
            if len(folds_available) == 1 and folds_available[0] == -1:
                trainval_candidates, test_samples = grouped_family_split(train_samples, 0.15, args.seed)
            else:
                test_samples = [s for s in train_samples if int(s.get("fold_id", -1)) == f_id]
                trainval_candidates = [s for s in train_samples if int(s.get("fold_id", -1)) != f_id]
            if args.val_split_mode == "grouped_family":
                tr_samples, va_samples = grouped_family_split(trainval_candidates, args.val_ratio, args.seed + f_id)
            else:
                tr_samples, va_samples = stratified_split(trainval_candidates, args.val_ratio, args.seed + f_id)
            print(f"[*] Split: Train={len(tr_samples)}, Val={len(va_samples)}, Test={len(test_samples)}")
            run_variants(tr_samples, va_samples, test_samples, fold_out, f_id, "same_dataset")
    else:
        folds_available = sorted(set(int(s.get("fold_id", -1)) for s in train_samples))
        if args.ignore_fold_id:
            folds_available = [-1]
            print("[*] ignore-fold-id enabled: forcing no-fold split mode.")
        test_lib = os.path.basename(test_paths[0]).split("_")[0]
        if len(folds_available) == 1 and folds_available[0] == -1:
            if args.val_split_mode == "grouped_family":
                tr_samples, va_samples = grouped_family_split(train_samples, args.val_ratio, args.seed)
            else:
                tr_samples, va_samples = stratified_split(train_samples, args.val_ratio, args.seed)
            fold_out = os.path.join(args.outdir, "fold_0")
            os.makedirs(fold_out, exist_ok=True)
            print(f"[*] Cross-dataset mode (no folds): Train={len(tr_samples)}, Val={len(va_samples)}, Test={len(first_test_samples)}")
            run_variants(tr_samples, va_samples, first_test_samples, fold_out, 0, test_lib)
        else:
            run_folds = folds_available if args.test_fold == -1 else [args.test_fold]
            print(f"[*] Cross-dataset K-Fold: {len(run_folds)} folds, Test={len(first_test_samples)}")
            for f_id in run_folds:
                fold_out = os.path.join(args.outdir, f"fold_{f_id}")
                os.makedirs(fold_out, exist_ok=True)
                va_samples = [s for s in train_samples if int(s.get("fold_id", -1)) == f_id]
                tr_samples = [s for s in train_samples if int(s.get("fold_id", -1)) != f_id]
                print(f"[*] Split: Train={len(tr_samples)}, Val={len(va_samples)}, Test={len(first_test_samples)}")
                run_variants(tr_samples, va_samples, first_test_samples, fold_out, f_id, test_lib)

    def aggregate_metrics(per_fold_metrics):
        if not per_fold_metrics:
            return {}
        keys = ["recall", "specificity", "precision", "f1", "balanced_acc", "acc", "tp", "tn", "fp", "fn"]
        rows = [m.get("final_test", {}) for m in per_fold_metrics]
        out = {"num_folds": len(rows)}
        for k in keys:
            vals = [float(r.get(k, 0.0)) for r in rows if k in r]
            if not vals:
                continue
            mean = sum(vals) / max(len(vals), 1)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
            out[k] = {"mean": mean, "std": math.sqrt(var), "values": vals}
        return out

    agg = aggregate_metrics(all_fold_results)
    by_setting = {}
    for setting_name, _ in run_settings:
        subset = [m for m in all_fold_results if m.get("setting") == setting_name]
        if subset:
            by_setting[setting_name] = aggregate_metrics(subset)

    summary = {
        "num_samples": len(train_samples),
        "num_folds": len(all_fold_results),
        "fold_results": all_fold_results,
        "aggregate": agg,
        "aggregate_by_setting": by_setting,
        "config": vars(args),
    }
    with open(os.path.join(args.outdir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "#" * 40)
    print("      AIG HIERARCHICAL SUMMARY")
    print("#" * 40)
    print(json.dumps(agg, indent=2))
    if by_setting:
        print("\n[*] Aggregate by setting")
        print(json.dumps(by_setting, indent=2))


if __name__ == "__main__":
    main()

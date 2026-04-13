from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

try:
    from .data import (
        LoadedLibrary,
        compute_cat_vocab_sizes,
        grouped_train_val_split,
        load_library_samples,
        ordered_directions,
        parse_direction_list,
        parse_lib_arg,
    )
    from .metrics import compute_binary_metrics
    from .models import NodeFeatureEncoder, build_model
except ImportError:
    from data import (
        LoadedLibrary,
        compute_cat_vocab_sizes,
        grouped_train_val_split,
        load_library_samples,
        ordered_directions,
        parse_direction_list,
        parse_lib_arg,
    )
    from metrics import compute_binary_metrics
    from models import NodeFeatureEncoder, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device, thr: float = 0.5):
    model.eval()
    ys = []
    ps = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch.y.view(-1).detach().cpu().numpy()
        ys.append(y)
        ps.append(prob)
    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=np.float32)
    y_prob = np.concatenate(ps, axis=0) if ps else np.zeros((0,), dtype=np.float32)
    return compute_binary_metrics(y_true=y_true, y_prob=y_prob, thr=thr), y_true, y_prob


def train_one_direction(
    model_name: str,
    src_lib: LoadedLibrary,
    tgt_lib: LoadedLibrary,
    cat_vocab_sizes: List[int],
    args,
    out_dir: Path,
) -> Dict[str, float]:
    train_samples, val_samples = grouped_train_val_split(
        src_lib.samples, val_ratio=args.val_ratio, seed=args.seed
    )
    test_samples = tgt_lib.samples

    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    encoder = NodeFeatureEncoder(
        cat_vocab_sizes=cat_vocab_sizes,
        cat_emb_dim=args.cat_emb_dim,
        num_dims=args.num_dims,
        clip_log_numeric=args.clip_log_numeric,
    )
    model = build_model(
        name=model_name,
        encoder=encoder,
        hidden=args.hidden_dim,
        layers=args.num_layers,
        dropout=args.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)

    pos_count = int(sum(int(s.y.item()) for s in train_samples))
    neg_count = max(0, len(train_samples) - pos_count)
    pos_weight = float(neg_count / max(pos_count, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_bal = -1.0
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            y = batch.y.view(-1).float()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_m, _, _ = run_eval(model, val_loader, device=device, thr=0.5)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_precision": val_m.precision,
            "val_recall": val_m.recall,
            "val_tnr": val_m.tnr,
            "val_f1": val_m.f1,
            "val_bal_acc": val_m.bal_acc,
            "val_fp": val_m.fp,
            "val_fn": val_m.fn,
        }
        history.append(rec)

        if val_m.bal_acc > best_val_bal:
            best_val_bal = val_m.bal_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_m, _, _ = run_eval(model, test_loader, device=device, thr=0.5)

    run_key = f"{model_name}__{src_lib.name}__to__{tgt_lib.name}"
    hist_path = out_dir / f"history__{run_key}.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "source": src_lib.name,
                "target": tgt_lib.name,
                "best_epoch": best_epoch,
                "best_val_bal_acc": best_val_bal,
                "history": history,
            },
            f,
            indent=2,
        )

    out = {
        "model": model_name,
        "source_lib": src_lib.name,
        "target_lib": tgt_lib.name,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_pos_count": pos_count,
        "train_neg_count": neg_count,
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_val_bal_acc": best_val_bal,
    }
    out.update(test_m.to_dict())
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="Train strong HT baselines in cross-library transfer.")
    ap.add_argument(
        "--lib",
        action="append",
        required=True,
        help="Library dataset mapping NAME=PATH (repeatable).",
    )
    ap.add_argument(
        "--directions",
        type=str,
        default="",
        help="Comma list, e.g. ASAP7->NANGATE15,NANGATE15->ASAP7. Empty = all ordered pairs.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="gnn4gate_like,trojansaint_like,fpgnn_like,raw_sage",
        help="Comma-separated model names.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-5)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--cat-emb-dim", type=int, default=16)
    ap.add_argument("--num-dims", type=int, default=8)
    ap.add_argument("--clip-log-numeric", action="store_true", default=True)
    ap.add_argument("--no-clip-log-numeric", action="store_false", dest="clip_log_numeric")
    ap.add_argument("--max-files-per-lib", type=int, default=0)
    ap.add_argument("--max-samples-per-lib", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="Force CPU.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lib_map = parse_lib_arg(args.lib)
    libs: Dict[str, LoadedLibrary] = {}
    for name, root in lib_map.items():
        loaded = load_library_samples(
            lib_name=name,
            root=root,
            max_files=args.max_files_per_lib,
            max_samples=args.max_samples_per_lib,
        )
        if not loaded.samples:
            raise RuntimeError(f"No valid samples loaded for {name} from {root}")
        libs[name] = loaded
        pos = int(sum(int(s.y.item()) for s in loaded.samples))
        print(f"[LIB] {name}: samples={len(loaded.samples)} pos={pos} neg={len(loaded.samples)-pos}")

    lib_names = list(libs.keys())
    directions = parse_direction_list(args.directions) if args.directions else ordered_directions(lib_names)
    for s, t in directions:
        if s not in libs or t not in libs:
            raise ValueError(f"Unknown direction {s}->{t}; known libs={lib_names}")

    cat_vocab_sizes = compute_cat_vocab_sizes(list(libs.values()), cat_dims=3)
    print(f"[INFO] categorical vocab sizes: {cat_vocab_sizes}")

    config = vars(args).copy()
    for k, v in list(config.items()):
        if isinstance(v, Path):
            config[k] = str(v)
    config["lib"] = args.lib
    config["directions"] = [f"{s}->{t}" for s, t in directions]
    config["timestamp"] = int(time.time())
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for model_name in model_names:
        for src_name, tgt_name in directions:
            print(f"\n[RUN] model={model_name}  {src_name}->{tgt_name}")
            row = train_one_direction(
                model_name=model_name,
                src_lib=libs[src_name],
                tgt_lib=libs[tgt_name],
                cat_vocab_sizes=cat_vocab_sizes,
                args=args,
                out_dir=args.out_dir,
            )
            rows.append(row)
            print(
                "[RESULT] "
                f"F1={row['f1']:.4f}  Prec={row['precision']:.4f}  Recall={row['recall']:.4f}  "
                f"TNR={row['tnr']:.4f}  BalAcc={row['bal_acc']:.4f}  FP={int(row['fp'])} FN={int(row['fn'])}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "per_direction_metrics.csv", index=False)

    summary_cols = ["precision", "recall", "tnr", "f1", "bal_acc", "fp", "fn"]
    summary = (
        df.groupby("model", as_index=False)[summary_cols]
        .mean(numeric_only=True)
        .sort_values("f1", ascending=False)
    )
    summary.to_csv(args.out_dir / "average_metrics_by_model.csv", index=False)

    print("\n[DONE] Wrote:")
    print(f"- {args.out_dir / 'per_direction_metrics.csv'}")
    print(f"- {args.out_dir / 'average_metrics_by_model.csv'}")


if __name__ == "__main__":
    main()

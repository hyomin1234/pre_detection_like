from __future__ import annotations

import gzip
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class LoadedLibrary:
    name: str
    root: Path
    samples: List[Data]


def _to_edge_index(raw_edge_index) -> torch.Tensor:
    """
    Normalize edge_index into shape [2, E].
    Accepts:
    - [src_list, dst_list]
    - [[u, v], [u, v], ...]
    - np/torch arrays with compatible shapes
    """
    arr = np.asarray(raw_edge_index, dtype=np.int64)
    if arr.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    if arr.ndim == 2 and arr.shape[0] == 2:
        out = arr
    elif arr.ndim == 2 and arr.shape[1] == 2:
        out = arr.T
    else:
        raise ValueError(f"Unsupported edge_index shape: {arr.shape}")
    return torch.from_numpy(out).long()


def _extract_group_name(sample: dict, fallback_design: str) -> str:
    s = sample.get("source_design")
    if isinstance(s, str) and s:
        return s
    meta = sample.get("meta", {})
    if isinstance(meta, dict):
        maybe = meta.get("source_design")
        if isinstance(maybe, str) and maybe:
            return maybe
    return fallback_design


def _iter_dataset_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.name.endswith(".pkl.gz"):
        yield root
        return
    for p in sorted(root.rglob("gnn_dataset.pkl.gz")):
        yield p


def load_library_samples(
    lib_name: str,
    root: Path,
    max_files: int = 0,
    max_samples: int = 0,
) -> LoadedLibrary:
    files = list(_iter_dataset_files(root))
    if not files:
        raise FileNotFoundError(f"No gnn_dataset.pkl.gz found under: {root}")
    if max_files > 0:
        files = files[:max_files]

    samples: List[Data] = []
    for fpath in files:
        with gzip.open(fpath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict) or "samples" not in obj:
            continue
        design_name = str(obj.get("source_design", fpath.parent.name))
        for s in obj["samples"]:
            if not isinstance(s, dict):
                continue
            x = np.asarray(s.get("x", []), dtype=np.float32)
            if x.ndim != 2 or x.shape[0] == 0:
                continue
            try:
                edge_index = _to_edge_index(s.get("edge_index", []))
            except Exception:
                continue

            if "graph_label" in s:
                y_val = int(s["graph_label"])
            elif "label" in s:
                y_val = int(s["label"])
            else:
                continue

            group = _extract_group_name(s, design_name)
            sample_id = str(s.get("sample_id", ""))
            d = Data(
                x=torch.from_numpy(x),
                edge_index=edge_index,
                y=torch.tensor([y_val], dtype=torch.float32),
            )
            d.group = group
            d.library = lib_name
            d.sample_id = sample_id
            samples.append(d)

            if max_samples > 0 and len(samples) >= max_samples:
                return LoadedLibrary(name=lib_name, root=root, samples=samples)

    return LoadedLibrary(name=lib_name, root=root, samples=samples)


def compute_cat_vocab_sizes(libs: Sequence[LoadedLibrary], cat_dims: int = 3) -> List[int]:
    max_vals = [0 for _ in range(cat_dims)]
    for lib in libs:
        for d in lib.samples:
            if d.x.shape[1] < cat_dims:
                continue
            cats = d.x[:, :cat_dims].long()
            for i in range(cat_dims):
                cur = int(cats[:, i].max().item())
                if cur > max_vals[i]:
                    max_vals[i] = cur
    return [m + 2 for m in max_vals]  # +1 for max index, +1 headroom


def grouped_train_val_split(
    samples: Sequence[Data], val_ratio: float, seed: int
) -> Tuple[List[Data], List[Data]]:
    groups = sorted({str(getattr(s, "group", "UNK")) for s in samples})
    if len(groups) < 2:
        # fallback sample-level split
        idx = list(range(len(samples)))
        rng = random.Random(seed)
        rng.shuffle(idx)
        k = max(1, int(len(idx) * val_ratio))
        val_idx = set(idx[:k])
        train = [samples[i] for i in idx if i not in val_idx]
        val = [samples[i] for i in idx if i in val_idx]
        return train, val

    rng = random.Random(seed)
    best = None
    for _ in range(64):
        shuffled = groups[:]
        rng.shuffle(shuffled)
        k = max(1, int(len(shuffled) * val_ratio))
        val_groups = set(shuffled[:k])
        train = [s for s in samples if str(getattr(s, "group", "UNK")) not in val_groups]
        val = [s for s in samples if str(getattr(s, "group", "UNK")) in val_groups]
        if len(train) == 0 or len(val) == 0:
            continue
        tr_pos = int(sum(int(s.y.item()) for s in train))
        va_pos = int(sum(int(s.y.item()) for s in val))
        if 0 < tr_pos < len(train) and 0 < va_pos < len(val):
            return train, val
        best = (train, val)
    if best is None:
        raise RuntimeError("Could not create a non-empty train/val split.")
    return best


def ordered_directions(lib_names: Sequence[str]) -> List[Tuple[str, str]]:
    dirs: List[Tuple[str, str]] = []
    for s in lib_names:
        for t in lib_names:
            if s != t:
                dirs.append((s, t))
    return dirs


def parse_direction_list(text: str) -> List[Tuple[str, str]]:
    out = []
    if not text:
        return out
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "->" not in tok:
            raise ValueError(f"Direction must be SRC->TGT, got: {tok}")
        s, t = tok.split("->", 1)
        out.append((s.strip(), t.strip()))
    return out


def parse_lib_arg(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--lib format must be NAME=PATH, got: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Empty lib name in --lib: {item}")
        out[k] = Path(v)
    return out


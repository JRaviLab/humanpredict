"""Leakage checks L1–L9 (PLAN.md section 3)."""
import numpy as np
import pandas as pd
import torch


def _to_device(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device, torch.float16)


@torch.no_grad()
def near_dup_pairs(emb: np.ndarray, threshold: float, device: str = "cuda", chunk: int = 4096) -> np.ndarray:
    """All pairs (i, j), i < j, of unit-norm rows with cosine >= threshold. Returns int64 [P, 2]."""
    x = _to_device(emb, device)
    pairs = []
    for start in range(0, len(x), chunk):
        sims = x[start:start + chunk] @ x.T
        i, j = torch.nonzero(sims >= threshold, as_tuple=True)
        i = i + start
        keep = i < j
        pairs.append(torch.stack([i[keep], j[keep]], dim=1).cpu())
    return torch.cat(pairs).numpy().astype(np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


@torch.no_grad()
def max_similarity(queries: np.ndarray, reference: np.ndarray, device: str = "cuda", chunk: int = 4096) -> np.ndarray:
    """For each query row, the highest cosine to any reference row."""
    ref = _to_device(reference, device)
    out = []
    for start in range(0, len(queries), chunk):
        q = _to_device(queries[start:start + chunk], device)
        out.append((q @ ref.T).max(dim=1).values.float().cpu())
    return torch.cat(out).numpy()


def similarity_summary(sims: np.ndarray, threshold: float) -> dict:
    qs = [0.5, 0.9, 0.99, 1.0]
    return {
        "n": int(len(sims)),
        "quantiles": {str(q): round(float(np.quantile(sims, q)), 4) for q in qs},
        f"n_ge_{threshold}": int((sims >= threshold).sum()),
    }


def check_accessions(labeled: pd.DataFrame, unlabeled_rows: pd.DataFrame) -> dict:
    """L1: accessions unique in labeled; no unlabeled accession is labeled."""
    dup = int(labeled["accession"].duplicated().sum())
    overlap = int(unlabeled_rows["accession"].isin(set(labeled["accession"])).sum())
    return {"labeled_duplicate_accessions": dup, "unlabeled_in_labeled": overlap, "passed": dup == 0 and overlap == 0}


def check_split_disjoint(labeled: pd.DataFrame, key: str, split_col: str = "split") -> dict:
    """No value of `key` appears in more than one split."""
    n_splits = labeled.groupby(key)[split_col].nunique()
    shared = int((n_splits > 1).sum())
    return {"key": key, "values_in_multiple_splits": shared, "passed": shared == 0}

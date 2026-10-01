"""Leakage-aware grouping and grouped stratified split (PLAN.md checks L3–L6, L9)."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold


def components(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component id per node for an undirected edge list [E, 2]."""
    if len(edges) == 0:
        return np.arange(n_nodes)
    graph = coo_matrix((np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
    return connected_components(graph, directed=False)[1]


class DisjointSet:
    def __init__(self, sizes: np.ndarray):
        self.parent = np.arange(len(sizes))
        self.size = sizes.astype(np.int64).copy()

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, roots: list[int]) -> None:
        root = max(roots, key=lambda r: self.size[r])
        for r in roots:
            if r != root:
                self.parent[r] = root
                self.size[root] += self.size[r]


def build_groups(row_keys: pd.DataFrame, text_component: pd.Series, link_cols: list[str], max_rows: int):
    """Group rows: near-duplicate text components are always kept together (hard
    constraint); rows sharing a value in `link_cols` (e.g. bioproject) are merged
    greedily, smallest value first, skipping any merge that would exceed `max_rows`.

    Generic texts ("BAL", "urine") shared across many projects otherwise chain
    hundreds of projects into one oversized group.

    Returns (group id per row, list of skipped link values).
    """
    comp = pd.factorize(row_keys["text_hash"].map(text_component))[0]
    ds = DisjointSet(np.bincount(comp))
    skipped = []
    for col in link_cols:
        members = row_keys.groupby(col, sort=True).indices
        for value in sorted(members, key=lambda v: (len(members[v]), v)):
            roots = list({ds.find(c) for c in np.unique(comp[members[value]])})
            if len(roots) < 2:
                continue
            if sum(ds.size[r] for r in roots) > max_rows:
                skipped.append({"column": col, "value": value, "rows": int(len(members[value]))})
                continue
            ds.union(roots)
    return np.array([ds.find(c) for c in comp]), skipped


def grouped_split(labels: np.ndarray, groups: np.ndarray, fractions: dict, seed: int,
                  n_folds: int = 20, force_train: np.ndarray | None = None) -> np.ndarray:
    """Assign 'train' / 'val' / 'test' so that no group spans two splits, stratified by label.

    Rows in `force_train` (e.g. groups too large for one fold) go to train; the
    val/test folds are sized relative to all rows.
    """
    split = np.full(len(labels), "train", dtype=object)
    free = np.ones(len(labels), bool) if force_train is None else ~force_train
    scale = len(labels) / free.sum()
    n_test = round(fractions["test"] * scale * n_folds)
    n_val = round(fractions["val"] * scale * n_folds)
    free_idx = np.flatnonzero(free)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(skf.split(np.zeros(len(free_idx)), labels[free_idx], groups[free_idx])):
        if fold < n_test:
            split[free_idx[idx]] = "test"
        elif fold < n_test + n_val:
            split[free_idx[idx]] = "val"
    return split


def group_size_summary(groups: np.ndarray, max_frac: float) -> dict:
    sizes = np.bincount(pd.factorize(groups)[0])
    largest = int(sizes.max())
    return {
        "n_groups": int(len(sizes)),
        "largest_group_rows": largest,
        "largest_group_frac": round(largest / len(groups), 4),
        "top5_sizes": sorted(sizes.tolist())[-5:],
        "passed": largest / len(groups) <= max_frac,
    }

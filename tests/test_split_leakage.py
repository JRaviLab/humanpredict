import numpy as np
import pandas as pd
import torch

from txt2onto.leakage import check_split_disjoint, max_similarity, near_dup_pairs
from txt2onto.split import build_groups, components, grouped_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_near_dup_pairs_and_max_similarity():
    x = unit([[1, 0, 0], [0.999, 0.01, 0], [0, 1, 0], [0, 0, 1]])
    pairs = near_dup_pairs(x, 0.98, DEVICE, chunk=2)
    assert pairs.tolist() == [[0, 1]]
    sims = max_similarity(x[2:], x[:2], DEVICE)
    assert np.all(sims < 0.1)


def test_components():
    comp = components(5, np.array([[0, 1], [1, 2]]))
    assert comp[0] == comp[1] == comp[2] and len({comp[3], comp[4], comp[0]}) == 3


def test_build_groups_links_projects_but_respects_cap():
    rows = pd.DataFrame({
        "text_hash": ["a", "b", "c", "d", "e", "f"],
        "bioproject": ["P1", "P1", "P2", "P2", "P3", None],
    })
    text_component = pd.Series([0, 1, 2, 3, 4, 4], index=list("abcdef"))  # e,f near-duplicates
    groups, skipped = build_groups(rows, text_component, ["bioproject"], max_rows=10)
    assert groups[0] == groups[1] and groups[2] == groups[3] and groups[4] == groups[5]
    assert groups[0] != groups[2] and not skipped
    groups, skipped = build_groups(rows, text_component, ["bioproject"], max_rows=1)
    assert len(set(groups[:4])) == 4 and len(skipped) == 2
    assert groups[4] == groups[5]  # near-duplicate text components are never broken


def test_grouped_split_keeps_groups_together_and_forces_train():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(400), 5)
    labels = (rng.random(400) < 0.6).astype(int).repeat(5)
    force = groups == 0
    split = grouped_split(labels, groups, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1, force_train=force)
    df = pd.DataFrame({"g": groups, "split": split})
    assert check_split_disjoint(df, "g")["passed"]
    assert set(split[force]) == {"train"}
    frac = pd.Series(split).value_counts(normalize=True)
    assert abs(frac["test"] - 0.15) < 0.03 and abs(frac["val"] - 0.15) < 0.03

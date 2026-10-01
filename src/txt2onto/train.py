"""Stage-1 training: StandardScaler + L2 logistic regression (PLAN.md section 4)."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCORING = {
    "f1": "f1",
    "neg_log_loss": "neg_log_loss",
    "average_precision": "average_precision",
    "precision": "precision",
    "recall": "recall",
}


def make_pipeline(stage_cfg: dict, C: float = 1.0, class_weight=None) -> Pipeline:
    # Scaler and classifier in one Pipeline, so CV folds never see statistics from held-out rows (L7).
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, l1_ratio=stage_cfg["l1_ratio"], class_weight=class_weight,
                                   solver=stage_cfg["solver"], max_iter=stage_cfg["max_iter"])),
    ])


def cv_splitter(stage_cfg: dict, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=stage_cfg["cv_folds"], shuffle=True, random_state=seed)


def grid_search(X, y, groups, stage_cfg: dict, seed: int) -> GridSearchCV:
    grid = stage_cfg["C_grid"]
    param_grid = {
        "clf__C": np.logspace(grid["logspace_start"], grid["logspace_stop"], grid["num"]),
        "clf__class_weight": stage_cfg["class_weight_grid"],
    }
    search = GridSearchCV(make_pipeline(stage_cfg), param_grid, scoring=SCORING, refit=stage_cfg["scoring"],
                          cv=cv_splitter(stage_cfg, seed), return_train_score=True, n_jobs=stage_cfg["n_jobs"])
    return search.fit(X, y, groups=groups)


def cv_results_frame(search: GridSearchCV) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    keep = {"param_clf__C": "C", "param_clf__class_weight": "class_weight", "mean_fit_time": "fit_time_s"}
    for split in ("train", "test"):
        for metric in SCORING:
            keep[f"mean_{split}_{metric}"] = f"{'cv' if split == 'test' else 'train'}_{metric}"
            keep[f"std_{split}_{metric}"] = f"{'cv' if split == 'test' else 'train'}_{metric}_std"
    res = res[list(keep)].rename(columns=keep)
    res["class_weight"] = res["class_weight"].fillna("none").astype(str)
    for col in [c for c in res.columns if "log_loss" in c and not c.endswith("_std")]:
        res[col] = -res[col]
    return res


def _learning_curve_point(X, y, train_idx, val_idx, fraction, fold, stage_cfg, params, seed):
    rng = np.random.default_rng(seed + fold)
    n = max(int(round(fraction * len(train_idx))), 50)
    sub = rng.choice(train_idx, size=min(n, len(train_idx)), replace=False)
    model = make_pipeline(stage_cfg, **params).fit(X[sub], y[sub])
    p_tr, p_va = model.predict_proba(X[sub])[:, 1], model.predict_proba(X[val_idx])[:, 1]
    return {
        "fraction": fraction, "fold": fold, "n_train": len(sub),
        "train_log_loss": log_loss(y[sub], p_tr), "cv_log_loss": log_loss(y[val_idx], p_va),
        "train_f1": f1_score(y[sub], p_tr >= 0.5), "cv_f1": f1_score(y[val_idx], p_va >= 0.5),
    }


def learning_curve(X, y, groups, stage_cfg: dict, params: dict, seed: int, fractions) -> pd.DataFrame:
    folds = list(cv_splitter(stage_cfg, seed).split(X, y, groups))
    rows = Parallel(n_jobs=stage_cfg["n_jobs"])(
        delayed(_learning_curve_point)(X, y, tr, va, f, k, stage_cfg, params, seed)
        for k, (tr, va) in enumerate(folds) for f in fractions
    )
    return pd.DataFrame(rows)


def optimization_trace(X_tr, y_tr, X_val, y_val, C: float, class_weight, max_iter: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Re-run the L-BFGS fit on the same objective sklearn minimizes, recording losses per iteration.

    Objective (sklearn's, up to a constant factor): sum_i s_i * logloss_i + ||w||^2 / (2C), with the
    intercept unpenalized and s_i the class weights. Features are standardized on train only.
    """
    scaler = StandardScaler().fit(X_tr)
    A = np.hstack([scaler.transform(X_tr).astype(np.float64), np.ones((len(X_tr), 1))])
    B = np.hstack([scaler.transform(X_val).astype(np.float64), np.ones((len(X_val), 1))])
    y = y_tr.astype(np.float64)
    if class_weight == "balanced":
        counts = np.bincount(y_tr, minlength=2)
        s = (len(y_tr) / (2 * counts))[y_tr]
    else:
        s = np.ones(len(y_tr))
    S = s.sum()

    def objective(theta):
        z = A @ theta
        loss = -(s * (y * log_expit(z) + (1 - y) * log_expit(-z))).sum()
        w = theta[:-1]
        grad = A.T @ (s * (expit(z) - y))
        grad[:-1] += w / C
        return (loss + w @ w / (2 * C)) / S, grad / S

    records = []

    def record(theta):
        p_tr, p_va = expit(A @ theta), expit(B @ theta)
        records.append({
            "iteration": len(records),
            "objective": objective(theta)[0],
            "train_log_loss": log_loss(y_tr, p_tr, labels=[0, 1]),
            "val_log_loss": log_loss(y_val, p_va, labels=[0, 1]),
            "train_f1": f1_score(y_tr, p_tr >= 0.5),
            "val_f1": f1_score(y_val, p_va >= 0.5),
        })

    theta0 = np.zeros(A.shape[1])
    record(theta0)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", callback=lambda xk: record(xk),
                      options={"maxiter": max_iter, "gtol": 1e-6})
    return pd.DataFrame(records), result.x

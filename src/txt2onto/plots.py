"""Static report figures (PNG). Colors follow a fixed entity mapping across panels."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TRAIN, HELD_OUT = BLUE, ORANGE          # train data / held-out data (CV fold or val)
HUMAN, NOT_HUMAN = BLUE, ORANGE

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.labelcolor": INK_2, "legend.fontsize": 8,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _end_label(ax, x, y, text, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=8, color=INK_2)


def _marker(ax, x, y, color, **kw):
    ax.plot(x, y, "o", ms=6.5, color=color, mec=SURFACE, mew=1.5, zorder=4, **kw)


def _vline(ax, x, text, y_frac=0.96, side="right"):
    ax.axvline(x, color=MUTED, lw=1, zorder=1)
    ax.annotate(text, (x, y_frac), xycoords=("data", "axes fraction"), xytext=(3 if side == "right" else -3, 0),
                textcoords="offset points", fontsize=8, color=INK_2, va="top", ha="left" if side == "right" else "right")


def training_figure(trace: pd.DataFrame, cv: pd.DataFrame, lc: pd.DataFrame, best: dict, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    cw = str(best["class_weight"] or "none")
    cvb = cv[cv["class_weight"] == cw].sort_values("C")

    # (a) optimization trace
    ax = axes[0, 0]
    it = trace["iteration"]
    ax.plot(it, trace["train_log_loss"], color=TRAIN, label="Train")
    ax.plot(it, trace["val_log_loss"], color=HELD_OUT, label="Validation")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("L-BFGS iteration")
    ax.set_ylabel("Log-loss")
    ax.set_title("a  Loss during optimization (selected C)")
    vmin = trace["val_log_loss"].idxmin()
    _marker(ax, it[vmin], trace["val_log_loss"][vmin], HELD_OUT)
    _end_label(ax, it[vmin], trace["val_log_loss"][vmin], f"val min {trace['val_log_loss'][vmin]:.3f} at iteration {it[vmin]}", dx=6, dy=-12)
    ax.legend(loc="upper right")

    # (b) regularization path: log-loss
    ax = axes[0, 1]
    ax.plot(cvb["C"], cvb["train_neg_log_loss"], color=TRAIN, label="Train folds")
    ax.plot(cvb["C"], cvb["cv_neg_log_loss"], color=HELD_OUT, label="Held-out folds")
    ax.fill_between(cvb["C"], cvb["cv_neg_log_loss"] - cvb["cv_neg_log_loss_std"],
                    cvb["cv_neg_log_loss"] + cvb["cv_neg_log_loss_std"], color=HELD_OUT, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse L2 strength)")
    ax.set_ylabel("Log-loss")
    ax.set_title("b  Log-loss vs regularization (grouped 5-fold CV)")
    _vline(ax, best["C"], f"selected C={best['C']:.3g}")
    ax.legend(loc="center right")

    # (c) regularization path: F1
    ax = axes[1, 0]
    ax.plot(cvb["C"], cvb["train_f1"], color=TRAIN, label="Train folds")
    ax.plot(cvb["C"], cvb["cv_f1"], color=HELD_OUT, label="Held-out folds")
    ax.fill_between(cvb["C"], cvb["cv_f1"] - cvb["cv_f1_std"], cvb["cv_f1"] + cvb["cv_f1_std"],
                    color=HELD_OUT, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse L2 strength)")
    ax.set_ylabel("F1 (human class, threshold 0.5)")
    ax.set_title("c  F1 vs regularization (selection criterion)")
    _vline(ax, best["C"], f"selected C={best['C']:.3g}", y_frac=0.12)
    ax.legend(loc="lower right")

    # (d) learning curve
    ax = axes[1, 1]
    g = lc.groupby("fraction").agg(n_train=("n_train", "mean"), tr=("train_log_loss", "mean"),
                                   cv=("cv_log_loss", "mean"), cv_sd=("cv_log_loss", "std")).reset_index()
    ax.plot(g["n_train"], g["tr"], color=TRAIN, label="Train subset")
    ax.plot(g["n_train"], g["cv"], color=HELD_OUT, label="Held-out folds")
    ax.fill_between(g["n_train"], g["cv"] - g["cv_sd"], g["cv"] + g["cv_sd"], color=HELD_OUT, alpha=0.12, lw=0)
    for col, color in (("tr", TRAIN), ("cv", HELD_OUT)):
        _marker(ax, g["n_train"].iloc[-1], g[col].iloc[-1], color)
        _end_label(ax, g["n_train"].iloc[-1], g[col].iloc[-1], f"{g[col].iloc[-1]:.3f}", dx=-30, dy=-9)
    ax.set_xscale("log")
    ax.set_xlabel("Labeled training rows")
    ax.set_ylabel("Log-loss")
    ax.set_title("d  Learning curve (selected C)")
    ax.legend(loc="upper right")

    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validation_figure(y: np.ndarray, p: np.ndarray, curve: pd.DataFrame, calib: list[dict],
                      threshold: float, taus: dict, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # (a) threshold curves
    ax = axes[0]
    for col, color, name in (("precision", BLUE, "Precision"), ("recall", ORANGE, "Recall"), ("f1", AQUA, "F1")):
        ax.plot(curve["threshold"], curve[col], color=color, label=name)
    _vline(ax, threshold, f"F1-optimal t={threshold:.2f}", y_frac=0.14)
    ax.set_xlabel("Threshold on P(human)")
    ax.set_ylabel("Validation score (human class)")
    ax.set_ylim(0, 1.02)
    ax.set_title("a  Precision / recall / F1 vs threshold")
    ax.legend(loc="lower left")

    # (b) score distributions
    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    for mask, color, name in ((y == 1, HUMAN, "Human"), (y == 0, NOT_HUMAN, "Not human")):
        ax.hist(p[mask], bins=bins, histtype="stepfilled", color=color, alpha=0.12, lw=0)
        ax.hist(p[mask], bins=bins, histtype="step", color=color, lw=1.6, label=f"{name} (n={mask.sum():,})")
    ax.set_yscale("log")
    for key, label, side in (("tau_neg", "τ⁻", "right"), ("tau_pos", "τ⁺", "left")):
        if taus.get(key) is not None:
            _vline(ax, taus[key], f"{label}={taus[key]:.2f}", y_frac=0.72, side=side)
    ax.set_xlabel("Predicted P(human)")
    ax.set_ylabel("Validation rows (log scale)")
    ax.set_title("b  Score distribution by true class")
    ax.legend(loc="upper center")

    # (c) reliability
    ax = axes[2]
    c = pd.DataFrame(calib)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
    ax.plot(c["mean_p"], c["frac_human"], color=BLUE, zorder=3)
    for _, r in c.iterrows():
        _marker(ax, r["mean_p"], r["frac_human"], BLUE)
    ax.annotate("perfect calibration", (0.62, 0.55), fontsize=8, color=MUTED, rotation=38)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted P(human) (10 bins)")
    ax.set_ylabel("Observed fraction human")
    ax.set_title("c  Calibration on validation")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, dpi=150)
    plt.close(fig)

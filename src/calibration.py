"""Probability calibration -- fit on a held-out calibration split that the
model never saw during training, per the same pattern used in the German
Credit notebook."""
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt


def calibrate(fitted_model, X_calib, y_calib, method: str = "isotonic"):
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    calibrated.fit(X_calib, y_calib)
    return calibrated


def calibration_report(y_true, probs_before, probs_after, n_bins: int = 10,
                        show: bool = True, save_path: str | None = None):
    brier_before = brier_score_loss(y_true, probs_before)
    brier_after = brier_score_loss(y_true, probs_after)
    print(f"Brier score before calibration: {brier_before:.4f}")
    print(f"Brier score after calibration:  {brier_after:.4f}")

    fig, ax = plt.subplots()
    for label, probs in [("Before", probs_before), ("After", probs_after)]:
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Calibration curve saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return {"brier_before": brier_before, "brier_after": brier_after}

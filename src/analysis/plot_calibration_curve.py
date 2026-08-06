"""Reliability diagram (calibration curve) for binary classifiers.

Standalone plotter so both the post-hoc analysis (analyze_stage4_train.py)
and the presentation-plot script (create_presentation_plots.py) can render
the same calibration view from one source.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.calibration import calibration_curve


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
    ax: Optional[Axes] = None,
    n_bins: int = 10,
) -> None:
    """Draw a reliability diagram comparing predicted probability to actual rate.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        save_path: If provided AND `ax` is None, save the standalone figure.
        ax: If provided, draw on this axes (no figure created, no save). The
            caller owns layout and saving.
        n_bins: Number of probability bins.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax.plot(prob_pred, prob_true, 'bo-', linewidth=2, markersize=8,
            label='Model Calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Model Calibration (Reliability Diagram)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

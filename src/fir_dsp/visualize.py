from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .core import build_fft_freq_grid, compute_frequency_response, linear_to_db
from .validation import ensure_1d_finite_array, ensure_non_negative_array, ensure_positive_sample_rate



def plot_frequency_response(
    fir: np.ndarray,
    fs: int = 48000,
    target_magnitude: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> None:
    fs = ensure_positive_sample_rate(fs)
    fir = ensure_1d_finite_array("fir", fir)
    n_fft = len(fir) if len(fir) % 2 == 0 else len(fir) + 1
    freqs = build_fft_freq_grid(n_fft, fs)
    actual_db = linear_to_db(compute_frequency_response(fir, n_fft))

    fig = plt.figure()
    plt.semilogx(freqs[1:], actual_db[1:], label="Actual FIR")

    if target_magnitude is not None:
        target = ensure_non_negative_array("target_magnitude", target_magnitude)
        if target.shape != actual_db.shape:
            raise ValueError(
                f"target_magnitude length {target.size} does not match FIR response length {actual_db.size}"
            )
        target_db = linear_to_db(target)
        plt.semilogx(freqs[1:], target_db[1:], label="Target")
        plt.legend()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("Frequency Response")
    plt.grid(True, which="both")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()




def plot_impulse(fir: np.ndarray, save_path: str | Path | None = None) -> None:
    fir = ensure_1d_finite_array("fir", fir)
    fig = plt.figure()
    plt.plot(fir)
    plt.title("Impulse Response")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

"""MNE Hilbert-amplitude envelopes for the spectral MEG tutorial."""

from __future__ import annotations

import mne
import numpy as np
from coco_pipe.io.structures import DataContainer
from scipy.ndimage import gaussian_filter1d

SPECTRAL_BANDS: dict[str, tuple[float, float]] = {
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 45.0),
}


def hilbert_log_amplitude_envelope(
    container: DataContainer,
    band: str = "alpha",
    *,
    crop: tuple[float, float] = (-0.2, 0.8),
    baseline: tuple[float, float] = (-0.2, 0.0),
    smoothing_s: float = 0.04,
    padding_s: float = 0.9,
) -> DataContainer:
    """Return one cropped, baseline-relative spectral-amplitude representation.

    The input comes from ``load_wakeman_henson`` and is already
    subject-wise noise whitened. MNE performs the band-pass and Hilbert
    amplitude steps on padded epochs before smoothing, log baselining, and
    cropping. The returned data remain ``obs × channel × time`` for PCA.
    """
    try:
        low, high = SPECTRAL_BANDS[band]
    except KeyError as exc:
        raise ValueError(
            f"Unknown spectral band {band!r}; choose {tuple(SPECTRAL_BANDS)}."
        ) from exc

    times = np.asarray(container.coords["time"], dtype=float)
    sfreq = 1.0 / float(np.median(np.diff(times)))
    if high >= sfreq / 2:
        raise ValueError(f"{high:g} Hz must be below Nyquist ({sfreq / 2:g} Hz).")

    if times[0] > crop[0] - padding_s + 0.51 / sfreq or times[-1] < (
        crop[1] + padding_s - 0.51 / sfreq
    ):
        raise ValueError(
            f"Need {padding_s:g} s of padding around the {crop[0]:g}..{crop[1]:g} s crop. "
            "Prepare the padded epochs once with "
            "load_wakeman_henson(..., prepare=True, spectral=True)."
        )

    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    if baseline_mask.sum() < 2:
        raise ValueError(f"Baseline {baseline} is absent from the source epochs.")

    data = np.asarray(container.X, dtype=float)
    names = [str(name) for name in container.coords["channel"]]
    info = mne.create_info(names, sfreq=sfreq, ch_types="misc")
    epochs = mne.EpochsArray(data, info, tmin=float(times[0]), verbose=False)
    epochs.filter(low, high, picks="all", verbose=False)
    epochs.apply_hilbert(picks="all", envelope=True, verbose=False)
    amplitude = epochs.get_data(copy=True)
    if smoothing_s:
        amplitude = gaussian_filter1d(
            amplitude, sigma=smoothing_s * sfreq, axis=-1, mode="reflect"
        )

    baseline_mean = amplitude[..., baseline_mask].mean(axis=-1, keepdims=True)
    floor = np.finfo(float).eps
    log_amplitude = 20 * np.log10(
        np.maximum(amplitude, floor) / np.maximum(baseline_mean, floor)
    )
    crop_mask = (times >= crop[0]) & (times <= crop[1])

    coords = {name: np.asarray(values).copy() for name, values in container.coords.items()}
    coords["time"] = times[crop_mask]
    return DataContainer(
        X=log_amplitude[..., crop_mask].astype(np.float32),
        dims=container.dims,
        coords=coords,
        y=None if container.y is None else np.asarray(container.y).copy(),
        ids=None if container.ids is None else np.asarray(container.ids).copy(),
        meta={
            **container.meta,
            "representation": "hilbert_log_amplitude_db",
            "band": band,
            "band_hz": [low, high],
            "baseline": list(baseline),
        },
    )


__all__ = ["SPECTRAL_BANDS", "hilbert_log_amplitude_envelope"]

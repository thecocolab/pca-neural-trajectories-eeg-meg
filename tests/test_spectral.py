from __future__ import annotations

import numpy as np
import pytest
from coco_pipe.io.structures import DataContainer

from pca_neural_trajectories.spectral import (
    hilbert_log_amplitude_envelope,
)


def _container(*, sfreq: float = 250.0, start: float = -1.1, stop: float = 1.7):
    times = np.arange(start, stop + 0.5 / sfreq, 1 / sfreq)
    carrier = np.sin(2 * np.pi * 10 * times)
    amplitude = np.where(times >= 0.1, 2.0, 1.0)
    data = np.stack([carrier * amplitude, 0.5 * carrier * amplitude])[None, ...]
    return DataContainer(
        X=data,
        dims=("obs", "channel", "time"),
        coords={"channel": ["MEG001", "MEG002"], "time": times, "subject": ["01"]},
        y=np.array([1]),
        ids=np.array(["trial-1"]),
        meta={"whitened": True},
    )


def test_hilbert_envelope_recovers_known_amplitude_modulation_and_metadata():
    result = hilbert_log_amplitude_envelope(_container())
    times = np.asarray(result.coords["time"])
    baseline = result.X[..., (times >= -0.18) & (times <= -0.04)].mean()
    active = result.X[..., (times >= 0.3) & (times <= 0.6)].mean()

    assert result.dims == ("obs", "channel", "time")
    assert times[0] >= -0.2 and times[-1] <= 0.8
    assert abs(baseline) < 0.5
    assert active > 4.5  # doubling amplitude is approximately +6 dB
    assert result.meta["representation"] == "hilbert_log_amplitude_db"
    assert result.meta["band_hz"] == [8.0, 12.0]


def test_hilbert_envelope_rejects_unknown_band_and_nyquist_violation():
    with pytest.raises(ValueError, match="Unknown spectral band"):
        hilbert_log_amplitude_envelope(_container(), band="theta")
    with pytest.raises(ValueError, match="Nyquist"):
        hilbert_log_amplitude_envelope(_container(sfreq=60.0), band="low_gamma")


def test_hilbert_envelope_rejects_missing_baseline():
    with pytest.raises(ValueError, match="Baseline.*absent"):
        hilbert_log_amplitude_envelope(_container(), baseline=(-2.0, -1.5))


def test_hilbert_envelope_rejects_short_unpadded_epochs_actionably():
    with pytest.raises(ValueError, match="spectral=True"):
        hilbert_log_amplitude_envelope(_container(start=-0.2, stop=0.8))


def test_hilbert_envelope_is_finite_near_zero():
    container = _container()
    container.X[:] = 0.0
    result = hilbert_log_amplitude_envelope(container)
    assert np.isfinite(result.X).all()
    assert np.allclose(result.X, 0.0)

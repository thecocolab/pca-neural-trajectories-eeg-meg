"""The two ds000117 facts that are easy to get wrong and expensive to get wrong."""

from __future__ import annotations

import pandas as pd
import pytest

from pca_neural_trajectories import wakeman_henson as wh


def _events(triggers: list[int], stim_types: list[str] | None = None) -> pd.DataFrame:
    """Minimal ds000117-shaped events table."""
    if stim_types is None:
        stim_types = [wh.LABEL_NAMES[wh.TRIGGER_INFO[t]["condition_id"]] for t in triggers]
    return pd.DataFrame(
        {
            "onset_sample": [26400 + 3300 * i for i in range(len(triggers))],
            "stim_type": stim_types,
            "trigger": triggers,
            "stim_file": [f"meg/x{i:03d}.bmp" for i in range(len(triggers))],
        }
    )


def test_map_events_assigns_conditions_and_repetitions():
    """The trigger table is the only place repetition lives — `stim_type` cannot see it."""
    mapped = wh.map_events(_events([5, 6, 7, 13, 14, 15, 17, 18, 19]))

    assert list(mapped["condition_id"]) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert list(mapped["condition"]) == ["Famous"] * 3 + ["Unfamiliar"] * 3 + ["Scrambled"] * 3
    assert list(mapped["repetition"]) == ["first", "immediate", "delayed"] * 3


def test_map_events_rejects_unknown_trigger():
    """An unmapped event has to fail, not quietly drop out of the analysis."""
    with pytest.raises(ValueError, match=r"Unknown ds000117 triggers: \[99\]"):
        wh.map_events(_events([5, 99], stim_types=["Famous", "Famous"]))


def test_fetch_names_the_calibration_files_under_derivatives(monkeypatch, tmp_path):
    """Maxwell filtering breaks without these, and they sit in the excluded tree."""
    import openneuro

    calls: list[dict] = []
    monkeypatch.setattr(openneuro, "download", lambda **kwargs: calls.append(kwargs))

    wh._fetch_wakeman_henson(tmp_path, subjects="01")

    assert calls[0]["tag"] == "1.0.6"
    assert calls[0]["include"] == [
        "sub-01/ses-meg",
        wh.CALIBRATION_FILE,
        wh.CROSSTALK_FILE,
        "sub-emptyroom",
    ]


def test_public_loader_routes_spectral_preparation(monkeypatch, tmp_path):
    calls = []
    sentinel = object()
    monkeypatch.setattr(wh, "_fetch_wakeman_henson", lambda *a, **k: calls.append(("fetch", a, k)))
    monkeypatch.setattr(
        wh, "_preprocess_subject", lambda *a, **k: calls.append(("preprocess", a, k))
    )
    def fake_load(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return sentinel

    monkeypatch.setattr(wh, "_load_wakeman_henson_container", fake_load)

    result = wh.load_wakeman_henson(
        tmp_path,
        subjects="01",
        prepare=True,
        spectral=True,
        sensor_set="sensors_occipital",
    )

    assert result is sentinel
    assert calls[0][0] == "fetch"
    spectral_config = calls[1][2]["config"]
    assert spectral_config["tmin"] == -1.2
    assert spectral_config["tmax"] == 1.8
    assert spectral_config["h_freq"] == 90.0
    assert spectral_config["baseline"] is None
    assert calls[-1][0] == "load"
    assert calls[-1][2]["sensor_set"] == "sensors_occipital"


def test_meg_sensor_sets_use_named_vectorview_regions():
    assert wh.MEG_SENSOR_SETS == {
        "all_sensors": None,
        "sensors_occipital": ("Left-occipital", "Right-occipital"),
        "sensors_temporal": ("Left-temporal", "Right-temporal"),
        "sensors_occipito_temporal": (
            "Left-occipital",
            "Right-occipital",
            "Left-temporal",
            "Right-temporal",
        ),
    }


def test_loader_rejects_unknown_sensor_set_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match="Unknown sensor_set"):
        wh._load_wakeman_henson_container(tmp_path, sensor_set="faces_roi")

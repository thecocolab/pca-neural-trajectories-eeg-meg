from __future__ import annotations

from types import SimpleNamespace

import pytest

from pca_neural_trajectories import eegbci


def test_loader_normalizes_bids_selectors(monkeypatch, tmp_path):
    dataset_kwargs = {}
    loaded = object()

    def fake_dataset(**kwargs):
        dataset_kwargs.update(kwargs)
        return SimpleNamespace(load=lambda: loaded)

    monkeypatch.setattr(eegbci, "BIDSDataset", fake_dataset)

    result = eegbci.load_eegbci_container(
        tmp_path,
        subjects=("sub-001", "002"),
        runs=(3, 12),
        conditions=(3, 6, 10),
    )

    assert result is loaded
    assert dataset_kwargs["root"] == tmp_path
    assert dataset_kwargs["subjects"] == ["001", "002"]
    assert dataset_kwargs["runs"] == ["03", "12"]
    assert dataset_kwargs["event_id"] == {
        "left_hand_exec": 3,
        "right_hand_imag": 6,
        "feet_imag": 10,
    }


def test_loader_rejects_unknown_condition(tmp_path):
    with pytest.raises(ValueError, match=r"Unknown EEGBCI conditions: \[99\]"):
        eegbci.load_eegbci_container(tmp_path, conditions=(3, 99))

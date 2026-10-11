from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from coco_pipe.io.structures import DataContainer

from pca_neural_trajectories import eegbci


def test_loader_normalizes_bids_selectors(monkeypatch, tmp_path):
    dataset_kwargs = {}
    # coco-pipe numbers events in the order it meets them, not with our codes.
    loaded = DataContainer(
        np.zeros((3, 2, 4)),
        dims=("obs", "channel", "time"),
        y=np.array([1, 2, 3]),
        meta={"event_id": {"feet_imag": 1, "left_hand_exec": 2, "right_hand_imag": 3}},
    )

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

    assert result.y.tolist() == [10, 3, 6]
    assert result.meta["event_id"] == dataset_kwargs["event_id"]
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

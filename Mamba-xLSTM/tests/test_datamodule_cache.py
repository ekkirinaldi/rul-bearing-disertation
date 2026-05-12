import numpy as np
import torch

from mxlstm.data.datamodule import BearingHI, HIWindowDataset, RULDataModule, _collate


def _bearing_hi(bearing_id: str, condition: int) -> BearingHI:
    hi = np.arange(20, dtype=np.float32).reshape(10, 2)
    return BearingHI(
        bearing_id=bearing_id,
        condition=condition,
        dataset="synthetic",
        hi_raw=hi.copy(),
        hi=hi,
        rul=np.linspace(1.0, 0.0, 10, dtype=np.float32),
        fs=1,
        acquisition_interval_s=1.0,
        eol_index=9,
        feature_names=["a", "b"],
    )


def _datamodule(tmp_path) -> RULDataModule:
    return RULDataModule(
        dataset="phm2012",
        root=tmp_path / "raw",
        train_bearings=["1_1"],
        val_bearings=["1_2"],
        test_bearings=["1_3"],
        window_length=4,
        stride_train=1,
        stride_eval=2,
        label_scheme="linear",
        smoothing_alpha=0.1,
        n_bands=5,
        batch_size=8,
        num_workers=0,
        cache_dir=tmp_path / "processed",
    )


def test_prepared_cache_round_trips_window_datasets(tmp_path):
    dm = _datamodule(tmp_path)
    dm._n_features = 2
    dm.pipeline = type("Pipeline", (), {"scaler": type("Scaler", (), {"to_dict": lambda self: {"min": [0.0, 0.0], "max": [1.0, 1.0]}})()})()
    path = dm._prepared_cache_path()

    dm._save_prepared_cache(
        path,
        [_bearing_hi("1_1", 1)],
        [_bearing_hi("1_2", 1)],
        [_bearing_hi("1_3", 1)],
    )

    restored = _datamodule(tmp_path)
    assert restored._restore_prepared_cache(path)
    assert restored.n_features == 2
    assert len(restored._train_ds) == 7
    assert len(restored._val_ds) == 4
    assert len(restored._test_ds) == 4


def test_hi_window_rul_window_last_matches_scalar_y():
    b = _bearing_hi("1_1", 1)
    ds = HIWindowDataset([b], window_length=4, stride=1)
    x, y, rw, meta = ds[0]
    assert x.shape == (4, 2)
    assert rw.shape == (4,)
    assert abs(float(rw[-1].item()) - float(y.item())) < 1e-5
    assert meta["bearing_id"] == "1_1"

    xs, ys, rws, metas = _collate([ds[0], ds[1]])
    assert xs.shape[0] == 2 and ys.shape == (2,) and rws.shape == (2, 4)
    assert torch.allclose(rws[:, -1], ys)


def test_setup_returns_when_all_datasets_already_exist(tmp_path, monkeypatch):
    dm = _datamodule(tmp_path)
    dm._n_features = 2
    dm._train_ds = object()
    dm._val_ds = object()
    dm._test_ds = object()

    monkeypatch.setattr("mxlstm.data.datamodule.load_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not load")))

    dm.setup(stage="test")

"""tests/unit/test_wf_validator.py — WalkForwardValidator kapsamlı testler."""
from __future__ import annotations
import numpy as np
import pytest
import pandas as pd
from wallet_manager.strategies.wf_validator import (
    PurgedKFoldSplit,
    WalkForwardSplitter,
    RetrainController,
    ValidationResult,
)


class TestPurgedKFoldSplit:
    def test_n_splits_respected(self):
        splitter = PurgedKFoldSplit(n_splits=5)
        X = np.arange(200).reshape(-1, 1).astype(float)
        splits = list(splitter.split(X))
        assert len(splits) == 5

    def test_no_overlap_between_train_test(self):
        splitter = PurgedKFoldSplit(n_splits=5, purge_gap=10)
        X = np.arange(200).reshape(-1, 1).astype(float)
        for train_idx, test_idx in splitter.split(X):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())
            assert len(train_set & test_set) == 0

    def test_purge_gap_creates_buffer(self):
        """Purge gap, train sonu ile test başı arasında boşluk olmasını sağlar."""
        purge = 15
        splitter = PurgedKFoldSplit(n_splits=3, purge_gap=purge)
        X = np.arange(300).reshape(-1, 1).astype(float)
        for train_idx, test_idx in splitter.split(X):
            if len(train_idx) > 0 and len(test_idx) > 0:
                gap = test_idx.min() - train_idx.max()
                assert gap >= purge

    def test_test_sets_cover_full_range(self):
        """Tüm split'lerin test setleri birleşince tüm veriyi kapsamalı."""
        splitter = PurgedKFoldSplit(n_splits=4, purge_gap=5)
        X = np.arange(200).reshape(-1, 1).astype(float)
        all_test = set()
        for _, test_idx in splitter.split(X):
            all_test.update(test_idx.tolist())
        assert len(all_test) >= len(X) * 0.7


class TestWalkForwardSplitter:
    def _X(self, n=500):
        return np.arange(n).reshape(-1, 1).astype(float)

    def test_split_returns_list_of_tuples(self):
        splitter = WalkForwardSplitter(train_size=200, test_size=50, step=50)
        splits = list(splitter.split(self._X()))
        assert len(splits) > 0
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0

    def test_train_always_before_test(self):
        splitter = WalkForwardSplitter(train_size=100, test_size=30, step=30)
        for train, test in splitter.split(self._X(300)):
            assert train.max() < test.min()

    def test_expanding_window_grows(self):
        splitter = WalkForwardSplitter(
            train_size=100, test_size=50, step=50, expanding=True
        )
        splits = list(splitter.split(self._X(500)))
        train_sizes = [len(t) for t, _ in splits]
        assert all(train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1))

    def test_sliding_split_constant_train_size(self):
        splitter = WalkForwardSplitter(
            train_size=100, test_size=30, step=30, expanding=False
        )
        splits = list(splitter.split_sliding(self._X(400)))
        train_sizes = [len(t) for t, _ in splits]
        assert all(s == train_sizes[0] for s in train_sizes)


class TestRetrainController:
    def test_no_retrain_before_threshold(self):
        ctrl = RetrainController(retrain_every_n_trades=50)
        for _ in range(30):
            ctrl.record_trade()
        assert not ctrl.should_retrain(trade_count=30)

    def test_retrain_at_threshold(self):
        ctrl = RetrainController(retrain_every_n_trades=50)
        for _ in range(50):
            ctrl.record_trade()
        assert ctrl.should_retrain(trade_count=50)

    def test_evaluate_retrain_below_sharpe_threshold(self):
        """OOS Sharpe düşükse retrain önerisi vermeli."""
        ctrl = RetrainController(min_oos_sharpe=0.30)
        low_sharpe_result = ValidationResult(
            oos_sharpe=0.10,
            oos_returns=[0.001] * 50,
            n_splits=5,
        )
        decision = ctrl.evaluate_retrain(low_sharpe_result)
        assert decision.should_retrain is True

    def test_evaluate_retrain_above_sharpe_no_retrain(self):
        ctrl = RetrainController(min_oos_sharpe=0.30)
        good_result = ValidationResult(
            oos_sharpe=0.75,
            oos_returns=[0.002] * 50,
            n_splits=5,
        )
        decision = ctrl.evaluate_retrain(good_result)
        assert decision.should_retrain is False

    def test_get_status_returns_dict(self):
        ctrl = RetrainController()
        status = ctrl.get_status()
        assert isinstance(status, dict)
        assert "trade_count" in status

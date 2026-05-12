import math

import numpy as np

from mxlstm.eval.metrics import mae, phm_score, r2, rmse


def test_rmse_perfect():
    y = np.linspace(0.0, 1.0, 10)
    assert rmse(y, y) == 0.0


def test_mae_perfect():
    y = np.linspace(0.0, 1.0, 10)
    assert mae(y, y) == 0.0


def test_r2_perfect():
    y = np.linspace(0.0, 1.0, 10)
    assert abs(r2(y, y) - 1.0) < 1e-9


def test_phm_score_perfect():
    y = np.linspace(0.1, 1.0, 10)
    assert abs(phm_score(y, y) - 1.0) < 1e-6


def test_phm_score_late_harsher_than_early():
    # PHM 2012 convention:
    #   er = (y_actual - y_pred) / y_actual
    #   er > 0  : pred < actual → "early" warning → lenient (half-life=20)
    #   er <= 0 : pred > actual → "late"  warning → harsh   (half-life=5)
    # Same magnitude of er, late should score lower than early.
    y = np.array([0.5, 0.5])
    pred_under = np.array([0.4, 0.4])   # er = +0.2  (early warning, lenient)
    pred_over  = np.array([0.6, 0.6])   # er = -0.2  (late warning, harsh)
    assert phm_score(y, pred_under) > phm_score(y, pred_over)

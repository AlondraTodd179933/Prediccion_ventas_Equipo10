import numpy as np

def test_predictions_not_nan():
    preds = np.array([1.0, 2.0, 3.0])
    assert not np.isnan(preds).any()
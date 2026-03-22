import pandas as pd
from sklearn.linear_model import Ridge

def test_model_trains():
    X = pd.DataFrame({'x': [1, 2, 3]})
    y = [1, 2, 3]
    model = Ridge()
    model.fit(X, y)
    assert model is not None

def test_prediction_shape():
    X = pd.DataFrame({'x': [1, 2, 3]})
    y = [1, 2, 3]
    model = Ridge().fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
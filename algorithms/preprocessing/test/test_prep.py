import pandas as pd

def test_dataframe_not_empty():
    df = pd.DataFrame({'sales': [10, 20, 30]})
    assert df.shape[0] > 0

def test_no_nulls_after_drop():
    df = pd.DataFrame({'sales': [10, None, 30]})
    df_clean = df.dropna()
    assert df_clean.isnull().sum().sum() == 0
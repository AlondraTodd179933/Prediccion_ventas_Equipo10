"""
Reusable feature engineering utilities.

Keeps train/inference scripts small and avoids duplicated code.
"""

from __future__ import annotations

import pandas as pd


def make_lag_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features used by the model.

    Requires columns: shop_id, item_id, date_block_num, item_cnt_month.

    Parameters
    ----------
    df_in : pd.DataFrame
        Monthly dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with lag_1, lag_2, lag_3, lag_mean_1_2 and NaNs removed.
    """
    required_base = {"shop_id", "item_id", "date_block_num", "item_cnt_month"}
    if not required_base.issubset(df_in.columns):
        return df_in.copy()

    df = df_in.sort_values(["shop_id", "item_id", "date_block_num"]).copy()
    df["lag_1"] = df.groupby(["shop_id", "item_id"])["item_cnt_month"].shift(1)
    df["lag_2"] = df.groupby(["shop_id", "item_id"])["item_cnt_month"].shift(2)
    df["lag_3"] = df.groupby(["shop_id", "item_id"])["item_cnt_month"].shift(3)
    df["lag_mean_1_2"] = df[["lag_1", "lag_2"]].mean(axis=1)

    return df.dropna().reset_index(drop=True)

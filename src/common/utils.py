import pandas as pd
from sklearn import metrics

def get_date_for_timestamp_file():
    def zfill_elements(arr):
        return list(map(lambda a: a.zfill(2),map(str, arr)))

    dt = pd.to_datetime('now').tz_localize('UTC').tz_convert("America/Argentina/Buenos_Aires")
    dt = zfill_elements([dt.year,dt.month,dt.day, dt.hour, dt.minute,dt.second])
    dt = "{}-{}-{}_{}:{}:{}".format(*dt)
    return dt


def get_metrics(y_real, y_pred):
    return {
        # Best possible score is 1.0, lower values are worse.
        "explained_variance_score": metrics.explained_variance_score(y_real, y_pred),
        # The best value is 0.0. it is robust to outliers
        "mae": metrics.mean_absolute_error(y_real, y_pred),
        # computes the coefficient of determination, Best possible score is 1.0 and it can be negative
        "r2": metrics.r2_score(y_real, y_pred), 
    }
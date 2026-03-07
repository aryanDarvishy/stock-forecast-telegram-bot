import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def fit_arima(train: pd.Series, test: pd.Series, order=(5, 1, 0)) -> np.ndarray:
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast.values

def fit_linear_model(train: pd.Series, test: pd.Series) -> np.ndarray:
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train.values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def evaluate(true, pred):
    return {
        "MAE": mean_absolute_error(true, pred),
        "MSE": mean_squared_error(true, pred),
        "R²": r2_score(true, pred)
    }
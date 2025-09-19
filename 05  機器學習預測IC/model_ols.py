import numpy as np
from sklearn.linear_model import LinearRegression

def model_OLS(X, Y, X_test):
    reg = LinearRegression().fit(X, Y.ravel())
    return float(reg.predict(X_test)[0])

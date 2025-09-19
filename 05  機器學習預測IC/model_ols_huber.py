from sklearn.linear_model import HuberRegressor

def model_OLS_Huber(X, Y, X_test):
    m = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True, max_iter=1000)
    m.fit(X, Y.ravel())
    return float(m.predict(X_test)[0])

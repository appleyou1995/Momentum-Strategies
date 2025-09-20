from sklearn.linear_model import HuberRegressor

def model_OLS_Huber(X, Y, X_test):
    m = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True, max_iter=1000)
    m.fit(X, Y.ravel())
    return float(m.predict(X_test)[0])


# epsilon       → 控制對異常值的敏感度，小一點更 robust，大一點更像 OLS。
# alpha         → 控制正則化強度，避免係數過大。
# fit_intercept → 是否估計截距，通常保留 True。
# max_iter      → 確保收斂，數據複雜時加大。
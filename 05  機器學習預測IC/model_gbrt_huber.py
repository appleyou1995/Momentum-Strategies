from sklearn.ensemble import GradientBoostingRegressor

def model_GBRT_Huber(X, Y, X_test, seed=999):
    m = GradientBoostingRegressor(
        loss='huber', max_depth=2,
        learning_rate=0.1, n_estimators=100,
        random_state=seed
    ).fit(X, Y.ravel())
    return float(m.predict(X_test)[0])

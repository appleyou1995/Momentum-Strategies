from sklearn.ensemble import RandomForestRegressor

def model_RF(X, Y, X_test, seed=999):
    m = RandomForestRegressor(
        max_depth=3, n_estimators=100,
        random_state=seed, n_jobs=-1
    ).fit(X, Y.ravel())
    return float(m.predict(X_test)[0])
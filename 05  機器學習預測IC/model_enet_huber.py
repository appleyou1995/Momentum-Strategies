from sklearn.linear_model import SGDRegressor

def model_ENet_Huber(X, Y, X_test, seed=999):
    m = SGDRegressor(
        loss='huber', penalty='elasticnet',
        alpha=3e-3, l1_ratio=0.9, epsilon=0.05,
        max_iter=3000, tol=1e-4, random_state=seed
    )
    m.fit(X, Y.ravel())
    return float(m.predict(X_test)[0])


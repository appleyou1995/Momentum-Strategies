from sklearn.cross_decomposition import PLSRegression

def model_PLS(X, Y, X_test, K=3):
    k_use = int(min(K, X.shape[1], max(1, X.shape[0]-1)))
    m = PLSRegression(n_components=k_use).fit(X, Y.ravel())
    return float(m.predict(X_test)[0])

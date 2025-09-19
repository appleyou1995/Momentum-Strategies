from sklearn.decomposition import PCA
from sklearn.pipeline      import make_pipeline
from sklearn.linear_model  import LinearRegression

def model_PCR(X, Y, X_test, K_max=20):
    K = int(min(K_max, X.shape[1], max(1, X.shape[0]-1)))
    pipe = make_pipeline(PCA(n_components=K), LinearRegression())
    pipe.fit(X, Y.ravel())
    return float(pipe.predict(X_test)[0])

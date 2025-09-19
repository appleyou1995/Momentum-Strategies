import numpy as np
import cvxpy as cp
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error

def choose_solver():
    # 選擇已安裝的解器，依偏好順序回傳
    installed = set(cp.installed_solvers())
    for s in ("OSQP", "SCS", "ECOS"):
        if s in installed:
            return s
    return None

# GLM + H (Huber) with Group-Lasso via CVXPY
# - group = 同一個「原始特徵」展開後的所有樣條基底

def spline_design_and_groups(X_raw, degree=3, n_knots=5):
    # 將原始 X 做 Spline 展開，並回傳 groups 向量（以原始特徵為組）。
    spl = SplineTransformer(degree=degree, n_knots=n_knots, include_bias=False)
    Xs = spl.fit_transform(X_raw)   # (n, p_spline)
    n_feat = X_raw.shape[1]
    width = Xs.shape[1] // n_feat   # 每個特徵展開的基底數
    groups = np.repeat(np.arange(n_feat), width)
    return Xs.astype(np.float32), groups.astype(int), spl

def fit_glm_group_huber_predict(X_tr, y_tr, X_va, y_va, X_te,
                                n_knots_grid=(5,), degree=3,
                                lam_grid=(1e-3,), delta=1.35):
    # 以驗證集挑選 (n_knots, lambda)，精確的 Huber + Group Lasso。
    # 回傳 X_te 的單點預測值（float）。
    best_loss, best_beta, best_spl = np.inf, None, None
    solver = choose_solver()

    for n_knots in n_knots_grid:
        # Spline 展開（訓練用的 spl 也拿來轉換 valid/test）
        Xs_tr, groups, spl = spline_design_and_groups(X_tr, degree=degree, n_knots=n_knots)
        Xs_va = spl.transform(X_va).astype(np.float32)
        Xs_te = spl.transform(X_te).astype(np.float32)

        n, p = Xs_tr.shape
        # 為了公平，對每組標準化（避免某組變數尺度較大吞掉懲罰）
        col_std = Xs_tr.std(axis=0, ddof=0)
        col_std[col_std == 0] = 1.0
        Xs_tr_std = Xs_tr / col_std
        Xs_va_std = Xs_va / col_std
        Xs_te_std = Xs_te / col_std

        # 權重 w_g = sqrt(p_g)（避免大組吃虧）
        w_g = {}
        for g in np.unique(groups):
            p_g = (groups == g).sum()
            w_g[g] = np.sqrt(p_g)

        for lam in lam_grid:
            beta = cp.Variable(p)
            # Huber 損失
            loss = cp.sum(cp.huber(y_tr - Xs_tr_std @ beta, delta))
            # Group-Lasso 懲罰
            pen = 0
            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                pen += w_g[g] * cp.norm2(beta[idx])
            obj = cp.Minimize(loss + lam * pen)
            prob = cp.Problem(obj)
            try:
                prob.solve(solver=solver, verbose=False)
            except Exception:
                # 換 solver 再試（以防某些環境某解器不可用）
                for s in ["SCS", "ECOS", "OSQP"]:
                    try:
                        prob.solve(solver=s, verbose=False)
                        break
                    except Exception:
                        continue

            if beta.value is None:
                continue

            # 驗證集損失（用 MSE 或 Huber 皆可；用 MSE 比較直觀）
            y_hat_va = (Xs_va_std @ beta.value).ravel()
            val_loss = mean_squared_error(y_va, y_hat_va)

            if val_loss < best_loss:
                best_loss = val_loss
                best_beta = beta.value.copy()
                best_spl  = spl
                best_std  = col_std.copy()

    # 以最佳參數做測試點預測
    Xs_te = best_spl.transform(X_te).astype(np.float32)
    Xs_te_std = Xs_te / best_std
    y_hat_te = float((Xs_te_std @ best_beta).ravel()[0])
    return y_hat_te

def model_GLM_Huber(X, Y, X_valid, Y_valid, X_test):
    return float(fit_glm_group_huber_predict(
        X_tr=X, y_tr=Y.ravel(),
        X_va=X_valid, y_va=Y_valid.ravel(),
        X_te=X_test,
        n_knots_grid=(5,), degree=3,
        lam_grid=(1e-3,), delta=1.35
    ))

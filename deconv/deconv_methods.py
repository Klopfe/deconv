import numpy as np
import pandas as pd
from sparse_ho.algo.forward import compute_beta
from sparse_ho import ImplicitForward
from sparse_ho import grad_search
from sparse_ho.models import SimplexSVR, NNSVR
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import GradientDescent
from sparse_ho.utils import Monitor
from cvxopt import matrix
from cvxopt import solvers
from sklearn.svm import NuSVR
from sklearn.model_selection import KFold
# from sklearn.linear_model import LinearRegression


def Linear_regression_constrained(X, y, intercept=False, sum_to_one=True):

    if intercept:
        X = np.hstack((
            np.reshape(np.repeat(1.0, X.shape[0]), (X.shape[0], 1)), X))
    l, n = X.shape

    # Quadratic term matrix
    Q = np.dot(np.transpose(X), X)
    Q = matrix(Q)

    # Linear term vector

    L = np.dot(np.transpose(X), y)
    L = matrix(-L)

    # Matrix of constraints (inequality)
    if intercept:
        G = np.zeros((n-1, n))
        G[:, np.arange(1, n)] = -np.eye(n-1, n-1)
        A = np.zeros(n)
        A[np.arange(1, n)] = 1.0
        h = np.repeat(0.0, n-1)

    else:
        G = -np.eye(n, n)
        A = np.repeat(1.0, n)
        h = np.repeat(0.0, n)

    G = matrix(G)
    # Matrix of constraints (equality)
    A = matrix(A, (1, n))

    # vector of inequality constraints

    h = matrix(h)
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    if sum_to_one:
        sol = solvers.qp(Q, L, G, h, A, b)
    else:
        sol = solvers.qp(Q, L, G, h)

    if intercept:
        solution = sol['x'][1:(n+1)]
    else:
        solution = sol['x'][:n]
    result = np.asarray(solution).flatten()

    return result


def SOLS(signature, data):
    n_try = data.shape[1]
    estimated_proportions = []

    for i in range(n_try):
        y = data[:, i]
        X = signature.copy()
        sol = Linear_regression_constrained(X, y, intercept=True)
        estimated_proportions.append(sol)

    return np.array(estimated_proportions)


def quantile_normalize(x):
    """
    input: dataframe with numerical columns
    output: dataframe with quantile normalized values
    """
    df = pd.DataFrame(x)
    df_sorted = pd.DataFrame(np.sort(df.values,
                                     axis=0),
                             index=df.index,
                             columns=df.columns)
    df_mean = df_sorted.mean(axis=1)
    df_mean.index = np.arange(1, len(df_mean) + 1)
    df_qn = df.rank(method="min").stack().astype(int).map(df_mean).unstack()
    return(np.array(df_qn))


def single_cibersort(signature, y, tol, max_iter):
    y = (y - np.mean(y)) / np.std(y)
    nus = [0.25, 0.5, 0.75]
    table_coefs = []
    rmses = []
    for nu in nus:
        model = NuSVR(
            nu=nu, C=1.0, kernel='linear', tol=tol, max_iter=max_iter)
        model.fit(signature, y)
        coefs = model.coef_[0, :].copy()
        if np.any(coefs < 0):
            if np.all(coefs < 0):
                coefs[0] = 1.0
            coefs[coefs < 0] = 0.0
        coefs = coefs / np.sum(coefs)
        table_coefs.append(coefs)
        rmses.append(np.sqrt((np.mean((signature @ coefs - y) ** 2))))

    best_index = np.argmin(np.array(rmses))

    return table_coefs[best_index]


def cibersort(signature, data):
    n_try = data.shape[-1]
    tol = 1e-4
    max_iter = 50_000
    estimated_proportions = []
    # data = quantile_normalize(data)
    signature = (signature - np.mean(signature)) / np.std(signature)
    for i in range(n_try):
        coefs_temp = single_cibersort(signature, data[:, i], tol, max_iter)
        estimated_proportions.append(coefs_temp)

    return np.array(estimated_proportions)


def deconv_ssvr(signature, data, sum_to_one=True):
    n_try = data.shape[1]
    tol = 1e-5
    max_iter = 50_000
    estimated_proportions = []
    X = signature.copy()
    mat = np.concatenate([X, data], axis=1)
    row_min = np.min(mat, axis=1)
    row_max = np.max(mat, axis=1)

    X = (X.T - row_min) / (row_max - row_min)
    X = X.T
    for i in range(n_try):
        y = data[:, i]
        y = (y - row_min) / (row_max - row_min)
        # find good initialization for epsilon
        coefs_ols = Linear_regression_constrained(X, y, sum_to_one=sum_to_one)
        res = y - X @ coefs_ols
        epsilon0 = np.quantile(np.abs(res), 0.25)
        # estimated_sigma = np.std(res[idx_val])
        if sum_to_one:
            model = SimplexSVR()
        else:
            model = NNSVR()
        criterion = HeldOutMSE(None, None)
        cross_val = CrossVal(criterion, cv=KFold(n_splits=min(5, len(y))))
        monitor = Monitor()
        algo = ImplicitForward(
            n_iter_jac=1_000, tol_jac=1e-3, max_iter=max_iter)
        optimizer = GradientDescent(
            n_outer=10, tol=0.001, p_grad_norm=0.5, verbose=True)
        C0 = 0.1
        grad_search(
            algo, cross_val, model, optimizer, X, y, np.array([C0, epsilon0]),
            monitor)

        supp, dense, _ = compute_beta(
            X, y, np.log(monitor.alphas[-1]),
            tol=tol, model=model, max_iter=max_iter)
        print(epsilon0, monitor.alphas[-1])
        proportions = np.zeros(X.shape[1])
        proportions[supp] = dense
        if not sum_to_one:
            if np.sum(proportions) != 0:
                proportions /= np.sum(proportions)
            else:
                proportions = np.repeat(1 / len(proportions), len(proportions))
        estimated_proportions.append(proportions)
    estimated_proportions = np.array(estimated_proportions)
    return estimated_proportions

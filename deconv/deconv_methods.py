import numpy as np
import pandas as pd
from sparse_ho.algo.forward import compute_beta
from sparse_ho import ImplicitForward
from sparse_ho import grad_search
from sparse_ho.models import SimplexSVR
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import GradientDescent
from sparse_ho.utils import Monitor
from cvxopt import matrix
from cvxopt import solvers
from sklearn.model_selection import KFold
from sklearn.svm import NuSVR


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
        sol = Linear_regression_constrained(X, y, intercept=False)
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
    max_iter = 50000
    estimated_proportions = []
    # data = quantile_normalize(data)
    signature = (signature - np.mean(signature)) / np.std(signature)
    for i in range(n_try):
        coefs_temp = single_cibersort(signature, data[:, i], tol, max_iter)
        estimated_proportions.append(coefs_temp)

    return np.array(estimated_proportions)


def deconv_ssvr(signature, data):
    kf = KFold(n_splits=5)
    n_try = data.shape[1]
    tol = 1e-5
    max_iter = 50000
    estimated_proportions = []
    X = signature.copy()
    concat_mat = np.concatenate((X, data), axis=1)
    row_min = np.min(concat_mat, axis=1)
    row_max = np.max(concat_mat, axis=1)
    X = (X.T - row_min) / (row_max - row_min)
    X = X.T

    for i in range(n_try):
        y = data[:, i]
        y = (y - row_min) / (row_max - row_min)
        # find good initialization for epsilon
        coefs_SOLS = Linear_regression_constrained(
            X, y, intercept=False, sum_to_one=True)
        absolute_residuals = np.abs(y - X @ coefs_SOLS.T)
        init_epsilon = np.quantile(absolute_residuals, 0.1)
        model = SimplexSVR()

        # criterion = HeldOutMSE(
        # np.arange(0, n_samples), np.arange(0, n_samples))
        criterion = HeldOutMSE(None, None)
        cross_val = CrossVal(criterion, cv=kf)
        monitor = Monitor()
        algo = ImplicitForward(
            n_iter_jac=1000, tol_jac=1e-5, max_iter=max_iter)
        # algo = Forward()
        optimizer = GradientDescent(
            n_outer=10, tol=tol, p_grad_norm=0.8, verbose=True)

        C0 = 0.1
        epsilon0 = init_epsilon
        grad_search(
            algo, cross_val, model, optimizer, X, y, np.array([C0, epsilon0]),
            monitor)

        supp, dense, _ = compute_beta(
            X, y, np.log(monitor.alphas[-1]),
            tol=tol, model=model, max_iter=max_iter)
        proportions = np.zeros(X.shape[1])
        proportions[supp] = dense

        estimated_proportions.append(proportions)

    estimated_proportions = np.array(estimated_proportions)
    return estimated_proportions

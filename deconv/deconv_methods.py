import numpy as np
import pandas as pd
from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho import ImplicitForward
from sparse_ho import grad_search, hyperopt_wrapper
from sparse_ho.models import SimplexSVR
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import LineSearchWolfe, GradientDescent
from sparse_ho.utils import Monitor
from sparse_ho.grid_search import grid_search
from sparse_ho.tests.cvxpylayer import ssvr_cvxpy
from sklearn.preprocessing import minmax_scale
from sklearn.svm import NuSVR 
import cvxpy as cp
from cvxopt import matrix
from cvxopt import solvers

def Linear_regression_constrained(X,y, intercept=False):
    
    if intercept:
        X=np.hstack((np.reshape(np.repeat(1.0,X.shape[0]),(X.shape[0],1)),X))
    l, n=X.shape
    
    #quadratic term matrix
    Q=np.dot(np.transpose(X),X)
    Q=matrix(Q)
    
    #Linear term vector

    L=np.dot(np.transpose(X),y)
    L=matrix(-L)
      
    #Matrix of constraints (inequality)
    if intercept:
        G = np.zeros((n-1, n))
        G[:, np.arange(1, n)] = -np.eye(n-1,n-1)
        A = np.zeros(n)
        A[np.arange(1, n)] = 1.0
        h=np.repeat(0.0,n-1)

    else:
        G=-np.eye(n,n)
        A=np.repeat(1.0,(n))
        h=np.repeat(0.0,n)


    G=matrix(G)
    #Matrix of constraints (equality)
    A=matrix(A,(1,n))


    #vector of inequality constraints

    h=matrix(h)
    b=matrix(1.0)

    solvers.options['show_progress'] = False

    sol= solvers.qp(Q,L,G,h,A,b)
    solution=sol['x'][:n]
    return(np.asarray(solution).flatten())

def SOLS(signature, data):
    n_try = data.shape[1]
    estimated_proportions = []
    # signature = (signature - np.mean(signature)) / np.std(signature)
    for i in range(n_try):
        y = data[:, i]
        # y = (y - np.mean(y)) / np.std(y)
        sol = Linear_regression_constrained(signature, y, intercept=False)
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
    df_qn =df.rank(method="min").stack().astype(int).map(df_mean).unstack()
    return(np.array(df_qn))


def single_cibersort(signature, y, tol, max_iter):
    y = (y - np.mean(y)) / np.std(y)
    nus = [0.25, 0.5, 0.75]
    table_coefs = []
    rmses = []
    for nu in nus:
        model = NuSVR(nu=nu, C=1.0, kernel='linear', tol=tol, max_iter=max_iter)
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
    tol=1e-3
    max_iter = 50000
    estimated_proportions = []
    # data = quantile_normalize(data)
    signature = (signature - np.mean(signature)) / np.std(signature)
    for i in range(n_try):
        coefs_temp = single_cibersort(signature, data[:, i], tol, max_iter)
        estimated_proportions.append(coefs_temp)
    
    return np.array(estimated_proportions)

def deconv_ssvr(signature, data):
    n_try = data.shape[1]
    tol = 1e-3
    max_iter = 10000
    n_samples = data.shape[0]
    idx_train = np.arange(0, n_samples)
    idx_val = np.arange(0, n_samples)
    X = signature.copy()
    X = (X - np.mean(X)) / np.std(X)
    estimated_proportions = []
    for i in range(n_try):
        y = data[:, i]
        y = (y - np.mean(signature)) / np.std(signature)
        model = SimplexSVR(max_iter=max_iter)
        criterion = HeldOutMSE(idx_train, idx_val)
        monitor = Monitor()
        algo = ImplicitForward(n_iter_jac=500, tol_jac=1e-3, max_iter=max_iter)
        optimizer = GradientDescent(
            n_outer=10, tol=tol, p_grad0=0.5, verbose=True)
        
        C0 = 0.1
        epsilon0 = 1.0 / np.std(signature)
        grad_search(
            algo, criterion, model, optimizer, X, y, np.array([C0, epsilon0]),
            monitor)
        supp, dense, jac1 = get_beta_jac_iterdiff(
            X, y, np.log(monitor.alphas[np.argmin(np.array(monitor.objs))]),
            tol=tol, model=model, max_iter=max_iter)
        proportions = np.zeros(X.shape[1])
        proportions[supp] = dense

        estimated_proportions.append(proportions)

    estimated_proportions =  np.array(estimated_proportions)
    return estimated_proportions
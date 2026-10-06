# 1-MEO-6
import numpy as np
from scipy.optimize import minimize

# max f  ->  min(-f)
def f(x):
    return -(5*np.log(x[0]) + 3*np.log(x[1]) - 2*x[1] - 2*x[0]*x[1])

def grad_f(x):
    f_x0 = -(5/x[0] - 2*x[1])
    f_x1 = -(3/x[1] - 2 - 2*x[0])
    return np.array([f_x0, f_x1])

x0 = np.array([1, 1])

res = minimize(f, x0, method='BFGS', jac=grad_f, tol=1e-8)
res

# %% Ячейка 0. Импорт
import numpy as np
from scipy.optimize import minimize


# %% Задача 1. min(5x² + y² − 4xy − 2x − 4y)  s.t.  2x + y = 5
# Целевая функция f(x) и её градиент
def f(x):
    return 5*x[0]**2 + x[1]**2 - 4*x[0]*x[1] - 2*x[0] - 4*x[1]
def grad_f(x):
    return np.array([10*x[0] - 4*x[1] - 2, 2*x[1] - 4*x[0] - 4])

# ограничения равенства задаём в виде словаря
constr = {'type': 'eq',
          'fun': lambda x: 2*x[0] + x[1] - 5,
          'jac': lambda x: np.array([2, 1]) }

# Начальное приближение
x0 = np.array([1, 1])

res = minimize(f, x0, method='SLSQP', jac=grad_f, constraints=[constr], tol=1e-9)

print(res)

# %% Ответ с округлением (один знак)
res.x.round(1)
# x = 1.0, y = 3.0


# %% Задача 2. min(5x² + 2y² − 6xy − 4x − 4y)  s.t.  2x + y = 6
# Целевая функция f(x) и её градиент
def f(x):
    return 5*x[0]**2 + 2*x[1]**2 - 6*x[0]*x[1] - 4*x[0] - 4*x[1]
def grad_f(x):
    return np.array([10*x[0] - 6*x[1] - 4, 4*x[1] - 6*x[0] - 4])

# ограничения равенства задаём в виде словаря
constr = {'type': 'eq',
          'fun': lambda x: 2*x[0] + x[1] - 6,
          'jac': lambda x: np.array([2, 1]) }

# Начальное приближение
x0 = np.array([1, 1])

res = minimize(f, x0, method='SLSQP', jac=grad_f, constraints=[constr], tol=1e-9)

print(res)

# %% Ответ с округлением (один знак)
res.x.round(1)
# x = 1.6, y = 2.8


# %% Задача 3. max(15x + 10y + 8z − 6x² − 9y² − 5z² − 2xy + 4xz + 2yz), без ограничений
# max F = min(−F), поэтому f = −F
def f(x):
    return (-15*x[0] - 10*x[1] - 8*x[2] + 6*x[0]**2 + 9*x[1]**2 + 5*x[2]**2
            + 2*x[0]*x[1] - 4*x[0]*x[2] - 2*x[1]*x[2])
def grad_f(x):
    return np.array([-15 + 12*x[0] + 2*x[1] - 4*x[2],
                     -10 + 2*x[0] + 18*x[1] - 2*x[2],
                     -8 - 4*x[0] - 2*x[1] + 10*x[2]])

# Начальное приближение
x0 = np.array([1, 1, 1])

# ограничений нет, поэтому constraints не задаём
res = minimize(f, x0, method='SLSQP', jac=grad_f, tol=1e-9)

print(res)

# %% Ответ с округлением (три знака)
res.x.round(3)
# x = 1.687, y = 0.544, z = 1.584


# %% Задача 4. Локальный минимум f(x, y) = 4x + 5xy − 5 ln x − 7 ln y
# Область определения: x > 0, y > 0
# Необходимое условие экстремума grad f = 0:
#   f'_x = 4 + 5y − 5/x = 0
#   f'_y = 5x − 7/y = 0
import sympy as sp

X, Y = sp.symbols('x y')
F = 4*X + 5*X*Y - 5*sp.log(X) - 7*sp.log(Y)

sol = sp.solve([sp.diff(F, X), sp.diff(F, Y)], [X, Y], dict=True)
print(sol)
# Получаем x = −1/2, y = −14/5: точка вне области x > 0, y > 0.
# Стационарных точек нет → локального минимума нет → ответ: x = 0, y = 0


# %% Задача 5. max(x + y + z)  s.t.  x − y + z = 0,  x² + y² + 2z² = 9
# max F = min(−F), поэтому f = −F
def f(x):
    return -x[0] - x[1] - x[2]
def grad_f(x):
    return np.array([-1, -1, -1])

# ограничения равенства задаём в виде словаря
constr1 = {'type': 'eq',
           'fun': lambda x: x[0] - x[1] + x[2],
           'jac': lambda x: np.array([1, -1, 1]) }

constr2 = {'type': 'eq',
           'fun': lambda x: x[0]**2 + x[1]**2 + 2*x[2]**2 - 9,
           'jac': lambda x: np.array([2*x[0], 2*x[1], 4*x[2]]) }

# Начальное приближение
x0 = np.array([1, 1, 1])

res = minimize(f, x0, method='SLSQP', jac=grad_f, constraints=[constr1, constr2], tol=1e-9)

print(res)

# %% Ответ с округлением (два знака)
res.x.round(2)
# x = 1.55, y = 2.32, z = 0.77


# %% Задача 6. max(6x + 12y − 5x² − y² + 4xy)  s.t.  2x + y = 15
# max F = min(−F), поэтому f = −F
def f(x):
    return -6*x[0] - 12*x[1] + 5*x[0]**2 + x[1]**2 - 4*x[0]*x[1]
def grad_f(x):
    return np.array([-6 + 10*x[0] - 4*x[1], -12 + 2*x[1] - 4*x[0]])

# ограничения равенства задаём в виде словаря
constr = {'type': 'eq',
          'fun': lambda x: 2*x[0] + x[1] - 15,
          'jac': lambda x: np.array([2, 1]) }

# Начальное приближение
x0 = np.array([1, 1])

res = minimize(f, x0, method='SLSQP', jac=grad_f, constraints=[constr], tol=1e-9)

print(res)

# %% Ответ с округлением (один знак)
res.x.round(1)
# x = 3.0, y = 9.0


# %% Задача 7. min(x + y + z)  s.t.  x − y + z = 0,  x² + 2y² + 2z² = 8
# Целевая функция f(x) и её градиент
def f(x):
    return x[0] + x[1] + x[2]
def grad_f(x):
    return np.array([1, 1, 1])

# ограничения равенства задаём в виде словаря
constr1 = {'type': 'eq',
           'fun': lambda x: x[0] - x[1] + x[2],
           'jac': lambda x: np.array([1, -1, 1]) }

constr2 = {'type': 'eq',
           'fun': lambda x: x[0]**2 + 2*x[1]**2 + 2*x[2]**2 - 8,
           'jac': lambda x: np.array([2*x[0], 4*x[1], 4*x[2]]) }

# Начальное приближение (для минимума берём отрицательные координаты)
x0 = np.array([-1, -1, -1])

res = minimize(f, x0, method='SLSQP', jac=grad_f, constraints=[constr1, constr2], tol=1e-9)

print(res)

# %% Ответ с округлением (два знака)
res.x.round(2)
# x = -1.15, y = -1.73, z = -0.58
# %% Ячейка 0. Импорт библиотек
import numpy as np
import sympy as sp
from scipy.optimize import minimize


# %% Задача 1. min 5x² + y² − 4xy − 2x − 4y  при  2x + y = 5
def f1(v):
    x, y = v
    return 5*x**2 + y**2 - 4*x*y - 2*x - 4*y

cons1 = [{'type': 'eq', 'fun': lambda v: 2*v[0] + v[1] - 5}]

res1 = minimize(f1, x0=[0, 0], constraints=cons1)
print('x, y =', res1.x.round(1), '  f =', round(res1.fun, 4))
# Ответ: x = 1.0, y = 3.0  (f = -12)


# %% Задача 2. min 5x² + 2y² − 6xy − 4x − 4y  при  2x + y = 6
def f2(v):
    x, y = v
    return 5*x**2 + 2*y**2 - 6*x*y - 4*x - 4*y

cons2 = [{'type': 'eq', 'fun': lambda v: 2*v[0] + v[1] - 6}]

res2 = minimize(f2, x0=[0, 0], constraints=cons2)
print('x, y =', res2.x.round(1), '  f =', round(res2.fun, 4))
# Ответ: x = 1.6, y = 2.8  (f = -16)


# %% Задача 3. max 15x + 10y + 8z − 6x² − 9y² − 5z² − 2xy + 4xz + 2yz  (численно)
def f3(v):
    x, y, z = v
    return (15*x + 10*y + 8*z - 6*x**2 - 9*y**2 - 5*z**2
            - 2*x*y + 4*x*z + 2*y*z)

# максимум f = минимум (−f)
res3 = minimize(lambda v: -f3(v), x0=[0, 0, 0])
print('x, y, z =', res3.x.round(3), '  f =', round(-res3.fun, 4))

# Проверка: гессиан отрицательно определён → максимум глобальный
H = np.array([[-12,  -2,   4],
              [ -2, -18,   2],
              [  4,   2, -10]])
print('собственные числа гессиана:', np.linalg.eigvalsh(H).round(3))

# Точное решение системы grad f = 0  (−H·p = b)
print('точно:', np.linalg.solve(-H, [15, 10, 8]).round(3))
# Ответ: x = 1.687, y = 0.544, z = 1.584


# %% Задача 4. Локальный минимум f(x, y) = 4x + 5xy − 5 ln x − 7 ln y
x, y = sp.symbols('x y')
f4 = 4*x + 5*x*y - 5*sp.log(x) - 7*sp.log(y)

stationary = sp.solve([sp.diff(f4, x), sp.diff(f4, y)], [x, y], dict=True)
print('стационарные точки:', stationary)

# Область определения: x > 0, y > 0
inside = [s for s in stationary if s[x] > 0 and s[y] > 0]
print('точки внутри области:', inside)
# Единственная точка (−0.5; −2.8) лежит вне области → решения нет
# Ответ: x = 0, y = 0


# %% Задача 5. max x + y + z  при  x² + y² + 2z² = 9,  x − y + z = 0
def f5(v):
    return v[0] + v[1] + v[2]

cons5 = [{'type': 'eq', 'fun': lambda v: v[0]**2 + v[1]**2 + 2*v[2]**2 - 9},
         {'type': 'eq', 'fun': lambda v: v[0] - v[1] + v[2]}]

res5 = minimize(lambda v: -f5(v), x0=[1, 1, 1], constraints=cons5)
print('x, y, z =', res5.x.round(2), '  f =', round(-res5.fun, 4))
# Ответ: x = 1.55, y = 2.32, z = 0.77  (f ≈ 4.648)


# %% Задача 6. max 6x + 12y − 5x² − y² + 4xy  при  2x + y = 15
def f6(v):
    x, y = v
    return 6*x + 12*y - 5*x**2 - y**2 + 4*x*y

cons6 = [{'type': 'eq', 'fun': lambda v: 2*v[0] + v[1] - 15}]

res6 = minimize(lambda v: -f6(v), x0=[0, 0], constraints=cons6)
print('x, y =', res6.x.round(1), '  f =', round(-res6.fun, 4))
# Ответ: x = 3.0, y = 9.0  (f = 108)


# %% Задача 7. min x + y + z  при  x² + 2y² + 2z² = 8,  x − y + z = 0
def f7(v):
    return v[0] + v[1] + v[2]

cons7 = [{'type': 'eq', 'fun': lambda v: v[0]**2 + 2*v[1]**2 + 2*v[2]**2 - 8},
         {'type': 'eq', 'fun': lambda v: v[0] - v[1] + v[2]}]

res7 = minimize(f7, x0=[-1, -1, -1], constraints=cons7)
print('x, y, z =', res7.x.round(2), '  f =', round(res7.fun, 4))
# Ответ: x = -1.15, y = -1.73, z = -0.58  (f ≈ -3.464)


# %% Проверка задач 5 и 7 методом Лагранжа (SymPy)
# Численный метод находит локальный экстремум, поэтому сверяем
# со всеми стационарными точками функции Лагранжа.
x, y, z, l1, l2 = sp.symbols('x y z lambda1 lambda2', real=True)

def lagrange_points(f, g1, g2):
    L = f - l1*g1 - l2*g2
    eqs = [sp.diff(L, v) for v in (x, y, z)] + [g1, g2]
    return sp.solve(eqs, [x, y, z, l1, l2], dict=True)

for name, g1 in [('Задача 5', x**2 + y**2 + 2*z**2 - 9),
                 ('Задача 7', x**2 + 2*y**2 + 2*z**2 - 8)]:
    print(name)
    for s in lagrange_points(x + y + z, g1, x - y + z):
        point = {str(k): round(float(s[k]), 2) for k in (x, y, z)}
        print('   ', point, '  f =', round(float((x + y + z).subs(s)), 3))
# В задаче 5 берём точку с бо́льшим f, в задаче 7 — с меньшим
#Вопрос 1, вариант А (графически, округлить до целого)
#Условие: min(5x + 2y) при 3x + 2y ≥ 10, x + 2y ≥ 6, x + 4y ≥ 8, x, y ≥ 0.

import numpy as np
from scipy.optimize import linprog

c = np.array([5, 2])
A = np.array([[-3, -2], [-1, -2], [-1, -4]])
b = np.array([-10, -6, -8])

res = linprog(c, A_ub=A, b_ub=b)

if not res.success:
    print('Нет решения -> ставь нули')
else:
    x, y = np.round(res.x) + 0.0
    print('x =', int(x), ' y =', int(y), ' целевая =', int(np.round(res.fun)))

##Вопрос 1, вариант Б (графически, округлить до целого)

#Условие: min(2x + 4y) при 5x + 2y ≥ 16, x + y ≥ 5, x + 5y ≥ 9, x, y ≥ 0.

import numpy as np
from scipy.optimize import linprog

c = np.array([2, 4])
A = np.array([[-5, -2], [-1, -1], [-1, -5]])
b = np.array([-16, -5, -9])

res = linprog(c, A_ub=A, b_ub=b)

if not res.success:
    print('Нет решения -> ставь нули')
else:
    x, y = np.round(res.x) + 0.0
    print('x =', int(x), ' y =', int(y), ' целевая =', int(np.round(res.fun)))

#Ответ: x = 4, y = 1, целевая = 12.

#Вопрос 2 (аналитически, округлить до 1 знака)

#Условие: max(3x + y − 2x² − 5y² + 6xy) при x + 2y ≤ 35.
import sympy as sp

x, y, lam = sp.symbols('x y lam')
f = 3*x + y - 2*x**2 - 5*y**2 + 6*x*y
g = x + 2*y - 35                      # ограничение: g <= 0

s = sp.solve([sp.diff(f, x), sp.diff(f, y)], [x, y], dict=True)[0]

if g.subs(s) <= 0:
    xs, ys = s[x], s[y]
else:
    L = f - lam*g
    s = sp.solve([sp.diff(L, x), sp.diff(L, y), g], [x, y, lam], dict=True)[0]
    xs, ys = s[x], s[y]

print('x =', round(float(xs), 1), ' y =', round(float(ys), 1))

#Ответ: x = 9.0, y = 5.5.

#Вопрос 3 (численно, округлить до 2 знаков)

#Условие: min(2x² + 3y² + 2z² + 2xy − 2xz) при x + 5y + 3z ≥ 3, x, y, z ≥ 0.


import numpy as np
from scipy.optimize import minimize

def f(v):
    x, y, z = v
    return 2*x**2 + 3*y**2 + 2*z**2 + 2*x*y - 2*x*z

cons = [{'type': 'ineq', 'fun': lambda v: v[0] + 5*v[1] + 3*v[2] - 3}]

res = minimize(f, x0=[1, 1, 1], bounds=[(0, None)]*3, constraints=cons,
               method='SLSQP', options={'ftol': 1e-12})

if not res.success:
    print('Нет решения -> ставь нули')
else:
    x, y, z = np.round(res.x, 2) + 0.0
    print('x =', x, ' y =', y, ' z =', z)

#Ответ: x = 0.16, y = 0.32, z = 0.41.

#Вопрос 4 (численно, ЛП, округлить до 2 знаков)

#Условие: min(3x + 2y + z) при 2x + 2y + z ≥ 9, 4x + y + 2z ≥ 11, x, y, z ≥ 0.


import numpy as np
from scipy.optimize import linprog

c = np.array([3, 2, 1])
A = np.array([[-2, -2, -1], [-4, -1, -2]])
b = np.array([-9, -11])

res = linprog(c, A_ub=A, b_ub=b)

if not res.success:
    print('Нет решения -> ставь нули')
else:
    x, y, z = np.round(res.x, 2) + 0.0
    print('x =', x, ' y =', y, ' z =', z, ' целевая =', round(res.fun, 2))

#Ответ: целевая = 9.00, x = 0. Для y и z вписывай то, что выведет код: у задачи несколько оптимальных решений, например (0; 2.33; 4.33) или (0; 0; 9).

#Задача с max(x + 2y − 3z) (численно, округлить до 2 знаков)


#Условие: max(x + 2y − 3z) при 2x² + y² + 2z² + 2xz ≤ 5, x, y, z ≥ 0.


import numpy as np
from scipy.optimize import minimize

def f(v):
    x, y, z = v
    return -(x + 2*y - 3*z)           # max -> min через минус

cons = [{'type': 'ineq', 'fun': lambda v: 5 - (2*v[0]**2 + v[1]**2 + 2*v[2]**2 + 2*v[0]*v[2])}]

res = minimize(f, x0=[0.5, 1, 0], bounds=[(0, None)]*3, constraints=cons,
               method='SLSQP', options={'ftol': 1e-12})

if not res.success:
    print('Нет решения -> ставь нули')
else:
    x, y, z = np.round(res.x, 2) + 0.0
    print('x =', x, ' y =', y, ' z =', z)

# Ответ: x = 0.53, y = 2.11, z = 0.00.




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ===== Задача =====
# min(5x + 2y)
# 3x + 2y >= 10
#  x + 2y >= 6
#  x + 4y >= 8
#  x, y >= 0
# linprog понимает только "<=", поэтому умножаем ограничения на -1

# ===== Решение =====
c = np.array([5, 2])                              # вектор целевой функции
A = np.array([[-3, -2], [-1, -2], [-1, -4]])      # матрица A_ub
b = np.array([-10, -6, -8])                       # вектор b_ub

res = linprog(c, A_ub=A, b_ub=b)                  # x, y >= 0 заданы по умолчанию
print(res)
print('Решение (x, y):', res.x.round())
print('Минимум 5x+2y =', round(res.fun))

# ===== Визуализация =====
x = np.linspace(-1, 10, 200)

# прямые-ограничения
plt.plot(x, (10 - 3*x) / 2, label='(1) 3x+2y=10')
plt.plot(x, (6 - x) / 2, label='(2) x+2y=6')
plt.plot(x, (8 - x) / 4, label='(3) x+4y=8')

# допустимая область: выше всех прямых (неограниченная, рисуем до y=8)
xs = np.linspace(0, 10, 300)
y_low = np.maximum.reduce([(10 - 3*xs) / 2, (6 - xs) / 2,
                           (8 - xs) / 4, np.zeros_like(xs)])
plt.fill_between(xs, y_low, 8, color='green', alpha=0.2, label='допустимая область')

# линия уровня целевой функции через оптимум: 5x + 2y = F
F = res.fun
plt.plot(x, (F - 5*x) / 2, 'r--', label=f'5x+2y={F:.0f}')

# оптимальная точка
plt.plot(res.x[0], res.x[1], 'ro')
plt.annotate(f'({res.x[0]:.0f}; {res.x[1]:.0f})',
             xy=(res.x[0], res.x[1]), xytext=(res.x[0] + 0.3, res.x[1] + 0.5))

# оси координат
plt.vlines(x=0, ymin=0, ymax=8, colors='b', linestyles='dotted')
plt.hlines(y=0, xmin=0, xmax=10, colors='b', linestyles='dotted')

# целочисленная сетка
plt.xticks(ticks=np.arange(-1, 11, 1))
plt.yticks(ticks=np.arange(-1, 9, 1))
plt.xlim(-1, 10)
plt.ylim(-1, 8)
plt.grid(linewidth=0.3)

plt.legend(loc=1)
plt.show()

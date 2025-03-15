# Задання даних
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 1
x1_test, x2_test = 1.5, 3

print("Марич Олег, ОІ-35, Варіант ", n);

x1 = [0, 0, 0, 1, 1, 2, 2, 2]
x2 = [1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5]
y = [2.3, 4 + 0.3 * n, 2 - 0.1 * n, 5 - 0.2 * n, 4 - 0.2 * n, 6.1 + 0.2 * n, 6.5 - 0.1 * n, 7.2]

N = len(x1)
sum_x1 = sum(x1)
sum_x2 = sum(x2)
sum_y = sum(y)
sum_x1x1 = sum(x1_i ** 2 for x1_i in x1)
sum_x1x2 = sum(x1[i] * x2[i] for i in range(N))
sum_x2x2 = sum(x2_i ** 2 for x2_i in x2)
sum_x1y = sum(x1[i] * y[i] for i in range(N))
sum_x2y = sum(x2[i] * y[i] for i in range(N))

A = [
    [N, sum_x1, sum_x2],
    [sum_x1, sum_x1x1, sum_x1x2],
    [sum_x2, sum_x1x2, sum_x2x2]
]
B = [sum_y, sum_x1y, sum_x2y]

# Розв'язання системи рівнянь методом Гаусса
for i in range(3):
    pivot = A[i][i]
    for j in range(i, 3):
        A[i][j] /= pivot
    B[i] /= pivot
    for k in range(i+1, 3):
        factor = A[k][i]
        for j in range(i, 3):
            A[k][j] -= factor * A[i][j]
        B[k] -= factor * B[i]

a2 = B[2] / A[2][2]
a1 = (B[1] - A[1][2] * a2) / A[1][1]
a0 = (B[0] - A[0][1] * a1 - A[0][2] * a2) / A[0][0]

# Обчислення значення функції у точці (1.5, 3)
y_pred = a0 + a1 * x1_test + a2 * x2_test

# Обчислення R^2
y_mean = sum_y / N
y_pred_all = [a0 + a1 * x1[i] + a2 * x2[i] for i in range(N)]
ss_total = sum((yi - y_mean) ** 2 for yi in y)
ss_residual = sum((y[i] - y_pred_all[i]) ** 2 for i in range(N))
r2 = 1 - (ss_residual / ss_total)

# Створення сітки значень x1, x2
x1_grid = [0, 1, 2]
x2_grid = [1.5, 2.5, 3.5]
points = [(x1i, x2i) for x1i in x1_grid for x2i in x2_grid]
y_grid = [a0 + a1 * x1i + a2 * x2i for x1i, x2i in points]

x1_vals = np.linspace(0, 2, 10)
x2_vals = np.linspace(1.5, 3.5, 10)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Y = a0 + a1 * X1 + a2 * X2


# Побудова графіка
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1_test, x2_test, y_pred, color='g', s=100, label='Шукана точка ({:.2f}, {:.2f}, {:.2f})'.format(x1_test, x2_test, y_pred))
ax.scatter(x1, x2, y, color='r', label='Реальні дані')
ax.scatter(*zip(*points), y_grid, color='b', marker='^', label='Сітка значень')
ax.plot_surface(X1, X2, Y, color='cyan', alpha=0.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title("Марич Олег, ОІ-35, Варіант " + str(n))
ax.legend()
plt.show()


print(f"a0 = {a0:.4f}")
print(f"a1 = {a1:.4f}")
print(f"a2 = {a2:.4f}")
print(f"Прогнозоване значення у точці (1.5, 3): y = {y_pred:.4f}")
print(f"Коефіцієнт детермінації R^2 = {1 - sum((y[i] - (a0 + a1 * x1[i] + a2 * x2[i])) ** 2 for i in range(N)) / sum((yi - sum_y / N) ** 2 for yi in y):.4f}")


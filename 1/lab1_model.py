import numpy as np
import matplotlib.pyplot as plt

# Дано з таблиці
x1 = np.array([0, 0, 0, 1, 1, 2, 2, 2])
x2 = np.array([1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5])
y = np.array([2.3, 8.2, 0.6, 2.2, 1.2, 8.9, 5.1, 7.2])

# Формуємо матрицю X для методу найменших квадратів
X = np.vstack([np.ones(len(x1)), x1, x2]).T

# Знаходимо коефіцієнти a0, a1, a2
coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

# Виведення рівняння
print(f"Знайдене рівняння: y = {coeffs[0]:.3f} + {coeffs[1]:.3f} * x1 + {coeffs[2]:.3f} * x2")

# Значення функції у точці x1 = 1.5, x2 = 3
x1_test, x2_test = 1.5, 3
y_test = coeffs[0] + coeffs[1] * x1_test + coeffs[2] * x2_test
print(f"Значення функції у точці (x1=1.5, x2=3): {y_test:.3f}")

# Обчислюємо коефіцієнт детермінації R²
y_pred = X @ coeffs
SSE = np.sum((y - y_pred)**2)
SST = np.sum((y - np.mean(y))**2)
R2 = 1 - (SSE / SST)
print(f"Коефіцієнт детермінації R²: {R2:.3f}")

# Побудова графіка
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, color='red', label='Дані')

# Створення сітки для побудови поверхні
x1_range = np.linspace(min(x1), max(x1), 10)
x2_range = np.linspace(min(x2), max(x2), 10)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Обчислюємо значення поверхні
Y = coeffs[0] + coeffs[1] * X1 + coeffs[2] * X2
ax.plot_surface(X1, X2, Y, alpha=0.5, color='cyan')

# Позначення осей і заголовок
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Модель найменших квадратів')
ax.legend()

# Виведення графіку
plt.show()
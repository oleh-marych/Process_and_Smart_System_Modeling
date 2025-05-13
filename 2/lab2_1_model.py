import numpy as np
import matplotlib.pyplot as plt

# Параметри
N = 14
a11 = 0.14
a12 = 0.0014
a21 = 0.0014
a22 = 0.56

# Початкові умови
x0 = 860
y0 = 560
t0 = 0
T = 150
h = 0.1


# Система рівнянь Лотки-Вольтерри
def dx_dt(x, y):
    return a11 * x - a12 * x * y


def dy_dt(x, y):
    return -a22 * y + a21 * x * y


# Метод Рунге-Кутта 4-го порядку
def runge_kutta(x0, y0, t0, T, h):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    t = t0
    x = x0
    y = y0

    while t < T:
        k1x = h * dx_dt(x, y)
        k1y = h * dy_dt(x, y)

        k2x = h * dx_dt(x + 0.5 * k1x, y + 0.5 * k1y)
        k2y = h * dy_dt(x + 0.5 * k1x, y + 0.5 * k1y)

        k3x = h * dx_dt(x + 0.5 * k2x, y + 0.5 * k2y)
        k3y = h * dy_dt(x + 0.5 * k2x, y + 0.5 * k2y)

        k4x = h * dx_dt(x + k3x, y + k3y)
        k4y = h * dy_dt(x + k3x, y + k3y)

        x += (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y += (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        t += h

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)

    return np.array(t_values), np.array(x_values), np.array(y_values)


# Виконання розрахунку
t_vals, x_vals, y_vals = runge_kutta(x0, y0, t0, T, h)

# Побудова графіків
plt.figure(figsize=(16, 5))

# x(t)
plt.subplot(1, 3, 1)
plt.plot(t_vals, x_vals, label='Жертви (x)', color='blue')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість жертв')
plt.title('Залежність x(t)')
plt.grid(True)

# y(t)
plt.subplot(1, 3, 2)
plt.plot(t_vals, y_vals, label='Хижаки (y)', color='red')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість хижаків')
plt.title('Залежність y(t)')
plt.grid(True)

# y(x)
plt.subplot(1, 3, 3)
plt.plot(x_vals, y_vals, label='Фазова траєкторія y(x)', color='green')
plt.xlabel('Кількість жертв (x)')
plt.ylabel('Кількість хижаків (y)')
plt.title('Залежність y(x)')
plt.grid(True)

plt.tight_layout()
plt.show()

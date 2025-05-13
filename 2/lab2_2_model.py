import numpy as np
import matplotlib.pyplot as plt

# Параметри
N = 14
H = 1000 - N
beta = 25 - N
gamma = N
x0 = 900 - N
y0 = 90 - N
z0 = H - x0 - y0

t0 = 0
T = 40
h = 0.1
steps = int((T - t0) / h) + 1

# Система рівнянь
def dx_dt(x, y):
    return -beta / H * x * y

def dy_dt(x, y):
    return beta / H * x * y - y / gamma

def dz_dt(y):
    return y / gamma

# Масиви результатів
t = np.linspace(t0, T, steps)
x = np.zeros(steps)
y = np.zeros(steps)
z = np.zeros(steps)

# Початкові умови
x[0], y[0], z[0] = x0, y0, z0

# Метод Рунге-Кутта 4-го порядку
for i in range(steps - 1):
    k1x = h * dx_dt(x[i], y[i])
    k1y = h * dy_dt(x[i], y[i])
    k1z = h * dz_dt(y[i])

    k2x = h * dx_dt(x[i] + k1x / 2, y[i] + k1y / 2)
    k2y = h * dy_dt(x[i] + k1x / 2, y[i] + k1y / 2)
    k2z = h * dz_dt(y[i] + k1y / 2)

    k3x = h * dx_dt(x[i] + k2x / 2, y[i] + k2y / 2)
    k3y = h * dy_dt(x[i] + k2x / 2, y[i] + k2y / 2)
    k3z = h * dz_dt(y[i] + k2y / 2)

    k4x = h * dx_dt(x[i] + k3x, y[i] + k3y)
    k4y = h * dy_dt(x[i] + k3x, y[i] + k3y)
    k4z = h * dz_dt(y[i] + k3y)

    x[i+1] = x[i] + (k1x + 2*k2x + 2*k3x + k4x) / 6
    y[i+1] = y[i] + (k1y + 2*k2y + 2*k3y + k4y) / 6
    z[i+1] = z[i] + (k1z + 2*k2z + 2*k3z + k4z) / 6

# Побудова графіків
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(t, x, 'g', label='Здорові (x)')
plt.xlabel('Час (дні)')
plt.ylabel('Кількість людей')
plt.title('x(t)')
plt.grid()
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t, y, 'r', label='Хворі (y)')
plt.xlabel('Час (дні)')
plt.title('y(t)')
plt.grid()
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(t, z, 'b', label='Ті, що одужали (z)')
plt.xlabel('Час (дні)')
plt.title('z(t)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

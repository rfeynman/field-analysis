import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- 1) Create 25 (x, y) data points ---
x = np.linspace(0.0, 10.0, 25)

rng = np.random.default_rng(0)
y = np.sin(x) + 0.1 * rng.normal(size=x.size)

# --- 2) Create multiple interpolation functions ---
f_linear = interp1d(x, y, kind="linear")
f_nearest = interp1d(x, y, kind="nearest")
f_quadratic = interp1d(x, y, kind="quadratic")
f_cubic = interp1d(x, y, kind="cubic")

# --- 3) Generate dense x for smooth plotting ---
x_dense = np.linspace(x.min(), x.max(), 400)

y_linear = f_linear(x_dense)
y_nearest = f_nearest(x_dense)
y_quadratic = f_quadratic(x_dense)
y_cubic = f_cubic(x_dense)

# --- 4) Plot everything ---
plt.figure()

#plt.plot(x_dense, y_linear, label="linear")
#plt.plot(x_dense, y_nearest, label="nearest")
#plt.plot(x_dense, y_quadratic, label="quadratic")
plt.plot(x_dense, y_cubic, label="cubic")

plt.scatter(x, y, label="data points")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Interpolation Methods")
plt.legend()
plt.show()

# --- 5) Example evaluation ---
print("At x = 5.0")
print("linear    :", float(f_linear(5.0)))
print("nearest   :", float(f_nearest(5.0)))
print("quadratic :", float(f_quadratic(5.0)))
print("cubic     :", float(f_cubic(5.0)))
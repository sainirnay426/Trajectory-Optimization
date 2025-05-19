import pykep as pk
import numpy as np
import matplotlib.pyplot as plt

# Constants
mu = 398600.4418  # km^3/s^2 (Earth)

# Orbit radii (circular)
R_i = 6378 + 300   # km (initial radius)
R_f = 6378 + 1100  # km (final radius)

# Define angular positions
theta_start = 0
theta_end = np.pi-0.001  # avoid exactly 180 degrees

# Compute Cartesian positions
r1 = [R_i * np.cos(theta_start), R_i * np.sin(theta_start), 0]
r2 = [R_f * np.cos(theta_end), R_f * np.sin(theta_end), 0]

# Time of flight (must be reasonable!)
tof = 2963.145  # seconds

lb = pk.lambert_problem(r1, r2, tof, mu, cw=False, max_revs=0)
v1 = lb.get_v1()[0]
v2 = lb.get_v2()[0]
print("v1 =", v1)
print("v2 =", v2)

num = 50
times = np.linspace(1e-5, tof, num)

states = np.array([pk.propagate_lagrangian(r1, v1, t, pk.MU_EARTH) for t in times])
states = np.array(states)  # shape: (num, 2, 3)

r_array = states[:, 0]  # shape: (num, 3)
v_array = states[:, 1]  # shape: (num, 3)

# Extract x, y and velocity components
x_vals = r_array[:, 0]
y_vals = r_array[:, 1]
xdot_vals = v_array[:, 0]
ydot_vals = v_array[:, 1]

r_vals = np.sqrt(x_vals ** 2 + y_vals ** 2)
theta_vals = np.arctan2(y_vals, x_vals)
dr_vals = (x_vals * xdot_vals + y_vals * ydot_vals) / r_vals
dtheta_vals = (x_vals * ydot_vals - y_vals * xdot_vals) / (r_vals ** 2)

plt.figure(0)
plt.title('trajectory')
plt.plot(x_vals, y_vals, color='purple', linewidth=2)
plt.xlim((-1E4, 1E4))
plt.ylim((-1E4, 1E4))

plt.figure(1)
plt.title('r')
plt.plot(r_vals)

plt.figure(2)
plt.title('theta')
plt.plot(theta_vals)

plt.figure(3)
plt.title('dr')
plt.plot(dr_vals)

plt.figure(4)
plt.title('dtheta')
plt.plot(dtheta_vals)

plt.show()
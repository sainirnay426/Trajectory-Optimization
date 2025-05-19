import math
import os
import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad

import pykep as pk
from pykep import DAY2SEC


def scale_factor(a):
    power = np.log10(np.abs(a))
    return 10 ** -np.floor(power)


def time_of_flight(p, e, theta_start, theta_end, u, use_sin):
    if use_sin:
        integrand = lambda th: 1 / (1 + e * np.sin(th)) ** 2
    else:
        integrand = lambda th: 1 / (1 + e * np.cos(th)) ** 2
    integral, _ = quad(integrand, theta_start, theta_end)
    return np.sqrt(p ** 3 / u) * integral


# Constants
G = 6.67430 * 10 ** (-20)  # gravitational constant (km)
M = 5.972 * 10 ** 24  # planet mass (earth)
Isp = 400  # s
g0 = 9.81
u = G * M
R = 6378  # earth radius (km)
a_i = 300  # initial altitude (km)
a_f = 1100  # final altitude (km)
theta_start = 0
theta_end = 7*np.pi/6

R_i = R + a_i
R_f = R + a_f

# ascending or descending
ascending = R_f > R_i

cos_0 = np.cos(theta_start)
cos_f = np.cos(theta_end)

if abs(cos_0) < 1e-3 and abs(cos_f) < 1e-3:
    trig_0 = np.sin(theta_start)
    trig_f = np.sin(theta_end)
    use_sin = True
else:
    trig_0 = cos_0
    trig_f = cos_f
    use_sin = False

A = np.array([
    [R_i * trig_0, -1],
    [R_f * trig_f, -1]
])
b = np.array([-R_i, -R_f])
solution = np.linalg.solve(A, b)

e_fit = solution[0]
p_fit = solution[1]

print('e-fit:', e_fit)
print('p-fit:', p_fit)

# 3. Get velocity magnitudes at R_i and R_f using vis-viva
a_transfer = p_fit / (1 - e_fit ** 2)

v0_elip = np.sqrt(u * (2 / R_i - 1 / a_transfer))
vf_elip = np.sqrt(u * (2 / R_f - 1 / a_transfer))

v0_circular = np.sqrt(u / R_i)
vf_circular = np.sqrt(u / R_f)

angular_vel_i = np.sqrt(u / (R_i ** 3))
angular_vel_f = np.sqrt(u / (R_f ** 3))

delta_v1 = np.abs(v0_elip - v0_circular)
delta_v2 = np.abs(vf_circular - vf_elip)

total_time = time_of_flight(p_fit, e_fit, theta_start, theta_end, u, use_sin)
h_time = np.pi * np.sqrt((a_transfer ** 3) / u)

num = int(50 * (total_time / h_time))

print("\ntrajectory time estimate:", total_time)
dt_val = total_time / num
print("# of steps:", num)
print("dt estimate:", dt_val)

m_empty = 1000  # kg
ve = Isp * g0
m_burn_2 = m_empty * np.exp(np.abs(delta_v2) * 1000 / ve)
prop_burn_2 = m_burn_2 - m_empty

m_burn_1 = m_burn_2 * np.exp(np.abs(delta_v1) * 1000 / ve)
prop_burn_1 = m_burn_1 - m_burn_2
max_fuel = prop_burn_1 + prop_burn_2

print("\ndelta_v1:", delta_v1)
print("delta_v2:", delta_v2)
print("prop_burn_1:", prop_burn_1)
print("prop_burn_2:", prop_burn_2)
print("max_fuel:", max_fuel)

burn_time = dt_val

thrust_1 = ((prop_burn_1 * ve) / burn_time) / 1000
thrust_2 = ((prop_burn_2 * ve) / burn_time) / 1000
print("thrust 1:", thrust_1)
print("thrust 2:", thrust_2)

max_thrust = max(thrust_1, thrust_2)

### STATE
ecc = e_fit
p = p_fit

theta_values = np.linspace(theta_start, theta_end, num=num)

r_values = np.zeros(num)
r_values = p / (1 + ecc * np.cos(theta_values))

dr_values = np.zeros(num)
dr_values = np.sqrt((u * 10 ** 9) / (p * 10 ** 3)) * ecc * np.sin(theta_values)

v_theta = np.sqrt(u / p) * (1 + ecc * np.cos(theta_values))  # Tangential velocity in elliptical orbit
dtheta_values = v_theta / r_values  # Convert to angular velocity


### CONTROLS
thrust_values = np.zeros(num)
thrust_values[1] = thrust_1
thrust_values[-2] = thrust_2

if ascending:
    thrust_ang_values = np.linspace(0, 0, num=num)
else:
    thrust_ang_values = np.linspace(np.pi, np.pi, num=num)

m_values = np.zeros(num)
m_scale = scale_factor(max_fuel)
m_values[0] = max_fuel

for i in range(1, num):
    if abs(thrust_values[i - 1] * 1000) > 0:
        dm = abs(thrust_values[i - 1] * 1000) / (Isp * g0) * dt_val
        m_values[i] = m_values[i - 1] - dm
    else:
        m_values[i] = m_values[i - 1]

x = r_values * np.cos(theta_values)
y = r_values * np.sin(theta_values)

th = np.linspace(0, 2 * np.pi, num=num)
orbit_0x = R_i * np.cos(th)
orbit_0y = R_i * np.sin(th)
orbit_fx = R_f * np.cos(th)
orbit_fy = R_f * np.sin(th)

folder_path = "inital_guess_rand_ellip_plots"
os.makedirs(folder_path, exist_ok=True)
files = os.listdir(folder_path)
for file in files:
    file_path = os.path.join(folder_path, file)
    os.remove(file_path)

plt.figure()
plt.title('trajectory')
plt.plot(x, y, color='purple', linewidth=2)
plt.plot(orbit_0x, orbit_0y, color='red', linestyle='dotted', linewidth=2)
plt.plot(orbit_fx, orbit_fy, color='blue', linestyle='dotted', linewidth=2)
plt.xlim((-1E4, 1E4))
plt.ylim((-1E4, 1E4))
file_path = os.path.join(folder_path, "trajectory.png")
plt.savefig(file_path)

plt.figure()
plt.title('r')
plt.plot(r_values)
file_path = os.path.join(folder_path, "radius.png")
plt.savefig(file_path)

plt.figure()
plt.title('theta')
plt.plot(theta_values)
file_path = os.path.join(folder_path, "theta.png")
plt.savefig(file_path)

plt.figure()
plt.title('dr')
plt.plot(dr_values)
file_path = os.path.join(folder_path, "dr.png")
plt.savefig(file_path)

plt.figure()
plt.title('dtheta')
plt.plot(dtheta_values)
file_path = os.path.join(folder_path, "dtheta.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust')
plt.plot(thrust_values)
file_path = os.path.join(folder_path, "thrust.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust-ang')
plt.plot(thrust_ang_values)
file_path = os.path.join(folder_path, "thrust_ang.png")
plt.savefig(file_path)

plt.figure()
plt.title('mass')
plt.plot(m_values)
file_path = os.path.join(folder_path, "mass.png")
plt.savefig(file_path)

plt.show()

import math
import os
import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl

# Constants
G = 6.67430 * 10 ** (-20)  # gravitational constant (km)
M = 5.972 * 10 ** 24  # planet mass (earth)
Isp = 400  # s
g0 = 9.81
u = G * M
R = 6378  # earth radius (km)
a_i = 300  # initial altitude (km)
a_f = 700  # final altitude (km)

R_i = R + a_i
R_f = R + a_f

# max_thrust = 2000
# m_empty = 1000  # kg
# max_fuel = 200  # kg

# ascending or descending
ascending = R_f > R_i

# Calculate Hohmann transfer parameters
a_transfer = (R_i + R_f) / 2  # Semi-major axis of transfer orbit
ecc = 1 - R_f / a_transfer  # Eccentricity
p = a_transfer * (1 - ecc ** 2)  # Semi-latus rectum

if ascending:
    periapsis = R_i
    apoapsis = R_f
else:
    periapsis = R_f
    apoapsis = R_i

ecc = (apoapsis - periapsis) / (apoapsis + periapsis)
p = a_transfer * (1 - ecc ** 2)

# Calculate delta-V requirements for burns
v_periapsis = np.sqrt(2 * u / periapsis - u / a_transfer)
v_apoapsis = np.sqrt(2 * u / apoapsis - u / a_transfer)
# print(v_periapsis)
# print(v_apoapsis)

if ascending:
    delta_v1 = v_periapsis - np.sqrt(u / R_i)
    delta_v2 = np.sqrt(u / R_f) - v_apoapsis
    thrust_direction = 1
else:
    delta_v1 = np.sqrt(u / R_i) - v_apoapsis
    delta_v2 = v_periapsis - np.sqrt(u / R_f)
    thrust_direction = -1

# print(delta_v1)
# print(delta_v2)

angular_vel_i = np.sqrt(u / (R_i ** 3))
angular_vel_f = np.sqrt(u / (R_f ** 3))

theta_end = np.pi
num = 50
time = (theta_end/np.pi) * np.pi * np.sqrt((a_transfer ** 3) / u)
dt = time/num

m_empty = 1000  # kg
ve = Isp * g0
delta_v = (delta_v1 + delta_v2) * 1000
mass_ratio = np.exp(delta_v / ve)
max_fuel = m_empty * (mass_ratio - 1)
print(max_fuel)

burn_time = (dt*2)
thrust_req = (max_fuel * ve) / burn_time
max_thrust = thrust_req/2
print(max_thrust)

r_scale = 1E-3
dr_scale = 1E-2
theta_scale = 1
dtheta_scale = 1E3
m_scale = 1E-2
F_scale = 1E-3

### STATE
theta_values = np.linspace(0, theta_end, num=num)

r_values = np.zeros(num)
r_values = p / (1 + ecc * np.cos(theta_values))

dr_values = np.zeros(num)
dr_values = np.sqrt(u / p) * ecc * np.sin(theta_values)

v_theta = np.sqrt(u / p) * (1 + ecc * np.cos(theta_values))  # Tangential velocity in elliptical orbit
dtheta_values = v_theta / r_values  # Convert to angular velocity

### CONTROLS
# # Thrust profile for continuous approximation of Hohmann
# thrust_values = np.zeros(num)
# # First burn region (initial 10% of trajectory)
# burn1_duration = int(num * 0.1)
# thrust_values[:burn1_duration] = thrust_direction * (max_thrust/2.3) * np.ones(burn1_duration)
# # Coast phase
# burn2_start = int(num * 0.9)
# thrust_values[burn1_duration:burn2_start] = 0
# # Second burn region (final 10% of trajectory)
# thrust_values[burn2_start:] = -thrust_direction * (max_thrust/2.3) * np.ones(num - burn2_start)

thrust_values = np.zeros(num)
thrust_values[1] = max_thrust
thrust_values[-2] = max_thrust

m_values = np.zeros(num)
m_values[0] = max_fuel
for i in range(1, num):
    if abs(thrust_values[i - 1]) > 0:
        dm = abs(thrust_values[i - 1]) / (Isp * g0) * dt
        m_values[i] = m_values[i - 1] - dm
    else:
        m_values[i] = m_values[i - 1]

x = r_values*np.cos(theta_values)
y = r_values*np.sin(theta_values)

folder_path = "inital_guess_plots"
os.makedirs(folder_path, exist_ok=True)
files = os.listdir(folder_path)
for file in files:
    file_path = os.path.join(folder_path, file)
    os.remove(file_path)

plt.figure(0)
plt.title('trajectory')
plt.plot(x, y, color='purple', linewidth=2)
plt.xlim((-1E4,1E4))
plt.ylim((-1E4,1E4))
file_path = os.path.join(folder_path, "trajectory.png")
plt.savefig(file_path)

plt.figure(1)
plt.title('r')
plt.plot(r_values)
file_path = os.path.join(folder_path, "radius.png")
plt.savefig(file_path)

plt.figure(2)
plt.title('theta')
plt.plot(theta_values)
file_path = os.path.join(folder_path, "theta.png")
plt.savefig(file_path)

plt.figure(3)
plt.title('dr')
plt.plot(dr_values)
file_path = os.path.join(folder_path, "dr.png")
plt.savefig(file_path)

plt.figure(4)
plt.title('dtheta')
plt.plot(dtheta_values)
file_path = os.path.join(folder_path, "dtheta.png")
plt.savefig(file_path)

plt.figure(5)
plt.title('thrust')
plt.plot(thrust_values)
file_path = os.path.join(folder_path, "thrust.png")
plt.savefig(file_path)

plt.figure(6)
plt.title('mass')
plt.plot(m_values)
file_path = os.path.join(folder_path, "mass.png")
plt.savefig(file_path)

plt.show()

















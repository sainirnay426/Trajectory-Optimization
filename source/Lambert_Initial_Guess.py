import numpy as np
import matplotlib.pyplot as plt
import pykep as pk
import os

# Constants
G = 6.67430 * 10 ** (-20)  # gravitational constant (km)
M = 5.972 * 10 ** 24  # planet mass (earth)
g0 = 9.81
u = G * M
R = 6378  # earth radius (km)

# Problem Parameters
m_empty = 1000
Isp = 400  # s
a_i = 300  # initial altitude (km)
a_f = 1100  # final altitude (km)
theta_start = 0
theta_end = np.pi

R_i = R + a_i
R_f = R + a_f

# ascending or descending
ascending = R_f > R_i

# Time of flight
h_time = np.pi * np.sqrt(((R_i + R_f) / 2) ** 3 / u) #Hohmann Transfer Time
tof = h_time * (theta_end / np.pi) #Fraction of Hohmann Transfer Time

# Initial and final position vectors
r1 = [R_i * np.cos(theta_start), R_i * np.sin(theta_start), 0]
r2 = [R_f * np.cos(theta_end), R_f * np.sin(theta_end), 0]

# Solve Lambert problem
lambert = pk.lambert_problem(r1, r2, tof, u)
v1 = lambert.get_v1()[0]
v2 = lambert.get_v2()[0]

v0_circular = np.sqrt(u / R_i)
vf_circular = np.sqrt(u / R_f)

v0_vec = v0_circular * np.array([-np.sin(theta_start), np.cos(theta_start)])
dv0_xy = np.array(v1[:2]) - v0_vec

r_hat = np.array([np.cos(theta_start), np.sin(theta_start)])
theta_hat = np.array([-np.sin(theta_start), np.cos(theta_start)])

dv_r0 = np.dot(dv0_xy, r_hat)
dv_theta0 = np.dot(dv0_xy, theta_hat)
dv0_p = [dv_r0, dv_theta0]

vf_vec = vf_circular * np.array([-np.sin(theta_end), np.cos(theta_end)])
dvf_xy = vf_vec - np.array(v2[:2])

r_hat = np.array([np.cos(theta_end), np.sin(theta_end)])
theta_hat = np.array([-np.sin(theta_end), np.cos(theta_end)])

dv_rf = np.dot(dvf_xy, r_hat)
dv_thetaf = np.dot(dvf_xy, theta_hat)
dvf_p = [dv_rf, dv_thetaf]

dv_1 = np.linalg.norm(dv0_p)
dv_2 = np.linalg.norm(dvf_p)

print("Cartesian:")
print("dv1:", dv0_xy)
print("dv2:", dvf_xy)
print("dv1 mag:", np.linalg.norm(dv0_xy))
print("dv2 mag:", np.linalg.norm(dvf_xy))

print("\nPolar:")
print("[radial, tang]")
print("dv1:", dv0_p)
print("dv2:", dvf_p)
print("dv1 mag:", np.linalg.norm(dv0_p))
print("dv2 mag:", np.linalg.norm(dvf_p))

# Time discretization

num = int(50 * (tof / h_time))

print("\ntrajectory time estimate:", tof)
dt_val = tof / num
t_vals = np.linspace(0, tof, num)


# Propagate the trajectory
xyz_vals = []
vxyz_vals = []

for t in t_vals:
    r, v = pk.propagate_lagrangian(r1, v1, t, u)
    xyz_vals.append(r)
    vxyz_vals.append(v)

xyz_vals = np.array(xyz_vals)
vxyz_vals = np.array(vxyz_vals)

x_vals, y_vals = np.array(xyz_vals[:, 0]), np.array(xyz_vals[:, 1])
xdot_vals, ydot_vals = np.array(vxyz_vals[:, 0]), np.array(vxyz_vals[:, 1])

r_vals = np.sqrt(x_vals ** 2 + y_vals ** 2)
theta_vals = np.arctan2(y_vals, x_vals) % (2 * np.pi)
dr_vals = (x_vals * xdot_vals + y_vals * ydot_vals) / r_vals
dtheta_vals = (x_vals * ydot_vals - y_vals * xdot_vals) / (r_vals ** 2)

ve = Isp * g0
m_burn_2 = m_empty * np.exp(np.abs(dv_2) * 1000 / ve)
prop_burn_2 = m_burn_2 - m_empty

m_burn_1 = m_burn_2 * np.exp(np.abs(dv_1) * 1000 / ve)
prop_burn_1 = m_burn_1 - m_burn_2
max_fuel = prop_burn_1 + prop_burn_2

print("\ndelta_v1:", dv_1)
print("delta_v2:", dv_2)
print("prop_burn_1:", prop_burn_1)
print("prop_burn_2:", prop_burn_2)
print("max_fuel:", max_fuel)

burn_time = dt_val

thrust_1 = ((prop_burn_1 * ve) / burn_time) / 1000
thrust_2 = ((prop_burn_2 * ve) / burn_time) / 1000
print("thrust 1:", thrust_1)
print("thrust 2:", thrust_2)

max_thrust = max(thrust_1, thrust_2)

m_vals = np.zeros(num)
m_vals[0] = max_fuel

thrust_vals = np.zeros(num)
thrust_vals[1] = thrust_1
thrust_vals[-2] = thrust_2

thrust_ang_values = np.zeros(num)

thrust_ang_1 = round(-np.arctan2(dv_r0, dv_theta0), 5)
thrust_ang_2 = round(-np.arctan2(dv_rf, dv_thetaf), 5)
print("thrust angle 1:", thrust_ang_1*(180/np.pi))
print("thrust angle 2:", thrust_ang_2*(180/np.pi))
thrust_ang_values[1] = thrust_ang_1
thrust_ang_values[-2] = thrust_ang_2

for i in range(1, num):
    if abs(thrust_vals[i - 1] * 1000) > 0:
        dm = abs(thrust_vals[i - 1] * 1000) / (Isp * g0) * dt_val
        m_vals[i] = m_vals[i - 1] - dm
    else:
        m_vals[i] = m_vals[i - 1]

a, e, i, RAAN, omega, theta0 = pk.ic2par(r1, v1, u)
print("\necc:", e)
print("inclination:", i)

# Compute eccentricity vector
h_vec = np.cross(r1, v1)
h_mag = np.linalg.norm(h_vec)
r1_mag = np.linalg.norm(r1)
v1_mag = np.linalg.norm(v1)
e_vec = (np.cross(v1, h_vec) / u) - (np.array(r1) / r1_mag)
e_hat = e_vec / np.linalg.norm(e_vec)

# Angle from x-axis to e_vec
phi = np.arctan2(e_hat[1], e_hat[0])  # rotation needed

# Plot transfer ellipse
theta = np.linspace(0, 2*np.pi, 50)
rt_vals = a * (1 - e**2) / (1 + e * np.cos(theta))
x_t = rt_vals * np.cos(theta)
y_t = rt_vals * np.sin(theta)

# Rotate to inertial frame using phi
transfer_x = x_t * np.cos(phi) - y_t * np.sin(phi)
transfer_y = x_t * np.sin(phi) + y_t * np.cos(phi)

r_p = a*(1-e)
periapsis = r_p * e_hat

# --- Plotting ---

th = np.linspace(0, 2 * np.pi, num=num)
orbit_0x = R_i * np.cos(th)
orbit_0y = R_i * np.sin(th)
orbit_fx = R_f * np.cos(th)
orbit_fy = R_f * np.sin(th)

folder_path = "initial_guess_lambert_plots"
os.makedirs(folder_path, exist_ok=True)
files = os.listdir(folder_path)
for file in files:
    file_path = os.path.join(folder_path, file)
    os.remove(file_path)

# Trajectory
plt.figure()
plt.plot(x_vals, y_vals, color='black', label='Trajectory')
plt.plot(orbit_0x, orbit_0y, color='red', linestyle='dotted', linewidth=2)
plt.plot(orbit_fx, orbit_fy, color='blue', linestyle='dotted', linewidth=2)
plt.plot(transfer_x, transfer_y, color='purple', linestyle='dotted', linewidth=2)
plt.scatter(r1[0], r1[1], color='green', label='start')
plt.scatter(r2[0], r2[1], color='red', label='end')
plt.scatter(periapsis[0], periapsis[1], color='purple', label='peri')
plt.xlabel('x (km)')
plt.ylabel('y (km)')
plt.axis('equal')
plt.title('Lambert Trajectory')
plt.grid(True)
file_path = os.path.join(folder_path, "trajectory.png")
plt.savefig(file_path)

# Velocity components
plt.figure()
plt.plot(t_vals, xdot_vals, label='xdot')
plt.plot(t_vals, ydot_vals, label='ydot')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/s)')
plt.title('Velocity Components Over Time')
plt.grid(True)
plt.legend()
file_path = os.path.join(folder_path, "velcoity_xy.png")
plt.savefig(file_path)

plt.figure()
plt.title('r')
plt.plot(r_vals)
file_path = os.path.join(folder_path, "radius.png")
plt.savefig(file_path)

plt.figure()
plt.title('theta')
plt.plot(theta_vals)
file_path = os.path.join(folder_path, "theta.png")
plt.savefig(file_path)

plt.figure()
plt.title('dr')
plt.plot(dr_vals)
file_path = os.path.join(folder_path, "dr.png")
plt.savefig(file_path)

plt.figure()
plt.title('dtheta')
plt.plot(dtheta_vals)
file_path = os.path.join(folder_path, "dtheta.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust')
plt.plot(thrust_vals)
file_path = os.path.join(folder_path, "thrust.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust_ang')
plt.plot(thrust_ang_values)
file_path = os.path.join(folder_path, "thrust_ang.png")
plt.savefig(file_path)

plt.figure()
plt.title('mass')
plt.plot(m_vals)
file_path = os.path.join(folder_path, "mass.png")
plt.savefig(file_path)

plt.show()

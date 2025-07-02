import numpy as np
import matplotlib.pyplot as plt
import pykep as pk
import os


def scale_factor(a):
    power = np.log10(np.abs(a))
    return 10 ** -np.floor(power)


def estimate_tof(r0, rf, u):
    r0 = np.array(r0)
    rf = np.array(rf)

    norm_r0 = np.linalg.norm(r0)
    norm_rf = np.linalg.norm(rf)
    r_avg = 0.5 * (norm_r0 + norm_rf)

    dot = np.dot(r0, rf) / (norm_r0 * norm_rf)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))

    tof = theta * np.sqrt(r_avg ** 3 / u)
    return tof  # in seconds


def compute_thrust_angles(r_vec, v_vec, dv_vec):
    # Normalize radial direction
    r_hat = r_vec / np.linalg.norm(r_vec)

    # Angular momentum vector and normal direction
    h_vec = np.cross(r_vec, v_vec)
    n_hat = h_vec / np.linalg.norm(h_vec)

    # Tangential direction (completing RTN frame)
    t_hat = np.cross(n_hat, r_hat)

    # Project dv into RTN frame
    dv_r = np.dot(dv_vec, r_hat)
    dv_t = np.dot(dv_vec, t_hat)
    dv_n = np.dot(dv_vec, n_hat)

    print("dv_vec:", dv_vec)
    print("added components:", dv_r + dv_t + dv_n)

    # Magnitude of dv
    dv_mag = np.linalg.norm(dv_vec)

    # Compute angles
    alpha = -np.arctan2(dv_r, dv_t)  # in-plane angle
    beta = -np.arcsin(np.clip(dv_n / dv_mag, -1.0, 1.0))  # out-of-plane elevation

    return alpha, beta


def propagate_orbit(r, v, mu, total_time, num_steps):
    ts = np.linspace(0, total_time, num_steps)
    positions = []
    for t in ts:
        r_t, _ = pk.propagate_lagrangian(r, v, t, mu)
        positions.append(r_t)
    return np.array(positions)


# Constants
G = 6.67430 * 10 ** (-20)  # gravitational constant (km)
M = 5.972 * 10 ** 24  # planet mass (earth)
g0 = 9.81
u = G * M
R = 6378  # earth radius (km)

# Problem Parameters
m_empty = 1000
Isp = 400  # s
h1 = 500
h2 = 1700/2

# # Initial orbit:
# a1 = R + h1
# e1 = 0
# i1 = 0            # Equatorial
# RAAN1 = 0
# arg_peri1 = 0
# true_anom1 = 0
#
# # Final orbit: elliptical, inclined
# a2 = R + h2
# e2 = 0.1
# i2 = 60           # 60 deg inclination
# RAAN2 = 30        # RAAN rotation
# arg_peri2 = 45    # Argument of periapsis
# true_anom2 = 90   # Halfway through ellipse

a1 = R + h1
e1 = 0
i1 = 28.5
RAAN1 = 0
arg_peri1 = 0
true_anom1 = 0

# Final elliptical orbit with 1200 km apogee
a2 = R + h2
e2 = (1200 - 500) / (1200 + 500)
i2 = 30
RAAN2 = 0
arg_peri2 = 0
true_anom2 = 180


# pos = [a, e, i, RAAN, omega, theta0]
p0 = [a1, e1, np.deg2rad(i1), np.deg2rad(RAAN1), np.deg2rad(arg_peri1), np.deg2rad(true_anom1)]
pf = [a2, e2, np.deg2rad(i2), np.deg2rad(RAAN2), np.deg2rad(arg_peri2), np.deg2rad(true_anom2)]

# p0 = [a1, 0.0, np.radians(28.5), 0.0, 0.0, 0.0]
# pf = [a2, 0.0, np.radians(90.0), 0.0, 0.0, np.radians(180.0)]

# xyz coordinate for position
r0, v0 = pk.par2ic(p0, u)
rf, vf = pk.par2ic(pf, u)

# Time of flight
tof = estimate_tof(r0, rf, u)

# Solve Lambert problem
lambert = pk.lambert_problem(r0, rf, tof, u)
v1 = lambert.get_v1()[0]
v2 = lambert.get_v2()[0]

# Δv vectors
dv0 = np.array(v1) - np.array(v0)
dvf = np.array(vf) - np.array(v2)

dv0_mag = np.linalg.norm(dv0)
dvf_mag = np.linalg.norm(dvf)

print("dv1:", dv0)
print("dv2:", dvf)
print("dv1 mag:", dv0_mag)
print("dv2 mag:", dvf_mag)

# Time discretization

num = int(50 * (tof / 2500))
print("\ntrajectory time estimate:", tof)
dt_val = tof / num
t_vals = np.linspace(0, tof, num)

# Propagate the trajectory
xyz_vals = []
vxyz_vals = []

for t in t_vals:
    r, v = pk.propagate_lagrangian(r0, v1, t, u)
    xyz_vals.append(r)
    vxyz_vals.append(v)

xyz_vals = np.array(xyz_vals)
vxyz_vals = np.array(vxyz_vals)

x_vals, y_vals, z_vals = np.array(xyz_vals[:, 0]), np.array(xyz_vals[:, 1]), np.array(xyz_vals[:, 2])
xdot_vals, ydot_vals, zdot_vals = np.array(vxyz_vals[:, 0]), np.array(vxyz_vals[:, 1]), np.array(vxyz_vals[:, 2])

a_trans, e_trans, i_trans, RAAN_trans, omega_trans, theta_trans = pk.ic2par(r0, v1, u)

ve = Isp * g0
m_burn_2 = m_empty * np.exp(np.abs(dvf_mag) * 1000 / ve)
prop_burn_2 = m_burn_2 - m_empty

m_burn_1 = m_burn_2 * np.exp(np.abs(dv0_mag) * 1000 / ve)
prop_burn_1 = m_burn_1 - m_burn_2
max_fuel = prop_burn_1 + prop_burn_2

print("\ndelta_v1:", dv0_mag)
print("delta_v2:", dvf_mag)
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

alpha_1, beta_1 = compute_thrust_angles(r0, v0, dv0)
alpha_2, beta_2 = compute_thrust_angles(rf, vf, dvf)

# Print in degrees
print("Initial α (deg):", np.degrees(alpha_1))
print("Initial β (deg):", np.degrees(beta_1))
print("Final α (deg):", np.degrees(alpha_2))
print("Final β (deg):", np.degrees(beta_2))

# Populate initial guess arrays
alpha_vals = np.zeros(num)
beta_vals = np.zeros(num)
alpha_vals[1] = alpha_1
beta_vals[1] = beta_1
alpha_vals[-2] = alpha_2
beta_vals[-2] = beta_2

for i in range(1, num):
    if abs(thrust_vals[i - 1] * 1000) > 0:
        dm = abs(thrust_vals[i - 1] * 1000) / (Isp * g0) * dt_val
        m_vals[i] = m_vals[i - 1] - dm
    else:
        m_vals[i] = m_vals[i - 1]

folder_path = "3D_initial_guess_lambert_plots"
os.makedirs(folder_path, exist_ok=True)
files = os.listdir(folder_path)
for file in files:
    file_path = os.path.join(folder_path, file)
    os.remove(file_path)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n_steps = 300

# initial orbit
T1 = 2 * np.pi * np.sqrt(a1 ** 3 / u)
r0_orbit = propagate_orbit(r0, v0, u, T1, n_steps)
ax.plot(r0_orbit[:, 0], r0_orbit[:, 1], r0_orbit[:, 2], label='Initial Orbit', color='blue')

# final orbit
T2 = 2 * np.pi * np.sqrt(a2 ** 3 / u)
rf_orbit = propagate_orbit(rf, vf, u, T2, n_steps)
ax.plot(rf_orbit[:, 0], rf_orbit[:, 1], rf_orbit[:, 2], label='Final Orbit', color='red')

# transfer orbit
T_transfer = 2 * np.pi * np.sqrt(a_trans ** 3 / u)
r_transfer = propagate_orbit(r0, v1, u, T_transfer, n_steps)
ax.plot(r_transfer[:, 0], r_transfer[:, 1], r_transfer[:, 2], '--', color='purple')

ax.plot(x_vals, y_vals, z_vals, label='Transfer Orbit', color='purple')

# Style
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.legend()
ax.set_title('3D Orbits')

file_path = os.path.join(folder_path, "trajectory.png")
plt.savefig(file_path)


plt.figure()
plt.subplot(3, 1, 1)
plt.title('x')
plt.plot(x_vals)

plt.subplot(3, 1, 2)
plt.title('y')
plt.plot(y_vals)

plt.subplot(3, 1, 3)
plt.title('z')
plt.plot(z_vals)

file_path = os.path.join(folder_path, "positions vals.png")
plt.savefig(file_path)


plt.figure()
plt.subplot(3, 1, 1)
plt.title('x-dot')
plt.plot(xdot_vals)

plt.subplot(3, 1, 2)
plt.title('y-dot')
plt.plot(ydot_vals)

plt.subplot(3, 1, 3)
plt.title('z-dot')
plt.plot(zdot_vals)

file_path = os.path.join(folder_path, "velocity vals.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust')
plt.plot(thrust_vals)
file_path = os.path.join(folder_path, "thrust.png")
plt.savefig(file_path)

plt.figure()
plt.title('alpha')
plt.plot(alpha_vals)
file_path = os.path.join(folder_path, "thrust_ang.png")
plt.savefig(file_path)

plt.figure()
plt.title('beta')
plt.plot(beta_vals)
file_path = os.path.join(folder_path, "thrust_ang.png")
plt.savefig(file_path)

plt.figure()
plt.title('mass')
plt.plot(m_vals)
file_path = os.path.join(folder_path, "mass.png")
plt.savefig(file_path)

plt.show()

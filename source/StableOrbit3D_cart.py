import os
import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad

import pykep as pk


### Planet centered at origin

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
    theta = np.arccos(np.clip(dot, -1.0, 1.0))  # safe acos

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
vx_vals, vy_vals, vz_vals = np.array(vxyz_vals[:, 0]), np.array(vxyz_vals[:, 1]), np.array(vxyz_vals[:, 2])

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
plt.title('vx')
plt.plot(vx_vals)

plt.subplot(3, 1, 2)
plt.title('vy')
plt.plot(vy_vals)

plt.subplot(3, 1, 3)
plt.title('vz')
plt.plot(vz_vals)

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

# ----------------------------- SIMULATION -----------------------------

recorder = csdl.Recorder(inline=True)
recorder.start()

### STATE

dt = csdl.Variable(value=dt_val)
dt.set_as_design_variable(lower=0.1, upper=dt_val * 1.5, scaler=1E-1)

x_scale = scale_factor(np.average(x_vals))
x = csdl.Variable(name='x', value=x_vals)
x.set_as_design_variable(scaler=x_scale)

y_scale = scale_factor(np.average(y_vals))
y = csdl.Variable(name='y', value=y_vals)
y.set_as_design_variable(scaler=y_scale)

z_scale = scale_factor(np.average(z_vals))
z = csdl.Variable(name='z', value=z_vals)
z.set_as_design_variable(scaler=z_scale)

vx_scale = scale_factor(np.average(vx_vals))
vx = csdl.Variable(name='vx', value=vx_vals)
vx.set_as_design_variable(scaler=vx_scale)

vy_scale = scale_factor(np.average(vy_vals))
vy = csdl.Variable(name='vy', value=vy_vals)
vy.set_as_design_variable(scaler=vy_scale)

vz_scale = scale_factor(np.average(vz_vals))
vz = csdl.Variable(name='vz', value=vz_vals)
vz.set_as_design_variable(scaler=vz_scale)

### CONTROLS
F_scale = scale_factor(max_thrust)
thrust = csdl.Variable(name='thrust', value=thrust_vals + 0.00001)
thrust.set_as_design_variable(lower=0, upper=max_thrust, scaler=F_scale)

alpha = csdl.Variable(name='alpha', value=alpha_vals)
alpha.set_as_design_variable(lower=-np.pi, upper=np.pi, scaler=1)

beta = csdl.Variable(name='beta', value=beta_vals)
beta.set_as_design_variable(lower=-np.pi/2, upper=np.pi/2, scaler=1)

m_scale = scale_factor(max_fuel)
m = csdl.Variable(name='m', value=m_vals)
m.set_as_design_variable(lower=0, scaler=m_scale)

# pos_scale =
# vel_scale =
# F_scale =
# m_scale =

print("\nx_scale:", x_scale)
print("y_scale:", y_scale)
print("z_scale:", z_scale)
print("xdot_scale:", vx_scale)
print("ydot_scale:", vy_scale)
print("zdot_scale:", vz_scale)
print("m_scale:", m_scale)
print("F_scale:", F_scale)
print("\n")

# Initial conditions
x_0 = x[0]
x_0.set_as_constraint(equals=x_vals[0])
y_0 = y[0]
y_0.set_as_constraint(equals=y_vals[0])
z_0 = z[0]
z_0.set_as_constraint(equals=z_vals[0])

vx_0 = vx[0]
vx_0.set_as_constraint(equals=vx_vals[0])
vy_0 = vy[0]
vy_0.set_as_constraint(equals=vy_vals[0])
vz_0 = vz[0]
vz_0.set_as_constraint(equals=vz_vals[0])

# Final conditions
x_f = x[-1]
x_f.set_as_constraint(equals=x_vals[-1])
y_f = y[-1]
y_f.set_as_constraint(equals=y_vals[-1])
z_f = z[-1]
z_f.set_as_constraint(equals=z_vals[-1])

vx_f = vx[-1]
vx_f.set_as_constraint(equals=vx_vals[-1])
vy_f = vy[-1]
vy_f.set_as_constraint(equals=vy_vals[-1])
vz_f = vz[-1]
vz_f.set_as_constraint(equals=vz_vals[-1])


m[-1].set_as_constraint(lower=0.01, upper=0.1)

T_total = dt * (num - 1)
T_total.set_as_constraint(upper=tof * 1.5)

x_res = csdl.Variable(value=np.zeros((num - 1)))
y_res = csdl.Variable(value=np.zeros((num - 1)))
z_res = csdl.Variable(value=np.zeros((num - 1)))
vx_res = csdl.Variable(value=np.zeros((num - 1)))
vy_res = csdl.Variable(value=np.zeros((num - 1)))
vz_res = csdl.Variable(value=np.zeros((num - 1)))

m_res = csdl.Variable(value=np.zeros((num - 1)))

for i in csdl.frange(num - 1):
    # Compute magnitudes
    r_mag = csdl.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2)
    v_mag = csdl.sqrt(vx[i] ** 2 + vy[i] ** 2 + vz[i] ** 2)
    m_total = m[i] + m_empty

    # Build CSDL vectors
    r_vec = csdl.vstack((x[i], y[i], z[i]))  # shape (3, 1)
    v_vec = csdl.vstack((vx[i], vy[i], vz[i]))  # shape (3, 1)

    # RTN unit vectors
    r_hat = csdl.div(r_vec, r_mag)
    h_vec = csdl.cross(r_vec, v_vec, axis=0)
    h_mag = csdl.sqrt(csdl.sum(h_vec ** 2))
    n_hat = csdl.div(h_vec, h_mag)
    t_hat = csdl.cross(n_hat, r_hat, axis=0)

    # Flatten for stacking into 3x3 matrix
    r_hat_flat = csdl.reshape(r_hat, shape=(3,))
    t_hat_flat = csdl.reshape(t_hat, shape=(3,))
    n_hat_flat = csdl.reshape(n_hat, shape=(3,))

    # Stack into 3x3 matrix (each column is a unit vector)
    RTN_to_XYZ = csdl.vstack((r_hat_flat, t_hat_flat, n_hat_flat))

    # Thrust direction in RTN frame (3,)
    T_r = csdl.cos(beta[i]) * csdl.sin(alpha[i])
    T_t = csdl.cos(beta[i]) * csdl.cos(alpha[i])
    T_n = csdl.sin(beta[i])
    thrust_dir_rtn = csdl.vstack((T_r, T_t, T_n))

    # Convert to inertial frame
    thrust_dir_xyz = csdl.matvec(RTN_to_XYZ, thrust_dir_rtn)

    # Compute thrust acceleration components
    Fx = thrust[i] * thrust_dir_xyz[0]
    Fy = thrust[i] * thrust_dir_xyz[1]
    Fz = thrust[i] * thrust_dir_xyz[2]

    ax = Fx * 1000 / m_total - u * x[i] / (r_mag ** 3)
    ay = Fy * 1000 / m_total - u * y[i] / (r_mag ** 3)
    az = Fz * 1000 / m_total - u * z[i] / (r_mag ** 3)

    dm = -thrust[i] * 1000 / (Isp * g0)

    # Dynamic constraints
    x_res = x_res.set(csdl.slice[i], x[i + 1] - x[i] - vx[i] * dt)
    y_res = y_res.set(csdl.slice[i], y[i + 1] - y[i] - vy[i] * dt)
    z_res = z_res.set(csdl.slice[i], z[i + 1] - z[i] - vz[i] * dt)

    vx_res = vx_res.set(csdl.slice[i], vx[i + 1] - vx[i] - ax * dt)
    vy_res = vy_res.set(csdl.slice[i], vy[i + 1] - vy[i] - ay * dt)
    vz_res = vz_res.set(csdl.slice[i], vz[i + 1] - vz[i] - az * dt)

    m_res = m_res.set(csdl.slice[i], m[i + 1] - m[i] - dm * dt)

x_res.set_as_constraint(equals=0, scaler=1)
y_res.set_as_constraint(equals=0, scaler=1)
z_res.set_as_constraint(equals=0, scaler=1)

vx_res.set_as_constraint(equals=0, scaler=1)
vy_res.set_as_constraint(equals=0, scaler=1)
vz_res.set_as_constraint(equals=0, scaler=1)

m_res.set_as_constraint(equals=0, scaler=1E3)

# takes difference between 2 adjacent values and squares
# smoothness = 1E-5 * csdl.sum((thrust_mag)**2)

# smoothness_r = (r_scale**2) * csdl.sum((r[1:] - r[:-1])**2)
thrust_ang_changes = csdl.sum((alpha[1:] - alpha[:-1]) ** 2) + csdl.sum((beta[1:] - beta[:-1]) ** 2)

mass_penalty = csdl.sum(m[0] - m[-1])
thrust_penalty = csdl.sum(thrust ** 2)

# j = mass_penalty + 0.5 * thrust_penalty
# j.set_as_objective(scaler=m_scale)

j = mass_penalty + 2 * thrust_ang_changes
j.set_as_objective(scaler=m_scale)

sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='StableOrbit', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1E-6}, turn_off_outputs=True)
optimizer = IPOPT(prob, solver_options={'max_iter': 1000, 'tol': 1E-4}, turn_off_outputs=True)
results = optimizer.solve()
optimizer.print_results()

recorder.execute()

print("Final residual norms:")
print("‣ x_res max:", np.max(np.abs(x_res.value)))
print("‣ y_res max:", np.max(np.abs(y_res.value)))
print("‣ z_res max:", np.max(np.abs(z_res.value)))
print("‣ vx_res max:", np.max(np.abs(vx_res.value)))
print("‣ vy_res max:", np.max(np.abs(vy_res.value)))
print("‣ vz_res max:", np.max(np.abs(vz_res.value)))
print("‣ m_res max:", np.max(np.abs(m_res.value)))

dt = dt.value
m = m.value
print("dt value:", dt)
print("fuel:", m[0])

x = x.value
y = y.value
z = z.value
vx = vx.value
vy = vy.value
vz = vz.value

thrust = thrust.value
alpha = alpha.value
beta = beta.value

folder_path = "3D_lambert_plots"
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

ax.plot(x, y, z, label='Transfer Orbit', color='purple')

# Style
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.legend()
ax.set_title('Orbit Transfer')

file_path = os.path.join(folder_path, "trajectory.png")
plt.savefig(file_path)


plt.figure()
plt.subplot(3, 1, 1)
plt.title('x')
plt.plot(x)

plt.subplot(3, 1, 2)
plt.title('y')
plt.plot(y)

plt.subplot(3, 1, 3)
plt.title('z')
plt.plot(z)

file_path = os.path.join(folder_path, "positions vals.png")
plt.savefig(file_path)


plt.figure()
plt.subplot(3, 1, 1)
plt.title('vx')
plt.plot(vx)

plt.subplot(3, 1, 2)
plt.title('vy')
plt.plot(vy)

plt.subplot(3, 1, 3)
plt.title('vz')
plt.plot(vz)

file_path = os.path.join(folder_path, "velocity vals.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust')
plt.plot(thrust)
file_path = os.path.join(folder_path, "thrust.png")
plt.savefig(file_path)

plt.figure()
plt.title('alpha')
plt.plot(alpha)
file_path = os.path.join(folder_path, "alpha.png")
plt.savefig(file_path)

plt.figure()
plt.title('beta')
plt.plot(beta)
file_path = os.path.join(folder_path, "beta.png")
plt.savefig(file_path)

plt.figure()
plt.title('mass')
plt.plot(m)
file_path = os.path.join(folder_path, "mass.png")
plt.savefig(file_path)

plt.show()

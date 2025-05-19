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

# ----------------------------- INITIAL GUESS -----------------------------

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

angular_vel_i = np.sqrt(u / (R_i ** 3))
angular_vel_f = np.sqrt(u / (R_f ** 3))

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

# print("Cartesian:")
# print("dv1:", dv0_xy)
# print("dv2:", dvf_xy)
# print("dv1 mag:", np.linalg.norm(dv0_xy))
# print("dv2 mag:", np.linalg.norm(dvf_xy))

print("\nDELTA V's:")
print("[radial, tang]")
print("dv1:", dv0_p)
print("dv2:", dvf_p)
print("dv1 mag:", np.linalg.norm(dv0_p))
print("dv2 mag:", np.linalg.norm(dvf_p))

# Time discretization
num = int(50 * (tof / h_time))
dt_val = tof / num
t_vals = np.linspace(0, tof, num)
print("\ntrajectory time estimate:", tof)
print("# of steps:", num)
print("dt value:", dt_val)

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
print("\nprop_burn_1:", prop_burn_1)
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

thrust_ang_vals = np.zeros(num)

thrust_ang_1 = round(-np.arctan2(dv_r0, dv_theta0), 5)
thrust_ang_2 = round(-np.arctan2(dv_rf, dv_thetaf), 5)
print("thrust angle 1:", thrust_ang_1*(180/np.pi))
print("thrust angle 2:", thrust_ang_2*(180/np.pi))
thrust_ang_vals[1] = thrust_ang_1
thrust_ang_vals[-2] = thrust_ang_2

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
plt.plot(thrust_ang_vals)
file_path = os.path.join(folder_path, "thrust_ang.png")
plt.savefig(file_path)


# ----------------------------- SIMULATION -----------------------------

### STATE

recorder = csdl.Recorder(inline=True)
recorder.start()

### STATE

dt = csdl.Variable(value=dt_val)
dt.set_as_design_variable(lower=0.1, upper=dt_val * 1.5, scaler=1E-1)

theta_scale = 1
theta = csdl.Variable(name='theta', value=theta_vals)
theta.set_as_design_variable(scaler=theta_scale)

r_scale = scale_factor(np.average(r_vals))
r = csdl.Variable(name='r', value=r_vals)
r.set_as_design_variable(lower=R, scaler=r_scale)

dr_scale = scale_factor(np.max(np.abs(dr_vals)))
dr = csdl.Variable(name='dr', value=dr_vals)
dr.set_as_design_variable(scaler=dr_scale)

dtheta_scale = scale_factor(np.average(np.abs(dtheta_vals)))
dtheta = csdl.Variable(name='dtheta', value=dtheta_vals)
dtheta.set_as_design_variable(scaler=dtheta_scale)

### CONTROLS
F_scale = scale_factor(max_thrust)
thrust = csdl.Variable(name='thrust', value=thrust_vals + 0.00001)
thrust.set_as_design_variable(lower=0, upper=max_thrust, scaler=F_scale)

thrust_ang = csdl.Variable(name='thrust_ang', value=thrust_ang_vals)
thrust_ang.set_as_design_variable(lower=0, upper=2 * np.pi, scaler=1)

m_scale = scale_factor(max_fuel)
m = csdl.Variable(name='m', value=m_vals)
m.set_as_design_variable(lower=0, scaler=m_scale)

# r_scale = 1E-3
# theta_scale = 1
# dr_scale = 1E1
# dtheta_scale = 1E3
# m_scale = 1E-1
# F_scale = 1

print("\nr_scale:", r_scale)
print("theta_scale:", theta_scale)
print("dr_scale:", dr_scale)
print("dtheta_scale:", dtheta_scale)
print("m_scale:", m_scale)
print("F_scale:", F_scale)
print("\n")

# For angular momentum
ddr = csdl.Variable(name='ddr', value=np.zeros(num))

# initial conditions
r_0 = r[0]
r_0.set_as_constraint(equals=R_i, scaler=scale_factor(R_i))
dr_0 = dr[0]
dr_0.set_as_constraint(equals=0, scaler=dr_scale)
theta_0 = theta[0]
theta_0.set_as_constraint(equals=theta_start, scaler=theta_scale)
dtheta_0 = dtheta[0]
dtheta_0.set_as_constraint(equals=angular_vel_i, scaler=scale_factor(angular_vel_i))

# final conditions
m_f = m[-1]
m_f.set_as_constraint(lower=0.01, upper=0.1, scaler=1E2)
# Ftheta_f = F_theta[-1]
# Ftheta_f.set_as_constraint(equals=0, scaler=F_scale)

r_f = r[-1]
r_f.set_as_constraint(equals=R_f, scaler=scale_factor(R_f))
dr_f = dr[-1]
dr_f.set_as_constraint(equals=0, scaler=dr_scale)
theta_f = theta[-1]
theta_f.set_as_constraint(equals=theta_end, scaler=theta_scale)
dtheta_f = dtheta[-1]
dtheta_f.set_as_constraint(equals=angular_vel_f, scaler=scale_factor(angular_vel_f))

T_total = dt * (num - 1)
T_total.set_as_constraint(upper=tof * 1.5)

r_res = csdl.Variable(value=np.zeros((num - 1)))
dr_res = csdl.Variable(value=np.zeros((num - 1)))
theta_res = csdl.Variable(value=np.zeros((num - 1)))
dtheta_res = csdl.Variable(value=np.zeros((num - 1)))
m_res = csdl.Variable(value=np.zeros((num - 1)))

for i in csdl.frange(num - 1):
    m_total = m[i] + m_empty

    Fr = thrust[i] * csdl.sin(thrust_ang[i])
    Ftheta = thrust[i] * csdl.cos(thrust_ang[i])

    ## acceleration (radial direction)
    # a_r = ddr + r*(dtheta)^2
    val = (Fr * 1000) / m_total - (u * 10 ** 9) / ((r[i] * 1000) ** 2) + (r[i] * 1000) * (dtheta[i] ** 2)

    # no radial thrust control (no F component)
    # val = - ((u * 10 ** 9) / ((r[i] * 1000) ** 2)) + (r[i] * 1000) * (dtheta[i] ** 2)
    ddr = ddr.set(csdl.slice[i], val)

    ## acceleration (theta direction)
    # a_theta = r*(ddtheta) + 2*dr*dtheta
    ddtheta = 1 / (r[i] * 1000) * ((Ftheta * 1000) / m_total - 2 * dr[i] * dtheta[i])
    # ddtheta = 1 / (r[i] * 1000) * (thrust_mag[i] / m_total - 2 * dr[i] * dtheta[i])

    # mass consumption
    Ftotal = csdl.sqrt(thrust[i] ** 2)
    dm = -(Ftotal * 1000) / (Isp * g0)

    # create the residuals for the dynamic constraints:
    r_res = r_res.set(csdl.slice[i], r[i + 1] - r[i] - dr[i] * 1e-3 * dt)
    dr_res = dr_res.set(csdl.slice[i], dr[i + 1] - dr[i] - ddr[i] * dt)
    theta_res = theta_res.set(csdl.slice[i], theta[i + 1] - theta[i] - dtheta[i] * dt)
    dtheta_res = dtheta_res.set(csdl.slice[i], dtheta[i + 1] - dtheta[i] - ddtheta * dt)
    m_res = m_res.set(csdl.slice[i], m[i + 1] - m[i] - dm * dt)

r_res.set_as_constraint(equals=0, scaler=1)
dr_res.set_as_constraint(equals=0, scaler=1)
theta_res.set_as_constraint(equals=0, scaler=1)
dtheta_res.set_as_constraint(equals=0, scaler=1)
m_res.set_as_constraint(equals=0, scaler=1E3)

# angular momentum constraint
# dh_dt_f = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]**2*ddr[-1]  #(for stable orbit angular momentum const, dh_dt = 0)
# dh_dt_f.set_as_constraint(equals=0, scaler=1E-3)

# objective function
# j = csdl.sum(csdl.sqrt(Fr ** 2 + Ftheta ** 2))

# takes difference between 2 adjacent values and squares
# smoothness = 1E-5 * csdl.sum((thrust_mag)**2)

# smoothness_r = (r_scale**2) * csdl.sum((r[1:] - r[:-1])**2)
thrust_changes = csdl.sum((thrust_ang[1:] - thrust_ang[:-1]) ** 2)

mass_penalty = csdl.sum(m[0] - m[-1])
thrust_penalty = csdl.sum(thrust ** 2)

# j = mass_penalty + 0.5 * thrust_penalty
# j.set_as_objective(scaler=m_scale)

j = mass_penalty + thrust_changes
j.set_as_objective(scaler=m_scale)

sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='StableOrbit', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1E-6}, turn_off_outputs=True)
optimizer = IPOPT(prob, solver_options={'max_iter': 1000, 'tol': 1E-4}, turn_off_outputs=True)
results = optimizer.solve()
optimizer.print_results()

recorder.execute()

print("Final residual norms:")
print("‣ r_res max:", np.max(np.abs(r_res.value)))
print("‣ dr_res max:", np.max(np.abs(dr_res.value)))
print("‣ theta_res max:", np.max(np.abs(theta_res.value)))
print("‣ dtheta_res max:", np.max(np.abs(dtheta_res.value)))
print("‣ m_res max:", np.max(np.abs(m_res.value)))

dt = dt.value
print("dt value:", dt)
r = r.value
theta = theta.value
x = r * np.cos(theta)
y = r * np.sin(theta)

th = np.linspace(0, 2 * np.pi, num=num)
orbit_0x = R_i * np.cos(th)
orbit_0y = R_i * np.sin(th)
orbit_fx = R_f * np.cos(th)
orbit_fy = R_f * np.sin(th)

dr = dr.value
dtheta = dtheta.value
m = m.value
thrust = thrust.value
thrust_ang = thrust_ang.value

Fr = thrust * np.sin(thrust_ang)
Ftheta = thrust * np.cos(thrust_ang)

thrust_ang = thrust_ang * 180 / np.pi

folder_path = "tangential_time_lambert_plots"
os.makedirs(folder_path, exist_ok=True)
files = os.listdir(folder_path)
for file in files:
    file_path = os.path.join(folder_path, file)
    os.remove(file_path)

# Plotting
plt.figure()
plt.title('radius')
plt.plot(r)
file_path = os.path.join(folder_path, "radius.png")
plt.savefig(file_path)

plt.figure()
plt.title('dr')
plt.plot(dr)
file_path = os.path.join(folder_path, "dr.png")
plt.savefig(file_path)

plt.figure()
plt.title('theta')
plt.plot(theta)
file_path = os.path.join(folder_path, "theta.png")
plt.savefig(file_path)

plt.figure()
plt.title('dtheta')
plt.plot(dtheta)
file_path = os.path.join(folder_path, "dtheta.png")
plt.savefig(file_path)

plt.figure()
plt.title('prop mass')
plt.plot(m)
file_path = os.path.join(folder_path, "prop mass.png")
plt.savefig(file_path)

plt.figure()
plt.title('trajectory')
plt.plot(x, y, color='purple', linewidth=2)
plt.plot(orbit_0x, orbit_0y, color='purple', linestyle='dotted', linewidth=2)
plt.plot(orbit_fx, orbit_fy, color='purple', linestyle='dotted', linewidth=2)
plt.xlim((-1E4, 1E4))
plt.ylim((-1E4, 1E4))
file_path = os.path.join(folder_path, "trajectory.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust')
plt.plot(thrust)
file_path = os.path.join(folder_path, "thrust.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust angle')
plt.plot(thrust_ang)
file_path = os.path.join(folder_path, "thrust_ang.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust (radial)')
plt.plot(Fr)
file_path = os.path.join(folder_path, "thrust_rad.png")
plt.savefig(file_path)

plt.figure()
plt.title('thrust (theta)')
plt.plot(Ftheta)
file_path = os.path.join(folder_path, "thrust_th.png")
plt.savefig(file_path)

plt.show()

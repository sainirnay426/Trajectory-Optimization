import os
import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl

### Planet centered at origin

def scale_factor(a):
    power = np.log10(np.abs(a))
    return 10 ** -np.floor(power)

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

if ascending:
    delta_v1 = v_periapsis - np.sqrt(u / R_i)
    delta_v2 = np.sqrt(u / R_f) - v_apoapsis
    thrust_direction = 1
else:
    delta_v1 = np.sqrt(u / R_i) - v_apoapsis
    delta_v2 = v_periapsis - np.sqrt(u / R_f)
    thrust_direction = -1

angular_vel_i = np.sqrt(u / (R_i ** 3))
angular_vel_f = np.sqrt(u / (R_f ** 3))

theta_end = np.pi
num = 50
total_time = (theta_end / np.pi) * np.pi * np.sqrt((a_transfer ** 3) / u)
print("\ntrajectory time estimate:", total_time)
dt_val = total_time / num
print("dt estimate:", dt_val)

m_empty = 1000  # kg
ve = Isp * g0
m_burn_2 = m_empty * np.exp(np.abs(delta_v2)*1000 / ve)
prop_burn_2 = m_burn_2 - m_empty

m_burn_1 = m_burn_2 * np.exp(np.abs(delta_v1)*1000 / ve)
prop_burn_1 = m_burn_1 - m_burn_2
max_fuel = prop_burn_1 + prop_burn_2

print("\ndelta_v1:", delta_v1)
print("delta_v2:", delta_v2)
print("prop_burn_1:", prop_burn_1)
print("prop_burn_2:", prop_burn_2)
print("max_fuel:", max_fuel)

burn_time = dt_val

thrust_1 = ((prop_burn_1 * ve) / burn_time)/1000
thrust_2 = ((prop_burn_2 * ve) / burn_time)/1000
print("thrust 1:", thrust_1)
print("thrust 2:", thrust_2)

max_thrust = max(thrust_1, thrust_2)

recorder = csdl.Recorder(inline=True)
recorder.start()

scale_factor(max(R_i, R_f))
# r_scale = 1E-3
# theta_scale = 1
# dr_scale = 1E1
# dtheta_scale = 1E3
# m_scale = 1E-1
# F_scale = 1

### STATE
dt = csdl.Variable(value=dt_val)
dt.set_as_design_variable(lower=0.1, upper=dt_val * 1.5, scaler=1E-1)

theta_values = np.linspace(0, theta_end, num=num)
theta_scale = 1
theta = csdl.Variable(name='theta', value=theta_values)
theta.set_as_design_variable(scaler=theta_scale)

r_values = np.zeros(num)
r_values = p / (1 + ecc * np.cos(theta_values))
r_scale = scale_factor(np.average(r_values))
r = csdl.Variable(name='r', value=r_values)
r.set_as_design_variable(lower=R, scaler=r_scale)

dr_values = np.zeros(num)
dr_values = np.sqrt(u / p) * ecc * np.sin(theta_values)
dr_scale = scale_factor(np.max(dr_values))
dr = csdl.Variable(name='dr', value=dr_values)
dr.set_as_design_variable(scaler=dr_scale)

v_theta = np.sqrt(u / p) * (1 + ecc * np.cos(theta_values))  # Tangential velocity in elliptical orbit
dtheta_values = v_theta / r_values  # Convert to angular velocity
dtheta_scale = scale_factor(np.average(dtheta_values))
dtheta = csdl.Variable(name='dtheta', value=dtheta_values)
dtheta.set_as_design_variable(scaler=dtheta_scale)

### CONTROLS

thrust_values = np.zeros(num)
thrust_sign = 1 if ascending else -1
thrust_values[1] = thrust_sign * thrust_1
thrust_values[-2] = thrust_sign * thrust_2

F_scale = scale_factor(max_thrust)
F_theta = csdl.Variable(name='F_theta', value=thrust_values + 0.00001)
# F_theta.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=F_scale)
F_theta.set_as_design_variable(lower=-1E-2, upper=2 * max_thrust, scaler=F_scale)

m_values = np.zeros(num)
m_scale = scale_factor(max_fuel)
m_values[0] = max_fuel

for i in range(1, num):
    if abs(thrust_values[i - 1] * 1000) > 0:
        dm = abs(thrust_values[i - 1]) / (Isp * g0) * dt_val
        m_values[i] = m_values[i - 1] - dm
    else:
        m_values[i] = m_values[i - 1]
m = csdl.Variable(name='m', value=m_values)
m.set_as_design_variable(lower=0, scaler=m_scale)

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
theta_0.set_as_constraint(equals=0, scaler=theta_scale)
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
theta_f.set_as_constraint(equals=np.pi, scaler=theta_scale)
dtheta_f = dtheta[-1]
dtheta_f.set_as_constraint(equals=angular_vel_f, scaler=scale_factor(angular_vel_f))

T_total = dt * (num - 1)
T_total.set_as_constraint(upper=total_time * 1.5)

r_res = csdl.Variable(value=np.zeros((num - 1)))
dr_res = csdl.Variable(value=np.zeros((num - 1)))
theta_res = csdl.Variable(value=np.zeros((num - 1)))
dtheta_res = csdl.Variable(value=np.zeros((num - 1)))
m_res = csdl.Variable(value=np.zeros((num - 1)))

for i in csdl.frange(num - 1):
    m_total = m[i] + m_empty
    ## acceleration (radial direction)
    # a_r = ddr + r*(dtheta)^2
    val = - (u * 10 ** 9) / ((r[i] * 1000) ** 2) + (r[i] * 1000) * (dtheta[i] ** 2)

    # no radial thrust control (no F component)
    # val = - ((u * 10 ** 9) / ((r[i] * 1000) ** 2)) + (r[i] * 1000) * (dtheta[i] ** 2)
    ddr = ddr.set(csdl.slice[i], val)

    ## acceleration (theta direction)
    # a_theta = r*(ddtheta) + 2*dr*dtheta
    ddtheta = 1 / (r[i] * 1000) * ((F_theta[i] * 1000) / m_total - 2 * dr[i] * dtheta[i])
    # ddtheta = 1 / (r[i] * 1000) * (thrust_mag[i] / m_total - 2 * dr[i] * dtheta[i])

    # mass consumption
    Ftotal = csdl.sqrt(F_theta[i] ** 2)
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
# m_res.set_as_constraint(equals=0, scaler=m_scale)
m_res.set_as_constraint(equals=0, scaler=1E6)

# angular momentum constraint
# dh_dt_f = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]**2*ddr[-1]  #(for stable orbit angular momentum const, dh_dt = 0)
# dh_dt_f.set_as_constraint(equals=0, scaler=1E-3)

# objective function
# j = csdl.sum(csdl.sqrt(Fr ** 2 + Ftheta ** 2))

# takes difference between 2 adjacent values and squares
# smoothness = 1E-5 * csdl.sum((thrust_mag)**2)

# smoothness_r = (r_scale**2) * csdl.sum((r[1:] - r[:-1])**2)
# thrust_changes = (1E1) * (csdl.sum((F_theta[1:] - F_theta[:-1])**2))
mass_penalty = csdl.sum(m[0] - m[-1])
thrust_penalty = csdl.sum(F_theta ** 2)

j = mass_penalty
j.set_as_objective(scaler=m_scale)

# j = (F_scale**2) * thrust_penalty
# j.set_as_objective(scaler=1)

# j = 1.5 * (F_scale**2) * thrust_penalty + 1E-2 * mass_penalty * m_scale
# j.set_as_objective(scaler=1)

sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='StableOrbit', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 2000, 'ftol': 1E-4}, turn_off_outputs=True)
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

dr = dr.value
dtheta = dtheta.value
m = m.value
F_theta = F_theta.value

# thrust_angle = thrust_angle.value
# thrust_mag = thrust_mag.value

folder_path = "tangential_time_single_result_plots"
os.makedirs(folder_path, exist_ok=True)
files = os.listdir(folder_path)
for file in files:
    file_path = os.path.join(folder_path, file)
    os.remove(file_path)

# Plotting
plt.figure(0)
plt.title('radius')
plt.plot(r)
file_path = os.path.join(folder_path, "radius.png")
plt.savefig(file_path)

plt.figure(1)
plt.title('dr')
plt.plot(dr)
file_path = os.path.join(folder_path, "dr.png")
plt.savefig(file_path)

plt.figure(2)
plt.title('theta')
plt.plot(theta)
file_path = os.path.join(folder_path, "theta.png")
plt.savefig(file_path)

plt.figure(3)
plt.title('dtheta')
plt.plot(dtheta)
file_path = os.path.join(folder_path, "dtheta.png")
plt.savefig(file_path)

plt.figure(4)
plt.title('prop mass')
plt.plot(m)
file_path = os.path.join(folder_path, "prop mass.png")
plt.savefig(file_path)

plt.figure(5)
plt.title('trajectory')
plt.plot(x, y, color='purple', linewidth=2)
file_path = os.path.join(folder_path, "trajectory.png")
plt.savefig(file_path)

plt.figure(6)
plt.title('thrust (theta)')
plt.plot(F_theta)
file_path = os.path.join(folder_path, "thrust (theta).png")
plt.savefig(file_path)

# plt.figure(7)
# plt.title('thrust angle')
# plt.plot(thrust_angle)

plt.show()

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
theta_end = np.pi

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

recorder = csdl.Recorder(inline=True)
recorder.start()

### STATE
ecc = e_fit
p = p_fit

dt = csdl.Variable(value=dt_val)
dt.set_as_design_variable(lower=0.1, upper=dt_val * 1.5, scaler=1E-1)

theta_values = np.linspace(theta_start, theta_end, num=num)
theta_scale = 1
theta = csdl.Variable(name='theta', value=theta_values)
theta.set_as_design_variable(scaler=theta_scale)

r_values = np.zeros(num)
r_values = p / (1 + ecc * np.cos(theta_values))
r_scale = scale_factor(np.average(r_values))
r = csdl.Variable(name='r', value=r_values)
r.set_as_design_variable(lower=R, scaler=r_scale)

dr_values = np.zeros(num)
dr_values = np.sqrt((u * 10 ** 9) / (p * 10 ** 3)) * ecc * np.sin(theta_values)
dr_scale = scale_factor(np.max(np.abs(dr_values)))
dr = csdl.Variable(name='dr', value=dr_values)
dr.set_as_design_variable(scaler=dr_scale)

v_theta = np.sqrt(u / p) * (1 + ecc * np.cos(theta_values))  # Tangential velocity in elliptical orbit
dtheta_values = v_theta / r_values  # Convert to angular velocity
dtheta_scale = scale_factor(np.average(np.abs(dtheta_values)))
dtheta = csdl.Variable(name='dtheta', value=dtheta_values)
dtheta.set_as_design_variable(scaler=dtheta_scale)

### CONTROLS
thrust_values = np.zeros(num)
thrust_sign = 1 if ascending else -1
thrust_values[1] = thrust_sign * thrust_1
thrust_values[-2] = thrust_sign * thrust_2

F_scale = scale_factor(max_thrust)
F_theta = csdl.Variable(name='F_theta', value=thrust_values + 0.00001)
F_theta.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=F_scale)

thrust_ang_values = np.linspace(0, 0, num=num)
thrust_ang = csdl.Variable(name='thrust_ang', value=thrust_ang_values)
thrust_ang.set_as_design_variable(lower=0, upper=2*np.pi, scaler=1)

# if ascending:
#     F_theta.set_as_design_variable(lower=-1E-2, upper=2 * max_thrust, scaler=F_scale)
# else:
#     F_theta.set_as_design_variable(lower=-2 * max_thrust, upper=1E-2, scaler=F_scale)

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
m_res.set_as_constraint(equals=0, scaler=1E3)

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

# j = mass_penalty + 0.5 * thrust_penalty
# j.set_as_objective(scaler=m_scale)

j = mass_penalty
j.set_as_objective(scaler=m_scale)

sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='StableOrbit', simulator=sim)
optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1E-6}, turn_off_outputs=True)
# optimizer = IPOPT(prob, solver_options={'max_iter': 1000, 'tol': 1E-4}, turn_off_outputs=True)
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

th = np.linspace(0, 2*np.pi, num=num)
orbit_0x = R_i * np.cos(th)
orbit_0y = R_i * np.sin(th)
orbit_fx = R_f * np.cos(th)
orbit_fy = R_f * np.sin(th)

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
plt.plot(orbit_0x, orbit_0y, color='purple', linestyle='dotted', linewidth=2)
plt.plot(orbit_fx, orbit_fy, color='purple', linestyle='dotted', linewidth=2)
plt.xlim((-1E4, 1E4))
plt.ylim((-1E4, 1E4))
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

import math

import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl

### Planet centered at origin

# Constants
G = 6.67430 * 10**(-20) * 10**9  #gravitational constant (m)
M = 5.972 * 10**24  #planet mass (earth)
Isp = 400 #s
g0 = 9.81
u = G*M
R = 6378 * 10**3 #earth radius (m)
a_i = 300 * 10**3 #initial altitude (m)
a_f = 700 * 10**3 #final altitude (m)

R_i = R + a_i
R_f = R + a_f

max_thrust = 5000
m_empty = 1000 #kg
max_fuel = 200 #kg

dt = 30
num = 150
recorder = csdl.Recorder(inline=True)
recorder.start()

r_scale = 1E-6
dr_scale = 1E-2
theta_scale = 1
dtheta_scale = 1E3
m_scale = 1E-2
F_scale = 1E-3

# State variables
r = csdl.Variable(name='r', value=np.linspace(R_i, R_f, num=num))
r.set_as_design_variable(lower=R, scaler=r_scale)

# Simple linear then constant then linear model
dr_values = np.zeros(num)

# Divide into three segments
first_seg = num // 3
second_seg = 2 * num // 3

# Estimate a reasonable maximum radial velocity (negative for descending)
max_dr = -500  # m/s (adjust as needed)

# First segment: linearly increase speed (downward)
dr_values[:first_seg] = np.linspace(0, max_dr, first_seg)

# Middle segment: maintain constant speed
dr_values[first_seg:second_seg] = max_dr

# Last segment: linearly decrease back to zero
dr_values[second_seg:] = np.linspace(max_dr, 0, num - second_seg)

dr = csdl.Variable(name='dr', value=dr_values)
dr.set_as_design_variable(scaler=dr_scale)

theta = csdl.Variable(name='theta', value=np.linspace(0, 3*np.pi/2, num=num))
theta.set_as_design_variable(scaler=theta_scale)

circular_i = np.sqrt(u/(R_i**3))
circular_f = np.sqrt(u/(R_f**3))
dtheta = csdl.Variable(name='dtheta', value=np.linspace(circular_i, circular_f, num=num))
dtheta.set_as_design_variable(scaler=dtheta_scale)

m = csdl.Variable(name='m', value=np.linspace(max_fuel, 0, num=num))
m.set_as_design_variable(lower=0, scaler=m_scale)

# controls
thrust_mag = csdl.Variable(name='thrust_mag', value=np.linspace(max_thrust/2, 0, num=num))
thrust_mag.set_as_design_variable(lower=0, upper=max_thrust, scaler=F_scale)
thrust_angle = csdl.Variable(name='thrust_angle', value=np.zeros(num))
thrust_angle.set_as_design_variable(lower=-np.pi, upper=np.pi, scaler=theta_scale)

# Fr = csdl.Variable(name='Fr', value=np.linspace(-max_thrust, 0, num=num))
# Fr.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=F_scale)
# Ftheta= csdl.Variable(name='Ftheta', value= np.linspace(max_thrust, 0, num=num))
# Ftheta.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=F_scale)

# For angular momentum
ddr = csdl.Variable(name='ddr', value=np.zeros((num)))

# initial conditions
r_0 = r[0]
r_0.set_as_constraint(equals=(R_i), scaler=r_scale)
dr_0 = dr[0]
dr_0.set_as_constraint(equals=0, scaler=dr_scale)
theta_0 = theta[0]
theta_0.set_as_constraint(equals=0, scaler=theta_scale)
dtheta_0 = dtheta[0]
dtheta_0.set_as_constraint(equals=circular_i, scaler=dtheta_scale)

# final conditions
m_f = m[-1]
m_f.set_as_constraint(equals=0, scaler=m_scale)
thrust_ang_f = thrust_angle[-1]
thrust_ang_f.set_as_constraint(equals=0, scaler=theta_scale)

Fr_f = thrust_mag[-1] * csdl.cos(theta[-1] + thrust_angle[-1])
Fr_f.set_as_constraint(equals=0, scaler=F_scale)
Ftheta_f = thrust_mag[-1] * csdl.sin(theta[-1] + thrust_angle[-1])
Ftheta_f.set_as_constraint(equals=0, scaler=F_scale)

r_f = r[-1]
r_f.set_as_constraint(equals=R_f, scaler=r_scale)
dr_f = dr[-1]
dr_f.set_as_constraint(equals=0, scaler=dr_scale)
theta_f = theta[-1]
theta_f.set_as_constraint(equals=3*np.pi/2, scaler=theta_scale)
dtheta_f = dtheta[-1]
dtheta_f.set_as_constraint(equals=circular_f, scaler=dtheta_scale)

# h = r[-1]^2*dtheta[-1]
# dh_dt = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]^2*ddr  (for stable orbit angular momentum const, dh_dt = 0)

for i in range(num - 1):
    Fr = thrust_mag[i] * csdl.cos(theta[i] + thrust_angle[i])
    Ftheta = thrust_mag[i] * csdl.sin(theta[i] + thrust_angle[i])

    m_total = m[i] + m_empty
    ## acceleration (radial direction)
    # a_r = ddr + r*(dtheta)^2
    val = Fr / m_total - ((u) / ((r[i]) ** 2)) + (r[i]) * (dtheta[i] ** 2)
    ddr = ddr.set(csdl.slice[i], val)

    ## acceleration (theta direction)
    # a_theta = r*(ddtheta) + 2*dr*dtheta`
    ddtheta = 1 / (r[i]) * (Ftheta / m_total - 2 * dr[i] * dtheta[i])

    # mass consumption
    Ftotal = csdl.sqrt(Fr ** 2 + Ftheta ** 2)
    dm = -Ftotal / (Isp * g0)

    # create the residuals for the dynamic constraints:
    r0 = r[i + 1] - r[i] - (dr[i]) * dt
    r0.set_as_constraint(equals=0, scaler=r_scale)

    r1 = dr[i + 1] - dr[i] - ddr[i] * dt
    r1.set_as_constraint(equals=0, scaler=dr_scale)

    r2 = theta[i + 1] - theta[i] - dtheta[i] * dt
    r2.set_as_constraint(equals=0, scaler=theta_scale)

    r3 = dtheta[i + 1] - dtheta[i] - ddtheta * dt
    r3.set_as_constraint(equals=0, scaler=dtheta_scale)

    r4 = m[i + 1] - m[i] - dm * dt
    r4.set_as_constraint(equals=0, scaler=m_scale)


# angular momentum constraint
# dh_dt_f = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]**2*ddr[-1]  #(for stable orbit angular momentum const, dh_dt = 0)
# dh_dt_f.set_as_constraint(equals=0, scaler=1E-3)

# objective function
# j = csdl.sum(csdl.sqrt(Fr ** 2 + Ftheta ** 2))

# takes difference between 2 adjacent values and squares
#smoothness = 0.01 * csdl.sum((thrust_angle[1:]-thrust_angle[:-1])**2)
j = csdl.sum(m[0]-m[-1]) #+ smoothness
j.set_as_objective(scaler=(m_scale))

sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='StableOrbit', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 2000, 'ftol': 1E-4}, turn_off_outputs=True)
optimizer = IPOPT(prob, solver_options={'max_iter': 500, 'tol': 1E-4}, turn_off_outputs=True)
results = optimizer.solve()
optimizer.print_results()

recorder.execute()

r = r.value
theta = theta.value
x = r*np.cos(theta)
y = r*np.sin(theta)

dr = dr.value
dtheta = dtheta.value
m = m.value
Fr = Fr.value
Ftheta = Ftheta.value

thrust_angle = thrust_angle.value
thrust_mag = thrust_mag.value

Fr = thrust_mag * np.cos(theta + thrust_angle)
Ftheta = thrust_mag * np.sin(theta + thrust_angle)

# Plotting
plt.figure(0)
plt.title('radius')
plt.plot(r)

plt.figure(1)
plt.title('dr')
plt.plot(dr)

plt.figure(2)
plt.title('theta')
plt.plot(theta)

plt.figure(3)
plt.title('dtheta')
plt.plot(dtheta)

plt.figure(4)
plt.title('prop mass')
plt.plot(m)

plt.figure(5)
plt.title('trajectory')
plt.plot(x, y, color='purple', linewidth=2)

plt.figure(6)
plt.title('thrust')
plt.plot(thrust_mag)

plt.figure(7)
plt.title('thrust angle')
plt.plot(thrust_angle)


plt.show()

# plt.figure(8)
# plt.title('ddr')
# plt.plot(ddr)
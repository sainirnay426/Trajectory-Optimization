import math

import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl

### Planet centered at origin

# Constants
G = 6.67430 * 10**(-20)  #gravitational constant (km)
M = 5.972 * 10**24  #planet mass (earth)
Isp = 400 #s
g0 = 9.81
u = G*M
R = 6378 #earth radius (km)
a_i = 700 #initial altitude (km)
a_f = 50 #final altitude (km)
v_circular = (u/(R+a_f))**0.5
max_thrust = 5 #kN
m_empty = 1000 #kg

dt = 60
num = 50

recorder = csdl.Recorder(inline=True)
recorder.start()

r_scale = 1E-3
dr_scale = 1E-2
theta_scale = 1
dtheta_scale = 1E3
m_scale = 1E-4
F_scale = 1

# State variables
r = csdl.Variable(name='r', value=np.linspace(R+a_i, R+a_f, num=num))
r.set_as_design_variable(lower=R, scaler=r_scale)
dr = csdl.Variable(name='dr', value=np.zeros((num)))
dr.set_as_design_variable(scaler=dr_scale)
theta = csdl.Variable(name='theta', value=np.linspace(0, 3*np.pi/2, num=num))
theta.set_as_design_variable(scaler=theta_scale)
dtheta = csdl.Variable(name='dtheta', value=np.zeros((num)))
dtheta.set_as_design_variable(scaler=dtheta_scale)
m = csdl.Variable(name='m', value=np.zeros((num)))
m.set_as_design_variable(lower=0, scaler=m_scale)

# controls
Fr = csdl.Variable(name='Fr', value=np.zeros(num)+ 0.001)
Fr.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=F_scale)
Ftheta= csdl.Variable(name='Ftheta', value= np.zeros(num) + 0.001)
Ftheta.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=F_scale)

# For angular momentum
ddr = csdl.Variable(name='ddr', value=np.zeros((num)))

# initial conditions
r_0 = r[0]
r_0.set_as_constraint(equals=(R+a_i), scaler=r_scale)
dr_0 = dr[0]
dr_0.set_as_constraint(equals=0, scaler=dr_scale)
theta_0 = theta[0]
theta_0.set_as_constraint(equals=0, scaler=theta_scale)
dtheta_0 = dtheta[0]
dtheta_0.set_as_constraint(equals=0, scaler=dtheta_scale)

# final conditions
m_f = m[-1]
m_f.set_as_constraint(equals=0, scaler=m_scale)
Fr_f = Fr[-1]
Fr_f.set_as_constraint(equals=0, scaler=F_scale)
Ftheta_f = Ftheta[-1]
Ftheta_f.set_as_constraint(equals=0, scaler=F_scale)

r_f = r[-1]
r_f.set_as_constraint(equals=(R+a_f), scaler=r_scale)
dr_f = dr[-1]
dr_f.set_as_constraint(equals=0, scaler=dr_scale)
theta_f = theta[-1]
theta_f.set_as_constraint(equals=3*np.pi/2, scaler=theta_scale)
dtheta_f = dtheta[-1]
dtheta_f.set_as_constraint(equals=(v_circular/(R+a_f)), scaler=dtheta_scale)

# h = r[-1]^2*dtheta[-1]
# dh_dt = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]^2*ddr  (for stable orbit angular momentum const, dh_dt = 0)

r_res = csdl.Variable(value=np.zeros((num - 1)))
dr_res = csdl.Variable(value=np.zeros((num - 1)))
theta_res = csdl.Variable(value=np.zeros((num - 1)))
dtheta_res = csdl.Variable(value=np.zeros((num - 1)))
m_res = csdl.Variable(value=np.zeros((num - 1)))

for i in csdl.frange(num - 1):
    m_total = m[i] + m_empty
    ## acceleration (radial direction)
    # a_r = ddr + r*(dtheta)^2
    val = (Fr[i]*1000) / m_total - (u*10**9) / ((r[i] * 1000) ** 2) + (r[i] * 1000) * (dtheta[i] ** 2)
    ddr = ddr.set(csdl.slice[i], val)

    ## acceleration (theta direction)
    # a_theta = r*(ddtheta) + 2*dr*dtheta`
    ddtheta = 1 / (r[i] * 1000) * ((Ftheta[i]*1000) / m_total - 2 * dr[i] * dtheta[i])

    # mass consumption
    Ftotal = csdl.sqrt(Fr[i] ** 2 + Ftheta[i] ** 2)
    dm = -(Ftotal*1000) / (Isp * g0)

    # create the residuals for the dynamic constraints:
    r_res = r_res.set(csdl.slice[i], r[i + 1] - r[i] - dr[i] * 1e-3 * dt)
    dr_res = dr_res.set(csdl.slice[i], dr[i + 1] - dr[i] - ddr[i] * dt)
    theta_res = theta_res.set(csdl.slice[i], theta[i + 1] - theta[i] - dtheta[i] * dt)
    dtheta_res = dtheta_res.set(csdl.slice[i], dtheta[i + 1] - dtheta[i] - ddtheta * dt)
    m_res = m_res.set(csdl.slice[i], m[i + 1] - m[i] - dm * dt)

r_res.set_as_constraint(equals=0, scaler=r_scale)
dr_res.set_as_constraint(equals=0, scaler=dr_scale)
theta_res.set_as_constraint(equals=0, scaler=theta_scale)
dtheta_res.set_as_constraint(equals=0, scaler=dtheta_scale)
m_res.set_as_constraint(equals=0, scaler=m_scale)

# angular momentum constraint
# dh_dt_f = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]**2*ddr[-1]  #(for stable orbit angular momentum const, dh_dt = 0)
# dh_dt_f.set_as_constraint(equals=0, scaler=1E-3)

# objective function
j = csdl.sum(m**2)
j.set_as_objective(scaler=(m_scale))

sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='StableOrbit', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 2000, 'ftol': 1E-4}, turn_off_outputs=True)
optimizer = IPOPT(prob, solver_options={'max_iter': 2000, 'tol': 1E-4}, turn_off_outputs=True)
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
plt.title('thrust (radial)')
plt.plot(Fr)

plt.figure(6)
plt.title('thrust (theta)')
plt.plot(Ftheta)

plt.figure(7)
plt.title('trajectory')
plt.plot(x, y, color='purple', linewidth=2)
plt.show()

# plt.figure(8)
# plt.title('ddr')
# plt.plot(ddr)
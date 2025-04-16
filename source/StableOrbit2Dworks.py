import math

import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl

### Planet centered at origin

# Constants
G = 6.67430 * 10**(-11)  #gravitational constant
M = 5.972 * 10**24  #planet mass (earth)
Isp = 100 #s
g0 = 9.81
u = G*M
R = 6378137 #earth radius
a_i = 700000 #initial altitude
a_f = 50000 #final altitude
v_circular = (u/(R+a_f))**0.5
max_thrust = 20000

dt = 40 # 0.04
num = 100 # 400
m_empty = 1000 #kg
recorder = csdl.Recorder(inline=True)
recorder.start()

# State variables
r = csdl.Variable(name='r', value=np.linspace(R+a_i, R+a_f, num=num))
r.set_as_design_variable(lower=R, scaler=1E-5)
dr = csdl.Variable(name='dr', value=np.zeros(num))
dr.set_as_design_variable(scaler=1E-3)
theta = csdl.Variable(name='theta', value=np.zeros(num))
theta.set_as_design_variable(scaler=1)
dtheta = csdl.Variable(name='dtheta', value=np.zeros(num))
dtheta.set_as_design_variable(scaler=1E3)
m = csdl.Variable(name='m', value=np.zeros(num))
m.set_as_design_variable(lower=0, scaler=1E-5)

# controls
Fr = csdl.Variable(name='Fr', value=np.zeros(num)+ 0.001)
Fr.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=1E-4)
Ftheta= csdl.Variable(name='Ftheta', value= np.zeros(num) + 0.001)
Ftheta.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=1E-4)

# For angular momentum
ddr = csdl.Variable(name='ddr', value=np.zeros(num))

# initial conditions
r_0 = r[0]
r_0.set_as_constraint(equals=(R+a_i), scaler=1E-5)
dr_0 = dr[0]
dr_0.set_as_constraint(equals=0, scaler=1E-2)
theta_0 = theta[0]
theta_0.set_as_constraint(equals=0, scaler=1)
dtheta_0 = dtheta[0]
dtheta_0.set_as_constraint(equals=0, scaler=1E3)

# final conditions
m_f = m[-1]
m_f.set_as_constraint(equals=0, scaler=1E-3)
Fr_f = Fr[-1]
Fr_f.set_as_constraint(equals=0, scaler=1E-4)
Ftheta_f = Ftheta[-1]
Ftheta_f.set_as_constraint(equals=0, scaler=1E-4)

r_f = r[-1]
r_f.set_as_constraint(equals=(R+a_f), scaler=1E-5)
dr_f = dr[-1]
dr_f.set_as_constraint(equals=0, scaler=1E-3)
theta_f = theta[-1]
theta_f.set_as_constraint(equals=3*np.pi/2, scaler=1)
dtheta_f = dtheta[-1]
dtheta_f.set_as_constraint(equals=(v_circular/(R+a_f)), scaler=1E3)

# h = r[-1]^2*dtheta[-1]
# dh_dt = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]^2*ddr  (for stable orbit angular momentum const, dh_dt = 0)

for i in range(num - 1):
    m_total = m[i] + m_empty
    ## acceleration (radial direction)
    # a_r = ddr + r*(dtheta)^2
    val = Fr[i] / m_total - u / (r[i] ** 2) + r[i] * (dtheta[i] ** 2)
    ddr = ddr.set(csdl.slice[i], val)

    ## acceleration (theta direction)
    # a_theta = r*(ddtheta) + 2*dr*dtheta`
    ddtheta = 1/r[i] * (Ftheta[i]/m_total - 2*dr[i]*dtheta[i])

    # mass consumption
    Ftotal = csdl.sqrt(Fr[i]**2 + Ftheta[i]**2)
    dm = -Ftotal/(Isp*g0)

    # create the residuals for the dynamic constraints:
    r0 = r[i + 1] - r[i] - dr[i] * dt
    r0.set_as_constraint(equals=0, scaler=1E-5)

    r1 = dr[i + 1] - dr[i] - ddr[i] * dt
    r1.set_as_constraint(equals=0, scaler=1E-3)

    r2 = theta[i + 1] - theta[i] - dtheta[i] * dt
    r2.set_as_constraint(equals=0, scaler=1)

    r3 = dtheta[i + 1] - dtheta[i] - ddtheta * dt
    r3.set_as_constraint(equals=0, scaler=1E3)

    r4 = m[i + 1] - m[i] - dm * dt
    r4.set_as_constraint(equals=0, scaler=1E-3)

# angular momentum constraint
# dh_dt_f = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]**2*ddr[-1]  #(for stable orbit angular momentum const, dh_dt = 0)
# dh_dt_f.set_as_constraint(equals=0, scaler=1E-3)

# objective function
j = csdl.sum(m**2)
j.set_as_objective(scaler=1E-6)

sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='StableOrbit', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 2000, 'ftol': 1E-4}, turn_off_outputs=True)
optimizer = IPOPT(prob, solver_options={'max_iter': 1500, 'tol': 1E-6}, turn_off_outputs=True)
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
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
Isp = 300 #s
g0 = 9.81
u = G*M
R = 6378 #earth radius (km)
a_i = 1000 #initial altitude (km)
a_f = 300 #final altitude (km)
v_circular = (u/(R+a_f))**0.5
max_thrust = 10000
m_empty = 1000 #kg

dt = 40 # 0.04
num = 100 # 400
recorder = csdl.Recorder(inline=True)
recorder.start()

x_scale = 1E-3
dx_scale = 1E-2
y_scale = 1E-3
dy_scale = 1E-2
m_scale = 1E-3
F_scale = 1E-3

# State variables
r_init = np.linspace(R+a_i, R+a_f, num=num)
theta_init = np.linspace(0, 3*np.pi/2, num=num)
x_init = r_init*np.cos(theta_init) + 0.001
y_init = r_init*np.sin(theta_init) + 0.001

x = csdl.Variable(name='x', value=x_init)
x.set_as_design_variable(lower=R, scaler=x_scale)
dx = csdl.Variable(name='dx', value=np.zeros(num))
dx.set_as_design_variable(scaler=dx_scale)
y = csdl.Variable(name='y', value=y_init)
y.set_as_design_variable(lower=R, scaler=y_scale)
dy = csdl.Variable(name='dy', value=np.zeros(num))
dy.set_as_design_variable(scaler=dy_scale)
m = csdl.Variable(name='m', value=np.zeros((num)))
m.set_as_design_variable(lower=0, scaler=m_scale)

# controls
Fx = csdl.Variable(name='Fr', value=np.zeros(num)+ 0.001)
Fx.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=F_scale)
Fy= csdl.Variable(name='Ftheta', value= np.zeros(num) + 0.001)
Fy.set_as_design_variable(lower=-max_thrust, upper=max_thrust, scaler=F_scale)

# initial conditions
x_0 = x[0]
x_0.set_as_constraint(equals=(R+a_i), scaler=x_scale)
y_0 = y[0]
y_0.set_as_constraint(equals=0, scaler=y_scale)
dx_0 = dx[0]
dx_0.set_as_constraint(equals=0, scaler=dx_scale)
dy_0 = dy[0]
dy_0.set_as_constraint(equals=0, scaler=dy_scale)

# final conditions
m_f = m[-1]
m_f.set_as_constraint(equals=0, scaler=m_scale)
Fx_f = Fx[-1]
Fx_f.set_as_constraint(equals=0, scaler=F_scale)
Fy_f = Fy[-1]
Fy_f.set_as_constraint(equals=0, scaler=F_scale)

x_f = x[-1]
x_f.set_as_constraint(equals=0, scaler=x_scale)
y_f = y[-1]
y_f.set_as_constraint(equals=(R+a_f), scaler=y_scale)

# no radial velocity (dot product of velocity and position = 0)
vr_f = x[-1]*dx[-1] + y[-1]*dy[-1]
vr_f.set_as_constraint(equals=0, scaler=x_scale*dx_scale)

# tangential velocity is constant
vt_f = (dx[-1]**2 + dy[-1]**2) ** 0.5 - v_circular
vt_f.set_as_constraint(equals=0, scaler=dx_scale)

# theta_f = theta[-1]
# dtheta_f.set_as_constraint(equals=(v_circular/(R+a_f)), scaler=1E3)

# h = r[-1]^2*dtheta[-1]
# dh_dt = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]^2*ddr  (for stable orbit angular momentum const, dh_dt = 0)

for i in range(num - 1):
    m_total = m[i] + m_empty
    rad = ((x[i] * 1000) ** 2 + (y[i] * 1000) ** 2) ** 0.5
    ## acceleration (x-direction)
    # ddx = Fx[i] / m_total - ((u * 10 ** 9) / (rad ** 2)) * (x[i] / rad)
    ddx = Fx[i] / m_total - ((u * 10 ** 9) / (rad ** 3)) * (x[i] * 1000)

    ## acceleration (y-direction)
    # ddy = Fy[i] / m_total - ((u * 10 ** 9) / (rad ** 2)) * (y[i] / rad)
    ddy = Fy[i] / m_total - ((u * 10 ** 9) / (rad ** 3)) * (y[i] * 1000)

    # mass consumption
    Ftotal = csdl.sqrt(Fx[i] ** 2 + Fy[i] ** 2)
    dm = -Ftotal / (Isp * g0)

    # create the residuals for the dynamic constraints:
    r0 = x[i + 1] - x[i] - (dx[i] * (1E-3)) * dt
    r0.set_as_constraint(equals=0, scaler=x_scale)

    r1 = dx[i + 1] - dx[i] - ddx * dt
    r1.set_as_constraint(equals=0, scaler=dx_scale)

    r2 = y[i + 1] - y[i] - (dy[i] * (1E-3)) * dt
    r2.set_as_constraint(equals=0, scaler=y_scale)

    r3 = dy[i + 1] - dy[i] - ddy * dt
    r3.set_as_constraint(equals=0, scaler=dy_scale)

    r4 = m[i + 1] - m[i] - dm * dt
    r4.set_as_constraint(equals=0, scaler=m_scale)

# angular momentum constraint
# dh_dt_f = 2*r[-1]*dr[-1]*dtheta[-1] + r[-1]**2*ddr[-1]  #(for stable orbit angular momentum const, dh_dt = 0)
# dh_dt_f.set_as_constraint(equals=0, scaler=1E-3)

# objective function
j = csdl.sum(m**2)
j.set_as_objective(scaler=m_scale**2)

sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='StableOrbit', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 2000, 'ftol': 1E-4}, turn_off_outputs=True)
optimizer = IPOPT(prob, solver_options={'max_iter': 1000, 'tol': 1E-6}, turn_off_outputs=True)
results = optimizer.solve()
optimizer.print_results()

recorder.execute()

x = x.value
y = y.value
dx = dx.value
dy = dy.value
m = m.value
Fx = Fx.value
Fy = Fy.value

r = (x**2 + y**2) ** 0.5

# Plotting
plt.figure(0)
plt.title('radius')
plt.plot(r)

plt.figure(1)
plt.title('prop mass')
plt.plot(m)

plt.figure(2)
plt.title('thrust (x)')
plt.plot(Fx)

plt.figure(3)
plt.title('thrust (y)')
plt.plot(Fy)

plt.figure(4)
plt.title('trajectory')
plt.plot(x, y, color='purple', linewidth=2)
plt.show()
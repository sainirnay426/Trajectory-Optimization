import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl

g = 9.81 # m/s^2
m = 100000 # kg
min_thrust = 880*1000 # N
max_thrust = 1*2210*1000 #kN
length = 50 # m
I = (1/12) * m * length**2
deg_to_rad = 0.01745329
max_gimble = 20 * deg_to_rad
min_gimble = -max_gimble

dt = 0.5 # 0.04
num = 40 # 400


recorder = csdl.Recorder(inline=True)
recorder.start()



x = csdl.Variable(name='x', value=np.zeros((num)))
x.set_as_design_variable(scaler=1)
dx = csdl.Variable(name='dx', value=np.zeros((num)))
dx.set_as_design_variable(scaler=1E-1)
y = csdl.Variable(name='y', value=np.zeros((num)))
y.set_as_design_variable(scaler=1E-3)
dy = csdl.Variable(name='dy', value=np.zeros((num)))
dy.set_as_design_variable(scaler=1E-1)
theta = csdl.Variable(name='theta', value=np.zeros((num)))
theta.set_as_design_variable(scaler=1E1)
dtheta = csdl.Variable(name='dtheta', value=np.zeros((num)))
dtheta.set_as_design_variable(scaler=1E1)
thrust = csdl.Variable(name='thrust', value=np.ones((num)) * 0.5)
thrust.set_as_design_variable(lower=0.4, upper=1, scaler=1)
beta = csdl.Variable(name='beta', value=np.zeros((num)))
beta.set_as_design_variable(lower=min_gimble, upper=max_gimble, scaler=1)

# initial conditions
x_0 = x[0]
x_0.set_as_constraint(equals=0, scaler=1)
dx_0 = dx[0]
dx_0.set_as_constraint(equals=0, scaler=1)
y_0 = y[0]
y_0.set_as_constraint(equals=1000, scaler=1E-3)
dy_0 = dy[0]
dy_0.set_as_constraint(equals=-80, scaler=1E-2)
theta_0 = theta[0]
theta_0.set_as_constraint(equals=-np.pi/2, scaler=1)
dtheta_0 = dtheta[0]
dtheta_0.set_as_constraint(equals=0, scaler=1)

# final conditions
x_f = x[-1]
x_f.set_as_constraint(equals=0, scaler=1)
dx_f = dx[-1]
dx_f.set_as_constraint(equals=0, scaler=1)
y_f = y[-1]
y_f.set_as_constraint(equals=0, scaler=1)
dy_f = dy[-1]
dy_f.set_as_constraint(equals=0, scaler=1)
theta_f = theta[-1]
theta_f.set_as_constraint(equals=0, scaler=1)
dtheta_f = dtheta[-1]
dtheta_f.set_as_constraint(equals=0, scaler=1)

for i in range(num - 1):
    # horizontal force
    F_x = max_thrust*thrust[i]*csdl.sin(beta[i] + theta[i])
    ddx = (F_x)/m

    # vertical force
    F_y = max_thrust*thrust[i]*csdl.cos(beta[i] + theta[i])
    ddy = (F_y)/m - g

    # torque
    T = -length/2*max_thrust*thrust[i]*csdl.sin(beta[i])
    ddtheta = T/I

    # create the residuals for the dynamic constraints:
    r0 = x[i+1] - x[i] - dx[i]*dt
    r0.set_as_constraint(equals=0, scaler=1E1)

    r1 = dx[i+1] - dx[i] - ddx*dt
    r1.set_as_constraint(equals=0, scaler=1E1)

    r2 = y[i+1] - y[i] - dy[i]*dt
    r2.set_as_constraint(equals=0, scaler=1E1)

    r3 = dy[i+1] - dy[i] - ddy*dt
    r3.set_as_constraint(equals=0, scaler=1E1)

    r4 = theta[i+1] - theta[i] - dtheta[i]*dt
    r4.set_as_constraint(equals=0, scaler=1E1)

    r5 = dtheta[i+1] - dtheta[i] - ddtheta*dt
    r5.set_as_constraint(equals=0, scaler=1E1)



# objective function
j = csdl.sum(thrust**2) + csdl.sum(beta**2) + 2*csdl.sum(dtheta**2)
j.set_as_objective(scaler=1E-2)


recorder.stop()


sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(problem_name='starship', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 2000, 'ftol': 1E-4}, turn_off_outputs=True)
optimizer = IPOPT(prob, solver_options={'max_iter': 1000, 'tol': 1E-6}, turn_off_outputs=True)
results = optimizer.solve()
optimizer.print_results()

recorder.execute()


# x = x.value
# y = y.value
# theta = theta.value

# plt.plot(x, y, color='purple', linewidth=2)

# for i in range(num):
#     marker = mpl.markers.MarkerStyle(marker='_')
#     marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(np.rad2deg(theta[i] + np.pi/2)))
#     plt.scatter(x[i], y[i], marker=marker, s=600, linewidth=4, color='black')


# plt.xlim([-250,250])
# plt.show()
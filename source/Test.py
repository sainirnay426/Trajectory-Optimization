import math

import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, COBYLA, NelderMead, PySLSQP
import matplotlib.pyplot as plt
import matplotlib as mpl

 # Constants
G = 6.67430 * 10**(-20)  #gravitational constant (km)
M = 5.972 * 10**24  #planet mass (earth)
Isp = 400 #s
g0 = 9.81
u = G*M
R = 6378 #earth radius (km)
a_i = 700 #initial altitude (km)
a_f = 300 #final altitude (km)

R_i = R + a_i
R_f = R + a_f

max_thrust = 5000
m_empty = 1000 #kg
max_fuel = 500 #kg

dt = 30
num = 200

circular_i = np.sqrt(u/(R_i**3))
circular_f = np.sqrt(u/(R_f**3))


dtheta = np.linspace(circular_i, circular_f, num=num)
theta = np.linspace(0, 3*np.pi/2, num=num)
r = np.linspace(R+a_i, R+a_f, num=num)

x = r*np.cos(theta)
y = r*np.sin(theta)

plt.figure(0)
plt.title('trajectory')
plt.plot(x, y, color='purple', linewidth=2)

plt.figure(1)
plt.title('dtheta')
plt.plot(dtheta)
plt.show()


















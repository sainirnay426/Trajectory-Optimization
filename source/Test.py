import pykep as pk
import numpy as np
import matplotlib.pyplot as plt

def scale_factor(a):
    power = np.log10(np.abs(a))
    return 10 ** -np.floor(power)


def llh_2_xyz(r_pos):
    lat = np.deg2rad(r_pos[0])
    lon = np.deg2rad(r_pos[1])
    r_mag = 6378 + r_pos[2]  # total radial distance
    x = r_mag * np.cos(lat) * np.cos(lon)
    y = r_mag * np.cos(lat) * np.sin(lon)
    z = r_mag * np.sin(lat)
    return [x, y, z]


# Constants
G = 6.67430 * 10 ** (-20)  # gravitational constant (km)
M = 5.972 * 10 ** 24  # planet mass (earth)
g0 = 9.81
u = G * M
R = 6378  # earth radius (km)

# Problem Parameters
m_empty = 1000
Isp = 400  # s

# pos = [lat, lon, h]
r0 = [0, 0, 300]
rf = [90, 0, 800]

xyz0 = llh_2_xyz(r0)  # Equatorial LEO
xyzf = llh_2_xyz(rf)  # Polar high orbit

print(xyz0)
print(xyzf)
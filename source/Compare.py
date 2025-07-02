import pykep as pk
import numpy as np
import matplotlib.pyplot as plt

ellipse = np.load('rand_initial_guess_debug.npz')
lambert = np.load('lambert_initial_guess_debug.npz')

# List of variable names to compare
var_names = ['r', 'theta', 'dr', 'dtheta', 'thrust', 'thrust_ang', 'm']

for var in var_names:
    arr1 = ellipse[var]
    arr2 = lambert[var]
    diff = arr1 - arr2

    print(f"\nVariable: {var}")
    print(f"  Shape: {arr1.shape}")
    print(f"  Data type: {arr1.dtype}")
    print(f"  Max abs difference: {np.max(np.abs(diff))}")
    print(f"  Mean abs difference: {np.mean(np.abs(diff))}")
    print(f"  Endpoints (ellipse, lambert): ({arr1[0]}, {arr2[0]}) ... ({arr1[-1]}, {arr2[-1]})")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(arr1, label='Ellipse')
    plt.plot(arr2, label='Lambert', linestyle='--')
    plt.title(f'{var} (Ellipse vs Lambert)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(diff, label=f'{var} difference')
    plt.title(f'{var} difference (Ellipse - Lambert)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

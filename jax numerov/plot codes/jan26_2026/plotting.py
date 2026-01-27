import matplotlib.pyplot as plt
import numpy as np

# --- Data from your log ---
r_box = np.array([
    1.0000, 1.1837, 1.3673, 1.5510, 1.7347, 1.9184, 2.1020, 2.2857, 2.4694, 2.6531, 
    2.8367, 3.0204, 3.2041, 3.3878, 3.5714, 3.7551, 3.9388, 4.1224, 4.3061, 4.4898, 
    4.6735, 4.8571, 5.0408, 5.2245, 5.4082, 5.5918, 5.7755, 5.9592, 6.1429, 6.3265, 
    6.5102, 6.6939, 6.8776, 7.0612, 7.2449, 7.4286, 7.6122, 7.7959, 7.9796, 8.1633, 
    8.3469, 8.5306, 8.7143, 8.8980, 9.0816, 9.2653, 9.4490, 9.6327, 9.8163, 10.0000
])

U_values = np.array([
    0.98082, 0.70320, 0.50460, 0.29345, 0.09564, -0.06807, -0.18451, -0.26708, -0.32662, -0.37014,
    -0.40228, -0.42622, -0.44417, -0.45770, -0.46792, -0.47567, -0.48155, -0.48602, -0.48941, -0.49198,
    -0.49393, -0.49541, -0.49652, -0.49736, -0.49799, -0.49846, -0.49882, -0.49908, -0.49927, -0.49941,
    -0.49951, -0.49959, -0.49964, -0.49967, -0.49970, -0.49971, -0.49972, -0.49972, -0.49972, -0.49972,
    -0.49971, -0.49971, -0.49970, -0.49969, -0.49968, -0.49967, -0.49966, -0.49965, -0.49963, -0.49962
])

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Main data curve
plt.plot(r_box, U_values, 'o-', color='blue', linewidth=2, markersize=5, label='Internal Energy U')

# Ionization Threshold Reference
plt.axvline(x=1.835, color='red', linestyle='--', alpha=0.7, label='Ionization Radius (~1.835)')
plt.axhline(y=0.0, color='black', linewidth=1, linestyle='-', alpha=0.5)
plt.axhline(y=-0.5, color='green', linestyle=':', label='Hydrogen Ground State (-0.5)')

# Formatting
plt.title('Equation of State: Compressed Hydrogen Atom', fontsize=14)
plt.xlabel('Box Radius $r_{box}$ (Bohr)', fontsize=12)
plt.ylabel('Internal Energy U (Hartree)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=11)

# Annotate regions
plt.text(1.2, 0.2, 'Free Regime\n(Compression)', fontsize=10, color='blue', ha='center')
plt.text(4.0, -0.2, 'Bound Regime\n(Relaxation)', fontsize=10, color='blue', ha='center')

plt.tight_layout()
plt.savefig("equation_of_state.png")
print("Plot saved as equation_of_state.png")
plt.show()
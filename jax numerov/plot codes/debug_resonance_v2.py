# housekeeping
import sys
import os

# 1. Fix the Import Path
# Get the folder where THIS script is (.../jax numerov/plot codes)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to (.../jax numerov)
jax_numerov_dir = os.path.dirname(script_dir)
# Now point to core files (.../jax numerov/core files)
core_path = os.path.join(jax_numerov_dir, 'core files')

if core_path not in sys.path:
    sys.path.append(core_path)
    
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from hybrid_functions_corrected import single_phase_shift_solver, mu_solver_corrected
from hybrid_functions_uncorrected import fd_occup_solver, bounded_states_solver

# Use a box size just past the transition
r_box = 1.835  # Slightly smaller to push resonance clearly into positive E
r_start = 1e-5
N_points = 500 # Higher resolution
Z = 1
l = 0
T = 1e-3

# Scan Energy from 0.001 to 2.0 Ha
energies_1 = jnp.linspace(0.001, 10.0, 1000)

# Vectorize the solver
vmap_delta = jax.vmap(lambda E: single_phase_shift_solver(E, r_box, r_start, N_points, l, Z))
delta_values = vmap_delta(energies_1)

energies_2 = jnp.linspace(0.001, 10, 100)

e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)

mu = mu_solver_corrected(e,m,d,r_box, r_start, N_points, Z, T)

vmap_occup = jax.vmap(fd_occup_solver, in_axes = (0, None, None))

occup = vmap_occup(energies_2, mu, T)

# Unwrap the phase shift to make it continuous (removes pi to -pi jumps)
delta_unwrapped = jnp.unwrap(delta_values)

# Calculate derivative numerically for visualization
d_delta = jnp.gradient(delta_unwrapped, energies_1)

# --- PLOT ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot 1: Raw Phase Shift
ax1.plot(energies_1, delta_unwrapped, 'b-', linewidth=2, label=r"$\delta(E)$ (Unwrapped)")
ax1.set_ylabel(r"Phase Shift $\delta$ (radians)")
ax1.set_title(f"Phase Shift vs Energy (r_box={r_box})")
ax1.grid(True)
ax1.legend()

# Plot 2: Derivative (Density of States Correction)
ax2.plot(energies_1, 2/jnp.pi * d_delta, 'r-', label="DOS Correction")
ax2.set_ylabel("Correction Density")
ax2.set_xlabel("Energy (Ha)")
ax2.axhline(0, color='k', linewidth=0.5)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(energies_2, occup)
plt.show()

print("code ended running")
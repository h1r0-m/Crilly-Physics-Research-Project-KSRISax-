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
from hybrid_functions_corrected import single_phase_shift_solver

# Use a box size just past the transition
r_box = 1.80  # Slightly smaller to push resonance clearly into positive E
r_start = 1e-5
N_points = 500 # Higher resolution
Z = 1
l = 0

# Scan Energy from 0.001 to 2.0 Ha
energies = jnp.linspace(0.001, 2.0, 1000)

# Vectorize the solver
vmap_delta = jax.vmap(lambda E: single_phase_shift_solver(E, r_box, r_start, N_points, l, Z))
delta_values = vmap_delta(energies)

# Unwrap the phase shift to make it continuous (removes pi to -pi jumps)
delta_unwrapped = jnp.unwrap(delta_values)

# Calculate derivative numerically for visualization
d_delta = jnp.gradient(delta_unwrapped, energies)

# --- PLOT ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot 1: Raw Phase Shift
ax1.plot(energies, delta_unwrapped, 'b-', linewidth=2, label=r"$\delta(E)$ (Unwrapped)")
ax1.set_ylabel(r"Phase Shift $\delta$ (radians)")
ax1.set_title(f"Phase Shift vs Energy (r_box={r_box})")
ax1.grid(True)
ax1.legend()

# Plot 2: Derivative (Density of States Correction)
ax2.plot(energies, 2/jnp.pi * d_delta, 'r-', label="DOS Correction")
ax2.set_ylabel("Correction Density")
ax2.set_xlabel("Energy (Ha)")
ax2.axhline(0, color='k', linewidth=0.5)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
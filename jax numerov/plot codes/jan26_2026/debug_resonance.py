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
from hybrid_functions_corrected import dos_correction_solver

# Use a box size just past the transition (where U jumped to 0.28)
r_box = 1.82 
r_start = 1e-5
N_points = 300
Z = 1
l = 0

# 1. Create a dense Log-Space Energy Grid to hunt for the spike
# From 1e-8 to 1e-1 (scanning the region just above zero)
energies = jnp.logspace(-8, -1, 500) 

# 2. Calculate DOS correction for each energy
# We map the function to work over the array
vmap_dos = jax.vmap(lambda E: dos_correction_solver(E, r_box, r_start, N_points, l, Z))
dos_values = vmap_dos(energies)

# 3. Plot
plt.figure(figsize=(10, 6))
plt.plot(energies, dos_values, '.-', label=f"DOS Correction (r={r_box})")
plt.xscale('log')
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel("Energy (Ha)")
plt.ylabel("Correction density (2/pi * d_delta/dE)")
plt.title(f"Hunting for the Resonance at r_box={r_box}")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

# 4. Print stats
print(f"Max DOS value: {jnp.max(dos_values)}")
print(f"Energy at Max: {energies[jnp.argmax(dos_values)]}")
print(f"Integral approx (Riemann): {jnp.trapz(dos_values, energies)}")
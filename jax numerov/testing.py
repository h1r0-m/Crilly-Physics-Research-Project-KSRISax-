# housekeeping
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from njax_functions import numerov_solver, U_solver
from cont_functions import U_solver_cont, bounded_states_solver, mu_solver

from rich.traceback import install
install()

Z_array = jnp.arange(1, 118)
r_box = 30
r_start = 1e-5
T_ha = 1e-3
N_points = 300

mu_array_3 = []
mu_list = [] 

for Z in Z_array:
    e, m, d = bounded_states_solver(r_box, r_start, N_points, Z)
    mu = mu_solver(e, m, d, r_box, Z, T_ha)
    mu_list.append(mu)

Z_np = np.array(Z_array)
mu_np = np.array(mu_list)

strict_increases = np.where(np.diff(mu_np) > 0)[0] + 1

peaks_indices = argrelextrema(mu_np, np.greater)[0]

plt.figure(figsize=(12, 7))

plt.plot(Z_np, mu_np, linestyle="-", color="red", label="Hybrid Model", alpha=0.7)
plt.scatter(Z_np, mu_np, color="red", s=10, marker=".")

if len(peaks_indices) > 0:
    plt.scatter(Z_np[peaks_indices], mu_np[peaks_indices], 
                color="blue", s=100, zorder=3, marker="o", facecolors='none', 
                label="Bumps")

    for idx in peaks_indices:
        plt.annotate(f"Z={Z_np[idx]}", 
                     (Z_np[idx], mu_np[idx]), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center', 
                     fontsize=9,
                     color="blue")

plt.xlabel("Z (Atomic Number)")
plt.ylabel("Chemical Potential (Ha)")
plt.title(f"Chemical Potential vs Z (T={T_ha*315775:.0f} K, r_box={r_box}, N={N_points})")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)

main_folder = "plots"
sub_folder = "jan22_2025_new"
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(script_dir, main_folder, sub_folder)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

plt.savefig(os.path.join(plot_dir, f"muVsZ_marked_temp{T_ha}.png"), dpi=300)
plt.close()

print("mu vs Z with peaks marked done.")
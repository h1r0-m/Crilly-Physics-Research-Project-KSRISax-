# housekeeping
import sys
import os
from datetime import datetime

# 1. Fix the Import Path
# Get the folder where THIS script is (.../jax numerov/plot codes/jan27_2026)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up TWO levels to reach the root (.../jax numerov)
jax_numerov_dir = os.path.dirname(os.path.dirname(script_dir))

# Now point to core files (.../jax numerov/core files)
core_path = os.path.join(jax_numerov_dir, 'core files')

if core_path not in sys.path:
    sys.path.append(core_path)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
from hybrid_functions_corrected import mu_solver_corrected, U_solver_corrected, bounded_states_solver
from njax_functions import numerov_solver

from rich.traceback import install
install()

def main():
    # U vs smaller r_box with higher N_points

    r_box_array = jnp.logspace(-1,2, num = 10)
    Z = 30
    r_start = 1e-5
    N_points = 300
    T_ha = 1e-3
    U_array_corrected = []

    count = 0

    for r in r_box_array:

        energies, psi = numerov_solver(r, r_start, N_points, 0, Z)

        e,m,d = bounded_states_solver(r, r_start, N_points, Z)

        mu = mu_solver_corrected(e,m,d,r, r_start, N_points, Z, T_ha)

        U = U_solver_corrected(e,m,d, r, r_start, N_points, mu, T_ha, Z)

        U_array_corrected.append(U)
        
        count += 1

        print(f"count = {count}, ground energy = {energies[0]}, r_box = {r}")

    plt.figure(figsize=(10,6))
    plt.plot(r_box_array, jnp.array(U_array_corrected) * 27.2114, linestyle = "-", marker = "x", color = "blue", label = "Non-hybrid")
    plt.xlabel("Box Radius (Ha)")
    plt.ylabel("Internal Energy (eV)")
    plt.title(f"Internal Energy vs r_box (T={T_ha*315775:.0f} K, N_points={N_points}, Z={Z})")
    plt.legend()
    plt.grid(True)

    # saving
    main_folder = "plots"
    sub_folder = "feb15_2026"

    # 1. Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_dir = os.path.join(jax_numerov_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    file_name = os.path.basename(__file__)
    temp_filename = os.path.join(plot_dir, f"{file_name}_T{T_ha}_N{N_points}_Z{Z}_{timestamp}.png")
    plt.savefig(temp_filename, dpi=300)

    print(f"--- {file_name} has completed running ---")

main()
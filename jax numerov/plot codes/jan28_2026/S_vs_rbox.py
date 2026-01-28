# housekeeping
import sys
import os

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
from hybrid_functions_corrected import mu_solver_corrected, U_solver_corrected, S_solver_corrected
from hybrid_functions_uncorrected import bounded_states_solver, mu_solver_uncorrected, U_solver_uncorrected

from rich.traceback import install
install()

def main():
    # U vs smaller r_box with higher N_points

    r_box_array = jnp.logspace(-1, 2, num = 30)
    Z = 1
    r_start = 1e-5
    N_points = 300
    T = 1e-3
    S_array_corrected = []

    count = 0

    for r in r_box_array:
        e,m,d = bounded_states_solver(r, r_start, N_points, Z)

        mu = mu_solver_corrected(e,m,d,r, r_start, N_points, Z,T)
        S = S_solver_corrected(e,m,d,r, r_start, N_points, mu, T, Z)
        S_array_corrected.append(S)
        
        count += 1

        print(f"count = {count}")

    plt.figure(figsize=(10,6))
    plt.plot(r_box_array, jnp.array(S_array_corrected), linestyle="-", marker="x", color="green", label= "Corrected")
    plt.xlabel("Box Radius (Ha)")
    plt.xscale('log')
    plt.ylabel("Entropy (Ha)")
    plt.title(f"Entropy vs r_box (T={T*315775:.0f} K, N_points={N_points}, Z={Z})")
    plt.legend()
    plt.grid(True)

    # saving
    main_folder = "plots"
    sub_folder = "jan28_2026"

    plot_dir = os.path.join(jax_numerov_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    file_name = os.path.basename(__file__)

    temp_filename = os.path.join(plot_dir, f"{file_name}_T{T}_N{N_points}_Z{Z}.png")
    plt.savefig(temp_filename, dpi=300)

    print(f"--- {file_name} has completed running ---")

main()
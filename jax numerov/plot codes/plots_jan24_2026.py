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
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
from hybrid_functions_corrected import mu_solver_corrected, U_solver_corrected
from hybrid_functions_uncorrected import bounded_states_solver, mu_solver_uncorrected, U_solver_uncorrected

from rich.traceback import install
install()

def main():

    # U vs smaller r_box with higher N_points

    r_box_array = jnp.logspace(-3, 1, num = 100)
    Z = 1
    r_start = 1e-5
    N_points = 300
    T_ha = 1e-3
    U_array_uncorrected = []
    U_array_corrected = []

    for r in r_box_array:
        e,m,d = bounded_states_solver(r, r_start, N_points, Z)

        mu = mu_solver_uncorrected(e, m, d, r, Z, T_ha)
        U = U_solver_uncorrected(r, e, m, d, mu, T_ha)
        U_array_uncorrected.append(U)
        
        mu = mu_solver_corrected(e,m,d,r, r_start, N_points, Z,T_ha)
        U = U_solver_corrected(e,m,d,r, r_start, N_points, mu, T_ha, Z)
        U_array_corrected.append(U)


    plt.figure(figsize=(10,6))
    plt.plot(r_box_array, jnp.array(U_array_uncorrected) * 27.2114, linestyle="-", marker="x", color="red", label= "Uncorrected")
    plt.plot(r_box_array, jnp.array(U_array_corrected) * 27.2114, linestyle="-", marker="x", color="red", label= "Corrected")
    plt.xlabel("Box Radius (Ha)")
    plt.xscale('log')
    plt.ylabel("Internal Energy (eV)")
    plt.title(f"Energy vs Box Size (T={T_ha*315775:.0f} K, N_points={N_points})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"UvsRbox_temp{T_ha}_N{N_points}.png"), dpi=300)
    plt.close()

    print("U vs smaller r_box done")

    # saving
    main_folder = "plots"
    sub_folder = "jan24_2025"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    plot_dir = os.path.join(script_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    temp_filename = os.path.join(plot_dir, f"Uvsrbox_corrected_N{N_points}.png")
    plt.savefig(temp_filename, dpi=300)

    file_name = os.path.basename(__file__)
    print(f"--- {file_name} has completed running ---")

main()
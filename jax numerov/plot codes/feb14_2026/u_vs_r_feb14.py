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
from hybrid_functions_corrected import mu_solver_corrected, U_solver_corrected
from hybrid_functions_uncorrected import bounded_states_solver, mu_solver_uncorrected, U_solver_uncorrected
from njax_functions import U_solver_nonhybrid

from rich.traceback import install
install()

def main():
    # U vs smaller r_box with higher N_points

    r_box_array = jnp.linspace(1, 2.2, 30)
    Z = 1
    r_start = 1e-5
    N_points = 300
    T_ha = 1e-3
    U_array_numerov = []
    U_array_uncorrected = []
    U_array_corrected = []

    count = 0

    for r in r_box_array:

        U = U_solver_nonhybrid(T_ha, r, r_start, N_points, Z, 300)
        U_array_numerov.append(U)

        e,m,d = bounded_states_solver(r, r_start, N_points, Z)

        mu = mu_solver_uncorrected(e, m, d, r, Z, T_ha)
        U = U_solver_uncorrected(r, e, m, d, mu, T_ha)
        U_array_uncorrected.append(U)
        
        mu = mu_solver_corrected(e,m,d,r, r_start, N_points, Z,T_ha)
        U = U_solver_corrected(e,m,d,r, r_start, N_points, mu, T_ha, Z)
        U_array_corrected.append(U)

        count += 1

        print(f"count = {count}")

    plt.figure(figsize=(10,6))

    # 1. Base layer: Solid and thick
    plt.plot(r_box_array, jnp.array(U_array_numerov) * 27.2114, 
             linestyle="-", marker = "x", linewidth=3, color="blue", alpha=0.6, 
             label="Non-hybrid (Numerov)")

    # 2. Comparison layer: Dotted/Dashed to show the "overlap"
    plt.plot(r_box_array, jnp.array(U_array_corrected) * 27.2114, 
             linestyle="--", linewidth=1.5, color="green", 
             marker="x", markersize=4, label="Corrected (Hybrid)")

    # 3. Reference layer: Different color/style to show the gap
    plt.plot(r_box_array, jnp.array(U_array_uncorrected) * 27.2114, 
             linestyle=":", color="red", marker="x", 
             label="Uncorrected (Hybrid)")

    # plt.axvline(x=1.82799, color="black", linestyle="-.", alpha=0.5, label="E_1 = 0 Reference")
    plt.xlabel("Box Radius (Ha)")
    plt.ylabel("Internal Energy (eV)")
    plt.title(f"Internal Energy vs r_box (T={T_ha*315775:.0f} K, N_points={N_points}, Z={Z})")
    plt.legend()
    plt.grid(True)

    # saving
    main_folder = "plots"
    sub_folder = "feb14_2026"

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
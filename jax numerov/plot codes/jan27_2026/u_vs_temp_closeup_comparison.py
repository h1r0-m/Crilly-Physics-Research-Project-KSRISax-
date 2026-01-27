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
from hybrid_functions_corrected import thermo_solver_corrected, mu_solver_corrected
from hybrid_functions_uncorrected import bounded_states_solver, mu_solver_uncorrected, U_solver_uncorrected

from rich.traceback import install
install()

def main():
    # U vs smaller r_box with higher N_points

    temp_array = jnp.linspace(0,0.3, num = 30)
    r_box = 30
    Z = 1
    r_start = 1e-5
    N_points = 300
    U_array_uncorrected = []
    U_array_corrected = []

    count = 0

    e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)

    for temp in temp_array:
        mu = mu_solver_uncorrected(e,m,d,r_box, Z, temp)
        U = U_solver_uncorrected(r_box, e,m,d,mu, temp)
        U_array_uncorrected.append(U)

        mu = mu_solver_corrected(e,m,d, r_box, r_start, N_points, Z, temp)
        _, U = thermo_solver_corrected(e,m,d, r_box, r_start, N_points, mu, temp, Z)
        U_array_corrected.append(U)

        count += 1

        print(f"count = {count}")

    temps_k = temp_array * 315775
    theory_x = jnp.linspace(jnp.min(temps_k), jnp.max(temps_k), 100)
    theory_y = 1.5 * 8.617e-5 * theory_x 

    plt.figure(figsize=(10,6))
    plt.plot(temp_array * 315775, jnp.array(U_array_uncorrected) * 27.2114, linestyle="-", marker="x", color="red", label= "Uncorrected")
    plt.plot(temp_array * 315775, jnp.array(U_array_corrected) * 27.2114, linestyle="-", marker="x", color="green", label= "Corrected")
    plt.plot(theory_x, theory_y, linestyle="--", color="blue", label="Ideal Gas (3/2 kT)")
    plt.plot
    plt.xlabel("Temperature (K)")
    plt.ylabel("Internal Energy (eV)")
    plt.title(f"Internal Energy vs Temperature (Closeup) (r_box={r_box}, N_points={N_points}, Z={Z})")
    plt.legend()
    plt.grid(True)

    # saving
    main_folder = "plots"
    sub_folder = "jan27_2026"

    plot_dir = os.path.join(jax_numerov_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    temp_filename = os.path.join(plot_dir, f"u_vs_temp_closeup_comparison_rbox{r_box}_N{N_points}_Z{Z}.png")
    plt.savefig(temp_filename, dpi=300)

    file_name = os.path.basename(__file__)
    print(f"--- {file_name} has completed running ---")

main()
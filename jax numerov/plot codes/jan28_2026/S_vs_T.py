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
from hybrid_functions_optimized import bounded_states_solver, mu_solver_corrected, U_solver_corrected, S_solver_corrected

from rich.traceback import install
install()

def sackur_tetrode_theoretical(T_K, r_box):
    """
    Calculates theoretical Entropy for 1 electron in a box using Sackur-Tetrode.
    Units: Atomic Units (Ha for Energy, but T is input in Kelvin).
    """
    # --- Constants ---
    # Conversion: Kelvin to Hartree (1 Ha = 3.1577e5 K)
    kelvin_to_ha = 1 / 315775.0
    
    # --- Convert T to Atomic Units ---
    T_ha = T_K * kelvin_to_ha
    
    # Avoid log(0)
    T_ha = np.maximum(T_ha, 1e-10)
    
    # --- System Parameters ---
    N = 1.0  # Single electron
    V = (4/3) * np.pi * (r_box ** 3) # Volume of sphere
    
    # --- Sackur-Tetrode Equation (Atomic Units) ---
    # S/kB = N * [ ln(V/N * (T/2pi)^1.5) + 5/2 + ln(2) ]
    
    # 1. Translational Contribution
    term_trans = np.log( (V/N) * (T_ha / (2 * np.pi))**1.5 )
    
    # 2. Constant Factor (5/2 for ideal gas)
    term_const = 2.5
    
    # 3. Spin Contribution (ln(2) for doublet)
    term_spin = np.log(2)
    
    S_theoretical = N * (term_trans + term_const + term_spin)
    
    return S_theoretical

def main():
    # U vs smaller r_box with higher N_points

    T_array = jnp.logspace(-1, 2, num = 30)
    Z = 1
    r_box = 30
    r_start = 1e-5
    N_points = 300
    S_array_corrected = []

    count = 0

    e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)

    for T in T_array:
        mu = mu_solver_corrected(e,m,d,r_box, r_start, N_points, Z,T)
        S = S_solver_corrected(e,m,d,r_box, r_start, N_points, mu, T, Z)
        S_array_corrected.append(S)
        
        count += 1

        print(f"count = {count}")


    t_theory_range = np.linspace(1000, 3.2e7, 200) 
    s_theory_smooth = sackur_tetrode_theoretical(t_theory_range, r_box)

    plt.figure(figsize=(10,6))
    plt.plot(T_array * 315775, jnp.array(S_array_corrected), linestyle="-", marker="x", color="red", label= "Hybrid (Corrected)")
    plt.plot(t_theory_range, s_theory_smooth, 'b-', label='Sackur-Tetrode (Theory)')
    plt.xlabel("Temperature (K)")
    plt.ylabel("Entropy (k_B)")
    # plt.title(f"Entropy vs Temperature (rbox={r_box}, N_points={N_points}, Z={Z})")
    plt.legend()
    plt.grid(True)

    # saving
    main_folder = "plots"
    sub_folder = "feb28_2026"

    plot_dir = os.path.join(jax_numerov_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    file_name = os.path.basename(__file__)

    temp_filename = os.path.join(plot_dir, f"{file_name}_T{T}_N{N_points}_Z{Z}.png")
    plt.savefig(temp_filename, dpi=300)

    print(f"--- {file_name} has completed running ---")

main()
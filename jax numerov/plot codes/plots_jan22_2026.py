# housekeeping
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
core_path = os.path.join(parent_dir, 'core files')

if core_path not in sys.path:
    sys.path.append(core_path)
    
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from njax_functions import numerov_solver, U_solver_nonhybrid
from hybrid_functions_uncorrected import U_solver_uncorrected, bounded_states_solver, mu_solver_uncorrected
from scipy.signal import argrelextrema

from rich.traceback import install
install()

def main():
    r_box_array = jnp.logspace(0, 1, num = 50)
    r_start = 1e-5
    N_points = 300
    l = 0
    Z = 1
    numerov_vect = jax.vmap(numerov_solver, in_axes = (0, None, None, None, None))
    energies, _ = numerov_vect(r_box_array, r_start, N_points, l, Z)

    plt.figure(figsize = (10,6))
    plt.plot(r_box_array, energies[:,0], marker = "x", color = "red", label = "Hybrid")
    plt.scatter(1.83, 0, color = "blue", label = "Intersection at (1.83, 0)")
    plt.xlabel("r_box (Ha)")
    plt.ylabel("E_1 (Ha)")
    plt.xscale("log")
    plt.ylim((-0.5,0.5))
    plt.legend()
    plt.grid(True)
    plt.axhline(y = 0, linestyle = "--", color = "green")
    plt.title(f"E_1 vs r_box, l = {l}")
    
    # saving
    main_folder = "plots"
    sub_folder = "jan22_2025_new"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    plot_dir = os.path.join(script_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")

    temp_filename = os.path.join(plot_dir, f"E1vsrbox_N{N_points}.png")
    plt.savefig(temp_filename, dpi=300)

    # U vs r_box

    r_box_array_2 = jnp.logspace(0, 1, num = 100)
    Z = 1
    r_start = 1e-5
    N_points = 300
    T_ha = 1e-3
    U_array_1 = []

    for r in r_box_array_2:
        e, m, d = bounded_states_solver(r, r_start, N_points, Z)
        mu = mu_solver_uncorrected(e, m, d, r, Z, T_ha)
        U = U_solver_uncorrected(r, e, m, d, mu, T_ha)
        U_array_1.append(U)

    plt.figure(figsize=(10,6))
    plt.plot(r_box_array_2, jnp.array(U_array_1) * 27.2114, linestyle="-", marker="x", color="red", label= "Hybrid")
    plt.axvline(x = 1.82799, color = "blue", linestyle = "--", label = "E_1 = 0 at r_box = 1.83")
    plt.xlabel("Box Radius (Ha)")
    plt.xscale('log')
    plt.ylabel("Internal Energy (eV)")
    plt.title(f"Internal Energy vs Box Size (T={T_ha*315775:.0f} K, N_points={N_points})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"UvsRbox_temp{T_ha}_N{N_points}_closeup.png"), dpi=300)
    plt.close()

    print("U vs smaller r_box done")

    # mu vs Z

    Z_array = jnp.arange(1, 118)
    r_box = 30
    r_start = 1e-5
    T_ha = 1e-3
    N_points = 300

    mu_array_3 = []
    mu_list = [] 

    for Z in Z_array:
        e, m, d = bounded_states_solver(r_box, r_start, N_points, Z)
        mu = mu_solver_uncorrected(e, m, d, r_box, Z, T_ha)
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

    plt.savefig(os.path.join(plot_dir, f"muVsZ_marked_temp{T_ha}.png"), dpi=300)
    plt.close()

    print("mu vs Z with peaks marked done.")

main()

# housekeeping
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
core_path = os.path.join(parent_dir, 'core files')

if core_path not in sys.path:
    sys.path.append(core_path)
    
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacrev, jit
import matplotlib.pyplot as plt
from njax_functions import lobpcg_solver
import time
import numpy as np

from rich.traceback import install
install()

def main():
    # initialization
    r_box = 30
    r_start = 1e-5
    N_points = 300
    l = 0
    k = 10

    energies, _ = lobpcg_solver(r_box, r_start, N_points, l, k)
    
    n_plot_lim = 10

    x_plot = jnp.linspace(1,n_plot_lim)
    y_plot = -1/(2*x_plot ** 2)

    plt.figure(figsize = (10,6))
    plt.plot(range(1,n_plot_lim + 1), energies[:n_plot_lim], marker = "x", label = "LOBPCG")
    plt.plot(x_plot, y_plot, linestyle = "-", label = "Theory (E = -1/(2n^2))")
    plt.xlabel("n")
    plt.ylabel("Energy (Ha)")
    plt.title(f"Hydrogen Energy Levels: r_box = {r_box}, N_points = {N_points}, l = {l}, E_1 = {energies[0]:.5f}")
    plt.grid(True)
    plt.legend()
    plt.xlim((1,n_plot_lim))
    
    # saving
    main_folder = "plots"
    sub_folder = "lobpcg"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    plot_dir = os.path.join(script_dir, main_folder, sub_folder)
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    file_name = f"elevel_lobpcg_rbox{r_box}_N{N_points}_l{l}.png"
    save_path = os.path.join(plot_dir, file_name)
    
    plt.savefig(save_path)
    plt.close()

    # error and time vs N_points
    N_points_plot = np.arange(60,1050,50)
    n_plot_test = len(N_points_plot)
    errors = np.zeros(n_plot_test)
    times = np.zeros(n_plot_test)

    for i in range(n_plot_test):
        # warming up JIT, so recording second run
        _, _ = lobpcg_solver(r_box, r_start, N_points_plot[i], l, k)

        start = time.time()
        energies, psi = lobpcg_solver(r_box, r_start, N_points_plot[i], l, k)
        energies.block_until_ready()
        end = time.time()
        errors[i] = abs(energies[0] - (-1/2))
        times[i] = end - start

    # error vs n
    plt.figure(figsize=(10,6))
    plt.plot(N_points_plot, errors, linestyle="-", marker="x", color="red", label = "LOBPCG")
    plt.xlabel("N_points", fontsize=12)
    plt.ylabel("abs(E1 - (-1/2))", fontsize=12)
    plt.yscale("log")
    plt.title(f"Error vs N_points (r_box={r_box}, l={l})", fontsize=14)
    plt.grid(True)
    plt.legend()

    error_filename = os.path.join(plot_dir, f"error_lobpcg_rbox{r_box}_l{l}.png")
    plt.savefig(error_filename, dpi=300)

    # time vs n
    plt.figure(figsize=(10,6))
    plt.plot(N_points_plot, times, linestyle="-", marker="x", color="blue", label = "LOBPCG")
    plt.xlabel("N_points", fontsize=12)
    plt.ylabel("Time (s)", fontsize=12)
    plt.title(f"Time vs N_points (r_box={r_box}, l={l})", fontsize=14)
    plt.grid(True)
    plt.legend()

    time_filename = os.path.join(plot_dir, f"time_lobpcg_rbox{r_box}_l{l}.png")
    plt.savefig(time_filename, dpi=300)

    # file run completion message
    current_script = os.path.basename(__file__)
    print(f"--- {current_script} has completed running ---")

main()
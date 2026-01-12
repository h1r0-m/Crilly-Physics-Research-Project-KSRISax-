# benchkeeping + file management
import jax

# enabling 64-bit precision
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import os
import time
import numpy as np
from njax_functions import numerov_solver

# enables colorful tracebacks which was useful for debugging
from rich.traceback import install
install() 

def main():
    # initilialization
    r_box = 30
    r_start = 1e-5
    N_points = 1000
    l = 0

    # error and time vs N_points
    N_points_plot = np.arange(50,1050,50)
    n_plot_test = len(N_points_plot)
    errors = np.zeros(n_plot_test)
    times = np.zeros(n_plot_test)

    for i in range(n_plot_test):
        # warming up JIT, so recording second run
        _, _ = numerov_solver(r_box, r_start, N_points_plot[i], l)

        start = time.time()
        energies, psi = numerov_solver(r_box, r_start, N_points_plot[i], l)
        energies.block_until_ready()
        end = time.time()
        errors[i] = abs(energies[0] - (-1/2))
        times[i] = end - start

    # error vs n
    plt.figure(figsize=(10,6))
    plt.plot(N_points_plot, errors, linestyle="-", marker="x", color="red")
    plt.xlabel("N_points", fontsize=12)
    plt.ylabel("abs(E1 - (-1/2))", fontsize=12)
    plt.yscale("log")
    plt.title(f"Error vs N_points (r_box={r_box}, l={l})", fontsize=14)
    plt.grid(True)

    # saving
    folder_name = "plots" 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, folder_name)

    # creating folder if it dosnt exist 
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")

    error_filename = os.path.join(plot_dir, f"error_rbox{r_box}_l{l}.png")
    plt.savefig(error_filename, dpi=300)

    # time vs n
    plt.figure(figsize=(10,6))
    plt.plot(N_points_plot, times, linestyle="-", marker="x", color="blue")
    plt.xlabel("N_points", fontsize=12)
    plt.ylabel("Time (s)", fontsize=12)
    plt.title(f"Time vs N_points (r_box={r_box}, l={l})", fontsize=14)
    plt.grid(True)

    time_filename = os.path.join(plot_dir, f"time_rbox{r_box}_l{l}.png")
    plt.savefig(time_filename, dpi=300)

    # file run completion message
    file_name = os.path.basename(__file__)
    print(f"--- {file_name} has completed running ---")

main()
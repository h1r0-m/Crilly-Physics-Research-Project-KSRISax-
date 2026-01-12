# %% Benchkeeping + file management
import jax

# enabling 64-bit precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacrev, jit
from functools import partial
import jax.scipy.linalg as jla
import matplotlib.pyplot as plt
import os
import time
import numpy as np

from numerov_jax_solver import numerov_solver, cholesky_solve

# enables colorful tracebacks which was useful for debugging
from rich.traceback import install
install() 

# %% main

# wanted functions to be all organized at the bottom so defined main sequence
def main():

    # initialization
    r_box = 30
    r_start = 1e-5
    N_points = 1000
    l = 0

    energies, psi = numerov_solver(r_box, r_start, N_points, l)

    # %% plotting

    # hydrogen energy levels

    n_plot_lim = 10

    x_plot = jnp.linspace(1,n_plot_lim)
    y_plot = -1/(2*x_plot ** 2)

    plt.figure(figsize = (10,6))
    plt.plot(range(1,n_plot_lim + 1), energies[:n_plot_lim], marker = "x", label = "Numerov Matrix (JAX)")
    plt.plot(x_plot, y_plot, linestyle = "-", label = "Theory (E = -1/(2n^2))")
    plt.xlabel("n")
    plt.ylabel("Energy (Ha)")
    plt.title(f"Hydrogen Energy Levels: r_box = {r_box}, N_points = {N_points}, l = {l}, E_1 = {energies[0]:.5f}")
    plt.grid(True)
    plt.legend()
    plt.xlim((1,n_plot_lim))
    
    # 1. Designate the folder name
    folder_name = "plots" 
    
    # 2. Get the current directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 3. Create the full path for the new folder
    plot_dir = os.path.join(script_dir, folder_name)
    
    # 4. Create the folder if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    save_path = os.path.join(plot_dir, f"hydrogen_rbox{r_box}_N{N_points}_l{l}_E1{energies[0]:.5f}.png")
    plt.savefig(save_path)
    plt.close()

    # error and time vs N_points

    N_points_plot = np.arange(50,1050,50)
    n_plot_test = len(N_points_plot)
    errors = np.zeros(n_plot_test)
    times = np.zeros(n_plot_test)

    # warming up JIT
    _, _ = numerov_solver(r_box, r_start, 100, l)
    
    for i in range(n_plot_test):
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

    print("done")

main()
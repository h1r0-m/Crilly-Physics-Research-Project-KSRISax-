# housekeeping
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacrev, jit
import os
import matplotlib.pyplot as plt
from njax_functions import numerov_solver, U_solver
import time
import numpy as np

from rich.traceback import install
install()

def main():
    # initialization
    r_box = 30.0
    r_start = 1e-5
    N_points_array = np.arange(50,1050,50)
    l = 0
    time_array = np.zeros(len(N_points_array))

    for i in range(len(N_points_array)):
        # defining function to obtain the gradient of
        # has to be float input and float output
        def E1_getter(r_box):
            energies, _ = numerov_solver(r_box, 1e-5, N_points_array[i], l)
            return energies[0]
        
        # obtaining gradient
        grad_func = jit(jacrev(E1_getter))

        # test run so that second run time is recorded
        _ = grad_func(r_box)

        start = time.time()
        grad = grad_func(r_box)
        grad.block_until_ready()
        end = time.time()
        time_array[i] = end - start

    # plotting
    plt.figure(figsize=(10,6))
    plt.plot(N_points_array, time_array, linestyle="-", marker="x", color="red", label = "Numerov Simulation")
    plt.xlabel("N_points", fontsize=12)
    plt.ylabel("Time Taken (s)", fontsize=12)
    plt.title(f"Time Taken for Gradient dE_1 / dr_box Calculation vs N_points (r_box={r_box}, l={l})", fontsize=14)
    plt.grid(True)
    plt.legend()

    # saving
    folder_name = "plots" 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, folder_name)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")

    temp_filename = os.path.join(plot_dir, f"gradtime_rbox{r_box}_l{l}.png")
    plt.savefig(temp_filename, dpi=300)

    print("done")

main()
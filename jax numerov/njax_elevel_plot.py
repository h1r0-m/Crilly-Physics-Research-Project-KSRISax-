# benchkeeping + file management
import jax

# enabling 64-bit precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
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

    # data retrieving
    energies, psi = numerov_solver(r_box, r_start, N_points, l)

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
    
    # saving
    folder_name = "plots" 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    plot_dir = os.path.join(script_dir, folder_name)
    
    # creating folder if it dosnt exist 
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    save_path = os.path.join(plot_dir, f"hydrogen_rbox{r_box}_N{N_points}_l{l}_E1{energies[0]:.5f}.png")
    plt.savefig(save_path)
    plt.close()

    # file run completion message
    file_name = os.path.basename(__file__)
    print(f"--- {file_name} has completed running ---")

main()
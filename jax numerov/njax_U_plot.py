# housekeeping
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacrev, jit
import os
import matplotlib.pyplot as plt
from njax_functions import numerov_solver, U_solver

from rich.traceback import install
install()

def main():
    # two plots: one with constant r_box but varying T and opposite

    # const r_box ( = 30) and varying T first:

    # initialization
    r_box = 30
    N_points = 1000
    l = 0
    r_start = 1e-5
    
    # running solver
    energies, psi = numerov_solver(r_box, r_start, N_points, l)
    
    # for varying temperature
    temps = jnp.logspace(-3,1, num = 50)
    
    # calculating internal energies
    U_array = jnp.zeros(len(temps))
    U_vect = jax.vmap(U_solver, in_axes = (None, 0))
    U_array = U_vect(energies, temps)

    # converting to Kelvin and eV
    temps *= 315775
    U_array *= 27.2114

    # theory values
    theory_x = jnp.linspace(0,4e6)
    theory_y = 3/2 * 8.617e-5 * theory_x + U_array[0]

    # global figure (no close up)

    plt.figure(figsize=(10,6))
    plt.plot(temps, U_array, linestyle="-", marker="x", color="red", label = "Numerov Simualation")
    plt.plot(theory_x, theory_y, linestyle = "-", color = "blue", label = "Theory: 3/2 k_B T + E_1")
    plt.xlabel("Temperature (K)", fontsize=12)
    plt.ylabel("Internal Energy (eV)", fontsize=12)
    plt.xlim((0,3e6))
    plt.title(f"Internal Energy vs Temperature (r_box={r_box}, l={l})", fontsize=14)
    plt.grid(True)

    # saving
    folder_name = "plots" 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, folder_name)

    # creating folder if it dosnt exist 
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")

    temp_filename = os.path.join(plot_dir, f"temp_rbox{r_box}_l{l}_global.png")
    plt.savefig(temp_filename, dpi=300)

    # close up plot (const r_box)
    plt.figure(figsize=(10,6))
    plt.plot(temps, U_array, linestyle="-", marker="x", color="red", label = "Numerov Simualation")
    plt.plot(theory_x, theory_y, linestyle = "-", color = "blue", label = "Theory: 3/2 k_B T + E_1")
    plt.xlabel("Temperature (K)", fontsize=12)
    plt.ylabel("Internal Energy (eV)", fontsize=12)
    plt.xlim((0,1e5))
    plt.ylim((-15,5))
    plt.title(f"Internal Energy vs Temperature (r_box={r_box}, l={l}) - Close Up", fontsize=14)
    plt.grid(True)
    plt.legend()

    # saving
    folder_name = "plots" 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, folder_name)

    # creating folder if it dosnt exist 
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")

    temp_filename = os.path.join(plot_dir, f"temp_rbox{r_box}_l{l}_closeup.png")
    plt.savefig(temp_filename, dpi=300)

    # const temp and varying r_box
    
    temp = 1e-3

    # for varying r_box
    r_box_array = jnp.logspace(-3, 2, num = 50)

    # running solver
    numerov_solver_vect = jax.vmap(numerov_solver, in_axes = (0, None, None, None))
    energies_mat, psi_mat = numerov_solver_vect(r_box_array, r_start, N_points, l)

    # calculating internal energies
    U_vect_2 = jax.vmap(U_solver, in_axes = (0, None))
    U_array_2 = U_vect_2(energies_mat, temp)
    U_array_2 *= 27.2114

    plt.figure(figsize=(10,6))
    plt.plot(r_box_array, U_array_2, linestyle="-", marker="x", color="red", label = "Numerov Simualation")
    plt.xlabel("r_box (Ha)", fontsize=12)
    plt.xscale("log")
    plt.ylabel("Internal Energy (eV)", fontsize=12)
    plt.title(f"Internal Energy vs r_box (temp = {temp*315775:.0f} K, l = {l})", fontsize=14)
    plt.grid(True)
    plt.legend()

    temp_filename = os.path.join(plot_dir, f"temp2_temp{temp}_l{l}.png")
    plt.savefig(temp_filename, dpi=300)

    # file run completion message
    file_name = os.path.basename(__file__)
    print(f"--- {file_name} has completed running ---")

main()
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
import time

from hybrid_functions_corrected import mu_solver_corrected, thermo_solver_corrected
from hybrid_functions_uncorrected import bounded_states_solver, mu_solver_uncorrected, U_solver_uncorrected

from rich.traceback import install
install()

def main():
    # --- SETUP ---
    r_box = 30.0
    N_points = 300
    r_start = 1e-5
    Z = 1.0
    T = 10

    # saving
    main_folder = "plots"
    sub_folder = "jan27_2026"

    plot_dir = os.path.join(jax_numerov_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    print(f"\n=== Starting Convergence Calibration (T={T} Ha) ===")
    
    l_max_values = jnp.arange(5,55,5)
    
    times_new = []
    errors_new = []

    Z = 1

    # Theory Target (Ideal Gas Limit)
    U_theory = 1.5 * T
    
    # --- 1. TEST OLD METHOD (Numerov) ---
    print(f"testing uncorrected")
    
    e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)
    mu = mu_solver_uncorrected(e,m,d, r_box, Z, T)
    _ = U_solver_uncorrected(r_box, e,m,d, mu,T).block_until_ready()

    start = time.time()
    u_val = U_solver_uncorrected(r_box, e,m,d,mu,T)
    u_val.block_until_ready()
    end = time.time()
    time_old = end - start
    error_old = abs((u_val - U_theory) / U_theory) * 100.0

    print(f"testing corrected:")

    for l in l_max_values:
        print(f"l_max = {l}")

        # --- WARMUP RUN (Compiles the function for this specific l) ---
        mu = mu_solver_corrected(e,m,d,r_box, r_start, N_points, Z, T, 50, l)
        mu.block_until_ready()
        _, u = thermo_solver_corrected(e,m,d,r_box, r_start, N_points, mu, T, Z, l)
        u.block_until_ready()
        
        # --- TIMING RUN (Measures pure execution) ---
        start = time.time()
        
        mu_true = mu_solver_corrected(e,m,d,r_box, r_start, N_points, Z, T, 50, l)
        mu_true.block_until_ready()
        _, u_true = thermo_solver_corrected(e,m,d,r_box, r_start, N_points, mu_true, T, Z, l)
        u_true.block_until_ready() # Force JAX to finish before stopping clock
        
        end = time.time()
        
        runtime = end - start
        error = abs((u_true - U_theory) / U_theory) * 100.0
        
        times_new.append(runtime)
        errors_new.append(error)

    plt.figure(figsize = (10,6))
    plt.axhline(y = time_old, linestyle = "--", color = "red", label = "Uncorrected Time")
    plt.plot(l_max_values, times_new, marker = "x", linestyle = "--", color = "green", label = "Corrected Time")
    plt.xlabel("l_max")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs l_max")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "runtime.png"), dpi=300)

    plt.figure(figsize = (10,6))
    plt.axhline(y = error_old, linestyle = "--", color = "red", label = "Uncorrected Error")
    plt.plot(l_max_values, errors_new, marker = "x", linestyle = "--", color = "green", label = "Corrected Error")
    plt.xlabel("l_max")
    plt.ylabel("Error (%)")
    plt.title("Error vs l_max")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "error.png"), dpi=300)

main()
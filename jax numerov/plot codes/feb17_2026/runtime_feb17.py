# housekeeping
import sys
import os
from datetime import datetime

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

from scipy.signal import argrelextrema
from hybrid_functions_optimized import bounded_states_solver, mu_solver_corrected, U_solver_corrected
from hybrid_functions_uncorrected import mu_solver_uncorrected, U_solver_uncorrected
from njax_functions import U_solver_nonhybrid

from rich.traceback import install
install()

def benchmark_runtime():
    # Simulation Parameters
    r_box = 30.0
    r_start = 1e-5
    Z = 1.0
    T = 1e-3
    l_max = 50
    
    # Range of N_points to test scaling
    n_points_range = [100, 300, 500, 1000, 2000]
    
    compilation_times = []
    execution_times = []

    print(f"{'N_points':<10} | {'Compile (s)':<12} | {'Execute (s)':<12}")
    print("-" * 40)

    for N in n_points_range:
        # 1. Pre-calculate bounded states (done once per N)
        e, m, d = bounded_states_solver(r_box, r_start, N, Z)
        mu = mu_solver_corrected(e, m, d, r_box, r_start, N, Z, T)

        # --- MEASURE COMPILATION (First Call) ---
        # We trigger JIT by calling the function for the first time
        start_compile = time.perf_counter()
        _ = U_solver_corrected(e, m, d, r_box, r_start, N, mu, T, Z, l_max).block_until_ready()
        end_compile = time.perf_counter()
        
        # --- MEASURE EXECUTION (Second Call) ---
        # Function is now cached in machine code
        start_exec = time.perf_counter()
        _ = U_solver_corrected(e, m, d, r_box, r_start, N, mu, T, Z, l_max).block_until_ready()
        end_exec = time.perf_counter()

        c_time = end_compile - start_compile
        e_time = end_exec - start_exec
        
        compilation_times.append(c_time)
        execution_times.append(e_time)
        
        print(f"{N:<10} | {c_time:<12.4f} | {e_time:<12.6f}")

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Execution Time (Primary Y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Number of Grid Points (N_points)')
    ax1.set_ylabel('Execution Time (seconds)', color=color)
    ax1.plot(n_points_range, execution_times, marker='o', linestyle='-', color=color, label='Execution')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Compilation Time (Secondary Y-axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Compilation Time (seconds)', color=color)
    ax2.plot(n_points_range, compilation_times, marker='x', linestyle='--', color=color, label='JIT Compilation')
    ax2.tick_params(axis='y', labelcolor=color)

    # saving
    main_folder = "plots"
    sub_folder = "feb18_2026"

    # 1. Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_dir = os.path.join(jax_numerov_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    
    file_name = os.path.basename(__file__)
    temp_filename = os.path.join(plot_dir, f"runtime_analysis_{timestamp}.png")
    plt.savefig(temp_filename, dpi=300)

    print(f"--- {file_name} has completed running ---")

if __name__ == "__main__":
    benchmark_runtime()
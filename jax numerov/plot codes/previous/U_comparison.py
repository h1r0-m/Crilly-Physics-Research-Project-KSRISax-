# this was made with the help of AI

# housekeeping

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
core_path = os.path.join(parent_dir, 'core files')

if core_path not in sys.path:
    sys.path.append(core_path)

import jax
# Enable 64-bit precision (Critical for Quantum Mechanics)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

# Import your modules
from njax_functions import numerov_solver, U_solver_nonhybrid
from hybrid_functions_uncorrected import U_solver_uncorrected, bounded_states_solver, mu_solver_uncorrected

from rich.traceback import install
install()

def run_convergence_test(plot_dir, r_box, N_points, r_start, temp_ha=1.0):
    """
    Runs a sweep over l_max to see how Runtime and Error scale.
    Uses 'Warm Up' runs to exclude JAX compilation time.
    """
    print(f"\n=== Starting Convergence Calibration (T={temp_ha} Ha) ===")
    
    l_max_values = [10, 20, 40, 60, 80, 100, 120, 150, 200]
    times_old = []
    errors_old = []
    
    # Theory Target (Ideal Gas Limit)
    U_theory = 1.5 * temp_ha
    
    # --- 1. TEST OLD METHOD (Numerov) ---
    print(f"Testing Old Method (Brute Force Numerov)...")
    
    for l in l_max_values:
        # --- WARMUP RUN (Compiles the function for this specific l) ---
        _ = U_solver_nonhybrid(temp_ha, r_box, r_start, N_points, l).block_until_ready()
        
        # --- TIMING RUN (Measures pure execution) ---
        start = time.time()
        
        u_val = U_solver_nonhybrid(temp_ha, r_box, r_start, N_points, l)
        u_val.block_until_ready() # Force JAX to finish before stopping clock
        
        end = time.time()
        
        runtime = end - start
        error = abs((u_val - U_theory) / U_theory) * 100.0
        
        times_old.append(runtime)
        errors_old.append(error)
        print(f"  l_max={l:3d} | Time: {runtime:.4f}s | Error: {error:.4f}%")

    # --- 2. TEST NEW METHOD (Hybrid) ---
    print(f"Testing New Method (Hybrid Continuum)...")
    
    # --- WARMUP RUN (Compiles the Hybrid Pipeline) ---
    # We run it once with the same parameters we intend to time
    e_dummy, m_dummy, d_dummy = bounded_states_solver(r_box, r_start, N_points, 1.0, l_max=5)
    mu_dummy = mu_solver_uncorrected(e_dummy, m_dummy, d_dummy, r_box, 1.0, temp_ha)
    _ = U_solver_uncorrected(r_box, e_dummy, m_dummy, d_dummy, mu_dummy, temp_ha).block_until_ready()
    
    # --- TIMING RUN ---
    start = time.time()
    
    energies, mask, deg = bounded_states_solver(r_box, r_start, N_points, 1.0, l_max=5)
    mu = mu_solver_uncorrected(energies, mask, deg, r_box, 1.0, temp_ha)
    u_new = U_solver_uncorrected(r_box, energies, mask, deg, mu, temp_ha)
    u_new.block_until_ready()
    
    end = time.time()
    
    time_new = end - start
    error_new = abs((u_new - U_theory) / U_theory) * 100.0
    print(f"  Hybrid     | Time: {time_new:.4f}s | Error: {error_new:.4f}%")

    # --- 3. PLOTTING (Same as before) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('l_max (Numerov Method)')
    ax1.set_ylabel('Runtime (s)', color=color)
    ln1 = ax1.plot(l_max_values, times_old, marker='x', color=color, label="Old Method Time")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ln2 = ax1.axhline(y=time_new, color='green', linestyle='--', label="New Method Time")

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Rel. Error (%)', color=color)
    ln3 = ax2.plot(l_max_values, errors_old, marker='x', color=color, linestyle=':', label="Old Method Error")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    
    ln4 = ax2.axhline(y=error_new, color='blue', linestyle='--', alpha=0.5, label="New Method Error")

    lines = ln1 + [ln2] + ln3 + [ln4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title("Calibration: Runtime vs Accuracy Tradeoff")
    fig.tight_layout()
    plt.savefig(os.path.join(plot_dir, "calibration_runtime_error.png"), dpi=300)
    plt.close()
    
    return l_max_values[-1], time_new, times_old[-1]

def main():
    # --- SETUP ---
    r_box = 30.0
    N_points = 300
    r_start = 1e-5
    Z = 1.0
    
    # Setup directories
    main_folder = "plots"
    sub_folder = "U_comparison"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, main_folder, sub_folder)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # --- STEP 1: CALIBRATION ---
    l_max_old, time_new, time_old = run_convergence_test(plot_dir, r_box, N_points, r_start)
    
    print(f"\n=== Selected Comparison Parameters ===")
    print("Comparing Hybrid Model vs Numerov (l_max=200)")
    print(f"Estimated Speedup: {time_old/time_new:.1f}x faster")

    # --- STEP 2: PHYSICS RUN (Varying Temp) ---
    print("\n--- Running Physics: Varying Temperature ---")
    
    temps_ha = jnp.logspace(-3, 1, num=50) # 0.001 to 10 Ha

    # Old Method
    U_old_vect = jax.vmap(U_solver_nonhybrid, in_axes=(0, None, None, None, None))
    U_old_ha = U_old_vect(temps_ha, r_box, r_start, N_points, l_max_old)

    # New Method
    energies, mask, degeneracies = bounded_states_solver(r_box, r_start, N_points, Z, l_max=5)
    mu_vect = jax.vmap(mu_solver_uncorrected, in_axes=(None, None, None, None, None, 0))
    mu_array = mu_vect(energies, mask, degeneracies, r_box, Z, temps_ha)
    U_new_vect = jax.vmap(U_solver_uncorrected, in_axes=(None, None, None, None, 0, 0))
    U_new_ha = U_new_vect(r_box, energies, mask, degeneracies, mu_array, temps_ha)

    # Plotting
    temps_k = temps_ha * 315775.0
    U_old_ev = U_old_ha * 27.2114
    U_new_ev = U_new_ha * 27.2114
    theory_x = jnp.linspace(jnp.min(temps_k), jnp.max(temps_k), 100)
    theory_y = 1.5 * 8.617e-5 * theory_x 

    plt.figure(figsize=(10,6))
    plt.plot(temps_k, U_old_ev, linestyle="-", marker="x", color="red", label="Numerov (l_max=200)")
    plt.plot(temps_k, U_new_ev, linestyle="-", marker="x", color="green", label="Hybrid Model")
    plt.plot(theory_x, theory_y, linestyle="--", color="blue", label="Ideal Gas (3/2 kT)")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Internal Energy (eV)")
    plt.title(f"Internal Energy Comparison (Speedup {time_old/time_new:.1f}x)")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig(os.path.join(plot_dir, "U_vs_T_final.png"), dpi=300)
    plt.close()

    # Close Up
    plt.figure(figsize=(10,6))
    plt.plot(temps_k, U_old_ev, linestyle="-", marker="x", color="red", label=f"Numerov (l_max = 200)")
    plt.plot(temps_k, U_new_ev, linestyle="-", marker="x", color="green", label="Hybrid Model")
    plt.plot(theory_x, theory_y, linestyle="--", color="blue", label="Theory")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Internal Energy (eV)")
    plt.xlim(0, 1e5)
    plt.ylim(-15, 20)
    plt.title("Close Up: Low-Mid Temperature Region")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "U_vs_T_closeup_final.png"), dpi=300)
    plt.close()

    # --- STEP 3: PHYSICS RUN (Varying Box Size) ---
    print("\n--- Running Physics: Varying Box Size ---")
    temp_ha = 1e-3
    r_box_array = jnp.logspace(-1, 2, num=30)

    U_old_vect_2 = jax.vmap(U_solver_nonhybrid, in_axes=(None, 0, None, None, None))
    U_old_2_ha = U_old_vect_2(temp_ha, r_box_array, r_start, N_points, l_max_old)

    U_new_2_list = []
    for r in r_box_array:
        e, m, d = bounded_states_solver(r, r_start, N_points, Z)
        mu_val = mu_solver_uncorrected(e, m, d, r, Z, temp_ha)
        u_val = U_solver_uncorrected(r, e, m, d, mu_val, temp_ha)
        U_new_2_list.append(u_val)
    U_new_2_ha = jnp.array(U_new_2_list)

    plt.figure(figsize=(10,6))
    plt.plot(r_box_array, U_old_2_ha * 27.2114, linestyle="-", marker="x", color="red", label= "Numerov (l_max = 200)")
    plt.plot(r_box_array, U_new_2_ha * 27.2114, linestyle="-", marker="x", color="green", label="Hybrid")
    plt.xscale("log")
    plt.xlabel("Box Radius (Ha)")
    plt.ylabel("Internal Energy (eV)")
    plt.title(f"Energy vs Box Size (T={temp_ha*315775:.0f} K)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "U_vs_rbox_final.png"), dpi=300)
    plt.close()

    print(f"\n--- All plots saved to {plot_dir} ---")

if __name__ == "__main__":
    main()
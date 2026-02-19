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
import jax.numpy as jnp
import matplotlib.pyplot as plt
from datetime import datetime
from hybrid_functions_optimized import thermo_solver_corrected, F_solver_corrected, bounded_states_solver, mu_solver_corrected

from rich.traceback import install
install()
# --- Import your solvers here ---
# from your_file import F_solver_corrected, thermo_solver_corrected, ...

def run_sweeps():
    # Setup
    Z = 1.0
    r_start = 1e-5
    N_points = 300
    l_max = 50
    
    # Sweep Arrays
    r_box_range = jnp.logspace(-1, 2, 30)
    T_range = jnp.logspace(-4, -1, 30)
    
    # Constants for fixed parameters
    T_fixed = 1e-3
    r_box_fixed = 30.0

    # --- Wrapper Functions for Gradients ---
    def get_U(T, energies, mask, degen, r_box, r_start, N_p, mu, Z, l_m):
        _, U, _ = thermo_solver_corrected(energies, mask, degen, r_box, r_start, N_p, mu, T, Z, l_m)
        return U

    def get_F(r_box, energies, mask, degen, r_start, N_p, mu, T, Z, l_m):
        # We need r_box as the first arg to differentiate wrt it
        return F_solver_corrected(energies, mask, degen, r_box, r_start, N_p, mu, T, Z, l_m)

    def get_F_T(T, energies, mask, degen, r_box, r_start, N_p, mu, Z, l_m):
        # We need T as the first arg to differentiate wrt it
        return F_solver_corrected(energies, mask, degen, r_box, r_start, N_p, mu, T, Z, l_m)

    # Gradient functions
    cv_fn = jax.grad(get_U, argnums=0)      # dU/dT
    p_fn = jax.grad(get_F, argnums=0)       # dF/dr
    s_grad_fn = jax.grad(get_F_T, argnums=0) # dF/dT

    # --- Helper to plot and save ---
    def save_plot(x, y, xlabel, ylabel, title, filename, color='blue', linestyle='-'):
        # saving
        main_folder = "plots"
        sub_folder = "feb18_2026"

        # 1. Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plot_dir = os.path.join(jax_numerov_dir, main_folder, sub_folder)

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            print(f"Created directory: {plot_dir}")

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker='x', color=color, linestyle=linestyle)
        plt.xscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        timestamp = datetime.now().strftime("%H%M%S")
        temp_filename = os.path.join(plot_dir, f"{filename}_{timestamp}.png")
        plt.savefig(temp_filename, dpi=300)
        plt.close()

    # --- SWEEP 1: vs r_box ---
    print("Running r_box sweep...")
    rb_data = {"F": [], "P": [], "Cv": [], "S_direct": [], "S_grad": [], "mu": []}
    
    for r in r_box_range:
        e, m, d = bounded_states_solver(r, r_start, N_points, Z)
        mu = mu_solver_corrected(e, m, d, r, r_start, N_points, Z, T_fixed)
        
        N, U, S = thermo_solver_corrected(e, m, d, r, r_start, N_points, mu, T_fixed, Z, l_max)
        F = U - T_fixed * S
        
        # Derivatives
        cv = cv_fn(T_fixed, e, m, d, r, r_start, N_points, mu, Z, l_max)
        df_dr = p_fn(r, e, m, d, r_start, N_points, mu, T_fixed, Z, l_max)
        pressure = -df_dr / (4 * jnp.pi * r**2)
        s_grad = -s_grad_fn(T_fixed, e, m, d, r, r_start, N_points, mu, Z, l_max)

        rb_data["F"].append(F * 27.2114)
        rb_data["P"].append(pressure)
        rb_data["Cv"].append(cv)
        rb_data["S_direct"].append(S)
        rb_data["S_grad"].append(s_grad)
        rb_data["mu"].append(mu * 27.2114)

    # Save r_box plots
    save_plot(r_box_range, rb_data["F"], "$r_{box}$", "F (eV)", "Free Energy vs $r_{box}$", "F_vs_r")
    save_plot(r_box_range, rb_data["P"], "$r_{box}$", "P (au)", "Pressure vs $r_{box}$", "P_vs_r", color='green')
    save_plot(r_box_range, rb_data["Cv"], "$r_{box}$", "$C_v$ (au)", "Heat Capacity vs $r_{box}$", "Cv_vs_r", color='red')
    save_plot(r_box_range, rb_data["mu"], "$r_{box}$", "$\mu$ (eV)", "Chemical Potential vs $r_{box}$", "mu_vs_r", color='purple')

    # --- SWEEP 2: vs Temperature ---
    print("Running Temperature sweep...")
    t_data = {"F": [], "P": [], "Cv": [], "S_direct": [], "S_grad": [], "mu": []}
    
    # Pre-solve bounded states (they don't depend on T)
    e_t, m_t, d_t = bounded_states_solver(r_box_fixed, r_start, N_points, Z)

    for t in T_range:
        mu = mu_solver_corrected(e_t, m_t, d_t, r_box_fixed, r_start, N_points, Z, t)
        N, U, S = thermo_solver_corrected(e_t, m_t, d_t, r_box_fixed, r_start, N_points, mu, t, Z, l_max)
        F = U - t * S
        
        cv = cv_fn(t, e_t, m_t, d_t, r_box_fixed, r_start, N_points, mu, Z, l_max)
        df_dr = p_fn(r_box_fixed, e_t, m_t, d_t, r_start, N_points, mu, t, Z, l_max)
        pressure = -df_dr / (4 * jnp.pi * r_box_fixed**2)
        s_grad = -s_grad_fn(t, e_t, m_t, d_t, r_box_fixed, r_start, N_points, mu, Z, l_max)

        t_data["F"].append(F * 27.2114)
        t_data["P"].append(pressure)
        t_data["Cv"].append(cv)
        t_data["S_direct"].append(S)
        t_data["S_grad"].append(s_grad)
        t_data["mu"].append(mu * 27.2114)

    # Save Temperature plots (using log scale for X)
    plt.xscale('log') # For T plots, log scale is usually better
    save_plot(T_range, t_data["F"], "T (Ha)", "F (eV)", "Free Energy vs Temperature", "F_vs_T")
    save_plot(T_range, t_data["P"], "T (Ha)", "P (au)", "Pressure vs Temperature", "P_vs_T", color='green')
    save_plot(T_range, t_data["Cv"], "T (Ha)", "$C_v$ (au)", "Heat Capacity vs Temperature", "Cv_vs_T", color='red')
    
    # Specialized Entropy Comparison Plot
    plt.figure(figsize=(8, 5))
    plt.plot(T_range, t_data["S_direct"], 'k-', label='Direct Solver')
    plt.plot(T_range, t_data["S_grad"], 'y--', label='$- \partial F / \partial T$')
    plt.xscale('log'); plt.xlabel("T (Ha)"); plt.ylabel("S (au)")
    plt.title("Entropy Consistency Check vs T"); plt.legend(); plt.grid(True)
    # saving
    main_folder = "plots"
    sub_folder = "feb18_2026"

    # 1. Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_dir = os.path.join(jax_numerov_dir, main_folder, sub_folder)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
    entropy_filename = os.path.join(plot_dir, f"S_consistency_T_{datetime.now().strftime('%H%M%S')}.png")
    plt.savefig(entropy_filename)

if __name__ == "__main__":
    run_sweeps()
import sys
import os
from datetime import datetime

# 1. Fix the Import Path
script_dir = os.path.dirname(os.path.abspath(__file__))
jax_numerov_dir = os.path.dirname(os.path.dirname(script_dir))
core_path = os.path.join(jax_numerov_dir, 'core files')

if core_path not in sys.path:
    sys.path.append(core_path)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from rich.traceback import install
install()

# Import core solvers
from hybrid_functions_optimized import (
    thermo_solver_corrected, 
    F_solver_corrected, 
    bounded_states_solver, 
    mu_solver_corrected
)

# Import the new gradient solvers
from thermo_quantities import P_solver_grad, S_solver_grad, Cv_solver_grad

def save_plot(x, y, xlabel, ylabel, title, filename, plot_dir, color='blue', linestyle='-', xscale='log'):
    """Helper function to format and save plots."""
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='x', color=color, linestyle=linestyle)
    plt.xscale(xscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%H%M%S")
    full_path = os.path.join(plot_dir, f"{filename}_{timestamp}.png")
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_sweeps():
    # Setup Parameters
    Z = 1.0
    r_start = 1e-5
    N_points = 300
    l_max = 50
    
    # Sweep Arrays
    r_box_range = jnp.logspace(-1, 2, 30)
    T_range = jnp.logspace(-4, -1, 30)
    
    # Constants for fixed parameters during complementary sweeps
    T_fixed = 1e-3
    r_box_fixed = 30.0

    # saving
    main_folder = "plots"
    date_folder = "feb19_2026"

    # 1. Get the current timestamp for the deeper folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"thermoplots_{timestamp}"

    # Build the deep path: .../jax numerov/plots/feb19_2026/thermoplots_20260219_093416
    plot_dir = os.path.join(jax_numerov_dir, main_folder, date_folder, run_folder)

    # makedirs with exist_ok=True is safer for nested paths
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Created directory: {plot_dir}")
    
    file_name = os.path.basename(__file__)
    print(f"Saving plots to: {plot_dir}")

    # =========================================================================
    # SWEEP 1: Thermodynamic Quantities vs r_box (at constant T)
    # =========================================================================
    print(f"Running r_box sweep at T = {T_fixed} Ha...")
    
    rb_data = {"F": [], "U": [], "S": [], "mu": [], "P": [], "Cv": []}
    
    for r in r_box_range:
        # 1. Eigenstates depend on r_box, so solve inside the loop
        e, m, d = bounded_states_solver(r, r_start, N_points, Z)
        
        # 2. Base solvers
        mu = mu_solver_corrected(e, m, d, r, r_start, N_points, Z, T_fixed)
        _, U, S = thermo_solver_corrected(e, m, d, r, r_start, N_points, mu, T_fixed, Z, l_max)
        F = U - T_fixed * S
        
        # 3. Gradient solvers (Total Derivatives)
        P = P_solver_grad(r, r_start, N_points, T_fixed, Z, l_max)
        Cv = Cv_solver_grad(T_fixed, e, m, d, r, r_start, N_points, Z, l_max)

        # Store data
        rb_data["F"].append(F * 27.2114)   # Convert Ha to eV for energy
        rb_data["U"].append(U * 27.2114)
        rb_data["S"].append(S)             # Entropy in atomic units
        rb_data["mu"].append(mu * 27.2114)
        rb_data["P"].append(P)             # Pressure in atomic units
        rb_data["Cv"].append(Cv)           # Heat capacity in atomic units

    # Suffix for r_box sweep titles
    title_r = f"\n(T = {T_fixed} Ha, Z = {Z}, $N_{{points}}$ = {N_points})"

    # Save r_box plots
    save_plot(r_box_range, rb_data["F"], "r_box (Bohr)", "Free Energy F (eV)", f"Free Energy vs $r_{{box}}${title_r}", "F_vs_r", plot_dir)
    save_plot(r_box_range, rb_data["U"], "r_box (Bohr)", "Internal Energy U (eV)", f"Internal Energy vs $r_{{box}}${title_r}", "U_vs_r", plot_dir)
    save_plot(r_box_range, rb_data["S"], "r_box (Bohr)", "Entropy S (a.u.)", f"Entropy vs $r_{{box}}${title_r}", "S_vs_r", plot_dir)
    save_plot(r_box_range, rb_data["mu"], "r_box (Bohr)", "Chemical Potential $\mu$ (eV)", f"Chemical Potential vs $r_{{box}}${title_r}", "mu_vs_r", plot_dir, color='purple')
    save_plot(r_box_range, rb_data["P"], "r_box (Bohr)", "Pressure P (a.u.)", f"Pressure vs $r_{{box}}${title_r}", "P_vs_r", plot_dir, color='green')
    save_plot(r_box_range, rb_data["Cv"], "r_box (Bohr)", "Heat Capacity $C_v$ (a.u.)", f"Heat Capacity vs $r_{{box}}${title_r}", "Cv_vs_r", plot_dir, color='red')

    # =========================================================================
    # SWEEP 2: Thermodynamic Quantities vs Temperature (at constant r_box)
    # =========================================================================
    print(f"Running Temperature sweep at r_box = {r_box_fixed} Bohr...")
    
    t_data = {"F": [], "U": [], "S_direct": [], "S_grad": [], "mu": [], "P": [], "Cv": []}
    
    # Eigenstates do NOT depend on T, so solve ONCE outside the loop to save compute
    e_t, m_t, d_t = bounded_states_solver(r_box_fixed, r_start, N_points, Z)

    for t in T_range:
        # 1. Base solvers
        mu = mu_solver_corrected(e_t, m_t, d_t, r_box_fixed, r_start, N_points, Z, t)
        _, U, S = thermo_solver_corrected(e_t, m_t, d_t, r_box_fixed, r_start, N_points, mu, t, Z, l_max)
        F = U - t * S
        
        # 2. Gradient solvers (Total Derivatives)
        P = P_solver_grad(r_box_fixed, r_start, N_points, t, Z, l_max)
        Cv = Cv_solver_grad(t, e_t, m_t, d_t, r_box_fixed, r_start, N_points, Z, l_max)
        S_grad = S_solver_grad(t, e_t, m_t, d_t, r_box_fixed, r_start, N_points, Z, l_max)

        # Store data
        t_data["F"].append(F * 27.2114)
        t_data["U"].append(U * 27.2114)
        t_data["S_direct"].append(S)
        t_data["S_grad"].append(S_grad)
        t_data["mu"].append(mu * 27.2114)
        t_data["P"].append(P)
        t_data["Cv"].append(Cv)

    # Suffix for Temperature sweep titles
    title_t = f"\n($r_{{box}}$ = {r_box_fixed} Bohr, Z = {Z}, $N_{{points}}$ = {N_points})"

    # Save Temperature plots
    save_plot(T_range, t_data["F"], "T (Ha)", "Free Energy F (eV)", f"Free Energy vs T{title_t}", "F_vs_T", plot_dir)
    save_plot(T_range, t_data["U"], "T (Ha)", "Internal Energy U (eV)", f"Internal Energy vs T{title_t}", "U_vs_T", plot_dir)
    save_plot(T_range, t_data["mu"], "T (Ha)", "Chemical Potential $\mu$ (eV)", f"Chemical Potential vs T{title_t}", "mu_vs_T", plot_dir, color='purple')
    save_plot(T_range, t_data["P"], "T (Ha)", "Pressure P (a.u.)", f"Pressure vs T{title_t}", "P_vs_T", plot_dir, color='green')
    save_plot(T_range, t_data["Cv"], "T (Ha)", "Heat Capacity $C_v$ (a.u.)", f"Heat Capacity vs T{title_t}", "Cv_vs_T", plot_dir, color='red')

    # -------------------------------------------------------------------------
    # Specialized Entropy Consistency Plot
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(T_range, t_data["S_direct"], 'k-', label='Direct Solver: S')
    plt.plot(T_range, t_data["S_grad"], 'y--', label='Gradient Solver: -dF/dT')
    plt.xscale('log')
    plt.xlabel("T (Ha)")
    plt.ylabel("Entropy S (a.u.)")
    plt.title(f"Entropy Consistency Check vs T{title_t}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%H%M%S")
    full_path = os.path.join(plot_dir, f"S_consistency_T_{timestamp}.png")
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All sweeps completed and plots saved.")

if __name__ == "__main__":
    run_sweeps()
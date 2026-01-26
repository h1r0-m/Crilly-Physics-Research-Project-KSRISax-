import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import time

# Import specific components from your files
from hybrid_functions_corrected import u_integration_solver, get_smart_grid, phase_shift_denominator, potential_solver, k_solver
from bessel_function import sph_bessel, sph_bessel_deriv, sph_neumann, sph_neumann_deriv

# --- Reconstruct Phase Shift Calculation ---
# FIX: Added static_argnames=['N_points'] to handle the integer argument correctly in JAX
@partial(jit, static_argnames=['N_points'])
def get_phase_shift_components(E, r_box, r_start, N_points, l, Z):
    # 1. Get Wavefunction Log-Derivative (K)
    u_array = u_integration_solver(E, r_box, r_start, N_points, l, Z)
    dr = (r_box - r_start) / (N_points - 1)
    
    # Boundary Derivative (5-point stencil)
    u_N   = u_array[-1]
    u_Nm1 = u_array[-2]
    u_Nm2 = u_array[-3]
    u_Nm3 = u_array[-4]
    u_Nm4 = u_array[-5]
    u_prime_end = (25*u_N - 48*u_Nm1 + 36*u_Nm2 - 16*u_Nm3 + 3*u_Nm4) / (12 * dr)
    
    # K (Log Derivative) and k_wavevector
    # K = u'/u - 1/r (for reduced radial wavefunction u)
    K = u_prime_end / (u_N + 1e-15) - 1.0 / r_box
    
    V_edge = potential_solver(r_box, Z)
    k_end = k_solver(E, V_edge)
    x = k_end * r_box
    
    # Bessel / Neumann values
    j_val = sph_bessel(l, x)
    j_der = sph_bessel_deriv(l, x)
    n_val = sph_neumann(l, x)
    n_der = sph_neumann_deriv(l, x)
    
    # Scattering Phase Shift Formula: tan(delta) = (K*j - k*j') / (K*n - k*n')
    numerator   = K * j_val - k_end * j_der
    denominator = K * n_val - k_end * n_der
    
    return numerator, denominator

def run_diagnostic():
    # --- DIAGNOSTIC CONFIGURATION ---
    r_box_problem = 1.835  # The transition region
    l_target = 0           # Usually l=0 (s-wave) is the first to bind
    
    # Physics Params
    r_start = 1e-5
    N_points = 300
    Z = 1
    T = 1e-3
    mu_guess = 0.0 # Dummy mu just for grid generation
    
    print(f"--- RUNNING DIAGNOSTIC FOR r_box = {r_box_problem} ---")
    
    # 1. Generate the 'Smart Grid' (The one being tested)
    print("Generating Solver Grid...")
    solver_grid = get_smart_grid(r_box_problem, r_start, N_points, mu_guess, T, l_target, Z, E_max=2.0)
    print(f"Solver generated {len(solver_grid)} points.")
    
    # 2. Generate 'Truth Grid' (High resolution scan)
    # Scan low energy to see the resonance
    print("Generating Truth Scan (1e-14 to 1e-1)...")
    truth_E = np.logspace(-14, -1, 1000) 
    
    # 3. Calculate Phase Shifts
    truth_phase = []
    
    # Using a simple python loop for safety in diagnostic
    for E in truth_E:
        num, den = get_phase_shift_components(E, r_box_problem, r_start, N_points, l_target, Z)
        # arctan2 gives correct quadrant [-pi, pi]
        delta = np.arctan2(float(num), float(den))
        truth_phase.append(delta)
        
    # Unwrap to make it a continuous curve
    truth_phase = np.unwrap(truth_phase) 
    
    # Normalize: Phase should go to 0 at high energy (or pi at low energy for bound states)
    # But for visualization, we just want to see the shape. 
    # Usually delta -> pi as E -> 0 if there is a resonance/bound state close.
    
    # 4. Calculate Phase Shifts on the Solver Grid
    solver_phase = []
    for E in solver_grid:
        num, den = get_phase_shift_components(E, r_box_problem, r_start, N_points, l_target, Z)
        delta = np.arctan2(float(num), float(den))
        solver_phase.append(delta)
        
    # --- PLOTTING ---
    plt.figure(figsize=(10, 7))
    
    # Plot 1: Phase Shift
    # Plot normalized by pi
    plt.semilogx(truth_E, np.array(truth_phase) / np.pi, 'b-', linewidth=2, label=r'True Phase Shift ($\delta/\pi$)')
    
    # Overlay solver points (we unwrap them roughly to match the curve for visual check)
    # A simple way is to plot raw and see if they fall on the line modulo 1
    # Or just plot them raw for now.
    plt.semilogx(solver_grid, np.array(solver_phase)/np.pi, 'ro', markersize=4, label='Solver Grid Points')
    
    # Add labels
    plt.xlabel('Energy (Ha) [Log Scale]')
    plt.ylabel(r'Phase Shift $\delta$ ($\times \pi$)')
    plt.title(f'Grid Diagnostic: r_box={r_box_problem}, l={l_target}')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Check if we hit the resonance (Phase passing through 0.5 pi)
    plt.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Resonance ($\pi/2$)')
    plt.axhline(-0.5, color='green', linestyle='--', alpha=0.5)
    
    plt.savefig("diagnostic_plot.png")
    print("Plot saved as diagnostic_plot.png")
    plt.show()
    print("Diagnostic Complete.")

if __name__ == "__main__":
    run_diagnostic()
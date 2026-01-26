import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import time

# Import your physics modules
from hybrid_functions_uncorrected import bounded_states_solver
from hybrid_functions_corrected import mu_solver_corrected, N_solver_corrected, U_solver_corrected

def run_simple_scan():
    # --- Configuration ---
    r_start = 1
    r_end   = 10
    steps   = 50  # 30 intervals (0.001 step size)
    
    # Physics Parameters
    r_start_integration = 1e-5
    N_points = 300
    Z = 1
    T = 1e-3

    print(f"--- Starting Simple Scan: r_box [{r_start} -> {r_end}] ---")
    
    r_values = np.linspace(r_start, r_end, steps)
    
    # Arrays to store plotting data
    plot_r = []
    plot_U = []

    for i, r_box in enumerate(r_values):
        # 1. Solve Bound States
        e, m, d = bounded_states_solver(r_box, r_start_integration, N_points, Z)
        
        # --- FIX: Robust Scalar Extraction ---
        # Flatten the array first to ensure we get the very first element regardless of shape
        mask_val = np.array(m).flatten()[0]
        energy_val = np.array(e).flatten()[0]
        
        is_masked = (int(mask_val) == 1)
        is_bound = (float(energy_val) < -1e-5)
        
        state_type = "BOUND" if (is_masked and is_bound) else "FREE "
        
        # 2. Solve Physics
        mu = mu_solver_corrected(e, m, d, r_box, r_start_integration, N_points, Z, T)
        N  = N_solver_corrected(e, m, d, r_box, r_start_integration, N_points, mu, T, Z)
        U  = U_solver_corrected(e, m, d, r_box, r_start_integration, N_points, mu, T, Z)
        
        # 3. Print to Terminal immediately
        print(f"r={r_box:.4f} | Type: {state_type} | U={U:.5f} | N={N:.5f} | mu={mu:.4f}")
        
        # Store for plotting
        plot_r.append(r_box)
        plot_U.append(float(U))

    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    plt.plot(plot_r, plot_U, 'o-', color='blue', label='Internal Energy U')
    
    plt.xlabel('Box Radius (r_box)')
    plt.ylabel('Internal Energy (Ha)')
    plt.xscale('log')
    plt.title(f'Continuity Check (T={T} Ha)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simple_scan()
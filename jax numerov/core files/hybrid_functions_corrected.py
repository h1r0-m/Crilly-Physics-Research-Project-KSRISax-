import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import scipy.optimize 
import numpy as np

# Import specific components
from hybrid_functions_uncorrected import V_solver, fd_occup_solver
from bessel_function import sph_bessel, sph_bessel_deriv, sph_neumann, sph_neumann_deriv
from fermi_dirac_integral import fermi_dirac_integral_half, fermi_dirac_integral_three_half
from jax.scipy.special import xlogy

from rich.traceback import install
install()

jax.config.update("jax_enable_x64", True) 

# ==========================================
# 1. Core Quantum Solvers (JIT Compiled)
# ==========================================

@jit
def potential_solver(r, Z):
    return -Z/r

@jit
def k_solver(E,V):
    result = jnp.where(E >= V, jnp.sqrt(2*(E-V)), 0)
    return result

@partial(jit, static_argnames=['N_points'])
def u_integration_solver(E, r_box, r_start, N_points, l, Z):
    """Numerov integration for radial wavefunction u(r)."""
    r_points = jnp.linspace(r_start, r_box, N_points)
    dr = r_points[1] - r_points[0]
    
    V_grid = potential_solver(r_points, Z)
    centrifugal = l * (l + 1) / (2 * r_points**2)
    k_sq_grid = 2 * (E - V_grid - centrifugal)
    
    def numerov_step(carry, i):
        u_prev, u_curr = carry
        k_prev = k_sq_grid[i-1]
        k_curr = k_sq_grid[i]
        k_next = k_sq_grid[i+1]
        
        term1 = (2 - 5/6 * dr**2 * k_curr) * u_curr
        term2 = (1 + 1/12 * dr**2 * k_prev) * u_prev
        denom = (1 + 1/12 * dr**2 * k_next)
        u_next = (term1 - term2) / denom
        return (u_curr, u_next), u_next

    u_0 = r_points[0]**(l + 1)
    u_1 = r_points[1]**(l + 1)
    
    scan_indices = jnp.arange(1, N_points - 1)
    _, u_rest = jax.lax.scan(numerov_step, (u_0, u_1), scan_indices)
    
    return jnp.concatenate([jnp.array([u_0, u_1]), u_rest])

@partial(jit, static_argnames = ['N_points', 'l'])
def phase_shift_raw_solver(E, r_box, r_start, N_points, l, Z):
    """Returns the numerator and denominator for tan(delta)."""
    u_array = u_integration_solver(E, r_box, r_start, N_points, l, Z)
    dr = (r_box - r_start) / (N_points - 1)
    
    # Boundary Derivative
    u_N   = u_array[-1]
    u_Nm1 = u_array[-2]
    u_Nm2 = u_array[-3]
    u_Nm3 = u_array[-4]
    u_Nm4 = u_array[-5]
    u_prime_end = (25*u_N - 48*u_Nm1 + 36*u_Nm2 - 16*u_Nm3 + 3*u_Nm4) / (12 * dr)
    
    K = u_prime_end / (u_N + 1e-15) - 1.0 / r_box
    V_edge = potential_solver(r_box, Z)
    k_end = k_solver(E, V_edge)
    x = k_end * r_box
    
    j_val = sph_bessel(l, x)
    j_der = sph_bessel_deriv(l, x)
    n_val = sph_neumann(l, x)
    n_der = sph_neumann_deriv(l, x)
    
    num = K * j_val - k_end * j_der
    den = K * n_val - k_end * n_der
    
    return num, den

@partial(jit, static_argnames = ['N_points', 'l'])
def phase_shift_denominator(E, r_box, r_start, N_points, l, Z):
    """Helper for root finding (resonance search)."""
    _, den = phase_shift_raw_solver(E, r_box, r_start, N_points, l, Z)
    return den

# ==========================================
# 2. Robust Grid Generation (Python)
# ==========================================

def get_smart_grid(r_box, r_start, N_points, mu, T, l, Z, E_max=5.0):
    """
    Generates a grid that captures resonances. 
    Uses a 'Log-Broad' strategy to ensure we bracket the resonance 
    even if we don't land exactly on the peak.
    """
    # Base grid
    grid_points = [1e-16, 1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, E_max]
    
    if 0 < mu < E_max:
        grid_points.extend([mu - 4*T, mu - T, mu, mu + T, mu + 4*T])

    # Resonance Search
    test_energies = jnp.logspace(jnp.log10(1e-16), jnp.log10(E_max), 200)
    denoms = jax.vmap(lambda E: phase_shift_denominator(E, r_box, r_start, N_points, l, Z))(test_energies)
    sign_changes = jnp.where(jnp.diff(jnp.sign(denoms)))[0]
    
    for idx in sign_changes:
        e_low = test_energies[idx]
        e_high = test_energies[idx+1]
        try:
            func_to_root = lambda e_val: float(phase_shift_denominator(e_val, r_box, r_start, N_points, l, Z))
            E_res = scipy.optimize.brentq(func_to_root, float(e_low), float(e_high))
            
            # Log-Broad Cluster: Place points at geometric factors around E_res
            # This ensures we catch the phase jump whether it's sharp or broad
            factors = [0.1, 0.5, 0.9, 0.99, 0.999]
            cluster = [E_res]
            for f in factors:
                cluster.append(E_res * f)
                cluster.append(E_res / f)
            
            grid_points.extend(cluster)
        except:
            pass

    # Sort and Filter
    grid_points = jnp.array(grid_points)
    grid_points = jnp.sort(grid_points)
    grid_points = grid_points[grid_points >= 1e-16]
    grid_points = grid_points[grid_points <= E_max]
    
    unique_mask = jnp.concatenate([jnp.array([True]), jnp.diff(grid_points) > 1e-14 * grid_points[1:]])
    return grid_points[unique_mask]

# ==========================================
# 3. New "Difference" Integrator
# ==========================================

@partial(jit, static_argnames=['N_points', 'l'])
def compute_correction_on_grid(grid, r_box, r_start, N_points, mu, T, l, Z):
    """
    Computes N_corr and U_corr using the phase-difference method.
    Robust against vertical phase jumps (resonances).
    """
    
    # 1. Calculate Raw Phase Components (Vectorized)
    nums, dens = jax.vmap(lambda E: phase_shift_raw_solver(E, r_box, r_start, N_points, l, Z))(grid)
    
    # 2. Compute Angle and Unwrap
    # arctan2 handles quadrants correctly. unwrap handles 2pi jumps.
    phases = jnp.arctan2(nums, dens)
    phases_unwrapped = jnp.unwrap(phases)
    
    # 3. Calculate Differences (Delta delta)
    # This captures the step height exactly, even if slope is infinite
    d_phases = jnp.diff(phases_unwrapped)
    
    # 4. DOS factor: (2/pi) * (2l+1) * d_delta
    # We remove dE from the integral because we use d_delta directly
    # Integral ~ sum( f * (d_delta/dE) * dE ) = sum( f * d_delta )
    factor = (2.0 / jnp.pi) * (2*l + 1)
    
    # 5. Integrate N (Trapezoidal on f)
    # Avg occupancy between steps
    occup = fd_occup_solver(grid, mu, T)
    occup_avg = 0.5 * (occup[1:] + occup[:-1])
    
    N_corr = factor * jnp.sum(occup_avg * d_phases)
    
    # 6. Integrate U (Trapezoidal on E*f)
    energy_occup = grid * occup
    energy_occup_avg = 0.5 * (energy_occup[1:] + energy_occup[:-1])
    
    U_corr = factor * jnp.sum(energy_occup_avg * d_phases)
    
    return N_corr, U_corr

# ==========================================
# 4. Main Solvers (Refactored)
# ==========================================

def N_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max=50):
    # 1. Bound
    occup = fd_occup_solver(energies, mu, T)
    N_bound = jnp.sum(degeneracies * occup * mask)
    
    # 2. Free Uncorrected
    V = V_solver(r_box)
    gamma_factor = jnp.sqrt(jnp.pi) / 2
    N_free_unc = ((jnp.sqrt(2)*V*T**(3/2))/(jnp.pi**2)) * gamma_factor * fermi_dirac_integral_half(mu/T)
    
    # 3. Correction (Sum over l)
    N_corr_total = 0.0
    for l in range(l_max):
        grid = get_smart_grid(r_box, r_start, N_points, mu, T, l, Z)
        n_c, _ = compute_correction_on_grid(grid, r_box, r_start, N_points, mu, T, l, Z)
        N_corr_total += n_c
        
    return N_bound + N_free_unc + N_corr_total

def U_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max=50):
    # 1. Bound
    occup = fd_occup_solver(energies, mu, T)
    U_bound = jnp.sum(degeneracies * occup * energies * mask)
    
    # 2. Free Uncorrected
    V = V_solver(r_box)
    gamma_factor = 3 * jnp.sqrt(jnp.pi) / 4
    U_free_unc = ((jnp.sqrt(2)*V*T**(5/2))/(jnp.pi**2)) * gamma_factor * fermi_dirac_integral_three_half(mu/T)
    
    # 3. Correction
    U_corr_total = 0.0
    for l in range(l_max):
        grid = get_smart_grid(r_box, r_start, N_points, mu, T, l, Z)
        _, u_c = compute_correction_on_grid(grid, r_box, r_start, N_points, mu, T, l, Z)
        U_corr_total += u_c
        
    return U_bound + U_free_unc + U_corr_total

# Keeping mu_solver effectively the same, just removing decorators 
# so it can handle the Python loop inside N_solver
def mu_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, Z, T, iteration_count=50, l_max=50):
    a = -50 * jnp.maximum(1, T * jnp.log(T)) * jnp.maximum(1, Z**2)
    b = 50 * jnp.maximum(1, 1 / (r_box ** 2))
    
    for _ in range(iteration_count):
        c = (a + b) / 2.0
        N_c = N_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, c, T, Z, l_max)
        
        if N_c < Z:
            a = c
        else:
            b = c
            
    return (a + b) / 2.0
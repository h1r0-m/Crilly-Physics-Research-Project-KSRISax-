# housekeeping
import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)

# Import your existing solvers (adjust the import path as needed)
from hybrid_functions_optimized import (
    bounded_states_solver, 
    mu_solver_corrected, 
    F_solver_corrected, 
    thermo_solver_corrected
)

# ==============================================================================
# 1. Total Functions (Wrappers for JAX to see the full chain of dependencies)
# ==============================================================================

def get_F_total_r(r_box, r_start, N_points, T, Z, l_max):
    """Free energy wrapper for volume/radius derivatives (Pressure)."""
    # r_box changes the eigenstates, so this must be evaluated inside!
    e, m, d = bounded_states_solver(r_box, r_start, N_points, Z, l_max)
    mu = mu_solver_corrected(e, m, d, r_box, r_start, N_points, Z, T, 50, l_max)
    F = F_solver_corrected(e, m, d, r_box, r_start, N_points, mu, T, Z, l_max)
    return F

def get_F_total_T(T, e, m, d, r_box, r_start, N_points, Z, l_max):
    """Free energy wrapper for Temperature derivatives (Entropy)."""
    # Eigenstates (e, m, d) don't depend on T, so we pass them in to save compute.
    mu = mu_solver_corrected(e, m, d, r_box, r_start, N_points, Z, T, 50, l_max)
    F = F_solver_corrected(e, m, d, r_box, r_start, N_points, mu, T, Z, l_max)
    return F

def get_U_total_T(T, e, m, d, r_box, r_start, N_points, Z, l_max):
    """Internal energy wrapper for Temperature derivatives (Heat Capacity)."""
    mu = mu_solver_corrected(e, m, d, r_box, r_start, N_points, Z, T, 50, l_max)
    _, U, _ = thermo_solver_corrected(e, m, d, r_box, r_start, N_points, mu, T, Z, l_max)
    return U

# ==============================================================================
# 2. Jitted Gradient Functions
# ==============================================================================
# argnums=0 tells JAX to differentiate with respect to the first argument.

dF_dr_grad = jax.jit(jax.grad(get_F_total_r, argnums=0), static_argnames=['N_points', 'l_max'])
dF_dT_grad = jax.jit(jax.grad(get_F_total_T, argnums=0), static_argnames=['N_points', 'l_max'])
dU_dT_grad = jax.jit(jax.grad(get_U_total_T, argnums=0), static_argnames=['N_points', 'l_max'])

# ==============================================================================
# 3. Clean Public Solvers for your Plotting Script
# ==============================================================================

def P_solver_grad(r_box, r_start, N_points, T, Z, l_max=50):
    """
    Calculates Pressure using P = -dF/dV = -(dF/dr) / (4*pi*r^2)
    """
    df_dr = dF_dr_grad(r_box, r_start, N_points, T, Z, l_max)
    dV_dr = 4.0 * jnp.pi * (r_box ** 2)
    return -df_dr / dV_dr

def S_solver_grad(T, e, m, d, r_box, r_start, N_points, Z, l_max=50):
    """
    Calculates Entropy using S = -dF/dT
    """
    return -dF_dT_grad(T, e, m, d, r_box, r_start, N_points, Z, l_max)

def Cv_solver_grad(T, e, m, d, r_box, r_start, N_points, Z, l_max=50):
    """
    Calculates Heat Capacity at constant volume using Cv = dU/dT
    """
    return dU_dT_grad(T, e, m, d, r_box, r_start, N_points, Z, l_max)
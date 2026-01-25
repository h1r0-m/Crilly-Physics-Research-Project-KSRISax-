import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from hybrid_functions_uncorrected import V_solver, fd_occup_solver
from bessel_function import sph_bessel, sph_bessel_deriv, sph_neumann, sph_neumann_deriv
from quad_integration_fixed_nodes import simple_GL, composite_GL_quad
from fermi_dirac_integral import fermi_dirac_integral_half, fermi_dirac_integral_three_half
from jax.scipy.special import xlogy

from rich.traceback import install
install()

jax.config.update("jax_enable_x64", True) 

@jit
def k_solver(E,V):
    result = jnp.where(E >= V, jnp.sqrt(2*(E-V)), 0)
    return result

@jit
def potential_solver(r, Z):
    return -Z/r

@partial(jit, static_argnames=['N_points'])
def u_integration_solver(E, r_box, r_start, N_points, l, Z):
    # 1. Setup Grid
    r_points = jnp.linspace(r_start, r_box, N_points)
    dr = r_points[1] - r_points[0]
    
    # 2. Pre-calculate Kinetic Term (k_squared) for the whole grid
    # This is much faster than calculating it inside the loop
    V_grid = potential_solver(r_points, Z)
    centrifugal = l * (l + 1) / (2 * r_points**2)
    
    # k^2(r) = 2 * (E - V(r) - l(l+1)/2r^2)
    k_sq_grid = 2 * (E - V_grid - centrifugal)
    # 3. Define the Scan Step for Numerov
    # We need k_sq at i-1, i, and i+1. 
    # The scan will iterate over indices i starting from 1 up to N-2
    
    def numerov_step(carry, i):
        u_prev, u_curr = carry
        
        k_prev = k_sq_grid[i-1]
        k_curr = k_sq_grid[i]
        k_next = k_sq_grid[i+1]
        
        # Numerov Formula
        term1 = (2 - 5/6 * dr**2 * k_curr) * u_curr
        term2 = (1 + 1/12 * dr**2 * k_prev) * u_prev
        denom = (1 + 1/12 * dr**2 * k_next)
        
        u_next = (term1 - term2) / denom
        
        return (u_curr, u_next), u_next
    # 4. Initial Conditions (Asymptotic behavior r^(l+1))
    u_0 = r_points[0]**(l + 1)
    u_1 = r_points[1]**(l + 1)
    # 5. Run the Scan
    # We iterate from index 1 to N_points-2 to generate u_2 ... u_last
    scan_indices = jnp.arange(1, N_points - 1)
    
    _, u_rest = jax.lax.scan(numerov_step, (u_0, u_1), scan_indices)
    
    # 6. Combine into full array
    u_array = jnp.concatenate([jnp.array([u_0, u_1]), u_rest])
    
    return u_array

# --- 2. Phase Shift Solver (Single Energy) ---
@partial(jit, static_argnames = ['N_points', 'l'])
def single_phase_shift_solver(E, r_box, r_start, N_points, l, Z):
    # 1. Get Wavefunction
    u_array = u_integration_solver(E, r_box, r_start, N_points, l, Z)
    dr = (r_box - r_start) / (N_points - 1)
    
    # 2. Calculate Derivative at Boundary (5-Point Backward Diff)
    # Using the last 5 points for O(h^4) precision
    u_N   = u_array[-1]
    u_Nm1 = u_array[-2]
    u_Nm2 = u_array[-3]
    u_Nm3 = u_array[-4]
    u_Nm4 = u_array[-5]
    
    u_prime_end = (25*u_N - 48*u_Nm1 + 36*u_Nm2 - 16*u_Nm3 + 3*u_Nm4) / (12 * dr)
    
    # 3. Logarithmic Derivative K (Radial Function R, not reduced u)
    # K = R'/R = (u'/u) - (1/r)
    # Handle u_N=0 case safely (though unlikely at boundary)
    K = u_prime_end / (u_N + 1e-15) - 1.0 / r_box
    # 4. Local Wavenumber (Muffin Tin Fix)
    V_edge = potential_solver(r_box, Z)
    k_end = k_solver(E, V_edge)
    x = k_end * r_box
    
    # 5. Bessel Functions & Derivatives
    j_val = sph_bessel(l, x)
    n_val = sph_neumann(l, x)
    j_der = sph_bessel_deriv(l, x)
    n_der = sph_neumann_deriv(l, x)
    # 6. Matching Formula
    numerator   = K * j_val - k_end * j_der
    denominator = K * n_val - k_end * n_der
    
    # Use atan (result is wrapped between -pi/2 and pi/2)
    # atan2 is better but requires decomposing tan(delta) into sin/cos which is harder here
    delta = jnp.arctan2(numerator, denominator)
    
    return delta

# --- 3. The Main Pipeline (Vectorized over Energy) ---
@partial(jit, static_argnames=['N_points', 'l'])
def d_delta_dE_solver(E, r_box, r_start, N_points, l, Z):
    """
    Input:
        energies: Array of E values (must be > V_edge for continuum)
    Output:
        d_delta_dE: Array of derivatives same shape as energies
    """
    def phase_shift_fn(E):
        return single_phase_shift_solver(E, r_box, r_start, N_points, l, Z)
    
    # This calculates d(delta)/dE automatically
    d_delta_dE = jax.grad(phase_shift_fn)(E)
    
    return d_delta_dE

@partial(jit, static_argnames = ['N_points', 'l'])
def dos_correction_solver(E, r_box, r_start, N_points, l, Z):
    """
    Calculates the Total Density of States g(E) for a single energy E.
    params = (r_box, l, Z)
    """
    
    # This calculates d(delta)/dE automatically
    d_delta_dE = d_delta_dE_solver(E, r_box, r_start, N_points, l, Z)
    
    dos_corr = (2.0 / jnp.pi) * (2*l + 1) * d_delta_dE
    
    return dos_corr

# --- Integrands for the Quad Function ---
@partial(jit, static_argnames = ['N_points', 'l'])
def N_corr_integrand(E, r_box, r_start, N_points, mu, T, l, Z):
    """Integrand for Number of Particles: g(E) * f(E)"""
    dos_corr = dos_correction_solver(E, r_box, r_start, N_points, l, Z)
    
    occup = fd_occup_solver(E, mu, T)
    return dos_corr * occup

@partial(jit, static_argnames = ['N_points', 'l'])
def U_corr_integrand(E, r_box, r_start, N_points, mu, T, l, Z):
    """Integrand for Energy: E * g(E) * f(E)"""
    # Reuse the N integrand and multiply by E
    N_corr_integrand_val = N_corr_integrand(E, r_box, r_start, N_points, mu, T, l, Z)
    return E * N_corr_integrand_val

@jit
def S_uncorr_integrand(E, r_box, mu, T):
    V = V_solver(r_box)
    occup = fd_occup_solver(E,mu,T)
    S_uncorr_integrand = -(jnp.sqrt(2) * V) / (jnp.pi ** 2) * jnp.sqrt(E) * (xlogy(occup, occup) + xlogy(1-occup, 1-occup))
    return S_uncorr_integrand

@partial(jit, static_argnames = ['N_points', 'l'])
def S_corr_integrand(E, r_box, r_start, N_points, mu, T, l, Z):
    """Integrand for Entropy"""
    dos_corr = dos_correction_solver(E, r_box, r_start, N_points, l, Z)
    occup = fd_occup_solver(E, mu, T)
    S_corr_integrand = dos_corr * (-(xlogy(occup, occup) + xlogy(1-occup, 1-occup)))
    
    return S_corr_integrand

def N_solver_corrected(energies, mask, degeneracies,
                       r_box, r_start, N_points,
                       mu, T, Z,
                       l_max: int = 50):
    # 1. Bound electrons
    V = V_solver(r_box)
    occup = fd_occup_solver(energies, mu, T)
    N_bound = jnp.sum(degeneracies * occup * mask)
    # 2. Uncorrected free
    gamma_factor = jnp.sqrt(jnp.pi) / 2
    N_free_uncorrected = (
        (jnp.sqrt(2) * V * T**(3/2)) / (jnp.pi**2)
        * gamma_factor
        * fermi_dirac_integral_half(mu/T)
    )
    # 3. Correction: sum over l in Python
    E_start = 1e-10
    E_end_candidate = mu + 20*T
    E_end = jnp.where(E_end_candidate > E_start, jnp.maximum(E_end_candidate, 5), 5)
    def integrand_E(E, r_box, r_start, mu, T, Z, l):
        return N_corr_integrand(E, r_box, r_start, N_points, mu, T, l, Z)
    N_free_correction = 0.0
    for l in range(l_max):
        # l is a Python int; N_corr_integrand sees it as static
        def func_E(E, r_box, r_start, mu, T, Z):
            return integrand_E(E, r_box, r_start, mu, T, Z, l)
        integral_l = composite_GL_quad(
            func_E,
            E_start, E_end,
            (r_box, r_start, mu, T, Z)
        )
        N_free_correction = N_free_correction + integral_l
    return N_bound + N_free_uncorrected + N_free_correction

def U_solver_corrected(energies, mask, degeneracies,
                       r_box, r_start, N_points,
                       mu, T, Z,
                       l_max: int = 50):
    """
    Corrected internal energy U = U_bound + U_free_uncorrected + U_free_correction
    - Sum over l is done in a Python loop.
    - U_corr_integrand is jitted with static N_points and l.
    """
    # 1. Bound-state contribution
    V = V_solver(r_box)
    occup = fd_occup_solver(energies, mu, T)
    U_bound = jnp.sum(degeneracies * occup * energies * mask)
    # 2. Uncorrected free contribution
    gamma_factor = 3 * jnp.sqrt(jnp.pi) / 4
    U_free_uncorrected = (
        (jnp.sqrt(2) * V * T**(5/2)) / (jnp.pi**2)
        * gamma_factor
        * fermi_dirac_integral_three_half(mu / T)
    )
    # 3. Correction term: sum over partial waves and integrate over E
    E_start = 1e-10
    E_end_candidate = mu + 20*T
    E_end = jnp.where(E_end_candidate > E_start, jnp.maximum(E_end_candidate, 5), 5)
    U_free_correction = 0.0
    for l in range(l_max):
        # For this l, define the E-integrand using the jitted U_corr_integrand
        def U_corr_E(E, r_box, r_start, mu, T, Z):
            # U_corr_integrand is jitted with static_argnames=['N_points', 'l']
            return U_corr_integrand(E, r_box, r_start, N_points, mu, T, l, Z)
        # Perform the energy integral for this l
        integral_l = composite_GL_quad(
            lambda E, r_box, r_start, mu, T, Z: U_corr_E(E, r_box, r_start, mu, T, Z),
            E_start, E_end,
            (r_box, r_start, mu, T, Z)
        )
        U_free_correction = U_free_correction + integral_l
    
    print(f"U_bound: {U_bound}")
    print(f"U_free_uncorrected: {U_free_uncorrected}")
    print(f"U_free_correction: {U_free_correction}")
    return U_bound + U_free_uncorrected + U_free_correction

def S_solver_corrected(energies, mask, degeneracies,
                       r_box, r_start, N_points,
                       mu, T, Z,
                       l_max: int = 50):
    """
    Corrected entropy S = S_bound + S_free_uncorrected + S_free_correction
    - Sum over l is done in a Python loop.
    - S_corr_integrand is jitted with static N_points and l.
    """
    # 1. Bound-state entropy
    V = V_solver(r_box)
    occup = fd_occup_solver(energies, mu, T)
    S_bound = -jnp.sum(
        degeneracies * mask *
        (xlogy(occup, occup) + xlogy(1 - occup, 1 - occup))
    )
    # 2. Uncorrected free entropy (continuum, uncorrected DOS)
    E_start = 1e-10
    E_end_candidate = mu + 20*T
    E_end = jnp.where(E_end_candidate > E_start, jnp.maximum(E_end_candidate, 5), 5)
    def S_uncorr_E(E, r_box, mu, T):
        return S_uncorr_integrand(E, r_box, mu, T)
    S_free_uncorrected = simple_GL(
        lambda E, r_box, mu, T: S_uncorr_E(E, r_box, mu, T),
        E_start, E_end,
        (r_box, mu, T)
    )
    # 3. Correction term: sum over partial waves l
    S_free_correction = 0.0
    for l in range(l_max):
        # For this l, define the E-integrand using the jitted S_corr_integrand
        def S_corr_E(E, r_box, r_start, mu, T, Z):
            # S_corr_integrand is jitted with static_argnames=['N_points', 'l']
            return S_corr_integrand(E, r_box, r_start, N_points, mu, T, l, Z)
        # Perform the energy integral for this l
        integral_l = composite_GL_quad(
            lambda E, r_box, r_start, mu, T, Z: S_corr_E(E, r_box, r_start, mu, T, Z),
            E_start, E_end,
            (r_box, r_start, mu, T, Z)
        )
        S_free_correction = S_free_correction + integral_l
    return S_bound + S_free_uncorrected + S_free_correction

@partial(jit, static_argnames=['N_points', 'iteration_count', 'l_max'])
def mu_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, Z, T, iteration_count = 50, l_max = 50):
    """
    finds chemical potential mu
    logic:
    can find N through N_solver, and we have to find mu such that N = Z (atomic number, assuming neutral atom)
    inputs:
    energies, mask, degeneracies, r_box - standard
    Z - atomic number
    T - temperature
    iteration_count - number of iterations for the bisection method, default = 50
    
    output:
    mu - chemical potential
    """
    # initial guesses for mu, scaled with r_box, Z, and T
    a = -50 * jnp.maximum(1, T*jnp.log(T)) * jnp.maximum(1, Z**2)
    b = 50 * jnp.maximum(1, 1/(r_box ** 2))
    
    def step(i, carry): # Note: fori_loop passes 'i' first
        a, b = carry
        c = (a + b) / 2
        N_c = N_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, c, T, Z, l_max)
        
        a_new = jnp.where(N_c < Z, c, a)
        b_new = jnp.where(N_c >= Z, c, b)
        return (a_new, b_new)
    # Run loop 0 to iteration_count
    a_final, b_final = jax.lax.fori_loop(0, iteration_count, step, (a, b))
    return (a_final + b_final) / 2

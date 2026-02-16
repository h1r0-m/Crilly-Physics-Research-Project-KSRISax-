import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jit, lax
from functools import partial
from bessel_function import sph_bessel, sph_bessel_deriv, sph_neumann, sph_neumann_deriv
from njax_functions import numerov_solver
from fermi_dirac_integral import fermi_dirac_integral_half, fermi_dirac_integral_three_half
from jax.scipy.special import xlogy

# Enable x64
jax.config.update("jax_enable_x64", True)

@jit
def V_solver(r_box):
    return (4/3) * jnp.pi * r_box ** 3

@jit
def fd_occup_solver(E, mu, T):
    return sigmoid((mu - E) / T)

@partial(jit, static_argnames=['N_points', 'l_max'])
def bounded_states_solver(r_box, r_start, N_points, Z, l_max=5):
    l_array = jnp.arange(l_max + 1)
    numerov_vect = jax.vmap(numerov_solver, in_axes=(None, None, None, 0, None))
    energies, _ = numerov_vect(r_box, r_start, N_points, l_array, Z)
    
    mask = jnp.where(energies < 0, 1, 0)
    l_grid = jnp.repeat(l_array[:, None], energies.shape[1], axis=1)
    degeneracies = 2 * (2 * l_grid + 1)
    
    return energies, mask, degeneracies

@jit
def potential_solver(r, Z):
    return -Z/r

@jit
def k_solver(E, V):
    # Using relu to ensure non-negative before sqrt, simpler than where for gradients
    val = 2 * (E - V)
    return jnp.sqrt(jnp.maximum(val, 0.0))

@partial(jit, static_argnames=['N_points'])
def u_integration_solver(E, r_box, r_start, N_points, l, Z):
    # Construct grid inside JIT
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

@partial(jit, static_argnames=['N_points'])
def phase_shift_raw_solver(E, r_box, r_start, N_points, l, Z):
    u_array = u_integration_solver(E, r_box, r_start, N_points, l, Z)
    dr = (r_box - r_start) / (N_points - 1)
    
    u_N   = u_array[-1]
    u_Nm1 = u_array[-2]
    u_Nm2 = u_array[-3]
    u_Nm3 = u_array[-4]
    u_Nm4 = u_array[-5]
    u_prime_end = (25*u_N - 48*u_Nm1 + 36*u_Nm2 - 16*u_Nm3 + 3*u_Nm4) / (12 * dr)
    
    V_edge = potential_solver(r_box, Z)
    k_end = k_solver(E, V_edge)
    x = k_end * r_box
    
    j_val = sph_bessel(l, x)
    j_der = sph_bessel_deriv(l, x)
    n_val = sph_neumann(l, x)
    n_der = sph_neumann_deriv(l, x)
    
    K_times_u = u_prime_end - (u_N / r_box)

    num_robust = K_times_u * j_val - u_N * k_end * j_der
    den_robust = K_times_u * n_val - u_N * k_end * n_der
    
    u_sign = jnp.sign(u_N)
    return num_robust * u_sign, den_robust * u_sign

@partial(jit, static_argnames=['N_points'])
def correction_grid_solver_jax(r_box, r_start, N_points, mu, T, l, Z, E_max):
    """
    JAX-compatible grid generation. 
    Instead of dynamically finding roots with Scipy (which breaks JIT),
    we create a fixed-size, high-density grid that concentrates points
    logarithmically and around mu.
    """
    
    # 1. Base Logarithmic Grid (Fixed size: 200 points)
    # We use enough points here to catch most features without needing brentq
    log_grid = jnp.logspace(jnp.log10(1e-4), jnp.log10(E_max), 200)
    
    # 2. Very low energy grid (Fixed size: 20 points)
    low_e_grid = jnp.linspace(1e-16, 1e-4, 20)
    
    # 3. Mu-centered Grid (Fixed size: 21 points)
    # This captures the Fermi-Dirac change. 
    # Logic: Create a grid from -4T to +4T, then shift by mu
    mu_window = jnp.linspace(-4.0, 4.0, 21) * T
    mu_grid = mu + mu_window
    
    # Filter points in mu_grid that are invalid (<=0 or > E_max)
    # We replace invalid points with a dummy value (e.g. 1e-10) 
    # Sorting later will handle order, and diff=0 will handle duplicates/dummies
    mu_grid = jnp.where((mu_grid > 1e-16) & (mu_grid < E_max), mu_grid, 1e-10)

    # 4. Concatenate all grids (Total fixed size = 241)
    full_grid = jnp.concatenate([low_e_grid, log_grid, mu_grid])
    
    # 5. Sort
    full_grid = jnp.sort(full_grid)
    
    # Note: We rely on the integration step (trapezoidal rule) to handle 
    # duplicate points gracefully (dx will be 0, adding 0 area).
    return full_grid

@partial(jit, static_argnames=['N_points'])
def correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z):
    # Vectorized computation over the fixed grid
    nums, dens = jax.vmap(lambda E: phase_shift_raw_solver(E, r_box, r_start, N_points, l, Z))(grid)
    
    phases = jnp.arctan2(nums, dens)
    phases_unwrapped = jnp.unwrap(phases)
    d_phases = jnp.diff(phases_unwrapped)
    
    factor = (2.0 / jnp.pi) * (2*l + 1)
    
    N_integrand = fd_occup_solver(grid, mu, T)
    N_avg = 0.5 * (N_integrand[1:] + N_integrand[:-1])
    N_corr = factor * jnp.sum(N_avg * d_phases)
    
    U_integrand = grid * N_integrand
    U_avg = 0.5 * (U_integrand[1:] + U_integrand[:-1])
    U_corr = factor * jnp.sum(U_avg * d_phases)

    # For Entropy, handle the logs safely
    term1 = xlogy(N_integrand, N_integrand)
    term2 = xlogy(1-N_integrand, 1-N_integrand)
    S_int = term1 + term2
    S_avg = 0.5 * (S_int[1:] + S_int[:-1])
    S_corr = -factor * jnp.sum(S_avg * d_phases)
    
    return N_corr, U_corr, S_corr

@jit
def S_free_uncorrected_solver(N_free_unc, U_free_unc, mu, T):
    T_safe = jnp.maximum(T, 1e-12)    
    S_free_unc = 1/T_safe * ((5/3) * U_free_unc - mu * N_free_unc)
    return S_free_unc

@partial(jit, static_argnames=['N_points', 'l_max'])
def thermo_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max=50):
    
    # 1. Bound States
    occup = fd_occup_solver(energies, mu, T)
    N_bound = jnp.sum(degeneracies * occup * mask)
    U_bound = jnp.sum(degeneracies * occup * energies * mask)
    S_bound = jnp.sum(-degeneracies * mask * (xlogy(occup, occup) + xlogy(1-occup, 1-occup)))
    
    # 2. Free Uncorrected
    V = V_solver(r_box)
    gamma_factor_N = jnp.sqrt(jnp.pi) / 2
    N_free_unc = ((jnp.sqrt(2)*V*T**(3/2))/(jnp.pi**2)) * gamma_factor_N * fermi_dirac_integral_half(mu/T)
    
    gamma_factor_U = 3 * jnp.sqrt(jnp.pi) / 4
    U_free_unc = ((jnp.sqrt(2)*V*T**(5/2))/(jnp.pi**2)) * gamma_factor_U * fermi_dirac_integral_three_half(mu/T)

    S_free_unc = S_free_uncorrected_solver(N_free_unc, U_free_unc, mu, T)

    # 3. Correction Terms (Using lax.scan for speed and compile efficiency)
    E_max = jnp.maximum(mu + 20 * T, 5.0)

    def scan_body(carry, l):
        # Current totals
        n_acc, u_acc, s_acc = carry
        
        # Get grid (fixed size)
        grid = correction_grid_solver_jax(r_box, r_start, N_points, mu, T, l, Z, E_max)
        
        # Compute corrections
        n_c, u_c, s_c = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # Add to accumulators
        new_carry = (n_acc + n_c, u_acc + u_c, s_acc + s_c)
        return new_carry, None

    # Initial carry (0.0, 0.0, 0.0)
    init_carry = (0.0, 0.0, 0.0)
    l_values = jnp.arange(l_max)
    
    (N_corr_total, U_corr_total, S_corr_total), _ = lax.scan(scan_body, init_carry, l_values)

    N_total = N_bound + N_free_unc + N_corr_total
    U_total = U_bound + U_free_unc + U_corr_total
    S_total = S_bound + S_free_unc + S_corr_total
    
    return N_total, U_total, S_total

# Helper for finding mu (Root finding requires loop)
@partial(jit, static_argnames=['N_points', 'iteration_count', 'l_max'])
def mu_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, Z, T, iteration_count=50, l_max=50):
    
    # Determine bounds
    a = -50 * jnp.maximum(1.0, T * jnp.log(T)) * jnp.maximum(1.0, Z**2)
    b = 50 * jnp.maximum(1.0, 1.0 / (r_box ** 2))
    
    def body_fun(i, val):
        a_curr, b_curr = val
        c = (a_curr + b_curr) / 2.0
        
        # We only need N here. We can optimize by making a specific N-only solver 
        # or just extracting the first output of thermo_solver.
        # For simplicity/correctness, we call the full solver but JAX's DCE (Dead Code Elimination)
        # might strip the U and S calculations if they aren't used, making it efficient.
        N_c, _, _ = thermo_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, c, T, Z, l_max)
        
        # Bisection update
        # If N_c < Z, we need more electrons -> increase chemical potential -> a = c
        # Use where to avoid python if/else
        a_next = jnp.where(N_c < Z, c, a_curr)
        b_next = jnp.where(N_c < Z, b_curr, c)
        
        return (a_next, b_next)

    # Run loop
    final_a, final_b = lax.fori_loop(0, iteration_count, body_fun, (a, b))
    
    return (final_a + final_b) / 2.0
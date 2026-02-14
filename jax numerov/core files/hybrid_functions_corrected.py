# housekeeping
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jit
from functools import partial
import scipy.optimize 
from bessel_function import sph_bessel, sph_bessel_deriv, sph_neumann, sph_neumann_deriv
from njax_functions import numerov_solver
from fermi_dirac_integral import fermi_dirac_integral_half, fermi_dirac_integral_three_half
from jax.scipy.special import xlogy

from rich.traceback import install
install()

jax.config.update("jax_enable_x64", True)

@jit
def V_solver(r_box):
    return (4/3) * jnp.pi * r_box ** 3

@jit
def fd_occup_solver(E, mu, T):
    # using sigmoid for numerical stability
    return sigmoid((mu - E) / T)

@partial(jit, static_argnames=['N_points', 'l_max'])
def bounded_states_solver(r_box, r_start, N_points, Z, l_max = 5):
    """
    pasted straight from the uncorrected code file

    inputs: 
    r_box, r_start, N_points - standard
    Z - atomic number
    l_max - what l to do numerov_solver until, assuming l = 0 to 5 covers all the bound states, but could be changed

    outputs:
    energies - a matrix of energy eigenvalues with each col corresponding to a certain l
    mask - a matrix same dimension as energies with either 1 or 0 depending on whether entry is bounded (E < 0) or not
            entry is 1 if bounded, otherwise 0
    degeneracies - same dimension as energies, with degeneracy factor for each entry
    """
    
    # using numerov solver for l = 0 to l_max
    l_array = jnp.arange(l_max + 1)
    numerov_vect = jax.vmap(numerov_solver, in_axes=(None, None, None, 0, None))
    energies, _ = numerov_vect(r_box, r_start, N_points, l_array, Z)
    
    # creating mask for bound states
    mask = jnp.where(energies < 0, 1, 0)
    
    # creating degeneracy matrix
    l_grid = jnp.repeat(l_array[:, None], energies.shape[1], axis=1) # same dimensions as energies, filled with corresponding l values
    degeneracies = 2 * (2 * l_grid + 1) # degeneracy matrix for each energy entry
    
    return energies, mask, degeneracies

@jit
def potential_solver(r, Z):
    """ to get the potential, didn't integrate it in the solver so i can easily change the potential later
     when using SCF solver
      
       just using simple coulomb potential: V = -Z/r for now
        
    inputs:
    r - radius (in Ha)
    Z - atomic number

    output:
    V - potential
            """
    return -Z/r

@jit
def k_solver(E,V):
    """ to obtain wave number, k = sqrt(2(E-V))
     
    derivation: 
    E = T + V = p^2/2m + V = (hk)^2 / 2m + V --> k = sqrt((2m(E-V))/h^2), in atomic units: k = sqrt(2(E-V))

    when calculating k of energy eigenvalues that were just positive, sometimes the inside of the sqrt was 
    positive, producing NaN results so just returning 0 if thats about to happen

    inputs:
    E - energy
    V - potential

    output:
    k - wave number
       """

    result = jnp.where(E >= V, jnp.sqrt(2*(E-V)), 0)
    return result

@partial(jit, static_argnames=['N_points'])
def u_integration_solver(E, r_box, r_start, N_points, l, Z):
    """ numerov interation for u(r) (reduced radial wave function), this is to calculate the phase shift 
    correction term, and ultimately the logarithmic derivative at the end: R'/R to match with the bessel
     function condition outside the box to then solve for delta (phase shift) (where R is the radial wave
     function, not reduced) 
     
    so the form is:
    d^2u / dr^2 + k^2 u = 0, where u = u(r) and k^2 = 2(E-V-l(l+1)/(2r^2))

    inputs:
    E - energy
    r_box, r_start, N_points - default
    l - angular quantum number
    Z - atomic number

    outputs:
    u_array - array of values with reduced radial wave function evaluated at every point within r_points
     """
    
    # radial points
    r_points = jnp.linspace(r_start, r_box, N_points)
    dr = r_points[1] - r_points[0]
    
    # finding potential, centrifugal, and k^2 term at each radial point
    V_grid = potential_solver(r_points, Z)
    centrifugal = l * (l + 1) / (2 * r_points**2)
    k_sq_grid = 2 * (E - V_grid - centrifugal)
    
    # defining function to loop until the end
    def numerov_step(carry, i):
        u_prev, u_curr = carry # using 2 previous values of u to calculate the next
        k_prev = k_sq_grid[i-1]
        k_curr = k_sq_grid[i]
        k_next = k_sq_grid[i+1]
        
        # using numerov method to integrate
        term1 = (2 - 5/6 * dr**2 * k_curr) * u_curr
        term2 = (1 + 1/12 * dr**2 * k_prev) * u_prev
        denom = (1 + 1/12 * dr**2 * k_next)
        u_next = (term1 - term2) / denom
        return (u_curr, u_next), u_next # syntax: carry to next iteration, what to store in list

    # using u(r) = C r^(l+1) for the first two radial points, appropriate for small r
    u_0 = r_points[0]**(l + 1)
    u_1 = r_points[1]**(l + 1)
    
    # obtaining u throughout
    scan_indices = jnp.arange(1, N_points - 1)
    _, u_rest = jax.lax.scan(numerov_step, (u_0, u_1), scan_indices)
    
    # returning full u array for all radial points
    return jnp.concatenate([jnp.array([u_0, u_1]), u_rest])

@partial(jit, static_argnames = ['N_points', 'l'])
def phase_shift_raw_solver(E, r_box, r_start, N_points, l, Z):
    """ returning numerator and denominator for phase shift 
    
    inputs:
    E - energy
    r_box, r_start, N_points - default
    l - angular quantum number
    Z - atomic number   

    outputs:
    num_robust - numerator of phase shift formula * u_N (u_N being reduced radial wave function at end)
    den_robust - denominator of phase shift formula * u_N

    with phase shift formula being:
    tan(delta) = (K j(x) - k j'(x)) / (K n(x) - k n'(x))
    
    where K = R'/R is the logarithmic derivative evaluated at r = r_box
    j,n are spherical bessel and neumann functions respectively
    j', n' are those functions' derivatives with respect to r
    x = k r_box, where k is the wave number

    revised to avoid K = u'/u singularity when u -> 0
    includes sign correction to prevent artificial pi-shifts
    """

    # obtaining u
    u_array = u_integration_solver(E, r_box, r_start, N_points, l, Z)
    dr = (r_box - r_start) / (N_points - 1)
    
    # finding derivative of wave function at boundary
    # using 4th order backward finite difference
    u_N   = u_array[-1]
    u_Nm1 = u_array[-2]
    u_Nm2 = u_array[-3]
    u_Nm3 = u_array[-4]
    u_Nm4 = u_array[-5]
    u_prime_end = (25*u_N - 48*u_Nm1 + 36*u_Nm2 - 16*u_Nm3 + 3*u_Nm4) / (12 * dr)
    
    # evaluating formula terms
    V_edge = potential_solver(r_box, Z)
    k_end = k_solver(E, V_edge)
    x = k_end * r_box
    
    j_val = sph_bessel(l, x)
    j_der = sph_bessel_deriv(l, x)
    n_val = sph_neumann(l, x)
    n_der = sph_neumann_deriv(l, x)
    
    # K_times_u = u' - u/r (to avoid numerical instability when calculating logarithmic derivative)
    K_times_u = u_prime_end - (u_N / r_box)

    # standard formula: (K*j - k*j') / (K*n - k*n')
    # multiplied by u:  ((Ku)*j - u*k*j') / ((Ku)*n - u*k*n')
    
    num_robust = K_times_u * j_val - u_N * k_end * j_der
    den_robust = K_times_u * n_val - u_N * k_end * n_der
    
    # implementing sign correction:
    # multiplied by u_N to fix the singularity
    # but if u_N < 0, we flipped the vector into the opposite quadrant (adding pi)
    # so we can calculate the sign of u_N to un-flip it
    # using jnp.sign: returns -1, 0, or 1. 
    
    u_sign = jnp.sign(u_N)
    
    return num_robust * u_sign, den_robust * u_sign

@partial(jit, static_argnames = ['N_points', 'l'])
def phase_shift_denominator(E, r_box, r_start, N_points, l, Z):
    """just a function for the denominator of the phase shift
    useful for the root-finding required in the next function correction_grid_solver
    
    inputs:
    E, r_box, r_start, N_points, l, Z - standard

    outputs:
    den - denominator from tan(delta) = - formula, phase shift formula
    """
    _, den = phase_shift_raw_solver(E, r_box, r_start, N_points, l, Z)
    return den

def correction_grid_solver(r_box, r_start, N_points, mu, T, l, Z, E_max):
    """
    function to create relevant grid for numerical integration. adds extra points where delta is expected to
    go through a phase shift and when f (occupancy) is expected to change as well

    inputs:
    r_box, r_start, N_points, mu, T, l, Z - standard
    E_max - max E we are integrating until for the correction term, doenst have to be inf because for example
    the integrand for N and U would be multiplied by the fermi-dirac occupation which decays as E increases

    outputs:
    grid_points - grid specialized for integration of this problem

    """
    # default grid, logarithmic spacing
    grid_points = [1e-16, 1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, E_max]
    
    # if mu is in relevant range of integration, adding grid points near it
    # f expected to change at around E = mu, and width of that change is also proportional to T
    if 0 < mu < E_max:
        grid_points.extend([mu - 4*T, mu - T, mu, mu + T, mu + 4*T])

    # search for when resonance occurs, or phase shift occurs
    # this happens when the denominator of the tan(delta) equation hits 0 because then tan(delta) --> inf
    # doing a coarse grid search first to find the region in which the root exists (and therefore sign changes)
    test_energies = jnp.logspace(jnp.log10(1e-16), jnp.log10(E_max), 200)
    denoms = jax.vmap(lambda E: phase_shift_denominator(E, r_box, r_start, N_points, l, Z))(test_energies)
    sign_changes = jnp.where(jnp.diff(jnp.sign(denoms)))[0] # returns array of indices where sign changes occur
    
    # and then after that, use scipy brentq (root solver) to find exact E of resonance
    for idx in sign_changes:
        e_low = test_energies[idx]
        e_high = test_energies[idx+1]
        try: # using try in case brentq fails for very small energy values
            func_to_root = lambda e_val: float(phase_shift_denominator(e_val, r_box, r_start, N_points, l, Z))
            E_res = scipy.optimize.brentq(func_to_root, float(e_low), float(e_high)) # performing root finding
            
            # places points around E_res to ensure we catch the phase shift
            factors = [0.1, 0.5, 0.9, 0.99, 0.999]
            cluster = [E_res]
            for f in factors:
                cluster.append(E_res * f)
                cluster.append(E_res / f)
            
            grid_points.extend(cluster)
        except:
            pass

    # sorting and filtering final grid 
    grid_points = jnp.array(grid_points)
    grid_points = jnp.sort(grid_points)
    grid_points = grid_points[grid_points >= 1e-16]
    grid_points = grid_points[grid_points <= E_max]
    
    # filtering out duplicates, since that can happen if E_res coincides with the Fermi edge for example
    unique_mask = jnp.concatenate([jnp.array([True]), jnp.diff(grid_points) > 1e-14 * grid_points[1:]])
    return grid_points[unique_mask]

@partial(jit, static_argnames=['N_points', 'l'])
def correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z):
    """
    computes correction term for N and U using change in delta directly 
    this is for each l, and then sum over l later until l_max in thermo_solver_corrected to get
    full correction term

    inputs:
    grid - grid of points used for integration obtained from correction_grid_solver function above
    r_box, r_start, N_points, mu, T, l, Z - standard

    outputs:
    N_corr - correction term for N
    U_corr - correction term for U

    """
    # obtaining numerator and denominator of phase shift equation for all points on the energy grid
    nums, dens = jax.vmap(lambda E: phase_shift_raw_solver(E, r_box, r_start, N_points, l, Z))(grid)
    
    # computing delta, and unwrapping so that we get a smooth and continuous curve for the phase shift
    # basically if difference is larger than pi, then it adds or subtracts 2pi to make it continous 
    phases = jnp.arctan2(nums, dens)
    phases_unwrapped = jnp.unwrap(phases)
    
    # calculating difference between phase shift of each point on energy grid
    d_phases = jnp.diff(phases_unwrapped)
    
    # for the dos factor: correction term = sum((2/pi) * (2l+1) * d_delta)
    factor = (2.0 / jnp.pi) * (2*l + 1)
    
    # integrating for N, N_corr = factor * sum(occup_avg * phase shift in that interval)
    N_integrand = fd_occup_solver(grid, mu, T)
    N_integrand_avg = 0.5 * (N_integrand[1:] + N_integrand[:-1])
    N_corr = factor * jnp.sum(N_integrand_avg * d_phases)
    
    # integrating for U, U_corr = factor * sum(average(occup * energy) * phase shift)
    U_integrand = grid * N_integrand
    U_integrand_avg = 0.5 * (U_integrand[1:] + U_integrand[:-1])
    U_corr = factor * jnp.sum(U_integrand_avg * d_phases)

    # integrating for S, S_corr = factor * sum(average(occup * log(occup) + (1-occup) * log(1-occup)) * phase shift)
    S_corr_integrand = xlogy(N_integrand, N_integrand) + xlogy(1-N_integrand, 1-N_integrand)
    S_corr_integrand_avg = 0.5 * (S_corr_integrand[1:] + S_corr_integrand[:-1])
    S_corr = -factor * jnp.sum(S_corr_integrand_avg * d_phases)
    
    return N_corr, U_corr, S_corr

@jit
def S_free_uncorrected_solver(N_free_unc, U_free_unc, mu, T):
    """ 
    for the free uncorrected entropy term, using the formula:
    S_free_unc = 1/T (5/3 U_free_unc - mu N_free_unc)

    inputs:
    N_free_unc - free uncorrected total electron number term
    U_free_unc - free uncorrected internal energy term
    mu - chemical potential
    T - temperature

    output:
    S_free_unc - free uncorrected term for entropy
       """
    # to avoid division by 0
    T_safe = jnp.maximum(T, 1e-12)   

    S_free_unc = 1/T_safe * ((5/3) * U_free_unc - mu * N_free_unc)

    return S_free_unc

def thermo_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max=50):
    """ 
    main solver for thermodynamic quantities: N, U, S (entropy)

    combined all in one because most of the calculation needed for U is the same as N, so waste of computational
    resources to calculate N and U separately if u want both quantities

    inputs:
    energies - grid of energy eigenvalues obtained from numerov solver
    mask - same dim as energies, 1 if bounded, 0 if free
    degeneracies - same dim as energies, 2*(2l+1) value corresponding to each entry in the grid
    r_box, r_start, N_points, mu, T, Z - standard
    l_max - max l to interate until for correction term

    outputs:
    N_total - total electron number, N_bound + N_free_uncorrected + N_free_correction
    U_total - total internal energy, U_bound + U_free_uncorrected + U_free_correction
    S_total - total entropy, S_bound + S_free_uncorrected + S_free_correction
        """

    # calculating bound state contributions of the physical quantities
    occup = fd_occup_solver(energies, mu, T)
    N_bound = jnp.sum(degeneracies * occup * mask)
    U_bound = jnp.sum(degeneracies * occup * energies * mask)
    S_bound = jnp.sum(-degeneracies * mask * (xlogy(occup, occup) + xlogy(1-occup, 1-occup)))
    
    # calculating free uncorrected contributions
    V = V_solver(r_box)
    gamma_factor_N = jnp.sqrt(jnp.pi) / 2
    N_free_unc = ((jnp.sqrt(2)*V*T**(3/2))/(jnp.pi**2)) * gamma_factor_N * fermi_dirac_integral_half(mu/T)
    
    gamma_factor_U = 3 * jnp.sqrt(jnp.pi) / 4
    U_free_unc = ((jnp.sqrt(2)*V*T**(5/2))/(jnp.pi**2)) * gamma_factor_U * fermi_dirac_integral_three_half(mu/T)

    S_free_unc = S_free_uncorrected_solver(N_free_unc, U_free_unc, mu, T)

    # calculating correction term
    N_corr_total = 0.0
    U_corr_total = 0.0
    S_corr_total = 0.0
    
    # setting max E to integrate until, at this E the occupancy is basically 0 so the integrand is also 0
    E_max = jnp.maximum(mu + 20 * T, 5)

    # adding contributions over each l
    for l in range(l_max):
        # generating relevant grid for integration
        grid = correction_grid_solver(r_box, r_start, N_points, mu, T, l, Z, E_max)
        
        # obtaining correction term for that l
        N_corr, U_corr, S_corr = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # adding to total of correction
        N_corr_total += N_corr
        U_corr_total += U_corr
        S_corr_total += S_corr
        
    # summing all contributions to get final physical quantity
    N_total = N_bound + N_free_unc + N_corr_total
    U_total = U_bound + U_free_unc + U_corr_total
    S_total = S_bound + S_free_unc + S_corr_total
    
    return N_total, U_total, S_total

def N_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max = 50):
    """ individual solver for N. if only solving for N then use this, if solving for other thermodynamic 
     quantities simulatenously, then use above thermo_solver_corrected
      
    inputs - same as above

    outputs:
    N_total - total electron number      
         """
    
    # calculating bound state contributions of the physical quantities
    occup = fd_occup_solver(energies, mu, T)
    N_bound = jnp.sum(degeneracies * occup * mask)
    
    # calculating free uncorrected contributions
    V = V_solver(r_box)
    gamma_factor_N = jnp.sqrt(jnp.pi) / 2
    N_free_unc = ((jnp.sqrt(2)*V*T**(3/2))/(jnp.pi**2)) * gamma_factor_N * fermi_dirac_integral_half(mu/T)
    
    # calculating correction term
    N_corr_total = 0.0
    
    # setting max E to integrate until, at this E the occupancy is basically 0 so the integrand is also 0
    E_max = jnp.maximum(mu + 20 * T, 5)

    # adding contributions over each l
    for l in range(l_max):
        # generating relevant grid for integration
        grid = correction_grid_solver(r_box, r_start, N_points, mu, T, l, Z, E_max)
        
        # obtaining correction term for that l
        N_corr, _, _ = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # adding to total of correction
        N_corr_total += N_corr
        
    # summing all contributions to get final physical quantity
    N_total = N_bound + N_free_unc + N_corr_total

    return N_total

def U_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max = 50):
    """ individual solver for U. if only solving for U then use this, if solving for other thermodynamic 
     quantities simulatenously, then use above thermo_solver_corrected
      
    inputs - same as above

    outputs:
    U_total - total internal energy
         """
    # calculating bound state contributions of the physical quantities
    occup = fd_occup_solver(energies, mu, T)
    U_bound = jnp.sum(degeneracies * occup * energies * mask)
    
    # calculating free uncorrected contributions
    V = V_solver(r_box)
    gamma_factor_U = 3 * jnp.sqrt(jnp.pi) / 4
    U_free_unc = ((jnp.sqrt(2)*V*T**(5/2))/(jnp.pi**2)) * gamma_factor_U * fermi_dirac_integral_three_half(mu/T)
    
    # calculating correction term
    U_corr_total = 0.0
    
    # setting max E to integrate until, at this E the occupancy is basically 0 so the integrand is also 0
    E_max = jnp.maximum(mu + 20 * T, 5)

    # adding contributions over each l
    for l in range(l_max):
        # generating relevant grid for integration
        grid = correction_grid_solver(r_box, r_start, N_points, mu, T, l, Z, E_max)
        
        # obtaining correction term for that l
        _, U_corr, _ = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # adding to total of correction
        U_corr_total += U_corr
        
    # summing all contributions to get final physical quantity
    U_total = U_bound + U_free_unc + U_corr_total
    
    return U_total

def S_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max = 50):
    """ individual solver for S. if only solving for S then use this, if solving for other thermodynamic 
     quantities simulatenously, then use above thermo_solver_corrected
      
    inputs - same as above

    outputs:
    S_total - total entropy
         """
    # calculating bound state contributions of the physical quantities
    occup = fd_occup_solver(energies, mu, T)
    S_bound = jnp.sum(-degeneracies * mask * (xlogy(occup, occup) + xlogy(1-occup, 1-occup)))
    
    # calculating free uncorrected contributions
    V = V_solver(r_box)
    gamma_factor_N = jnp.sqrt(jnp.pi) / 2
    N_free_unc = ((jnp.sqrt(2)*V*T**(3/2))/(jnp.pi**2)) * gamma_factor_N * fermi_dirac_integral_half(mu/T)
    
    gamma_factor_U = 3 * jnp.sqrt(jnp.pi) / 4
    U_free_unc = ((jnp.sqrt(2)*V*T**(5/2))/(jnp.pi**2)) * gamma_factor_U * fermi_dirac_integral_three_half(mu/T)

    S_free_unc = S_free_uncorrected_solver(N_free_unc, U_free_unc, mu, T)

    # calculating correction term
    S_corr_total = 0.0
    
    # setting max E to integrate until, at this E the occupancy is basically 0 so the integrand is also 0
    E_max = jnp.maximum(mu + 20 * T, 5)

    # adding contributions over each l
    for l in range(l_max):
        # generating relevant grid for integration
        grid = correction_grid_solver(r_box, r_start, N_points, mu, T, l, Z, E_max)
        
        # obtaining correction term for that l
        _, _, S_corr = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # adding to total of correction
        S_corr_total += S_corr
        
    # summing all contributions to get final physical quantity
    S_total = S_bound + S_free_unc + S_corr_total
    
    return S_total

def F_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max = 50):
    """ 
    solver for Helmholtz free energy

    formula:
    F = U - TS

    inputs:
    same as above

    output:
    F - helmholtz free energy
        """
        
    _, U, S = thermo_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max)
    F = U - T * S
    return F

def mu_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, Z, T, iteration_count=50, l_max=50):
    """ 
    to find chemical potential (mu) using bisection method

    inputs:
    e,m,d,r_box, r_start, N_points, Z, T - standard
    iteration_count - how many iterations for bisection method, default = 50
    l_max - max l to iterate until for correction term, default = 50

    output:
    mu - chemical potential
        """
    
    # initial boundaries
    a = -50 * jnp.maximum(1, T * jnp.log(T)) * jnp.maximum(1, Z**2)
    b = 50 * jnp.maximum(1, 1 / (r_box ** 2))
    
    # implementing bisection method (same logic as uncorrected version)
    for _ in range(iteration_count):
        c = (a + b) / 2.0
        N_c = N_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, c, T, Z, l_max)
        
        if N_c < Z:
            a = c
        else:
            b = c
            
    return (a + b) / 2.0
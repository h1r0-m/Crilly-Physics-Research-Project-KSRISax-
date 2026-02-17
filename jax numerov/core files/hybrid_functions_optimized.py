
""" optimized phase shift dos corrected solvers, same functions as hybrid_functions_corrected.py but much faster """

# housekeeping
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jit, lax
from functools import partial
from bessel_function import sph_bessel, sph_bessel_deriv, sph_neumann, sph_neumann_deriv
from njax_functions import numerov_solver
from fermi_dirac_integral import fermi_dirac_integral_half, fermi_dirac_integral_three_half
from jax.scipy.special import xlogy

jax.config.update("jax_enable_x64", True)

@jit
def V_solver(r_box):
    """ function for calculating volume of spherical atom
    
    input: r_box
    output: V or volume   
    """

    return (4/3) * jnp.pi * r_box ** 3

@jit
def fd_occup_solver(E, mu, T):
    """ function for calculating fermi-dirac occupation
     
    formula:
    f(E,mu,T) = 1/(exp((E-mu)/T) + 1)

    inputs:
    E - energy
    mu - chemical potential
    T - temperature 

    output:
    f - fermi-dirac occupation
       """

    return sigmoid((mu - E) / T)

@partial(jit, static_argnames=['N_points', 'l_max'])
def bounded_states_solver(r_box, r_start, N_points, Z, l_max=5, is_log_grid = True):
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
    numerov_vect = jax.vmap(numerov_solver, in_axes=(None, None, None, 0, None, None))
    energies, _ = numerov_vect(r_box, r_start, N_points, l_array, Z, is_log_grid)
    
    # creating mask for bound states
    mask = jnp.where(energies < 0, 1, 0)

    # creating degeneracy matrix
    l_grid = jnp.repeat(l_array[:, None], energies.shape[1], axis=1)
    degeneracies = 2 * (2 * l_grid + 1)
    
    return energies, mask, degeneracies

@jit
def potential_solver(r, Z):
    """ to get the potential, didn't integrate it in the solver so i can easily change the potential later when using SCF solver
    
    just using simple coulomb potential: V = -Z/r for now
        
    inputs:
    r - radius (in Ha)
    Z - atomic number

    output:
    V - potential
            """
    return -Z/r

@jit
def k_solver(E, V):
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
    
    # just making sure it doesnt produce NaN if inside of sqrt is negative
    val = 2 * (E - V)
    return jnp.sqrt(jnp.maximum(val, 0.0))

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

@partial(jit, static_argnames=['N_points'])
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

    revised to avoid K = u'/u singularity when u --> 0
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

@jit
def correction_grid_solver_jax(mu, T, E_max):
    """
    function for grid generation that is compatible for jit compilation
    instead of finding where resonance occurs, can just create a fixed-size,
    high-density grid that concentrates points both closer to E = 0 - for resonance) and around E = mu (for when the fermi-dirac occupation changes most suddenly)

    inputs:
    mu - chemical potential
    T - temperature
    E_max - max E for grid generation, usually when fermi-dirac occupation becomes negligible

    output:
    full_grid - grid of energy points that can later be used to calculate correction terms
    """
    
    # base logarithmic grid of 300 points, can make finer if wanted
    log_grid = jnp.logspace(jnp.log10(1e-4), jnp.log10(E_max), 300)
    
    # very low energy grid of 200 points to capture resonance around E = 0
    low_e_grid = jnp.linspace(1e-16, 1e-4, 200)
    
    # grid for mu of 41 points to capture change in fermi-dirac occupation
    # this "width" of change depends on T, because higher T means more spread out
    # and lower T means more sudden change. therefore this grid width is from -4T
    # to +4T from E = mu to capture this behavior accurately
    mu_window = jnp.linspace(-4.0, 4.0, 41) * T
    mu_grid = mu + mu_window
    
    # filtering points to mu_grid that are invalid (e.g. if E <= 0 or E > E_max)
    # replacing invalid points with a dummy value (e.g. 1e-10), cuz arrays have
    # to be fixed size for jit compilation
    mu_grid = jnp.where((mu_grid > 1e-16) & (mu_grid < E_max), mu_grid, 1e-10)

    # concatenating all grids 
    full_grid = jnp.concatenate([low_e_grid, log_grid, mu_grid])
    
    # sorting the grid 
    full_grid = jnp.sort(full_grid)
    
    # just a note, but even if there are duplicate points in the grid, shouldnt
    # affect calculation because the trapezoidal rule later to calculate
    # correction would just add 0 contribution if change in E is 0 in that energy interval

    return full_grid

@partial(jit, static_argnames=['N_points'])
def correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z):
    """
    computes correction term for N and U using change in delta directly 
    this is for each l, and then sum over l later until l_max in thermo_solver_corrected to get full correction term

    inputs:
    grid - grid of points used for integration obtained from correction_grid_solver function above
    r_box, r_start, N_points, mu, T, l, Z - standard

    outputs:
    N_corr - correction term for N (# of particle/electron)
    U_corr - correction term for U (internal energy)
    S_corr - correction term for S (entropy)
    """
    # vectorizing computation of phase shift over the grid
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
    N_avg = 0.5 * (N_integrand[1:] + N_integrand[:-1])
    N_corr = factor * jnp.sum(N_avg * d_phases)
    
    # integrating for U, U_corr = factor * sum(average(occup * energy) * phase shift)
    U_integrand = grid * N_integrand
    U_avg = 0.5 * (U_integrand[1:] + U_integrand[:-1])
    U_corr = factor * jnp.sum(U_avg * d_phases)

    # integrating for S, S_corr = factor * sum(average(occup * log(occup) + (1-occup) * log(1-occup)) * phase shift)
    term1 = xlogy(N_integrand, N_integrand)
    term2 = xlogy(1-N_integrand, 1-N_integrand)
    S_int = term1 + term2
    S_avg = 0.5 * (S_int[1:] + S_int[:-1])
    S_corr = -factor * jnp.sum(S_avg * d_phases)
    
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

@partial(jit, static_argnames=['N_points', 'l_max'])
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

    # calculating bound state contributions of the physical contributions
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

    # endpoint for energy grid, when fermi-dirac occupation becomes negligible
    # f = 1/(exp((E-mu)/T) + 1) say when that becomes e^-20, such that
    # exp((E-mu)/T) = exp(20) - 1 approx exp(20) --> E-mu/T = 20 and
    # E = 20*T + mu, therefore choosing that as E_max, default value of 5
    E_max = jnp.maximum(mu + 20 * T, 5.0)

    # getting grid
    grid = correction_grid_solver_jax(mu, T, E_max)

    # using lax.scan to compute correction term across l 
    def scan_body(carry, l):
        # current correction term totals
        n_acc, u_acc, s_acc = carry
        
        # computing additional correction term
        n_c, u_c, s_c = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # adding to carry
        new_carry = (n_acc + n_c, u_acc + u_c, s_acc + s_c)

        return new_carry, None

    # initiol carry for correction terms
    init_carry = (0.0, 0.0, 0.0)

    l_values = jnp.arange(l_max)
    
    # computing correction term for all quantities
    (N_corr_total, U_corr_total, S_corr_total), _ = lax.scan(scan_body, init_carry, l_values)

    # summing all contributions to get final physical quantity
    N_total = N_bound + N_free_unc + N_corr_total
    U_total = U_bound + U_free_unc + U_corr_total
    S_total = S_bound + S_free_unc + S_corr_total
    
    return N_total, U_total, S_total

@partial(jit, static_argnames=['N_points', 'l_max'])
def N_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max=50):
    """ individual solver for N. if only solving for N then use this, if solving for other thermodynamic 
     quantities simulatenously, then use above thermo_solver_corrected
      
    inputs - same as above

    outputs:
    N_total - total electron number      
         """

    # calculating bound state contributions of the physical contributions
    occup = fd_occup_solver(energies, mu, T)
    N_bound = jnp.sum(degeneracies * occup * mask)
    
    # calculating free uncorrected contributions
    V = V_solver(r_box)
    gamma_factor_N = jnp.sqrt(jnp.pi) / 2
    N_free_unc = ((jnp.sqrt(2)*V*T**(3/2))/(jnp.pi**2)) * gamma_factor_N * fermi_dirac_integral_half(mu/T)
    
    # endpoint for energy grid, when fermi-dirac occupation becomes negligible
    # f = 1/(exp((E-mu)/T) + 1) say when that becomes e^-20, such that
    # exp((E-mu)/T) = exp(20) - 1 approx exp(20) --> E-mu/T = 20 and
    # E = 20*T + mu, therefore choosing that as E_max
    # default value of 5 because this might turn out negative if mu is negative
    # which will break the calculation since it includes jnp.log10(E_max)
    E_max = jnp.maximum(mu + 20 * T, 5.0)

    # getting grid
    grid = correction_grid_solver_jax(mu, T, E_max)

    # using lax.scan to compute correction term across l 
    def scan_body(carry, l):
        # current correction term totals
        n_acc = carry
        
        # computing additional correction term
        n_c, _, _ = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # adding to carry
        new_carry = n_acc + n_c

        return new_carry, None

    # initiol carry for correction terms
    init_carry = 0.0

    l_values = jnp.arange(l_max)
    
    # computing correction term for all quantities
    N_corr_total, _ = lax.scan(scan_body, init_carry, l_values)

    # summing all contributions to get final physical quantity
    N_total = N_bound + N_free_unc + N_corr_total

    return N_total

@partial(jit, static_argnames=['N_points', 'l_max'])
def U_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max=50):
    """ individual solver for U. if only solving for U then use this, if solving for other thermodynamic 
     quantities simulatenously, then use above thermo_solver_corrected
      
    inputs - same as above

    outputs:
    U_total - total internal energy
         """

    # calculating bound state contributions of the physical contributions
    occup = fd_occup_solver(energies, mu, T)
    U_bound = jnp.sum(degeneracies * occup * energies * mask)
    
    # calculating free uncorrected contributions
    V = V_solver(r_box)
    gamma_factor_U = 3 * jnp.sqrt(jnp.pi) / 4
    U_free_unc = ((jnp.sqrt(2)*V*T**(5/2))/(jnp.pi**2)) * gamma_factor_U * fermi_dirac_integral_three_half(mu/T)

    # endpoint for energy grid, when fermi-dirac occupation becomes negligible
    # f = 1/(exp((E-mu)/T) + 1) say when that becomes e^-20, such that
    # exp((E-mu)/T) = exp(20) - 1 approx exp(20) --> E-mu/T = 20 and
    # E = 20*T + mu, therefore choosing that as E_max, default value of 5
    E_max = jnp.maximum(mu + 20 * T, 5.0)

    # getting grid
    grid = correction_grid_solver_jax(mu, T, E_max)

    # using lax.scan to compute correction term across l 
    def scan_body(carry, l):
        # current correction term totals
        u_acc = carry
        
        # computing additional correction term
        _, u_c, _ = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # adding to carry
        new_carry = u_acc + u_c

        return new_carry, None

    # initiol carry for correction terms
    init_carry = 0.0

    l_values = jnp.arange(l_max)
    
    # computing correction term for all quantities
    U_corr_total, _ = lax.scan(scan_body, init_carry, l_values)

    # summing all contributions to get final physical quantity
    U_total = U_bound + U_free_unc + U_corr_total
    
    return U_total

@partial(jit, static_argnames=['N_points', 'l_max'])
def S_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, mu, T, Z, l_max=50):
    """ individual solver for S. if only solving for S then use this, if solving for other thermodynamic 
     quantities simulatenously, then use above thermo_solver_corrected
      
    inputs - same as above

    outputs:
    S_total - total entropy
         """

    # calculating bound state contributions of the physical contributions
    occup = fd_occup_solver(energies, mu, T)
    S_bound = jnp.sum(-degeneracies * mask * (xlogy(occup, occup) + xlogy(1-occup, 1-occup)))
    
    # calculating free uncorrected contributions
    V = V_solver(r_box)
    gamma_factor_N = jnp.sqrt(jnp.pi) / 2
    N_free_unc = ((jnp.sqrt(2)*V*T**(3/2))/(jnp.pi**2)) * gamma_factor_N * fermi_dirac_integral_half(mu/T)
    
    gamma_factor_U = 3 * jnp.sqrt(jnp.pi) / 4
    U_free_unc = ((jnp.sqrt(2)*V*T**(5/2))/(jnp.pi**2)) * gamma_factor_U * fermi_dirac_integral_three_half(mu/T)

    S_free_unc = S_free_uncorrected_solver(N_free_unc, U_free_unc, mu, T)

    # endpoint for energy grid, when fermi-dirac occupation becomes negligible
    # f = 1/(exp((E-mu)/T) + 1) say when that becomes e^-20, such that
    # exp((E-mu)/T) = exp(20) - 1 approx exp(20) --> E-mu/T = 20 and
    # E = 20*T + mu, therefore choosing that as E_max, default value of 5
    E_max = jnp.maximum(mu + 20 * T, 5.0)

    # getting grid
    grid = correction_grid_solver_jax(mu, T, E_max)

    # using lax.scan to compute correction term across l 
    def scan_body(carry, l):
        # current correction term totals
        s_acc = carry
        
        # computing additional correction term
        s_c = correction_value_solver(grid, r_box, r_start, N_points, mu, T, l, Z)
        
        # adding to carry
        new_carry = s_acc + s_c

        return new_carry, None

    # initiol carry for correction terms
    init_carry = 0.0

    l_values = jnp.arange(l_max)
    
    # computing correction term for all quantities
    S_corr_total, _ = lax.scan(scan_body, init_carry, l_values)

    # summing all contributions to get final physical quantity
    S_total = S_bound + S_free_unc + S_corr_total
    
    return S_total

@partial(jit, static_argnames=['N_points', 'l_max'])
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

@partial(jit, static_argnames=['N_points', 'iteration_count', 'l_max'])
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
    a = -50 * jnp.maximum(1.0, T * jnp.log(T)) * jnp.maximum(1.0, Z**2)
    b = 50 * jnp.maximum(1.0, 1.0 / (r_box ** 2))
    
    # implementing bisection method
    def body_fun(i, val):
        # current endpoints
        a_curr, b_curr = val

        # midpoint
        c = (a_curr + b_curr) / 2.0
        
        # calculating N at mu = c
        N_c = N_solver_corrected(energies, mask, degeneracies, r_box, r_start, N_points, c, T, Z, l_max)
        
        # if N_c < Z, then N = Z must be somehwere between mu = c and mu = b, so change a to c
        # if N_c > Z, opposite to change b to c
        a_next = jnp.where(N_c < Z, c, a_curr)
        b_next = jnp.where(N_c < Z, b_curr, c)
        
        return (a_next, b_next)

    # running loop
    final_a, final_b = lax.fori_loop(0, iteration_count, body_fun, (a, b))
    
    return (final_a + final_b) / 2.0
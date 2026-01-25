import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from njax_functions import numerov_solver, U_solver_nonhybrid
from fermi_dirac_integral import fermi_dirac_integral_half, fermi_dirac_integral_three_half # from Crilly's repo, for the fermi dirac integrals

from rich.traceback import install
install()

jax.config.update("jax_enable_x64", True)

# just to make my life easier

@jit
def V_solver(r_box):
    return (4/3) * jnp.pi * r_box ** 3

@jit
def fd_occup_solver(E, mu, T):
    occup = 1/(jnp.exp((E-mu)/T) + 1)
    return occup

# main solver to find the bound states

@partial(jit, static_argnames=['N_points', 'l_max'])
def bounded_states_solver(r_box, r_start, N_points, Z, l_max = 5):
    """
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
    numerov_vect = vmap(numerov_solver, in_axes=(None, None, None, 0, None))
    energies, _ = numerov_vect(r_box, r_start, N_points, l_array, Z)
    
    # creating mask for bound states
    mask = jnp.where(energies < 0, 1, 0)
    
    # creating degeneracy matrix
    l_grid = jnp.repeat(l_array[:, None], energies.shape[1], axis=1) # same dimensions as energies, filled with corresponding l values
    degeneracies = 2 * (2 * l_grid + 1) # degeneracy matrix for each energy entry
    
    return energies, mask, degeneracies

# required for finding mu

@jit
def N_solver_uncorrected(r_box, energies, mask, degeneracies, mu, T):
    """  
    logic:

    N = N_bound + N_free

    N_bound = sum over all bounded n,l states of (degeneracy factor * occupancy) 

    N_free = integral of E from 0 to infinity (density of states * occupancy) dE
         = sqrt(2) V / pi^2 * int_0^inf E^1/2 / (exp((E-mu)/T)+1) dE
         using change of variables of u = E/T, 
         = sqrt(2) V T^3/2 / pi^2 * int_0^+inf u^1/2 / (exp(u-mu/T) + 1) du
         = sqrt(2) V T^3/2 / pi^2 * F_1/2 (mu/T), where F_j(eta) is the fermi-dirac equation such that:
         F_j(eta) = int_0^inf x^j / (exp(x-eta) + 1) dx

    however, the fermi-dirac integral in fermi_dirac_integral.py is normalized, such that F_j(eta) = 1/gamma(j+1) * int --
    so a gamma factor is multiplied below such that it is correct

    inputs:
    r_box, energies, mask, degeneracies - standard
    mu - chemical potential
    T - temperature

    output:
    N - total electron number in the atom
    """

    # getting volume
    V = V_solver(r_box)
    
    # finding N for bounded electrons
    occup = fd_occup_solver(energies, mu, T)
    N_bound = jnp.sum(degeneracies * occup * mask)
    
    # calculating gamma factor
    gamma_factor = jnp.sqrt(jnp.pi) / 2

    # finding N for free electrons
    N_free = ((jnp.sqrt(2) * V * T**(3/2)) / (jnp.pi**2)) * gamma_factor * fermi_dirac_integral_half(mu/T)

    # total electrons
    N = N_bound + N_free

    return N

# for finding internal energy of the atom, through separate continuum treatmenet
@jit
def U_solver_uncorrected(r_box, energies, mask, degeneracies, mu, T):
    """ similar to N_solver, but factors multiplied by the energy
    
    logic:
    U = U_bound + U_free

    U_bound = sum over all bounded n,l states (E * degeneracy factor * occupancy)

    U_free = int_0^+inf (density of state * E * occupancy)
        = sqrt(2) * V * T^5/2 / pi^2 * F_3/2 (mu/T)

    inputs:
    r_box, energies, mask, degeneracies - standard
    mu - chemical potential
    T - temperature

    outputs:
    U - internal energy of the atom
       """
    # finding volume
    V = V_solver(r_box)
    
    # bound internal energy
    occup = fd_occup_solver(energies, mu, T)
    U_bound = jnp.sum(energies * degeneracies * occup * mask)
    
    # free internal energy
    gamma_factor = 3 * jnp.sqrt(jnp.pi) / 4
    U_free = ((jnp.sqrt(2) * V * T**(5/2)) / (jnp.pi**2)) * gamma_factor * fermi_dirac_integral_three_half(mu/T)

    # total internal energy
    U = U_bound + U_free

    jax.debug.print("U_bound: {x}", x=U_bound)
    jax.debug.print("U_free: {x}", x = U_free)

    return U

# finding mu through bisection method

@partial(jit, static_argnames=['iteration_count'])
def mu_solver_uncorrected(energies, mask, degeneracies, r_box, Z, T, iteration_count = 50):
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
    
    # performing bisection method
    for _ in range(iteration_count):
        # midpoint
        c = (a + b) / 2
        
        # calculating N at mu = c
        N_c = N_solver_uncorrected(r_box, energies, mask, degeneracies, c, T)
        
        # if N_c < Z, then N = Z must be somehwere between mu = c and mu = b, so change a to c
        # if N_c > Z, opposite to change b to c
        a = jnp.where(N_c < Z, c, a)
        b = jnp.where(N_c >= Z, c, b)
    
    c = (a + b) / 2

    return c
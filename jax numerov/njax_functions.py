# housekeeping

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import jit
from functools import partial

# %% functions

# @jit for faster running but N_points is a static argument so using @partial
@partial(jit, static_argnames = ['N_points'])
def numerov_solver(r_box, r_start, N_points, l):
    """ 
    inputs(all in atomic units):
    r_box: endpoint for r_points, infinite potential (hard wall)
    r_start: r_points(0) <-- not 0 because singularity
    N_points: number of points including the boundaries
    l: orbital quantum number 

    outputs:
    energies: column of the energies for each energy state
    psi: matrix of (N_points-2) x (N_points-2), with the nth column representing
    the wave function for the nth energy state
    """
    
    # defining distance interval and creating array for points of analysis
    d = (r_box - r_start) / (N_points - 1)
    r_points = jnp.linspace(r_start, r_box, N_points)

    # creating matrices for numerov
    A_lower = jnp.ones(N_points-3)
    A_mid = -2 * jnp.ones(N_points-2)
    A_upper = jnp.ones(N_points-3)
    A = (jnp.diag(A_lower, k = -1) + jnp.diag(A_mid, k = 0) + jnp.diag(A_upper, k = 1)) / d ** 2

    B_lower = jnp.ones(N_points - 3)
    B_mid = 10 * jnp.ones(N_points-2)
    B_upper = jnp.ones(N_points-3)
    B = (jnp.diag(B_lower, k = -1) + jnp.diag(B_mid, k = 0) + jnp.diag(B_upper, k = 1)) / 12

    # potential terms, V_eff = coulomb + centrifugal for now
    V_eff_vec = -1 / r_points + l * (l+1) / (2 * r_points ** 2)
    V_eff = jnp.diag(V_eff_vec[1:-1], k = 0)

    # constructing Hamiltonian matrix
    H = -1/2 * A + B @ V_eff

    # obtaining eigenvals / vecs from cholesky decomposition
    energies, psi = cholesky_solve(H,B)

    return energies, psi

@jit
def cholesky_solve(A,B):
    """ 
    basically a generalized eigenvalue/vec solver, for the form Ax = lambda B x
    performing cholesky decomposition, uses the eigh function
    
    background:
    B = L L^T --> Ax = lambda L L^T x --> L^-1 A x = lambda L^T x = lambda y
    y defined to be y = L^T x 
    L^-1 A L^-T y = A_tilde y = lambda y --> can use eigh function from jax to obtain y
    obtain x through x = L^-T y

    inputs:
    A,B: the matrices mentioned above

    outputs:
    eigvals, eigenvecs of the original equation (in that order)
    
    """
    L = jla.cholesky(B, lower=True)

    # can use jla.inv but if a matrix is triangular, can use solve_triangular
    # to basically solve Lx = I --> x = L^-1 (more computationally efficient)
    L_inv = jla.solve_triangular(L,jnp.eye(L.shape[0]), lower = True)

    A_tilde = L_inv @ A @ L_inv.T

    eigvals, eigvecs_tilde = jla.eigh(A_tilde)

    eigvecs = L_inv.T @ eigvecs_tilde

    return eigvals, eigvecs

@partial(jit, static_argnames = ['N_points', 'l_max'])
def U_solver(temp, r_box, r_start, N_points, l_max):
    """ 
    implementing boltzmann statistics for the calculation of
    internal energy as a function of temperature and r_max

    inputs: 
    temp - temperature (in Ha, 1 Ha = 315775 K)
    r_box - box size for solving the Schrodinger equation (in Ha)
    r_start - start point for r_points, r_points[0]
    N_points - # of r_points used for numerov solver
    l_max - max l we consider until, when looping through the numerov solver
    to obtain energy eigenvalues for different l

    outputs: 
    U - scalar value of the internal energy (in Ha, 1 Ha = 27.2 eV)
    """

    l_array = jnp.arange(l_max + 1)

    energies = jnp.zeros((len(l_array), N_points-2))

    # everything constant except for the last input for numerov_solver which is l
    numerov_vect = jax.vmap(numerov_solver, in_axes = (None, None, None, 0))
    energies, _ = numerov_vect(r_box, r_start, N_points, l_array)

    # calculating degeneracy factors, basically how many slots open for each l
    # 2 for spin, and 2*l+1 for magnetic quantum number so:
    g_l = 2*(2*l_array + 1)

    # resizing matrix so it can be used for matrix multiplication
    g_l_matrix = g_l[:, None]

    # calculating boltzmann factor using the degeneracy factor and boltzmann statistics
    f_i = g_l_matrix * jnp.exp(-energies / temp)

    # calculating normalization constant to divide everything by, such that
    # all the probabilities will add up to 1
    Z = jnp.sum(f_i)

    U = jnp.sum(energies * f_i) / Z

    return U

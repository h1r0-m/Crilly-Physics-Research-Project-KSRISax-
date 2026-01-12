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
    performing cholesky decomposition
    
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

@jit
def U_solver(energies, temp):
    """ 
    implementing boltzmann statistics for the calculation of
    internal energy as a function of temperature and r_max

    inputs: 
    energies - 1D array of the eigenvalues from numerov_solver
    temp - scalar temperature value (in Ha, 1 Ha = 315,775 K)
    r_max - scalar value representing confinement and density

    outputs: 
    U - scalar value of the internal energy (in Ha, 1 Ha = 27.2 eV)
    """

    f_i = jnp.exp(-energies/temp) / sum(jnp.exp(-energies/temp))

    U = sum(energies * f_i)

    return U
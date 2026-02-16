# housekeeping

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import jit, lax
from jax.nn import sigmoid
from functools import partial
from jax.experimental import sparse
jax.config.update("jax_enable_x64", True)

# %% functions

# @jit for faster running but N_points is a static argument so using @partial
@partial(jit, static_argnames = ['N_points', 'use_log_grid'])
def numerov_solver(r_box, r_start, N_points, l, Z = 1, use_log_grid = True):
    """ 
    solves the radial schrodinger equation using numerov method.
    supports both linear and logarithmic grids

    inputs(all in atomic units):
    r_box: endpoint for r_points, infinite potential (hard wall)
    r_start: r_points(0) <-- not 0 because singularity
    N_points: number of points including the boundaries
    l: orbital quantum number 
    Z: atomic number, charge in the nucleus

    outputs:
    energies: column of the energies for each energy state
    psi: matrix of (N_points-2) x (N_points-2), with the nth column representing
    the wave function for the nth energy state
    """
    
    if use_log_grid:
        # setting up logarithmic grid, defining uniform grid in x = ln(r)
        x_start = jnp.log(r_start)
        x_end   = jnp.log(r_box)
        dx      = (x_end - x_start) / (N_points - 1)
        x_points = jnp.linspace(x_start, x_end, N_points)
        r_points = jnp.exp(x_points)
        
        # numerov matrices (A and B) using dx
        diag_main = -2 * jnp.ones(N_points - 2)
        diag_off  =  1 * jnp.ones(N_points - 3)
        
        A = (jnp.diag(diag_main, k=0) + jnp.diag(diag_off, k=-1) + jnp.diag(diag_off, k=1)) / dx**2
        
        B_main = 10 * jnp.ones(N_points - 2)
        B_off  =  1 * jnp.ones(N_points - 3)
        B = (jnp.diag(B_main, k=0) + jnp.diag(B_off, k=-1) + jnp.diag(B_off, k=1)) / 12

        # Hamiltonian Construction for Log Grid
        # equation is: -1/2 d^2/dx^2 + U_eff = E * r^2
        
        # r squared (needed for the RHS of the generalized eigenvalue problem)
        r2_vec = r_points[1:-1]**2
        R2 = jnp.diag(r2_vec)
        
        # potential V(r)
        V_r = -Z / r_points[1:-1] # Coulomb
        
        # effective potential in log coordinates
        # U_diag = r^2 * V(r) + 1/2 * (l + 1/2)^2
        centrifugal_log = 0.5 * (l + 0.5)**2
        U_vec = (r2_vec * V_r) + centrifugal_log
        U_diag = jnp.diag(U_vec)
        
        # H = -1/2 * A + B * U_eff
        H = -0.5 * A + B @ U_diag
        
        # the metric matrix (RHS)
        # solving H * psi = E * (B * r^2) * psi
        S = B @ R2 
        
        # solving generalized eigenvalue problem
        energies, psi_transformed = cholesky_solve(H, S)
        
        # psi_transformed is u(r)/sqrt(r)
        return energies, psi_transformed

    else:
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
        V_eff_vec = -Z / r_points + l * (l+1) / (2 * r_points ** 2)
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

@partial(jit, static_argnames=['N_points', 'l_max'])
def U_solver_nonhybrid(temp, r_box, r_start, N_points, Z_atom, l_max=10):
    """
    Calculates internal energy using Fermi-Dirac statistics.
    Solves for chemical potential (mu) to satisfy charge neutrality.
    """
    l_array = jnp.arange(l_max + 1)
    
    # 1. Get Energies (using Log Grid for accuracy)
    # Note: We pass use_log_grid=True
    numerov_vect = jax.vmap(numerov_solver, in_axes=(None, None, None, 0, None, None))
    energies, _ = numerov_vect(r_box, r_start, N_points, l_array, Z_atom, True)
    
    # 2. Degeneracy Factors (2 * (2l + 1))
    g_l = 2 * (2 * l_array + 1)
    g_l_matrix = g_l[:, None] # Reshape for broadcasting
    
    # 3. Define N(mu) function
    def calculate_N(mu_val):
        # Fermi-Dirac: 1 / (exp((E-mu)/T) + 1)
        # using sigmoid((mu-E)/T) for numerical stability
        occ = sigmoid((mu_val - energies) / temp)
        return jnp.sum(occ * g_l_matrix)

    # 4. Solve for mu (Bisection Method)
    # We need to find mu such that Total N = Z_atom
    # Search range: roughly around the ground state energy up to positive
    mu_min = jnp.min(energies) - 2.0  
    mu_max = 10.0 
    
    def bisection_step(i, bounds):
        low, high = bounds
        mid = (low + high) / 2.0
        N_mid = calculate_N(mid)
        # If we have too few electrons, we need higher mu -> low = mid
        new_low = jnp.where(N_mid < Z_atom, mid, low)
        new_high = jnp.where(N_mid < Z_atom, high, mid)
        return (new_low, new_high)
        
    final_low, final_high = lax.fori_loop(0, 50, bisection_step, (mu_min, mu_max))
    mu_correct = (final_low + final_high) / 2.0
    
    # 5. Calculate U using correct mu
    occ_final = sigmoid((mu_correct - energies) / temp)
    
    # U = Sum(Energy * Occupancy * Degeneracy)
    U_total = jnp.sum(energies * occ_final * g_l_matrix)
    
    return U_total

@partial(jit, static_argnames = ['N_points', 'k'])
def lobpcg_solver(r_box, r_start, N_points, l, k):
    """ 
    inputs(all in atomic units):
    r_box: endpoint for r_points, infinite potential (hard wall)
    r_start: r_points(0) <-- not 0 because singularity
    N_points: number of points including the boundaries
    l: orbital quantum number 
    k: number of eigenvalues to obtain

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

    # cholesky decomposition to make it a standard eigenvalue problem Au = lambda u
    L = jla.cholesky(B, lower = True)
    L_inv = jla.solve_triangular(L,jnp.eye(L.shape[0]), lower = True)
    A_tilde = L_inv @ H @ L_inv.T

    # by default, the lobpcg solver finds the largest eigenvalues, so flipping the sign
    # of the matrix and finding its largest eigenvalues is equivalent to finding the
    # smallest eigenvalues of the original matrix
    A_tilde_neg = -A_tilde

    # generating random matrix for original search
    key = jax.random.PRNGKey(0)
    X0 = jax.random.normal(key, (N_points-2, k))

    # 3rd output is number of iterations performed, which is not necessary
    energies_neg, psi, _ = jax.experimental.sparse.linalg.lobpcg_standard(A_tilde_neg, X0)

    energies = -energies_neg
    
    return energies, psi
# %% Benchkeeping + file management
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import jax.scipy.linalg as jla
import matplotlib.pyplot as plt
import os
import time
import numpy as np

# enables colorful tracebacks which was useful for debugging
from rich.traceback import install
install() 

# %% main

# wanted functions to be all organized at the bottom so defined main sequence
def main():

    # initialization
    r_box = 30
    r_start = 1e-5
    N_points = 1000
    l = 0

    energies, psi = numerov_solver(r_box, r_start, N_points, l)

    # %% plotting

    # hydrogen energy levels

    n_plot_lim = 10

    x_plot = jnp.linspace(1,n_plot_lim)
    y_plot = -1/(2*x_plot ** 2)

    plt.figure(figsize = (10,6))
    plt.plot(range(1,n_plot_lim + 1), energies[:n_plot_lim], marker = "x", label = "Numerov Matrix (JAX)")
    plt.plot(x_plot, y_plot, linestyle = "-", label = "Theory (E = -1/(2n^2))")
    plt.xlabel("n")
    plt.ylabel("Energy (Ha)")
    plt.title(f"Hydrogen Energy Levels: r_box = {r_box}, N_points = {N_points}, l = {l}, E_1 = {energies[0]:.5f}")
    plt.grid(True)
    plt.legend()
    plt.xlim((1,n_plot_lim))
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f"hydrogen_rbox{r_box}_N{N_points}_l{l}_E1{energies[0]:.5f}.png")
    plt.savefig(save_path)
    plt.close()

    # error and time vs N_points

    N_points_plot = np.arange(50,1050,50)
    n_plot_test = len(N_points_plot)
    errors = np.zeros(n_plot_test)
    times = np.zeros(n_plot_test)

    for i in range(n_plot_test):
        start = time.time()
        energies, psi = numerov_solver(r_box, r_start, N_points_plot[i], l)
        energies.block_until_ready()
        end = time.time()
        errors[i] = abs(energies[0] - (-1/2))
        times[i] = end - start

    # error vs n
    plt.figure(figsize=(10,6))
    plt.plot(N_points_plot, errors, linestyle="-", marker="x", color="red")
    plt.xlabel("N_points", fontsize=12)
    plt.ylabel("abs(E1 - (-1/2))", fontsize=12)
    plt.yscale("log")
    plt.title(f"Error vs N_points (r_box={r_box}, l={l})", fontsize=14)
    plt.grid(True)

    error_filename = os.path.join(script_dir, f"error_rbox{r_box}_l{l}.png")
    plt.savefig(error_filename, dpi=300)

    # time vs n
    plt.figure(figsize=(10,6))
    plt.plot(N_points_plot, times, linestyle="-", marker="x", color="blue")
    plt.xlabel("N_points", fontsize=12)
    plt.ylabel("Time (s)", fontsize=12)
    plt.title(f"Time vs N_points (r_box={r_box}, l={l})", fontsize=14)
    plt.grid(True)

    time_filename = os.path.join(script_dir, f"time_rbox{r_box}_l{l}.png")
    plt.savefig(time_filename, dpi=300)
    
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

main()
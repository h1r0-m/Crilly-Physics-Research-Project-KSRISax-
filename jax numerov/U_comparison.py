import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from njax_functions import numerov_solver, U_solver
from cont_functions import bounded_states_solver, mu_solver, U_solver_cont

from rich.traceback import install
install()

jax.config.update("jax_enable_x64", True)

def main():
    # Setup
    r_box = 30
    r_start = 1e-5
    N_points = 300
    Z = 1
    T = 1
    
    energies, mask, degeneracies = bounded_states_solver(r_box, r_start, N_points, Z)
    
    mu = mu_solver(energies, mask, degeneracies, r_box, Z, T)
    
    U_new = U_solver_cont(r_box, energies, mask, degeneracies, mu, T)
    
    U_old = U_solver(T, r_box, r_start, N_points, 200)

    print(f"mu: {mu}")
    print(f"U (old): {U_old}")
    print(f"U (new): {U_new}")

main()
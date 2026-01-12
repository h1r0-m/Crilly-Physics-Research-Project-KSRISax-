# housekeeping

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jacrev, jit

from numerov_jax_solver import numerov_solver, U_solver

from rich.traceback import install
install()

def main():
    r_box = 30.0
    r_start = 1e-5
    N_points = 500
    l = 0
    temp = 1

    energies, psi = numerov_solver(r_box, r_start, N_points, l)

    U = U_solver(energies, temp)
    print(U)

main()
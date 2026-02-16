# housekeeping
import sys
import os
from datetime import datetime

# 1. Fix the Import Path
# Get the folder where THIS script is (.../jax numerov/plot codes/jan27_2026)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up TWO levels to reach the root (.../jax numerov)
jax_numerov_dir = os.path.dirname(os.path.dirname(script_dir))

# Now point to core files (.../jax numerov/core files)
core_path = os.path.join(jax_numerov_dir, 'core files')

if core_path not in sys.path:
    sys.path.append(core_path)

import jax
import jax.numpy as jnp
from hybrid_functions_corrected import U_solver_corrected, bounded_states_solver, mu_solver_corrected

def main():

    r_box = 1.0
    T = 1e-3
    r_start = 1e-5
    N_points = 300
    Z = 1

    e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)
    mu = mu_solver_corrected(e,m,d,r_box, r_start, N_points, Z, T)

    U_grad = jax.grad(U_solver_corrected, argnums = 3)

    U_grad_val = U_grad(e,m,d,r_box, r_start, N_points, mu, T, Z)

    print(U_grad_val)


main()
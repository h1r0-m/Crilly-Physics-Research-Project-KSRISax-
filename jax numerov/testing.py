# housekeeping
import sys
import os

# 1. Fix the Import Path
# Get the folder where THIS script is (.../jax numerov/plot codes)
jax_numerov_dir = os.path.dirname(os.path.abspath(__file__))
# Now point to core files (.../jax numerov/core files)
core_path = os.path.join(jax_numerov_dir, 'core files')

if core_path not in sys.path:
    sys.path.append(core_path)
    
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema
from hybrid_functions_corrected import mu_solver_corrected, U_solver_corrected
from hybrid_functions_uncorrected import bounded_states_solver, mu_solver_uncorrected, U_solver_uncorrected

from rich.traceback import install
install()

def main():
    # some bound states (corrected)
    print("some bounded states (corrected):")

    r_box = 1.84
    Z = 1
    r_start = 1e-5
    N_points = 300
    T_ha = 1e-3
    e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)
    mu = mu_solver_corrected(e,m,d,r_box, r_start, N_points, Z, T_ha)   
    U = U_solver_corrected(e,m,d,r_box, r_start, N_points, mu, T_ha, Z)

    print(f"mu (some bound, corrected) = {mu}")
    print(f"U (some bound, corrected) = {U}")

    # some bound states (uncorrected)
    print("\n some bound states (uncorrected):")

    mu = mu_solver_uncorrected(e, m, d, r_box, Z, T_ha)
    U = U_solver_uncorrected(r_box, e, m, d, mu, T_ha)

    print(f"mu (some bound, uncorrected) = {mu}")
    print(f"U (some bound, uncorrected) = {U}")

    # all unbounded (corrected)
    
    print("\n  all free (corrected):")

    r_box = 1.83

    e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)
    mu = mu_solver_corrected(e,m,d,r_box, r_start, N_points, Z, T_ha)   
    U = U_solver_corrected(e,m,d,r_box, r_start, N_points, mu, T_ha, Z)
    
    print(f"mu (all free) = {mu}")
    print(f"U (all free) = {U}")
    
    # all unbounded  states (uncorrected)
    print("\n all free states (uncorrected):")

    mu = mu_solver_uncorrected(e, m, d, r_box, Z, T_ha)
    U = U_solver_uncorrected(r_box, e, m, d, mu, T_ha)

    print(f"mu (all free, uncorrected) = {mu}")
    print(f"U (all free, uncorrected) = {U}")
    
    file_name = os.path.basename(__file__)
    print(f"--- {file_name} has completed running ---")

main()
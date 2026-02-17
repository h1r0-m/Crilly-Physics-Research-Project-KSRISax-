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
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.signal import argrelextrema
from hybrid_functions_optimized import mu_solver_corrected, thermo_solver_corrected, bounded_states_solver, U_solver_corrected, u_integration_solver
from hybrid_functions_corrected import mu_solver_corrected as msc, U_solver_corrected as Usc
from hybrid_functions_uncorrected import mu_solver_uncorrected, U_solver_uncorrected
from njax_functions import U_solver_nonhybrid
from old_grid_test import correction_grid_solver_jax

from rich.traceback import install
install()

def main():

    r_box_array = jnp.linspace(1,2.2, num = 10)
    r_start = 1e-5
    N_points = 300
    Z = 1
    T_ha = 1e-3
    u_array_end = []
        
    for r_box in r_box_array:
        e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)
        mu = mu_solver_corrected(e,m,d,r_box, r_start, N_points, Z, T_ha, Z)

        E_max = jnp.maximum(mu + 20*T_ha, 5.0)
        grid = correction_grid_solver_jax(mu, T_ha, E_max)

        u_array = jax.vmap(lambda E: u_integration_solver(E, r_box, r_start, N_points, 0, Z))(grid)
        u_array_end.append(jnp.min(abs(u_array[:, -1])))

    plt.figure()
    plt.plot(r_box_array, u_array_end, "x-", color = "blue")
    plt.xlabel("r_box")
    plt.ylabel("minimum distance to 0 for u_array[-1]")
    plt.grid(True)
    plt.show()

    # plt.figure()
    # plt.plot(range(len(u_array)), u_array[:, -1], "x-", color = "blue")
    # plt.xlabel("i")
    # plt.ylabel("u at r_box")
    # plt.grid(True)
    # plt.show()

    # plt.figure()
    # r_points = jnp.linspace(r_start, r_box, N_points)
    # plt.plot(r_points, u_array[0,:], "x-", color = "blue")
    # plt.grid(True)
    # plt.xlabel("r")
    # plt.ylabel("u")
    # plt.show()

main()
# housekeeping
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from njax_functions import numerov_solver, U_solver
from cont_functions import U_solver_cont, bounded_states_solver, mu_solver

from rich.traceback import install
install()

# mu vs T

r_box = 30
r_start = 1e-5
T_ha_array = jnp.logspace(-3, 1, num=50)
Z = 1
N_points = 300

e_1, m_1, d_1 = bounded_states_solver(r_box, r_start, N_points, Z)
mu_vect = jax.vmap(mu_solver, in_axes = (None, None, None, None, None, 0))
mu_array = mu_vect(e_1, m_1, d_1, r_box, Z, T_ha_array)

plt.figure(figsize=(10,6))
plt.plot(T_ha_array * 315775, mu_array, linestyle="-", marker="x", color="red", label = "Hybrid")
plt.xlabel("Temperature (K)", fontsize=12)
plt.ylabel("Chemical Potential (Ha)", fontsize=12)
plt.title(f"Chemical Potential vs Temperature (r_box={r_box}, N_points={N_points})", fontsize=14)
plt.grid(True)
plt.legend()

# saving
main_folder = "plots"
sub_folder = "jan21_2025"

script_dir = os.path.dirname(os.path.abspath(__file__))

plot_dir = os.path.join(script_dir, main_folder, sub_folder)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

temp_filename = os.path.join(plot_dir, f"muVsTemp_rbox{r_box}_N{N_points}.png")
plt.savefig(temp_filename, dpi=300)

print("mu vs T done")

# mu vs r_box

r_box_array = jnp.logspace(-3,1,num = 50)
r_start = 1e-5
T_ha = 1e-3
N_points = 300
Z = 1

mu_array_2 = []
for r in r_box_array:
    e, m, d = bounded_states_solver(r, r_start, N_points, Z)
    mu = mu_solver(e, m, d, r, Z, T_ha)
    mu_array_2.append(mu)

plt.figure(figsize=(10,6))
plt.plot(r_box_array, mu_array_2, linestyle="-", marker="x", color="red", label= "Hybrid")
plt.xlabel("Box Radius (Ha)")
plt.ylabel("Chemical Potential (Ha)")
plt.title(f"Energy vs Box Size (T={T_ha*315775:.0f} K, N_points={N_points})")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, f"muVsRbox_temp{T_ha}_N{N_points}.png"), dpi=300)
plt.close()

print("mu vs r_box done")

# U vs smaller r_box with higher N_points

r_box_array_2 = jnp.logspace(-3, 1, num = 100)
Z = 1
r_start = 1e-5
N_points = 1000
T_ha = 1e-3
U_array_1 = []

for r in r_box_array_2:
    e, m, d = bounded_states_solver(r, r_start, N_points, Z)
    mu = mu_solver(e, m, d, r, Z, T_ha)
    U = U_solver_cont(r, e, m, d, mu, T_ha)
    U_array_1.append(U)

plt.figure(figsize=(10,6))
plt.plot(r_box_array_2, jnp.array(U_array_1) * 27.2114, linestyle="-", marker="x", color="red", label= "Hybrid")
plt.xlabel("Box Radius (Ha)")
plt.xscale('log')
plt.ylabel("Internal Energy (eV)")
plt.title(f"Energy vs Box Size (T={T_ha*315775:.0f} K, N_points={N_points})")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, f"UvsRbox_temp{T_ha}_N{N_points}.png"), dpi=300)
plt.close()

print("U vs smaller r_box done")

file_name = os.path.basename(__file__)
print(f"--- {file_name} has completed running ---")
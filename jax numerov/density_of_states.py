import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from njax_functions import numerov_solver
import os

# initialization
r_box = 30
r_start = 1e-5
N_points = 1000 
l_max = 100     

# solving for energy eigenvalues for each l until l_max
l_array = jnp.arange(l_max + 1)
numerov_vect = jax.vmap(numerov_solver, in_axes=(None, None, None, 0))
energies, _ = numerov_vect(r_box, r_start, N_points, l_array)

# degeneracy factors
g_l = 2 * (2 * l_array + 1) 

# broadcasting degeneracy to match the shape of energies_matrix
# repeating the degeneracy value across the n columns for each l row
weights_matrix = jnp.repeat(g_l[:, None], energies.shape[1], axis=1)

# flattening data so it becomes 1D
flat_energies = np.array(energies.flatten())
flat_weights = np.array(weights_matrix.flatten())

# filtering out high energies for visual reasons
mask = flat_energies < 10
flat_energies = flat_energies[mask]
flat_weights = flat_weights[mask]

# plotting histogram
plt.figure(figsize=(10, 6))

counts, bins, _ = plt.hist(flat_energies, bins=200, weights=flat_weights, 
                           color='skyblue', edgecolor='black', alpha=0.7, density=True, label="Simulation DOS")

plt.xlabel("Energy (Ha)")
plt.ylabel("Weighted Count")
plt.title(f"Density of States (l_max={l_max}, r_box={r_box}, N_points={N_points})")
plt.legend()
plt.grid(True)

# saving
folder_name = "plots" 
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(script_dir, folder_name)

# creating folder if it dosnt exist 
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

temp_filename = os.path.join(plot_dir, f"dos_rbox{r_box}_lmax{l_max}_N{N_points}.png")
plt.savefig(temp_filename, dpi=300)
# housekeeping
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import jit, lax
from functools import partial
from bessel_function import sph_bessel, sph_bessel_deriv, sph_neumann, sph_neumann_deriv
from njax_functions import numerov_solver
from fermi_dirac_integral import fermi_dirac_integral_half, fermi_dirac_integral_three_half
from jax.scipy.special import xlogy

jax.config.update("jax_enable_x64", True)

@jit
def correction_grid_solver_jax(mu, T, E_max):
    """
    function for grid generation that is compatible for jit compilation
    instead of finding where resonance occurs, can just create a fixed-size,
    high-density grid that concentrates points both closer to E = 0 - for resonance) and around E = mu (for when the fermi-dirac occupation changes most suddenly)

    inputs:
    mu - chemical potential
    T - temperature
    E_max - max E for grid generation, usually when fermi-dirac occupation becomes negligible

    output:
    full_grid - grid of energy points that can later be used to calculate correction terms
    """
    
    # base logarithmic grid of 200 points, can make finer if wanted
    log_grid = jnp.logspace(jnp.log10(1e-4), jnp.log10(E_max), 200)
    
    # very low energy grid of 20 points to capture resonance around E = 0
    low_e_grid = jnp.linspace(1e-16, 1e-4, 20)
    
    # grid for mu of 21 points to capture change in fermi-dirac occupation
    # this "width" of change depends on T, because higher T means more spread out
    # and lower T means more sudden change. therefore this grid width is from -4T
    # to +4T from E = mu to capture this behavior accurately
    mu_window = jnp.linspace(-4.0, 4.0, 21) * T
    mu_grid = mu + mu_window
    
    # filtering points to mu_grid that are invalid (e.g. if E <= 0 or E > E_max)
    # replacing invalid points with a dummy value (e.g. 1e-10), cuz arrays have
    # to be fixed size for jit compilation
    mu_grid = jnp.where((mu_grid > 1e-16) & (mu_grid < E_max), mu_grid, 1e-10)

    # concatenating all grids 
    full_grid = jnp.concatenate([low_e_grid, log_grid, mu_grid])
    
    # sorting the grid 
    full_grid = jnp.sort(full_grid)
    
    # just a note, but even if there are duplicate points in the grid, shouldnt
    # affect calculation because the trapezoidal rule later to calculate
    # correction would just add 0 contribution if change in E is 0 in that energy interval

    return full_grid

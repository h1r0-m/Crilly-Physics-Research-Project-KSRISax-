# housekeeping
from jax import jit, lax
import jax.numpy as jnp

l_max = 50  # max l we will ever use for now, so no need to calculate bessel/neumann functions higher than this

@jit
def sph_bessel(l, x):
    """
    computing spherical bessel function j_l(x) for values of l from 0 to l_max
    using fixed range scanning for jit compiling

    inputs:
    l - orbital quantum number
    x - point at which we are evaluating the function

    outputs:
    j_l(x) - value of spherical bessel function evaluated

    """
    # replacing x with a very small number if its basically 0. 
    # to avoid division by 0
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)

    # j_0 and j_1 (initial values)
    j0 = jnp.sin(x) / x
    j1 = j0 / x - jnp.cos(x) / x

    # using recursive formulas:
    # f_l+1 (x) = (2l+1) / x f_l(x) - f_l-1(x)
    def body(carry, i):
        j_prev, j_curr = carry
        j_next = (2 * i + 1) / x * j_curr - j_prev
        return (j_curr, j_next), j_next

    # running reccurence
    (j_prev, j_curr), js_rest = lax.scan(body, (j0, j1), jnp.arange(1, l_max))

    # combining initial values and next to full arrays of j_l(x)
    js = jnp.concatenate([jnp.array([j0, j1]), js_rest], axis=0)

    # making sure l is between 0 and l_max
    # if l < 0, return value for l = 0
    # if l > l_max, return value for l = l_max
    l_clamped = jnp.clip(l, 0, l_max)

    return js[l_clamped]

@jit
def sph_neumann(l, x):
    """
    computing spherical neumann function n_l(x) for values of l from 0 to l_max
    
    inputs:
    l, x - same idea as bessel function

    outputs:
    n_l(x) - value of spherical neumann function evaluated

    """
    # check if x is basically 0
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)

    # initial values
    n0 = -jnp.cos(x) / x
    n1 = n0 / x - jnp.sin(x) / x

    # recursive formula same as spherical bessel function above (j_l(x))
    def body(carry, i):
        n_prev, n_curr = carry
        n_next = (2 * i + 1) / x * n_curr - n_prev
        return (n_curr, n_next), n_next

    # going through l = 0 to l_max
    (n_prev, n_curr), ns_rest = lax.scan(body, (n0, n1), jnp.arange(1, l_max))
    
    # combining initial values and rest to get an array with all values of n_l(x)
    ns = jnp.concatenate([jnp.array([n0, n1]), ns_rest], axis=0)

    # checking if l is between 0 and l_max
    l_clamped = jnp.clip(l, 0, l_max)

    return ns[l_clamped]

@jit
def sph_bessel_deriv(l, x):
    """ calculating derivative of spherical bessel function with respect to r
     
    formula:
    f_l'(x) = l/x f_l(x) - f_l+1 (x)

    inputs:
    l, x - same idea as before

    outputs:
    j_l'(x) - value of derivative of spherical bessel function

        """
    # checking if x is basically 0
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)

    # calculating
    j_l   = sph_bessel(l, x)
    j_lp1 = sph_bessel(l + 1, x)

    return (l / x) * j_l - j_lp1

@jit
def sph_neumann_deriv(l, x):
    """ for derivative of spherical neumann function n_l'(x) 
    
    inputs:
    l, x - same

    output:
    n_l'(x)
    """
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)
    n_l   = sph_neumann(l, x)
    n_lp1 = sph_neumann(l + 1, x)
    return (l / x) * n_l - n_lp1
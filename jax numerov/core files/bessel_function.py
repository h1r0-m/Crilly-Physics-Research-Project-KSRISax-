from jax import jit, lax
import jax.numpy as jnp

@jit
def sph_bessel(l, x):
    """
    Computes spherical Bessel function j_l(x) using stable recurrence.
    """
    # Small x approximation to avoid singularity
    # (or you can use the exact sin/cos formula if x > l)
    # But for general use, we need the recurrence.
    
    # 1. Base cases: j_0 and j_1
    # Check for x near zero to avoid division by zero errors
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x) 
    
    j0 = jnp.sin(x) / x
    j1 = j0 / x - jnp.cos(x) / x

    def body_fun(i, vals):
        j_prev, j_curr = vals
        # Recurrence: j_{n+1} = (2n+1)/x * j_n - j_{n-1}
        j_next = (2 * i + 1) / x * j_curr - j_prev
        return (j_curr, j_next)

    # If l=0, return j0
    # If l=1, return j1
    # If l > 1, loop up to l
    
    # Using lax.cond to handle l=0, l=1 cases efficiently
    final_val = lax.cond(
        l == 0, lambda _: j0,
        lambda _: lax.cond(
            l == 1, lambda _: j1,
            lambda _: lax.fori_loop(1, l, body_fun, (j0, j1))[1],
            operand=None
        ),
        operand=None
    )
    
    return final_val

@jit
def sph_neumann(l, x):
    """
    Computes spherical Neumann function n_l(x) (also called y_l).
    """
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)
    
    n0 = -jnp.cos(x) / x
    n1 = n0 / x - jnp.sin(x) / x
    
    def body_fun(i, vals):
        n_prev, n_curr = vals
        n_next = (2 * i + 1) / x * n_curr - n_prev
        return (n_curr, n_next)

    final_val = lax.cond(
        l == 0, lambda _: n0,
        lambda _: lax.cond(
            l == 1, lambda _: n1,
            lambda _: lax.fori_loop(1, l, body_fun, (n0, n1))[1],
            operand=None
        ),
        operand=None
    )
    
    return final_val

@jit
def sph_bessel_deriv(l, x):
    """
    Computes derivative j'_l(x) using the identity:
    j'_l(x) = (l/x) * j_l(x) - j_{l+1}(x)
    """
    # Safety for x=0 (though your x = k*R should be > 0)
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)
    
    j_l   = sph_bessel(l, x)
    j_lp1 = sph_bessel(l + 1, x)
    
    return (l / x) * j_l - j_lp1

@jit
def sph_neumann_deriv(l, x):
    """
    Computes derivative n'_l(x) using the identity:
    n'_l(x) = (l/x) * n_l(x) - n_{l+1}(x)
    """
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)
    
    n_l   = sph_neumann(l, x)
    n_lp1 = sph_neumann(l + 1, x)
    
    return (l / x) * n_l - n_lp1

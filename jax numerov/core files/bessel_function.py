from jax import jit, lax
import jax.numpy as jnp

L_MAX = 50  # max l you will ever use (>= your l_max in solvers)

@jit
def sph_bessel(l, x):
    """
    Computes spherical Bessel j_l(x) for 0 <= l <= L_MAX.
    Uses a fixed-length scan over l, so AD is compatible.
    """
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)

    # j_0 and j_1
    j0 = jnp.sin(x) / x
    j1 = j0 / x - jnp.cos(x) / x

    def body(carry, i):
        j_prev, j_curr = carry
        j_next = (2 * i + 1) / x * j_curr - j_prev
        return (j_curr, j_next), j_next

    # Run recurrence up to L_MAX-1
    (j_prev, j_curr), js_rest = lax.scan(
        body,
        (j0, j1),
        jnp.arange(1, L_MAX)  # i = 1,...,L_MAX-1
    )

    # Build full array: [j0, j1, j2,..., j_L_MAX]
    js = jnp.concatenate(
        [jnp.array([j0, j1]), js_rest],
        axis=0
    )  # shape (L_MAX+1,)

    # Safety: clamp l into [0, L_MAX]
    l_clamped = jnp.clip(l, 0, L_MAX)

    # Select j_l
    return js[l_clamped]


@jit
def sph_neumann(l, x):
    """
    Computes spherical Neumann n_l(x) for 0 <= l <= L_MAX.
    Same pattern as sph_bessel.
    """
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)

    n0 = -jnp.cos(x) / x
    n1 = n0 / x - jnp.sin(x) / x

    def body(carry, i):
        n_prev, n_curr = carry
        n_next = (2 * i + 1) / x * n_curr - n_prev
        return (n_curr, n_next), n_next

    (n_prev, n_curr), ns_rest = lax.scan(
        body,
        (n0, n1),
        jnp.arange(1, L_MAX)
    )

    ns = jnp.concatenate(
        [jnp.array([n0, n1]), ns_rest],
        axis=0
    )  # shape (L_MAX+1,)

    l_clamped = jnp.clip(l, 0, L_MAX)
    return ns[l_clamped]


@jit
def sph_bessel_deriv(l, x):
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)
    j_l   = sph_bessel(l, x)
    j_lp1 = sph_bessel(l + 1, x)
    return (l / x) * j_l - j_lp1


@jit
def sph_neumann_deriv(l, x):
    x = jnp.where(jnp.abs(x) < 1e-10, 1e-10, x)
    n_l   = sph_neumann(l, x)
    n_lp1 = sph_neumann(l + 1, x)
    return (l / x) * n_l - n_lp1
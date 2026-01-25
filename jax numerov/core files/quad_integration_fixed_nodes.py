import jax
import jax.numpy as jnp

# Example: 32-point Gauss–Legendre nodes and weights on [-1, 1]
# You can choose 64, 128 etc depending on accuracy needs.
# These can be precomputed or imported from somewhere.
GL32_x = jnp.array([
    -0.9972638618, -0.9856115115, -0.9647622556, -0.9349060759,
    -0.8963211558, -0.8493676137, -0.7944837960, -0.7321821187,
    -0.6630442669, -0.5877157572, -0.5068999089, -0.4213512761,
    -0.3318686023, -0.2392873623, -0.1444719616, -0.0483076657,
     0.0483076657,  0.1444719616,  0.2392873623,  0.3318686023,
     0.4213512761,  0.5068999089,  0.5877157572,  0.6630442669,
     0.7321821187,  0.7944837960,  0.8493676137,  0.8963211558,
     0.9349060759,  0.9647622556,  0.9856115115,  0.9972638618
])

GL32_w = jnp.array([
    0.0070186100, 0.0162743947, 0.0253920653, 0.0342738629,
    0.0428358980, 0.0509980593, 0.0586840935, 0.0658222228,
    0.0723457941, 0.0781938958, 0.0833119242, 0.0876520930,
    0.0911738787, 0.0938443991, 0.0956387200, 0.0965400885,
    0.0965400885, 0.0956387200, 0.0938443991, 0.0911738787,
    0.0876520930, 0.0833119242, 0.0781938958, 0.0723457941,
    0.0658222228, 0.0586840935, 0.0509980593, 0.0428358980,
    0.0342738629, 0.0253920653, 0.0162743947, 0.0070186100
])

def fixed_GL_quad(func, a, b, params):
    """
    Simple fixed-order Gauss-Legendre quadrature of func over [a,b].
    func: callable (E, *params) -> scalar
    a, b: scalars
    params: tuple/list of extra parameters
    """
    # Map nodes from [-1,1] to [a,b]
    mid  = 0.5 * (a + b)
    half = 0.5 * (b - a)

    xs = mid + half * GL32_x           # shape (32,)
    # Vectorize func over E
    f_vec = jax.vmap(lambda x: func(x, *params))(xs)  # shape (32,)

    return half * jnp.sum(GL32_w * f_vec)
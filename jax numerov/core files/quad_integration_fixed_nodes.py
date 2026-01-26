import jax
import jax.numpy as jnp
from functools import partial

# --- 32-Point Gauss-Legendre Nodes ---
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

def simple_GL(func, a, b, params):
    """Basic linear GL quad over [a, b]"""
    # Standard mapping from [-1, 1] to [a, b]
    mid  = 0.5 * (a + b)
    half = 0.5 * (b - a)
    xs = mid + half * GL32_x
    
    # Evaluate function at all nodes
    f_vec = jax.vmap(lambda x: func(x, *params))(xs)
    
    return half * jnp.sum(GL32_w * f_vec)

def composite_GL_quad(func, a, b, params):
    """
    Decade-by-Decade Integration (Vectorized).
    Uses jax.vmap to compute all log-segments in parallel, 
    solving the compilation bottleneck.
    """
    cutoff = 0.5 
    
    # 1. Define the Decades (Low energy segments)
    # We use a static array of decade boundaries
    decades = jnp.array([1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5])
    
    # Prepare start and end points for each segment
    starts = decades[:-1]
    ends   = decades[1:]
    
    # 2. Vectorize the integral over these segments
    # This replaces the Python loop with a single JAX operation
    def integrate_segment(s, e):
        return simple_GL(func, s, e, params)

    # vmap over starts and ends -> returns array of integrals
    segment_integrals = jax.vmap(integrate_segment)(starts, ends)
    
    val_log = jnp.sum(segment_integrals)

    # 3. Add the linear tail (High energy)
    # Only integrate if b > cutoff
    val_lin = jax.lax.cond(
        b > cutoff,
        lambda: simple_GL(func, cutoff, b, params),
        lambda: 0.0
    )
    
    return val_log + val_lin

def high_density_linear_quad(func, a, b, num_points=100000):
    """
    Batched Trapezoidal Integration.
    Splits the massive grid into smaller chunks (batches) to prevent 
    Out-Of-Memory errors when differentiating complex functions.
    """
    # 1. Configuration
    BATCH_SIZE = 5000  # Safe batch size (fits in memory)
    
    # Calculate batches
    num_batches = int(num_points / BATCH_SIZE)
    if num_batches < 1: num_batches = 1
    
    # Recalculate exact total points
    total_points = num_batches * BATCH_SIZE
    
    # 2. Define the grid
    xs = jnp.linspace(a, b, total_points)
    dx = xs[1] - xs[0]
    
    # 3. Define the batch worker
    def process_batch(start_idx):
        # Slice the chunk dynamically
        chunk = jax.lax.dynamic_slice(xs, (start_idx,), (BATCH_SIZE,))
        
        # Evaluate function (Vectorized over just this chunk)
        ys = jax.vmap(func)(chunk)
        
        return jnp.sum(ys)

    # 4. Run loop over batches
    # lax.fori_loop keeps memory usage low (it doesn't unroll)
    def body_fun(i, accumulated_sum):
        start_idx = i * BATCH_SIZE
        batch_sum = process_batch(start_idx)
        return accumulated_sum + batch_sum

    total_sum_y = jax.lax.fori_loop(0, num_batches, body_fun, 0.0)
    
    # 5. Apply Trapezoidal Correction
    y_first = func(xs[0])
    y_last  = func(xs[-1])
    
    total_area = total_sum_y * dx - 0.5 * dx * (y_first + y_last)
    
    return total_area

def composite_GL_quad_dynamic(func, segments, params):
    """
    Vectorized integration over dynamic segments provided by the user.
    segments: jnp.array of shape (N+1,) defining N intervals.
    """
    starts = segments[:-1]
    ends   = segments[1:]
    
    def integrate_segment(s, e):
        # Only integrate if the segment has non-zero length
        return jax.lax.cond(
            e > s + 1e-14, 
            lambda: simple_GL(func, s, e, params),
            lambda: 0.0
        )

    segment_integrals = jax.vmap(integrate_segment)(starts, ends)
    return jnp.sum(segment_integrals)
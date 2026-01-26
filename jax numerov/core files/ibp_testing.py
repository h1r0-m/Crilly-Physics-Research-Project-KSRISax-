from jax import jit
from hybrid_functions_uncorrected import bounded_states_solver
from hybrid_functions_corrected import mu_solver_corrected, N_solver_corrected, U_solver_corrected

def main():
    r_box = 1.8
    r_start = 1e-5
    N_points = 300
    Z = 1
    T = 1e-3

    print(f"for r_box = {r_box}:")

    e,m,d = bounded_states_solver(r_box, r_start, N_points, Z)
    mu = mu_solver_corrected(e,m,d, r_box, r_start, N_points, Z, T)
    N = N_solver_corrected(e,m,d,r_box, r_start, N_points, mu, T, Z)
    U = U_solver_corrected(e,m,d, r_box, r_start, N_points, mu, T, Z)

    print(f"mu = {mu}")
    print(f"N = {N}")
    print(f"Z = {Z}")
    print(f"U = {U}")

    print("file is done")

main()
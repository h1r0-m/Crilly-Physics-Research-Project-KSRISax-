import numpy as np
import matplotlib.pyplot as plt

def sackur_tetrode_theoretical(T_K, r_box):
    """
    Calculates theoretical Entropy for 1 electron in a box using Sackur-Tetrode.
    Units: Atomic Units (Ha for Energy, but T is input in Kelvin).
    """
    # --- Constants ---
    # Conversion: Kelvin to Hartree (1 Ha = 3.1577e5 K)
    kelvin_to_ha = 1 / 315775.0
    
    # --- Convert T to Atomic Units ---
    T_ha = T_K * kelvin_to_ha
    
    # Avoid log(0)
    T_ha = np.maximum(T_ha, 1e-10)
    
    # --- System Parameters ---
    N = 1.0  # Single electron
    V = (4/3) * np.pi * (r_box ** 3) # Volume of sphere
    
    # --- Sackur-Tetrode Equation (Atomic Units) ---
    # S/kB = N * [ ln(V/N * (T/2pi)^1.5) + 5/2 + ln(2) ]
    
    # 1. Translational Contribution
    term_trans = np.log( (V/N) * (T_ha / (2 * np.pi))**1.5 )
    
    # 2. Constant Factor (5/2 for ideal gas)
    term_const = 2.5
    
    # 3. Spin Contribution (ln(2) for doublet)
    term_spin = np.log(2)
    
    S_theoretical = N * (term_trans + term_const + term_spin)
    
    return S_theoretical

# --- 1. Dense Data Extracted Visually from 'S_vs_T.py_T100.0_N300_Z1.png' ---
# Reading the 'x' markers from left to right
# T is in units of 1e7 K
t_simulated = np.array([
    0.0,
    0.02e7, 0.04e7, 0.06e7, 0.08e7, # The dense vertical rise
    0.10e7, 0.12e7, 0.15e7, 0.18e7, # The "knee"
    0.25e7, 0.35e7, 
    0.50e7, # Grid line 0.5
    0.75e7, 
    1.00e7, # Grid line 1.0
    1.25e7,
    1.50e7, # Grid line 1.5
    2.00e7, # Grid line 2.0
    2.50e7, # Grid line 2.5
    3.15e7  # Final point
])

# Corresponding Entropy (Ha) estimates
s_simulated_estimate = np.array([
    5.60, 
    7.10, 8.40, 9.80, 10.80, 
    11.90, 12.60, 13.50, 14.10, 
    14.90, 15.60, 
    16.15, 
    16.85, 
    17.25, 
    17.60, 
    17.95, 
    18.30, 
    18.65, 
    18.95
])

# --- 2. Calculate Theoretical Curve ---
r_box = 30.0 
s_theory_at_points = sackur_tetrode_theoretical(t_simulated, r_box)
error = s_theory_at_points - s_simulated_estimate

# --- 3. Output Results ---
print(f"{'T (Kelvin)':<15} | {'S (Simulated)':<15} | {'S (Theoretical)':<15} | {'Diff'}")
print("-" * 60)
for i in range(1, len(t_simulated)): # Skip T=0
    print(f"{t_simulated[i]:<15.1e} | {s_simulated_estimate[i]:<15.2f} | {s_theory_at_points[i]:<15.2f} | {error[i]:.2f}")

# --- 4. Plot Comparison ---
t_theory_range = np.linspace(1000, 3.2e7, 200)
s_theory_smooth = sackur_tetrode_theoretical(t_theory_range, r_box)

plt.figure(figsize=(10, 6))
plt.plot(t_theory_range, s_theory_smooth, 'b-', label='Sackur-Tetrode (Theory)')
plt.plot(t_simulated, s_simulated_estimate, 'rx', markersize=8, label='Extracted Points')
plt.xlabel('Temperature (K)')
plt.ylabel('Entropy (Ha)')
plt.title(f'Verification: Dense Data Extraction vs Theory (r_box={r_box})')
plt.legend()
plt.grid(True)
plt.show()
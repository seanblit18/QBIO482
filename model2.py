import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Define the ODE system with delays for CORT and G6Pase
def hpa_gluc_axis(Y, t, k):
    k1_base, k2, k3, k4, k5_base, k6, eta, Kd, A, Km, tau = k
    
    # Circadian modulation of k1 (CRH) and k5 (G6Pase)
    k1 = k1_base * (1 + A * np.cos(2 * np.pi * t / 24))
    k5 = k5_base * (1 + A * np.cos(2 * np.pi * t / 24))
    
    # Access delayed values
    CRH, ACTH, CORT, G6Pase = Y(t)
    CORT_delayed = Y(t - tau)[2]  # CORT at t - tau
    
    # ODEs
    dCRH_dt = k1 * (1 - eta * CORT / (Kd + CORT)) - k2 * CRH
    dACTH_dt = k3 * CRH - k4 * ACTH
    dCORT_dt = k5 * ACTH - k6 * CORT_delayed
    dG6Pase_dt = k5 * (CORT_delayed / (Km + CORT_delayed)) - k6 * G6Pase
    
    return np.array([dCRH_dt, dACTH_dt, dCORT_dt, dG6Pase_dt])

# History function (constant history before t=0)
def history(t, y0):
    return np.array(y0)

# Parameters
k1_base = 0.2    # Base production rate of CRH
k2 = 0.5         # CRH degradation and ACTH production rate
k3 = 0.4         # ACTH degradation and CORT production rate
k4 = 0.3         # CORT degradation rate
k5_base = 0.25   # Base production rate of G6Pase
k6 = 0.35        # G6Pase degradation rate
eta = 0.8        # Feedback strength
Kd = 0.2         # Dissociation constant for CORT feedback
A = 0.3          # Amplitude of circadian oscillation
Km = 0.15        # Michaelis constant for CORT activation of G6Pase
tau = 3       # Time delay in hours

# Parameter vector
k = [k1_base, k2, k3, k4, k5_base, k6, eta, Kd, A, Km, tau]

# Time points (0 to 72 hours)
t = np.linspace(0, 72, 1000)

# Initial conditions: [CRH, ACTH, CORT, G6Pase]
y0 = [0.8, 0.7, 0.6, 0.5]

# Solve the DDEs
solution = ddeint(hpa_gluc_axis, lambda t: history(t, y0), t, fargs=(k,))

# Extract variables for plotting directly from the solution array
CRH = solution[:, 0]
ACTH = solution[:, 1]
CORT = solution[:, 2]
G6Pase = solution[:, 3]

# Plot the timecourse
plt.figure(figsize=(8, 4))
plt.plot(t, CRH, label='CRH')
plt.plot(t, ACTH, label='ACTH')
plt.plot(t, CORT, label='CORT')
plt.plot(t, G6Pase, label='G6Pase')
plt.xlabel('Time (hours)')
plt.ylabel('Concentration (ug/dl)')
plt.title('Computational simulation of the HPA-gluc axis system with delays')
plt.grid(True)
plt.legend()
plt.show()
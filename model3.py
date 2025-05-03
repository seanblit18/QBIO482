import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Define the ODE system with multiple delays
def hpa_gluc_axis(Y, t, k):
    k1_base, k2, k3, k4, k5_base, k6, eta, Kd, A, Km, tau1, tau2, tau3, tau4 = k
    
    # Circadian modulation of k1 (CRH) and k5 (G6Pase)
    k1 = k1_base * (1 + A * np.cos(2 * np.pi * t / 24))
    k5 = k5_base * (1 + A * np.cos(2 * np.pi * t / 24))
    
    # Access current and delayed values
    CRH, ACTH, CORT, G6Pase = Y(t)
    CORT_delayed_tau1 = Y(t - tau1)[2]  # CORT at t - tau1 for CRH
    CRH_delayed = Y(t - tau2)[0]        # CRH at t - tau2 for ACTH
    ACTH_delayed = Y(t - tau3)[1]       # ACTH at t - tau3 for CORT
    CORT_delayed_tau4 = Y(t - tau4)[2]  # CORT at t - tau4 for G6Pase
    
    # ODEs with delays
    dCRH_dt = k1 * (1 - eta * CORT_delayed_tau1 / (Kd + CORT_delayed_tau1)) - k2 * CRH
    dACTH_dt = k3 * CRH_delayed - k4 * ACTH
    dCORT_dt = k5 * ACTH_delayed - k6 * CORT
    dG6Pase_dt = k5 * (CORT_delayed_tau4 / (Km + CORT_delayed_tau4)) - k6 * G6Pase
    
    return np.array([dCRH_dt, dACTH_dt, dCORT_dt, dG6Pase_dt])

# History function (constant history before t=0)
def history(t, y0):
    return np.array(y0)

# Parameters
k1_base = 0.2    # Base production rate of CRH
k2 = 0.5         # CRH degradation rate
k3 = 0.50     # CRH to ACTH production rate
k4 = 0.4         # ACTH degradation rate
k5_base = 0.25   # Base production rate of G6Pase (and reused for CORT production)
k6 = 0.35        # G6Pase degradation rate (and reused for CORT degradation)
eta = 0.8        # Feedback strength
Kd = 0.2         # Dissociation constant for CORT feedback
A = 0.3          # Amplitude of circadian oscillation
Km = 0.15        # Michaelis constant for CORT activation of G6Pase
tau1 = 1.0       # Delay for CORT feedback in CRH equation
tau2 = 0.5       # Delay for CRH in ACTH equation
tau3 = 0.8       # Delay for ACTH in CORT equation
tau4 = 1.2       # Delay for CORT in G6Pase equation

# Parameter vector
k = [k1_base, k2, k3, k4, k5_base, k6, eta, Kd, A, Km, tau1, tau2, tau3, tau4]

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
plt.title('Computational simulation of the HPA-gluc axis system with multiple delays')
plt.grid(True)
plt.legend()
plt.show()
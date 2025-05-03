import numpy as np
from ddeint import ddeint
import matplotlib.pyplot as plt

# Define the ODE system with multiple delays
def hpa_gluc_axis(Y, t, k):
    k1_base, k2, k3, k4, k5_base, k6, eta, Kd, A, Km, tau1, tau2, tau3, tau4 = k
    k1 = k1_base * (1 + A * np.cos(2 * np.pi * t / 24))
    k5 = k5_base * (1 + A * np.cos(2 * np.pi * t / 24))
    CRH, ACTH, CORT, G6Pase = Y(t)
    CORT_delayed_tau1 = Y(t - tau1)[2]
    CRH_delayed = Y(t - tau2)[0]
    ACTH_delayed = Y(t - tau3)[1]
    CORT_delayed_tau4 = Y(t - tau4)[2]
    dCRH_dt = k1 * (1 - eta * CORT_delayed_tau1 / (Kd + CORT_delayed_tau1)) - k2 * CRH
    dACTH_dt = k3 * CRH_delayed - k4 * ACTH
    dCORT_dt = k5 * ACTH_delayed - k6 * CORT
    dG6Pase_dt = k5 * (CORT_delayed_tau4 / (Km + CORT_delayed_tau4)) - k6 * G6Pase
    return np.array([dCRH_dt, dACTH_dt, dCORT_dt, dG6Pase_dt])

# History function (constant history before t=0)
def history(t, y0):
    return np.array(y0)

# Common parameters
k1_base = 0.2
k2 = 0.5
k4 = 0.4
k6 = 0.35
eta = 0.8
Kd = 0.2
A = 0.3
Km = 0.15
tau1 = 1.0
tau2 = 0.5
tau3 = 0.8
tau4 = 1.2

# Three sets of parameters for k3 and k5_base
conditions = [
    {'k3': 0.50, 'k5_base': 0.25, 'label': 'G6Pase Condition I'},
    {'k3': 0.50, 'k5_base': 0.50, 'label': 'G6Pase Condition II'},
    {'k3': 1.00, 'k5_base': 0.50, 'label': 'G6Pase Condition III'}
]

# Time points (0 to 72 hours)
t = np.linspace(0, 72, 1000)

# Initial conditions: [CRH, ACTH, CORT, G6Pase]
y0 = [0.8, 0.7, 0.6, 0.5]

# Plot setup
plt.figure(figsize=(8, 4))

# Simulate and plot G6Pase for each condition
for cond in conditions:
    k3 = cond['k3']
    k5_base = cond['k5_base']
    k = [k1_base, k2, k3, k4, k5_base, k6, eta, Kd, A, Km, tau1, tau2, tau3, tau4]
    solution = ddeint(hpa_gluc_axis, lambda t: history(t, y0), t, fargs=(k,))
    G6Pase = solution[:, 3]  # Extract G6Pase
    plt.plot(t, G6Pase, label=cond['label'])

# Customize plot
plt.xlabel('Time (hours)')
plt.ylabel('G6Pase Concentration (ug/dl)')
plt.title('G6Pase Dynamics with Different k3 and k5_base Parameters')
plt.grid(True)
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
#x1=crh, x2 = acth, x3 = cortisol, x4=g6pase

def rep_sde(x, t, k, h, i):

    k1,k2,k3,k4,k5,kd,n,s1,s2,s3,S = k
    x1, x2, x3, x4 = x  

    dx1 = S * k1 * (1 - n*(x3/(kd+x3))) - (k2 * x1)
    dx2 = (k2 * x1) - (k3 * x2)
    dx3 = (k3 * x2) - (k4 * x3)
    dx4 = k5 * x3 

    x1_noise = s1 * x1  * np.random.normal(0, 1)
    x2_noise = s2 * x2  * np.random.normal(0, 1)
    x3_noise = s3 * x3 * np.random.normal(0, 1)
    x4_noise = s3 * x4 * np.random.normal(0, 1)

    return np.array([dx1+x1_noise, dx2+x2_noise, dx3+x3_noise, dx4+x4_noise])

#params
start = 0
end = 500
h = 0.001
k = [0.5,0.01,0.04,0.01,0.01,10,0.2,0,0,0,1]  # [k1,k2,k3,k4,k5,kd,n,s1,s2,s3,S]

t = np.arange(start, end, h)
x = np.zeros((len(t), 4))  
x0 = [1,10,5,0]  # Initial conditions
x[0, :] = x0  

#  Euler's step approach by adding noise
for i in range(len(t) - 1):

    dx = rep_sde(x[i, :], t[i], k, h, i)

    x[i + 1, 0] = x[i, 0] + h * dx[0] 
    x[i + 1, 1] = x[i, 1] + h * dx[1] 
    x[i + 1, 2] = x[i, 2] + h * dx[2] 
    x[i + 1, 3] = x[i, 3] + h * dx[3] 

x1, x2, x3, x4 = x.T

plt.plot(t, x1, label='CRH')
plt.plot(t, x2, label='ACTH')
plt.plot(t, x3, label='Cortisol')
plt.plot(t, x4, label='G6Pase')
plt.title('HPA-Gluconeogenesis Dynamics Timecourse')
plt.xlabel('Time (hours)')
plt.ylabel('Hormone Concentration (ug/dl)')
plt.legend()
plt.show()
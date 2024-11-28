import numpy as np
import matplotlib.pyplot as plt

def psp_kernel(t, Q, d, tau, beta):
    return (Q / (d * np.sqrt(t))) * np.exp(-beta * (d**2) / t) * np.exp(-t / tau) * (t > 0)

Q = 5
d = 1.5
tau = 20
beta = 1.1
time = np.linspace(0.1, 100, 1000)

psp_values = psp_kernel(time, Q, d, tau, beta)

plt.plot(time, psp_values, color='blue')
plt.title("PSP Kernel")
plt.xlabel("Time (t)")
plt.ylabel("ε(t)")
plt.show()

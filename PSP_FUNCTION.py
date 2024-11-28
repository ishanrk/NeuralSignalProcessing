import numpy as np
import matplotlib.pyplot as plt

# Define the modified AHP kernel
def modified_ahp_kernel(t, R, gamma, asymptote):
    return  (R) * np.exp(-t / gamma) * (t >= 0)

# Parameters
R = -1.0       # Initial drop at t=0
gamma = 1.5    # Decay rate
asymptote = -1 # Leveling out value
time = np.linspace(0, 10, 1000)  # Time range

# Compute the modified AHP kernel
ahp_vals = modified_ahp_kernel(time, R, gamma, asymptote)
# Plot the modified AHP kernel
plt.figure(figsize=(8, 4))
plt.plot(time, ahp_vals, label="Modified AHP Kernel", color="blue")
plt.axhline(y=asymptote, color="red", linestyle="--", label="")
plt.axvline(x=0, color="gray", linestyle="--", label="t=0")
plt.title("Modified AHP Kernel with Leveling Out")
plt.xlabel("Time (t)")
plt.ylabel("AHP(t)")
plt.legend()
plt.grid(True)
plt.show()

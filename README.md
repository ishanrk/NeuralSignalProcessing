# Spike Response Model (SRM) for a Single Neuron

This repository contains a Python implementation of a Spike Response Model (SRM) for a single neuron. The model simulates the neuron's response to an injected current and includes mechanisms for post-spike responses and integration with filter kernels.

![image](https://github.com/user-attachments/assets/95123a0e-b4e3-4eff-89ef-f8f6ddaf9763)

---

## Features
- **Spike Response Model (SRM):** Models a single neuron's membrane potential, `u(t)`, as a combination of post-spike responses (`η(t)`) and filtered injected current.
- **Injected Current:** Simulates random Gaussian noise as input to the neuron.
- **Filter Kernel:** Low-pass filter to smooth the injected current.
- **Spike Generation:** Tracks spikes based on a defined threshold and incorporates post-spike effects into the membrane potential.
- **Plotting:** Visualizes the membrane potential over time.

---

## Equation

The membrane potential is calculated as:

$\[
u(t) = \eta(t - \hat{t}) + \int_0^\infty \kappa(t - \hat{t}, s) I(t - s) \, ds
\]$

- $\( \eta(t) \)$: Post-spike response.
- $\( \kappa(t, s) \)$: Filter kernel.
- $\( I(t) \)$: Injected current.

---

## Code Overview

### **Key Components:**

1. **Post-Spike Response (\( \eta(t) \)):**
   ```python
   def eta(t2, eta_0=0.5, tau_m=20.0):
       # Computes post-spike response with exponential decay and Dirac delta approximations.

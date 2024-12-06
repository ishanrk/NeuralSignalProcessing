# Neuron Models and Signal Processing

This repository contains information about different Neuron Models including Leaky Integrate and Fire, Hodgkin Huxley Model, and Spike Rresponse Models. It also has basic information on designing experiements and getting their associated raster plots. There is a section about encoding continuous signals to spike trains and vice versa. At the very beginning you can see an introduction section,

---

## Table of Contents
1. [Introduction](#introduction)
   - [Kernels and Filters](#kernels-and-filters)
2. [Spike Response Model (SRM)](#spike-response-model-srm)
   - [Single Neuron Model](#SRM-for-a-single-neuron)
   - [Multiple Synaptic Inputs](#SRM_0-for-multiple-synaptic-inputs)
3. [Experiments](#experiments)
4. [Hopfield Nets and Memory Models](#hopfield-nets-and-memory-models)
5. [Author](#author)

---
# Introduction
This section will give a brief introduction to what kernels are. It will also introduce you to different filters and what they do.

## Kernels and Filters

### What is a Kernel?
A **kernel** is a mathematical function that defines the shape of a transformation applied to data. Kernels are commonly used in signal processing and neural models to describe how past inputs influence the current state.

### What is a Filter?
A **filter** applies a kernel to modify or transform an input signal. Filters often use **convolution**, which combines the input signal with the kernel to produce a smoothed or transformed output.

#### Mathematical Representation of Convolution
$\[
y(t) = \int_0^\infty \kappa(s) \cdot I(t - s) \, ds
\]$
Where:
- $\( \kappa(s) \)$: Kernel function.
- $\( I(t) \)$: Input signal.
- $\( y(t) \)$: Filtered signal.

---

### Example Kernels
Below are examples of commonly used kernels:

1. **Exponential Kernel** (used in low-pass filters):
   $\[
   \kappa(s) = \lambda \cdot e^{-s/\tau}
   \]$

2. **Gaussian Kernel** (used in smoothing):
   $\[
   \kappa(s) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot e^{-s^2/(2\sigma^2)}
   \]$

3. **Spike Response Kernel** (used in SRM models):
   $\[
   \kappa(s) = \frac{s}{\tau^2} e^{-s/\tau}
   \]$

---

### Kernel Plots
Below are visualizations of different kernels:

#### Exponential Kernel
![image](https://github.com/user-attachments/assets/6e375e82-2289-486e-a7d0-2e55d64f987a)

#### Gaussian Kernel
![image](https://github.com/user-attachments/assets/ce100ef8-76c6-45a8-a0ab-623b1f7f9fca)

These kernels are essential for filtering and understanding neural response models.

## Convolution Between Two Functions

### What is Convolution?
**Convolution** is a mathematical operation that integrates one function with a shifted version of another, commonly used in signal processing to filter or smooth signals.

### Example
Here, we convolve a sine wave (signal) with a Gaussian kernel. The convolution smooths the oscillations of the sine wave, resulting in a filtered version of the signal.

#### Convolution Formula
$\[
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \cdot g(t - \tau) \, d\tau
\]$

- $\( f \)$: Signal (sine wave).
- $\( g \)$: Convolution kernel (Gaussian).

### Plots
1. **Signal (Sine Wave), Kernel (Gaussian Kernel), and Convolution Result:**
  ![image](https://github.com/user-attachments/assets/1596d967-e9bd-4b55-b079-643a720f1e10)



# Spike Response Model (SRM)

The Spike Response Model (SRM) is a framework for simulating neuronal dynamics. It calculates the membrane potential 𝑢(𝑡) based on input currents, synaptic inputs, and post-spike responses.

## SRM for a Single Neuron
The SRM models the membrane potential of a single neuron as:
$\[
u(t) = \eta(t - \hat{t}) + \int_0^\infty \kappa(t - \hat{t}, s) I(t - s) \, ds
\]$

## Equation

- $\( \eta(t) \)$: Post-spike response.
- $\( \kappa(t, s) \)$: Filter kernel.
- $\( I(t) \)$: Injected current.

## Features
- **Spike Response Model (SRM):** Models a single neuron's membrane potential, `u(t)`, as a combination of post-spike responses (`η(t)`) and filtered injected current.
- **Injected Current:** Simulates random Gaussian noise as input to the neuron.
- **Filter Kernel:** Low-pass filter to smooth the injected current.
- **Spike Generation:** Tracks spikes based on a defined threshold and incorporates post-spike effects into the membrane potential.
- **Plotting:** Visualizes the membrane potential over time.

## Code Overview

### Components
- $\eta(t - \hat{t})$: Post-spike response kernel, accounting for the refractory period after a spike.
- $\kappa(t - \hat{t}, s)$: Filter kernel, modeling the integration of injected current 𝐼(𝑡)
- $\hat{t}$: Time of the most recent spike. 

1. **Post-Spike Response $(\( \eta(t) \))$:**
   ```python
   def eta(t2, eta_0=0.5, tau_m=20.0):
      eta_values = np.zeros_like(t2)  # Initialize the eta array with zeros
    
    # Dirac delta spikes at t = 5 ms and t = 20 ms
       spike_times = [5, 20, 70 ,90]
    
    for spike_time in spike_times:
        # Find the index of the spike time closest to the time array
        spike_index = np.argmin(np.abs(t2 - spike_time))
        eta_values[spike_index] = 10  # Dirac delta approximation with value 10
        
        # Exponential decay component for time after the spike
        for i in range(spike_index + 1, len(t2)):
            # Only apply exponential decay for times after the spike time
            eta_values[i] += -eta_0 * np.exp(-(t2[i] - spike_time) / 20)
    
    return eta_values
       # Computes post-spike response with exponential decay and Dirac delta approximations.
2. **Decay Kernel $(\( \kappa(t) \))$:**
   ```python
   def filter_kernel(s, tau, lambda_val=1.0):
    return lambda_val * np.exp(-s / tau) * s/tau # low pass filter: allows for slowly moving changes, blocks fast changes
3. **Convolutonal Kernel Response $\int_0^\infty \kappa(t - \hat{t}, s) I(t - s) \, ds$:**
    ```python
    def response_to_current(current, tau, lambda_val=1.0, dt=0.1):
   
    time = np.arange(0, len(current) * dt, dt)
    
    # Initialize the response array
    response = np.zeros_like(current)
    
    # Loop over each time step to compute the response
    for t in range(1, len(current)):
        # Integrate the current with the filter kernel
        # We'll sum over past values of the current, weighted by the filter
        integral = 0
        for s in np.arange(0, t * dt, dt):  # Integration variable (s)
            integral += filter_kernel(s, tau) * current[t - int(s / dt)]
        response[t] = integral * dt  # Update the response
        
    return response
### Plotting
![image](https://github.com/user-attachments/assets/5db9ec80-dec8-43fd-adcb-5a4b9320f15e)

---
## SRM_0 for Multiple Synaptic Inputs

In the case of input generated by synaptic current pulses caused by presynaptic spike arrival, the Spike Response Model (SRM) can be written as:

$\[
u(t) = \eta(t - \hat{t}) + \sum_{j} \sum_{f} w_{j} \varepsilon(t - \hat{t}, t - t_{j}^{f})
\]$

Where:
- $\( w_j \)$ is the weight of synapse $\( j \)$,
- $\( t_j^{f} \)$ is the arrival time of the \( f \)-th spike at synapse $\( j \)$,
- $\( \varepsilon(t - \hat{t}, t - t_j^f) \)$ is the time course of the postsynaptic potential caused by the spike arrival at synapse $\( j \)$.

To see the connection with the equation above, suppose that the input to the Spike Response Model consists not of an imposed current, but of synaptic input currents of amplitude $\( w_j \)$ and time course $\( \alpha(t - t_j^f) \)$, where $\( t_j^f \)$ is the spike arrival time at synapse $\( j \)$. The input current is:

Convolution of the kernel $\( \kappa \)$ with the current $\( \alpha(t - t_j^f) \)$ yields the postsynaptic potential $\( \varepsilon(t - \hat{t}, t - t_j^f) \)$.

### Code Example for Multiple Synaptic Inputs with Spike Arrival

The following Python code implements the SRM for multiple synaptic inputs, including the calculation of the membrane potential in response to multiple spikes:

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants for the PSP kernel
Q = 5
d = 1.5
tau = 20
beta = 1.1

# AHP kernel constants
R = -1
gamma = 1.5

time = np.linspace(0, 100, 1000)  # 1000 points
dt = time[1] - time[0] 

# PSP kernel (ε(t))
def psp_kernel(t, Q, d, tau, beta):
    return (Q / (d * np.sqrt(t))) * np.exp(-beta * (d**2) / t) * np.exp(-t / tau) * (t > 0)

# AHP kernel (η(t))
def ahp_kernel(t, R, gamma):
    return  (R) * np.exp(-t / gamma) * (t > 0)

# Synapse firing times
SYNAPSE1_INPUTS = [8, 10 ,12]  # Excitatory neuron
SYNAPSE2_INPUTS = [13,14,15]  # Inhibitory neuron
SYNAPSE_LIST = [SYNAPSE1_INPUTS, SYNAPSE2_INPUTS]

u = np.zeros_like(time)

# Simulation
for t_idx, t in enumerate(time):
    # Find AHP
    if t_idx > 0:
        u[t_idx] += ahp_kernel(time[:t_idx] - time[t_idx - 1], R, gamma).sum()

    # Summation over all synapses
    for synapse_idx, firing_times in enumerate(SYNAPSE_LIST):
        for tf in firing_times:
            if tf < t:  # Only valid for past spikes
                if synapse_idx == 1:
                    u[t_idx] += psp_kernel(t - tf, Q, d, tau, beta)  # Inhibitory
                else:
                    u[t_idx] += psp_kernel(t - tf, Q, d, tau, beta)  # Excitatory

    # Threshold of 1, if crossed, add spike of 10 and paste an AHP kernel
    if u[t_idx] >= 1:
        u[t_idx] += 10  # Add a spike
        u[t_idx:] += ahp_kernel(time[:len(time) - t_idx], R, gamma)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, u, label="u(t)")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (u(t))")
plt.title("Membrane Potential u(t) Over Time with Spikes")
plt.axhline(1, color='red', linestyle='--', label="Threshold")
for spike_time in SYNAPSE1_INPUTS:
    plt.axvline(spike_time, color='green', linestyle='--', alpha=0.7, label="Synapse 1 Spike" if spike_time == SYNAPSE1_INPUTS[0] else "")

# Add markers for Synapse 2 spike times
for spike_time in SYNAPSE2_INPUTS:
    plt.axvline(spike_time, color='orange', linestyle='--', alpha=0.7, label="Synapse 2 Spike" if spike_time == SYNAPSE2_INPUTS[0] else "")

plt.legend()
plt.grid()
plt.show()
```
### Plotting
![image](https://github.com/user-attachments/assets/dc7084dc-9736-43cb-b9d7-015be4ce476c)


## PSP and AHP Kernels

The **Postsynaptic Potential (PSP)** kernel and **Afterhyperpolarization (AHP)** kernel are crucial components in the modeling of synaptic input and the resulting membrane potential. These kernels are used to calculate the effect of presynaptic spikes on the postsynaptic neuron.
The kernels referenced fromn: Arunava Banerjee. Computational Neuroscience: Methods and Modeling. Retrieved from https://www.cise.ufl.edu/~arunava/

### PSP Kernel Equation

The **PSP kernel** $(\(\varepsilon(t)\))$ is used to model the postsynaptic potential due to a synaptic input. It is given by the following equation:

$\[
\varepsilon(t) = \frac{Q}{d \sqrt{t}} \exp\left( -\frac{\beta d^2}{t} \right) \exp\left( -\frac{t}{\tau} \right) \cdot \mathbb{1}_{t > 0}
\]$

Where:
- $\( Q \)$ is the amplitude of the synaptic current,
- $\( d \)$ is a scaling factor,
- $\( \beta \)$ is a constant for the time course,
- $\( \tau \)$ is the time constant,
- $\( \mathbb{1}_{t > 0} \)$ is the Heaviside step function ensuring the kernel is valid for positive times.

![image](https://github.com/user-attachments/assets/00e91138-ce0f-41d3-905c-dce5b995caaf)


### AHP Kernel Equation

The **AHP kernel** $(\(\eta(t)\))$ is used to model the afterhyperpolarization effect that occurs after a spike. It is described by:
$
\[
\eta(t) = R \exp\left( -\frac{t}{\gamma} \right) \cdot \mathbb{1}_{t > 0}
\]
$
Where:
- $\( R \)$ is the amplitude of the AHP,
- $\( \gamma \)$ is the time constant for the decay of the AHP,
- $\( \mathbb{1}_{t > 0} \)$ ensures the kernel is valid for positive times.
![image](https://github.com/user-attachments/assets/38166dfc-de37-4e8a-971a-e4a8649c5d7e)

### Code to Plot PSP and AHP Kernels

Below is the Python code to generate plots for the PSP and AHP kernels:

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants for the PSP kernel
Q = 5
d = 1.5
tau = 20
beta = 1.1

# AHP kernel constants
R = -1
gamma = 1.5

time = np.linspace(0, 100, 1000)  # 1000 points

# PSP kernel (ε(t))
def psp_kernel(t, Q, d, tau, beta):
    return (Q / (d * np.sqrt(t))) * np.exp(-beta * (d**2) / t) * np.exp(-t / tau) * (t > 0)

# AHP kernel (η(t))
def ahp_kernel(t, R, gamma):
    return  (R) * np.exp(-t / gamma) * (t > 0)

# Plotting both PSP and AHP Kernels
plt.figure(figsize=(12, 6))

# PSP kernel plot
plt.subplot(1, 2, 1)
plt.plot(time, psp_kernel(time, Q, d, tau, beta), label="PSP Kernel (ε(t))", color='b')
plt.title("PSP Kernel")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

# AHP kernel plot
plt.subplot(1, 2, 2)
plt.plot(time, ahp_kernel(time, R, gamma), label="AHP Kernel (η(t))", color='r')
plt.title("AHP Kernel")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
```
# Experiments
Using the SRM_0 model listed above, you can design your own network of spiking neurons with some defined as input neurons that you can feed a spike train to. Then for each neuron in each layer in your network, you can measure its spiking times and plot a raster plot to show spiking frequency. I've listed an example below that shows the raster plot of 10 spiking neurons fed an input spike train of [8,10,12,70].
![image](https://github.com/user-attachments/assets/e7d33043-5e72-4361-857b-df6976713a6d)

# Hopfield Nets and Memory Models
This project implements a Hopfield network capable of storing binary images and recalling them when fed corrupted versions. The network is trained on a set of input patterns and iteratively updates its states to converge on the closest stored memory.

## Features

- **Hopfield Network Implementation**: Supports training on multiple binary images and recall from corrupted inputs.
- **Custom Corruption**: Simulates image corruption by altering a specific portion (e.g., the left quarter) of the image.
- **Image Display**: Visualizes the original, corrupted, and recalled images for comparison.

---
# Author
Ishan Kumthekar

import numpy as np
import matplotlib.pyplot as plt

# constants for the PSP kernel
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

# synapse firing times
SYNAPSE1_INPUTS = [8, 10 ,12] # exhibitory neuron
SYNAPSE2_INPUTS = [13,14,15] # inhibitory neuron
SYNAPSE_LIST = [SYNAPSE1_INPUTS, SYNAPSE2_INPUTS]

u = np.zeros_like(time)

# simulation
for t_idx, t in enumerate(time):
    # find ahp
    if t_idx > 0:
        u[t_idx] += ahp_kernel(time[:t_idx] - time[t_idx - 1], R, gamma).sum()

    # summation over all synapses
    for synapse_idx, firing_times in enumerate(SYNAPSE_LIST):
        for tf in firing_times:
            if tf < t:  # only valid for past spikse
                if synapse_idx==1:
                    u[t_idx] -= psp_kernel(t - tf, Q, d, tau, beta)
                else:
                    u[t_idx] += psp_kernel(t - tf, Q, d, tau, beta)

    # threshold of 1, if crossed add spike of 10 and paste a ahp kernel
    if u[t_idx] >= 1:
        u[t_idx] += 10  # Add a spike
        u[t_idx:] += ahp_kernel(time[:len(time) - t_idx], R, gamma)

# plot
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

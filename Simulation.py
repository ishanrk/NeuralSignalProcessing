import numpy as np
import matplotlib.pyplot as plt


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

def filter_kernel(s, tau, lambda_val=1.0):
  
    return lambda_val * np.exp(-s / tau) * s/tau # low pass filter: allows for slowly moving changes, blocks fast changes
    # 


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

# Time array (0 to 100 ms)
time = np.linspace(0, 100, 1000)
current = np.random.normal(0, 0.9, len(time))  # current with Gaussian noise



tau = 2  # Membrane time constant (ms)


response = response_to_current(current, tau)

eta_values = eta(time)  
u_values=[]
for i in range(0,1000):
    u_values.append(eta_values[i]+response[i])



spike_times = []
dt = 0.1

for t in range(0,1000):
    eta_values[t] =0
for t in range(0,1000):
    integral = 0
    for s in np.arange(0, t * dt, dt):  # Integration variable (s)
            integral += filter_kernel(s, tau) * current[t - int(s / dt)]
    response[t] = integral * dt
    u_values[t]=response[t] + eta_values[t]
    if(u_values[t]>=0.4):
        spike_times.append(t)
        for i in range(t + 1, 1000):
            # Only apply exponential decay for times after the spike time
            eta_values[i] += -1.5 * np.exp(-(i - t) / 2)
        u_values[t]+=10
    
plt.figure(figsize=(10, 6))


plt.plot(time, u_values, label=r'$\eta(t_1, t_2)$: Post-Spike Response, threshold = 0.4')
plt.title("Post-Spike Response ")
plt.xlabel("Time (ms)")
plt.ylabel("u(t)")

plt.legend()
plt.grid(True)
plt.show()


# eta kernel =  -1.5 * np.exp(-(i - t) / 20). shape pasted after spike or after potential
# refractory period of 5 ms

# kappa kerenl for membrane filter: filter_kernel(s, tau, lambda_val=1.0):
  
  #lambda_val * np.exp(-s / tau)
  # this is convolved with input current from 0 to infinity to time t to get the membrane potential

  # now how do you connect different neurons together and model it since there is no injected current now
  # how do you model experiments and what does the training data normally look like
  # apart from theoretical neuroscience are there any good resources for comp neuro (neuronal dynamics by gerstner)
  # What does the study of plasticity involve and how is it studied computationally (changing weights of synapses)
  # hopfield net 
  # any recommended math, since parts of dayan and abbot are relatively math heavy/notation heavy
  # real analysis 1, mathematical probability, dynamical systems: intro to dynamical systems katok 
  # any specific classes you would recommend since registration is here
  # for ml and more specifically for linear regression and multivariate regression, there is a guarantee that the model will converge 
  # convergence guarantee for having data that has rank atleast equal to the number of dimensions?
  # each synapyse will get a set of dirac deltas coming in, different kappas for different synapses. parameter for distance to measure synapse strength towards response
  # summation of dirac deltas over all synapses,for weights look up thr Q formula
  # look at past of other neurons and set a time step to measure the potential for all neurons in the network
  # 10% of neurons get spikes (assumption)
  # input(t): input current into some neurons (Stimulus to current) . (record of spike trains)




## SRM 0 WITH DIFFERENT SYNAPTIC INPUTS



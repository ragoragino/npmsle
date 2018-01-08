import npsmle
import numpy as np
import functools
import scipy.optimize
import time

"""
PARAMETERS
"""
N_obs = 1000
N_sim = 500
alpha = 0.06
beta = 0.5
sigma = 0.15
delta = 1
start = alpha
step = 10
h = 0.35

np.random.seed(123)

"""
SIMULATION OF THE VASICEK MODEL
"""
def simulation_vasicek(alpha, beta, sigma, N, x0, delta, step):
    dt = delta / step
    process = np.zeros(N, dtype=np.double)
    process[0] = x0
    for i in range(1, N):
        process[i] = process[i - 1]
        for _ in range(0, step):
            process[i] += alpha * (beta - process[i]) * dt + sigma * (dt ** 0.5) * np.random.normal()

    return process

vasicek = simulation_vasicek(alpha, beta, sigma, N_obs, start, delta, step)
sim_vasicek = np.zeros(N_sim, dtype=np.double)  # pre-allocate array for a simulated vasicek process

random_size = 10000
random_buffer = np.random.normal(0, 1, random_size)
random_buffer_index = np.random.randint(0, random_size, random_size, dtype=np.int32)

# Initial parameters into optimization
theta_size = 3
theta_low = [0.0, 0.0, 0.0]
theta_high = [1.0, 1.0, 1.0]
initial_params = np.zeros(3, dtype=np.double)
for i in range(theta_size):
    initial_params[i] = np.random.uniform(theta_low[i], theta_high[i])

# Construct the optimization function
partial_loglik_vasicek = functools.partial(npsmle.loglik_vasicek, process=vasicek,
                                           sim_process=sim_vasicek, random_buffer=random_buffer,
                                           random_buffer_index=random_buffer_index, process_length=N_obs,
                                           sim_process_length=N_sim, step=step, delta=delta)

timing_frame = 20
timing = 0.0
for i in range(timing_frame):
    begin = time.time()
    # Optimization routine
    for i in range(theta_size):
        initial_params[i] = np.random.uniform(theta_low[i], theta_high[i])
    begin = time.time()
    res = scipy.optimize.minimize(fun=partial_loglik_vasicek, x0=initial_params,
                                      method="Nelder-Mead", options={'maxiter': 200, 'disp': True})
    timing += (time.time() - begin) / timing_frame

    print(res)

print("ELAPSED TIME: ", timing)






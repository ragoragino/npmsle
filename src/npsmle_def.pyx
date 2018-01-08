import numpy as np
cimport numpy as np
cimport npsmle_dec as npsmle
import cython
import ctypes
from libc.time cimport time, time_t

"""
DEFINITIONS
"""

cdef unsigned int cseed = time(NULL)

@cython.boundscheck(False)
@cython.wraparound(False)
def loglik_vasicek(np.ndarray[dtype=double, ndim=1, mode="c"] params not None,
	np.ndarray[dtype=double, ndim=1, mode="c"] process not None,
	np.ndarray[dtype=double, ndim=1, mode="c"] sim_process not None,
	np.ndarray[dtype=double, ndim=1, mode="c"] random_buffer not None,
	np.ndarray[dtype=int, ndim=1, mode="c"] random_buffer_index not None,
	process_length, sim_process_length, step, delta):
  
	loglikelihood = npsmle.simulated_ll_vasicek(&process[0], &sim_process[0],  
	process_length, sim_process_length, step, params[0], params[1], params[2], 
	delta, &random_buffer[0], &random_buffer_index[0])

	return loglikelihood
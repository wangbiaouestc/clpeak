'''
Demonstrates under more controlled situations that the output of divergent warps
can be affected by the order of thread index as well as the ordering of ifs and elses in
the code itself.
'''

# Import Cheetah.
from Cheetah.Template import Template

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

import itertools
import numpy as np
import sys

group_kernel_string = \
"""
/**
 * Splits a warp in two at a threshold value.  Templating is used so that the lower-index
 *  threads can execute either the top or the bottom condition, and the threshold is variable.
 */
__global__ void group_kernel(int* out) {
	int tid = threadIdx.x;

	extern __shared__ int data[];
	
	if (tid $inequality $threshold) {
		// __syncthreads() used to try to ensure prior memory transactions completed
		__syncthreads();
		if (tid == 0 || tid == blockDim.x)
			data[0] = 0;
		__syncthreads();
	} else {
		__syncthreads();
		if (tid == 0 || tid == blockDim.x)
			data[0] = 1;
		__syncthreads();
	}
	
	if (tid == 0) {
		out[0] = data[0];
	}
}
"""

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

def main():
  template = Template(group_kernel_string)
  out_d = gpu.empty(1, dtype=np.int32)
  small_last = True

  for inequality in ['<', '<=', '>', '>=']:
    template.inequality = inequality
    for threshold in xrange(2,31):
      template.threshold = threshold

      group_kernel = cuda_compile(template, 'group_kernel')
    
      group_kernel(out_d, block=(32,1,1), grid=(1,1), shared=4)
      out = out_d.get()[0]

      if inequality in ['<', '<='] and out != 0 or inequality in ['>', '>='] and out != 1:
        print 'Higher tid write persisted! (unusual)', inequality, threshold, out
        small_last = False

  if small_last:
    print 'Lower tid write always persisted!'

if __name__ == '__main__':
  main()

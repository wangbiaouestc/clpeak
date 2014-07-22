'''
Finds the full ordering in which conditionals execute within the kernel.  Demonstrates
the effect of the ordering of ifs and elses on output in conditional code.
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

kernel_string = \
"""
/**
 * Probes the order in which conditional branches execute.  Must be called with 33 threads.
 * out[0] will store the index of the thread that first executes.
 * out[1] will store the index of the thread that second executes.
 * ... and so on for ... out[31]
 */
__global__ void kernel(int* order, int* out) {
	int tid = threadIdx.x;
	int count_idx = blockDim.x;
	
	extern __shared__ int data[];
	
	if (tid == 0) data[count_idx] = 0;

	if (tid == order[0]) {
		__syncthreads();
		data[data[count_idx]] = tid;
		data[count_idx] += 1;
	} 
	#for $idx in xrange(1, 31):
	else if (tid == order[#echo $idx#]) {
		__syncthreads();
		data[data[count_idx]] = tid;
		data[count_idx] += 1;
	}
	#end for
	else {
		__syncthreads();
		data[data[count_idx]] = tid;
		data[count_idx] += 1;
	}

	out[tid] = data[tid];
}
"""

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)
    
def main():
  n = 32
  num_permutation_runs = 5000
  
  template = Template(kernel_string)
  kernel = cuda_compile(template, 'kernel')
  out_d = gpu.empty(n, dtype=np.int32)

  num_runs_so_far = 0
  all_top_to_bottom = True

  # Try many different orderings
  for order in itertools.permutations(np.arange(0,n)):
    order = np.array(order, dtype=np.int32)
    order_d = gpu.to_gpu(order)

    kernel(order_d, out_d, block=(n,1,1), grid=(1,1), shared=(n+1)*4)
    out = out_d.get()

    # If out and order are not equal
    if not reduce(lambda x,y: x and y, out==order):
      print 'Execution was not top to bottom'
      all_top_to_bottom = False
      
    num_runs_so_far += 1
    if num_runs_so_far >= num_permutation_runs:
      break

  if all_top_to_bottom:
    print 'All executions went from top to bottom'

if __name__ == '__main__':
  main()


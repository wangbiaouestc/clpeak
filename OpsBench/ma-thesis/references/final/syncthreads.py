from Cheetah.Template import Template

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

import numpy as np
import sys

n = 1024  # Number of threads and length of array.
in_arr  = np.int32(np.random.randint(10000, size=n)) # Array going to GPU.

kernel_string = \
"""
/**
 * max_kernel -- returns the maximum value of the array 'in'
 *
 * Uses __syncthreads() to try to force lower tids to go second, but fails when
 *   threads within a single warp diverge.
 */
__global__ void max_kernel(int* in, int* out) {
	int tid = threadIdx.x;

	extern __shared__ int data[];
	data[tid] = in[tid];
	__syncthreads();

	int last = $n;
	while (last > 0) {
		int half = last / 2;
		if (tid < last) {
			if (tid < half) {
				// Copies the max over
				__syncthreads();
				data[tid] = data[tid + half];
			} else {
				// Calculates the max
				data[tid] = max(data[tid], data[tid - half]);
				__syncthreads();
			}
		} else {
			__syncthreads();
		}

		last = half;
		__syncthreads();
	}

	if (tid == 0) out[0] = data[0];

}
"""

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)
    
def main():
  template = Template(kernel_string)
  template.n = n
  max_kernel = cuda_compile(template, 'max_kernel')
  out_arr = np.zeros(1, dtype=np.int32)
  
  in_d  = gpu.to_gpu(in_arr)
  out_d = gpu.to_gpu(out_arr)

  max_kernel(in_d, out_d, block=(n,1,1), grid=(1,1), shared=n*4)
  out = out_d.get()
  print 'Kernel returns:', out[0]
  print 'Actual max:    ', np.max(in_arr)

if __name__ == '__main__':
  main()


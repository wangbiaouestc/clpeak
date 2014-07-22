'''
Figures out the associativity of the cache (sizes of lines, number of sets,
associativity, and so on).
'''
from Cheetah.Template import Template

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

import numpy as np
import sys

# We change the MIN / MAX size based on what cache we are exploring. For 
# instance, we center the min and max over 768KB if we are probing the L2 cache.
# We use a much smaller value for L1 (16 or 48KB).
STRIDE = 2
SIZE_INCREMENT = 2 * 4
MIN_SIZE = 760000
MAX_SIZE = 860000
tpb = 1 
bpg = 1
unrolling = 256
num_repeats = 128
L1 = True

kernel_string = \
"""
__device__ long long int perform_stride(int* array) {
	int k = 0;
	int i;
	
	long long int start = clock64();

  for(i = 0; i < $num_repeats; i++) {
	  #for _ in xrange($unrolling):
	  k = array[k];
	  #end for
	}

	long long int end = clock64();

	// Prevent compiler from optimizing out the loop
	if ( k < 0) {
		return -1;
	} else {
		// Do we need to worry about overflow?
		return start < end ? end-start : end + (0xffffffffffffffff - start);
	}
}

__global__ void stride_kernel(int* array, int warm, long long int* time) {
	if (threadIdx.x > 0) {
	  return;
	}
	__syncthreads();
	
	if (warm != 0) perform_stride(array);

	time[blockIdx.x] = perform_stride(array);
}
"""

def cuda_compile(source_string, function_name):
  # Turn L1 caching off to simplify analysis of L2.
  if L1:
    options = None
  else:
    options = ['-Xptxas','-dlcm=cg']
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string, options=options)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

def init_array(nbytes, stride):
  num_elems = nbytes / 4
  array = np.arange(stride, num_elems+stride, dtype=np.int64)
  array = array % num_elems
  return np.int32(array)

def main():
  if len(sys.argv) != 2:
    print 'Usage: python associativity.py output_file_name'
    exit()

  # Array sizes.
  sizes = np.arange(MIN_SIZE, MAX_SIZE + 1, SIZE_INCREMENT)
  num_iters = unrolling*num_repeats
  template = Template(kernel_string)
  time_d = gpu.empty(bpg, dtype=np.int64)
  warm = np.int32(1)
  
  # Compile just once.
  template.unrolling  = unrolling
  template.num_repeats = num_repeats
  stride_kernel = cuda_compile(template, 'stride_kernel')

  # PREFER_SHARED allocated 16kB for L1, 48kB for shared mem. PREFER_L1 reverses these
  cu.Context.set_cache_config(cu.func_cache.PREFER_SHARED)

  with open(sys.argv[1], 'w') as f:
    for nbytes in sizes:
      # Get the array and send to GPU
      array = init_array(nbytes, STRIDE)
      array_d = gpu.to_gpu(array)

      # Run the kernel and record the result
      stride_kernel(array_d, warm, time_d, block=(tpb,1,1), grid=(bpg,1))
      cycles = time_d.get()
      f.write('%d\t%d\t%d\t%d\t%d\t%d\n' % (nbytes, STRIDE, tpb, bpg, cycles[0], num_iters))
      print 'Done with an array size of %d B.' % (int(nbytes))

      # Free the memory held by the array on the GPU
      array_d.gpudata.free()
        
if __name__ == '__main__':
  main()


'''
Performs latency tests with varying array sizes and strides.
'''
from Cheetah.Template import Template

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

import numpy as np
import sys

# Turn L1 caching on? If L1 is on, we are analyzing the effects of stride
# length. Otherwise, we are analyzing the effects of turning the L1 cache off.
L1 = True
L2 = True
log_stride = True # Increment stride logarithmically?
MIN_STRIDE = 1
MAX_STRIDE = 24
tpb = 1 # Block size (Threads per block).
bpg = 1 # Grid size (Blocks per grid).
unrolling = 256 # How much to unroll.
num_repeats = 1 # Repeats within unrolling.
SIZES = [16384, 49152, 49*2**10, 400*2**10, 786432, 770*2**10, 2**26]
BIG_L1 = True # If true, use 48K for L1 cache, 16K for shared mem. Otherwise, reverse.
WARM = False # Should we warm the cache?

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
		// Handle (unlikely) overflow
		return start < end ? end-start : end + (0xffffffffffffffff - start);
	}
}

__global__ void stride_kernel(int* array, int warm, long long int* time) {
	// Only run on one thread per block
	if (threadIdx.x == 0) {
		long long int ret = 0;
		if (warm != 0) ret = perform_stride(array);

		// Always true; prevents compiler removing first call
		if (ret >= 0)  time[blockIdx.x] = perform_stride(array);
	}
}
"""

def cuda_compile(source_string, function_name):
  # Turn L1 caching off if appropriate.
  if L1 and L2:
    options = None
  else:
    options = ['-Xptxas']
    if L2 and not L1:
      # 'cg' = Cache globally -- disables L1 but not L2
      options.append('-dlcm=cg')
    if not L2:
      # 'cs' = Cache streaming -- tells GPU to treat data as if it will not be reused
      options.append('-dlcm=cs')

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
    print 'Usage: python stride.py output_file_name'
    exit()

  sizes = np.array(SIZES, dtype=np.int32)
  num_iters = unrolling*num_repeats
  template = Template(kernel_string)
  time_d = gpu.empty(bpg, dtype=np.int64)

  if WARM:
    warm = np.int32(1)
  else:
    warm = np.int32(0)
  
  # Compile just once if we are analyzing the effects of turning off the L1 cache.
  template.unrolling  = unrolling
  template.num_repeats = num_repeats
  stride_kernel = cuda_compile(template, 'stride_kernel')

  # Use 48 KB for cache
  if BIG_L1:
    cu.Context.set_cache_config(cu.func_cache.PREFER_L1)
  else:
    cu.Context.set_cache_config(cu.func_cache.PREFER_SHARED)

  # Determine the output file.
  output_file = open(sys.argv[1], 'w')

  if log_stride:
    strides = np.array([2**x for x in xrange(MIN_STRIDE, MAX_STRIDE + 1)], dtype=np.int32)
  else:
    strides = np.arange(MIN_STRIDE, MAX_STRIDE + 1, dtype=np.int32)
  with output_file as f:
    for nbytes in sizes:
      for stride in strides:
        # Get the array and send to GPU
        array = init_array(nbytes, stride)
        array_d = gpu.to_gpu(array)

        # Run the kernel and record the result
        stride_kernel(array_d, warm, time_d, block=(tpb,1,1), grid=(bpg,1))
        cycles = time_d.get()
        f.write('%d\t%d\t%d\t%d\t%d\t%d\n' % (nbytes, stride, tpb, bpg, cycles[0], num_iters))
        # Free the memory held by the array on the GPU
        array_d.gpudata.free()
        
      print 'Done with an array size of %d B.' % (int(nbytes))
        
if __name__ == '__main__':
  main()

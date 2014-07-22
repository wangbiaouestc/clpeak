'''
Tests for what happens when many threads attempt to access the same L2 cache.
'''
from Cheetah.Template import Template

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

import numpy as np
import sys

MAX_BPG    = 14 # Max blocks per grid (in [1, #SMs]).
tpb = 1 # Threads per block.
num_iters = 1024 # Iterations.
nbytes = 256 * 2**10 # Size of the array.
LOG_STRIDE = True # Treat min/max stride as exponents if true.
MIN_STRIDE = 1
MAX_STRIDE = 24

kernel_string = \
"""
__device__ long long int perform_stride(int* array) {
	int k = 0;
	long long int start = clock64();

	#for _ in xrange($num_iters):
	k = array[k];
	#end for

	long long int end = clock64();

	// Prevent compiler from optimizing out the loop
	if ( k < 0) {
		return -1;
	} else {
		// Do we need to worry about overflow?
		return start < end ? end-start : end + (0xffffffffffffffff - start);
	}
}

__global__ void stride_kernel(
	#for $i in xrange($bpg): 
	int* array$i, 
	#end for 
	int warm, 
	long long int* time) {

	int* array;

	#for $i in xrange($bpg):
	if (blockIdx.x == #echo $i#) {
		array = array$i;
	}
	#end for

	if (threadIdx.x == 0) {
		if (warm != 0) perform_stride(array);
		time[blockIdx.x] = perform_stride(array);
	}
}
"""

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

def init_array(nbytes, stride):
  num_elems = nbytes / 4
  array = np.arange(stride, num_elems+stride, dtype=np.int64)
  array = array % num_elems
  return np.int32(array)

def launch(kernel, bpg, array, *args, **kwargs):
  # Allocate bpg arrays on the GPU and prepend references to the provided args
  new_args = []
  for _ in xrange(bpg):
    new_args.append(gpu.to_gpu(array))
  new_args.extend(args)
  kernel(*new_args, **kwargs)

  # Free the memory held by the arrays on the GPU
  for array_d in new_args[:bpg]:
    array_d.gpudata.free()

def main():
  if len(sys.argv) != 2:
    print 'Usage: python multi_stride.py output_file_name'
    exit()

  template = Template(kernel_string)
  time_d = gpu.empty(MAX_BPG, dtype=np.int64)
  warm = np.int32(1)
  
  # Unroll the striding loop
  template.num_iters = num_iters

  # Use 48 KB for cache; reserve 10 KB shared mem to enforce one thread per SM
  cu.Context.set_cache_config(cu.func_cache.PREFER_L1)
  shared = 10 * 2**10
  
  if LOG_STRIDE:
    strides = np.array([2**x for x in xrange(MIN_STRIDE, MAX_STRIDE + 1)], dtype=np.int32)
  else:
    strides = np.arange(MIN_STRIDE, MAX_STRIDE+1, dtype=np.int32)
  with open(sys.argv[1], 'w') as f:
    for bpg in xrange(1,MAX_BPG):
      template.bpg = bpg
      stride_kernel = cuda_compile(template, 'stride_kernel')
      # Note to self: any inner loops go below here (can share some kernel compilation)
      
      for stride in strides:
        # Initialize the stride array
        array = init_array(nbytes, stride)

        # Run the kernel and record the result
        launch(stride_kernel, bpg, array, warm, time_d, block=(tpb,1,1), grid=(bpg,1), shared=shared)
        cycles = time_d.get()
        f.write('%d\t%d\t%d\t%d\t%d\t%d\n' % (nbytes, stride, tpb, bpg, cycles[0], num_iters))

      print 'Done with %d threads' % bpg

if __name__ == '__main__':
  main()


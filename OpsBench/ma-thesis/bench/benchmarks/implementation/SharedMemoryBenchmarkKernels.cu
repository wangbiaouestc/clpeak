/*
 * SharedMemoryBenchmarkKernels.cu
 *
 *  Created on: Nov 14, 2013
 *      Author: michael
 */

// bench includes
#include "util/interface/Repeat.hpp"
#include "cudatools/interface/ErrorHandler.hpp"

__device__
long long int performStride(int* array, unsigned elems) {
	int k = 0;
	const unsigned repeats = LOOP_REPEATS;
	
	extern __shared__ int sharedMem[];

	for (unsigned i = 0; i < elems; i++) {
		sharedMem[i] = array[i];
	}

	long long int start = clock64();

	for(unsigned i = 0; i < repeats; i++) {
		repeat(UNROLL_REPEATS, k = sharedMem[k]; )
	}
	
	long long int end = clock64();

	if (k < 0) {
		return -1;
	} else {
		return start < end ? end-start : end + (0xffffffffffffffff - start);
	}
}

__global__
void cudaSharedMemStride(int* array, unsigned elems, long long int* time) {
	*time = performStride(array, elems);
}

void cudaSharedMemStrideWrapper(int* array, unsigned sharedMemBytes, long long int* latency, const dim3& grid, const dim3& block) {
	cudaSharedMemStride<<<grid, block, sharedMemBytes>>>(array, sharedMemBytes / sizeof(int), latency);
}


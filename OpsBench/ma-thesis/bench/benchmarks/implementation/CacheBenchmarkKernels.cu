// bench includes
#include "util/interface/Repeat.hpp"
#include "cudatools/interface/ErrorHandler.hpp"
#include "benchmarks/interface/CacheBenchmarkKernelImpl.cu.h"
#include "benchmarks/interface/CacheBenchmark.hpp"

// uintptr_t type
#include <stdint.h>

#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__
void cudaSetCacheFuncDummy() {
	return;
}

void cudaSetCacheFuncDummyWrapper(cudaFuncCache config, const dim3& grid, const dim3& block) {
	cudaFuncSetCacheConfig(cudaSetCacheFuncDummy, config);
	cudaSetCacheFuncDummy<<<grid, block>>>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__
void cudaDeterminePointerSize(unsigned* ptrSize) {
	*ptrSize = sizeof(ptrSize);
}

void cudaDeterminePointerSizeWrapper(unsigned* ptrSize, const dim3& grid, const dim3& block) {	
	cudaDeterminePointerSize<<<grid, block>>>(ptrSize);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////// Code below from cudabmk benchmarks (see GT200 paper) ////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void cuda8BPointerChaseCudaBmkWrapper(unsigned long long** deviceArray, long long int* latency, bool doWarmup, const dim3& grid, const dim3& block) {
	cudaPointerChaseCudaBmk<<<grid, block>>>(deviceArray, latency, doWarmup);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////// Below code from C2070 microbenchmarking website ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__
long long int performStride(int* array) {
	int k = 0;
	const unsigned repeats = LOOP_REPEATS;
	
	long long int start = clock64();
	
	for (unsigned i = 0; i < repeats; i++) {
		repeat(UNROLL_REPEATS, k = array[k]; )
	}

	long long int end = clock64();

	if (k < 0) {
		return -1;
	} else {
		return start < end ? end-start : end + (0xffffffffffffffff - start);
	}
}

__global__
void cudaGlobalMemStride(int* array, bool warm, long long int* time) {
	long long int ret = 0;
	if (warm) {
		ret = performStride(array);
	}
	// Always true; prevents compiler removing first call
	if (ret >= 0) {
		*time = performStride(array);
	}
}

void cudaGlobalMemStrideWrapper(int* deviceArray, bool doWarmup, long long int* latency, const dim3& grid, const dim3& block) {
	cudaGlobalMemStride<<<grid, block>>>(deviceArray, doWarmup, latency);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__
void cudaLocalMemStrideShared(int stride, unsigned elems, long long int* time, unsigned suspectedHWStride) {
	// shared index to be passed around between threads
	__shared__ int k;
	if (threadIdx.x == 0) { k = 0; }

	// local memory array for each thread
	int localArray[THREADLOCAL_MEM_MAX_INTS];

	// precomputed shifting and masking data for each thread
	int threadIdMask = suspectedHWStride - 1;
	int indexShift = 0;
	int tmp = suspectedHWStride;
	while (tmp >>= 1) ++indexShift;

	// every thread copies the striding data into its local memory
	for (unsigned i = 0; i < elems && i < THREADLOCAL_MEM_MAX_INTS; i++) {
		localArray[i] = (i * suspectedHWStride + threadIdx.x + stride) % (elems * suspectedHWStride);
	}
	
	long long int start, end;

	__syncthreads();

	start = clock64();
	
	__syncthreads();

	for (unsigned i = 0; i < LOOP_REPEATS; i++) {
		repeat(UNROLL_REPEATS, if ((k & threadIdMask) == threadIdx.x) { k = localArray[k >> indexShift]; } __syncthreads(); )
	}

	end = clock64();

	if (k < 0) {
		if (threadIdx.x == 0) { *time = -1; }
	} else {
		if (threadIdx.x == 0) { *time =  start < end ? end-start : end + (0xffffffffffffffff - start); }
	}
}

void cudaLocalMemStrideSharedWrapper(int stride, unsigned elems, long long int* time, unsigned suspectedHWStride, const dim3& grid, const dim3& block) {
	cudaLocalMemStrideShared<<<grid, block>>>(stride, elems, time, suspectedHWStride);
}

__global__
void cudaLocalMemStride(int* array, unsigned elems, long long int* time) {
	int k = 0;
	const unsigned repeats = LOOP_REPEATS;
	
	int localArray[THREADLOCAL_MEM_MAX_INTS];	
	for (unsigned i = 0; i < elems && i < THREADLOCAL_MEM_MAX_INTS; i++) {
		localArray[i] = array[i];
	}
	
	long long int start = clock64();
	
	for (unsigned i = 0; i < repeats; i++) {
		repeat(UNROLL_REPEATS, k = localArray[k]; )
	}

	long long int end = clock64();

	if (k < 0) {
		*time = -1;
	} else {
		*time =  start < end ? end-start : end + (0xffffffffffffffff - start);
	}
}

void cudaLocalMemStrideWrapper(int* deviceArray, unsigned elems, long long int* latency, const dim3& grid, const dim3& block) {
	cudaLocalMemStride<<<grid, block>>>(deviceArray, elems, latency);
}

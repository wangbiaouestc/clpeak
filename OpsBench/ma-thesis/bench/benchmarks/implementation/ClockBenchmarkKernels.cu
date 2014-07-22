/*
 * ClockBenchmarkKernels.cu
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#include "util/interface/Repeat.hpp"

__global__
void cudaClockBenchmarkAdvanced(unsigned* d_timing, unsigned* d_dummy, unsigned a, unsigned b, unsigned iterations) {
	unsigned int t1 = a;
	unsigned int t2 = b;
	unsigned int startTime = 0, stopTime = 0;

	for (int i = 0; i < iterations; i++) {
		startTime = clock();
		repeat64(t1+=t2;t2+=t1;)
		stopTime = clock();
	}

	*d_dummy = t1+t2;
	d_timing[(blockIdx.x*blockDim.x + threadIdx.x)*2] = startTime;
	d_timing[(blockIdx.x*blockDim.x + threadIdx.x)*2+1] = stopTime;
}

void cudaClockBenchmarkAdvancedWrapper(
		unsigned* d_timing,
		unsigned* d_dummy,
		unsigned a,
		unsigned b,
		unsigned iterations,
		dim3 grid, dim3 block) {

	cudaClockBenchmarkAdvanced<<<grid, block>>>(d_timing, d_dummy, a, b, iterations);
}

__global__
void cudaClockBenchmarkNaive(unsigned* deviceMem) {

	unsigned start, end;

	start = clock();
	end = clock();

	deviceMem[0] = start;
	deviceMem[1] = end;
}

void cudaClockBenchmarkNaiveWrapper(unsigned* deviceMem, dim3 grid, dim3 block) {
	cudaClockBenchmarkNaive<<<grid, block>>>(deviceMem);
}

__global__
void cudaDetermineCycleTime(unsigned long long* elapsedCycles, long long int *dummyResult, long long int dummyA, long long int dummyB) {
	const unsigned long long waitCycles = 1 << 29;
	
	long long int start = clock64();
	while ((clock64() - start) < waitCycles) {
		dummyA += dummyB;
		dummyB += dummyA;
	}	
	long long int end = clock64();
	
	*elapsedCycles = end - start;
	*dummyResult = dummyB;
}

void cudaDetermineCycleTimeWrapper(unsigned long long* elapsedCycles, long long int* dummyResult, const dim3& grid, const dim3& block) {
	cudaDetermineCycleTime<<<grid, block>>>(elapsedCycles, dummyResult, 6, 4);
}

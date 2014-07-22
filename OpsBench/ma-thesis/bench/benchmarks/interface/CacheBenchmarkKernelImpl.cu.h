/*
 * CacheBenchmarkKernelImpl.cuh
 *
 *  Created on: Nov 8, 2013
 *      Author: michael
 */

#ifndef CUDABENCHMARKKERNEL_CUH_
#define CUDABENCHMARKKERNEL_CUH_

#include "util/interface/Repeat.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////// Code below from cudabmk benchmarks (see GT200 paper) ////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__
void cudaPointerChaseCudaBmk(unsigned long long** my_array, long long int* time, bool warmup) {

	unsigned int start_time, end_time;
	volatile unsigned long long sum_time = 0;
	unsigned long long* j = (unsigned long long*)my_array;
	const unsigned repeats = LOOP_REPEATS;

	for (unsigned k = 0; k < repeats; k++) {
		start_time = clock();
		repeat(UNROLL_REPEATS, j=*(unsigned long long**)j; )
		end_time = clock();
		sum_time += end_time - start_time;
	}

	*my_array = j;
	*time = sum_time;
}

#endif /* CUDABENCHMARKKERNEL_CUH_ */

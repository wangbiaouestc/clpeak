// bench includes
#include "benchmarks/interface/TextureBenchmark.hpp"

#include "cudatools/interface/ErrorHandler.hpp"

#include "util/interface/Repeat.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

texture<int, 1, cudaReadModeElementType> Surface;

__global__
void cudaTextureMemoryStride(long long int* latency) {
	int k = 0;
	
	long long int start = clock64();
	for (unsigned i = 0; i < LOOP_REPEATS; i++) {
		repeat(UNROLL_REPEATS, k = tex1Dfetch(Surface, k); )
	}
	long long int end = clock64();
	
	if (k < 0) {
		*latency = -1;
	} else {
		*latency =  start < end ? end-start : end + (0xffffffffffffffff - start);
	}
}

void cudaTextureMemoryStrideWrapper(int* deviceStrides, unsigned elems, long long int* deviceLatency, const dim3& grid, const dim3& block) {
	
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
	
	Surface.addressMode[0] = cudaAddressModeWrap;
	Surface.addressMode[1] = cudaAddressModeWrap;
	Surface.filterMode = cudaFilterModePoint;
	Surface.normalized = false;
	
	check( cudaBindTexture(0, Surface, deviceStrides, channelDesc, elems * sizeof(int)) );
	
	cudaTextureMemoryStride<<<grid, block>>>(deviceLatency);
}
/*
 * CacheBenchmark.h
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#ifndef CACHEBENCHMARK_H_
#define CACHEBENCHMARK_H_

// bench includes
#include "benchmarks/interface/Benchmark.hpp"
#include "cudatools/interface/ErrorHandler.hpp"
#include "util/interface/Clustering.hpp"

// uintptr_t type
#include <stdint.h>

// cuda includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// true for compute capability 2.0 and higher
#define THREADLOCAL_MEM_MAX_BYTES 96*1024
#define THREADLOCAL_MEM_MAX_INTS THREADLOCAL_MEM_MAX_BYTES/4

/////////////////////////////////////////////////////////////////////////////////////////////

void cudaSetCacheFuncDummyWrapper(cudaFuncCache config, const dim3& grid, const dim3& block);
void cudaDeterminePointerSizeWrapper(unsigned* ptrSize, const dim3& grid, const dim3& block);
// GT200 paper code
// void cuda4BPointerChaseCudaBmkWrapper(unsigned long long* deviceArray, long long int* latency, bool doWarmup, const dim3& grid, const dim3& block);
void cuda8BPointerChaseCudaBmkWrapper(unsigned long long** deviceArray, long long int* latency, bool doWarmup, const dim3& grid, const dim3& block);
// C2070 microbenchmarking code
void cudaGlobalMemStrideWrapper(int* deviceArray, bool doWarmup, long long int* latency, const dim3& grid, const dim3& block);
void cudaLocalMemStrideWrapper(int* deviceArray, unsigned elems, long long int* latency, const dim3& grid, const dim3& block);
void cudaLocalMemStrideSharedWrapper(int stride, unsigned elems, long long int* time, unsigned suspectedHWStride, const dim3& grid, const dim3& block);

/////////////////////////////////////////////////////////////////////////////////////////////

namespace bench {

	class CacheBenchmark: public Benchmark {
	public:
		CacheBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out);
		virtual ~CacheBenchmark();
	
	public:
		bool run();
	
	private:
		unsigned _findDevicePointerSize();
		void _performPointerChase(bool logStrides, bool warmup);

		void _performLocalMemHWStrideCheck(bool logStrides);
	
	private:
		unsigned m_devicePtrSize;
		cudaFuncCache m_cacheConfig;

		const unsigned m_outputIndents;
		
		const bool m_autoDetectMemoryLatencies;
		unsigned m_maxHierarchyLevels;
		util::Clustering::DataVector m_latencies;
	};
	
}

#endif /* CACHEBENCHMARK_H_ */

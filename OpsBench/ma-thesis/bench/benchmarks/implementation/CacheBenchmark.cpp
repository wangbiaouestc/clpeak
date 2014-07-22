/*
 * CacheBenchmark.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#include <cstring>
#include <vector>
#include <cmath>

#include "benchmarks/interface/CacheBenchmark.hpp"
#include "util/interface/Util.hpp"
#include "util/interface/Repeat.hpp"
#include "cudatools/interface/ErrorHandler.hpp"

#ifdef REPORT_LEVEL
#undef REPORT_LEVEL
#endif

#define REPORT_LEVEL 1
#include "util/interface/Debug.hpp"

#define CACHE_BENCH_STABILIZE_REPEATS 4

////////////////////////////////////////////////////////////////////////////////////////////////////

bench::CacheBenchmark::CacheBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out):
	Benchmark(_deviceConfig, _options, _out),
	m_devicePtrSize(0),
	m_cacheConfig(cudaFuncCachePreferNone),
	m_outputIndents(2),
	m_autoDetectMemoryLatencies(m_options.contains("--auto")) {
	
	if (m_options.contains("--cache-prefer")) {
		const std::string& cacheFuncOption = m_options.at("--cache-prefer");
		if (cacheFuncOption == "l1") {
			m_cacheConfig = cudaFuncCachePreferL1;
		} else if (cacheFuncOption == "shared") {
			m_cacheConfig = cudaFuncCachePreferShared;
		} else if (cacheFuncOption == "equal") {
			m_cacheConfig = cudaFuncCachePreferEqual;
		}
	}
	
	if (m_autoDetectMemoryLatencies) {
		m_maxHierarchyLevels = util::lexical_cast<unsigned>(m_options.at("--auto"));
	}
}

bench::CacheBenchmark::~CacheBenchmark() {

}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool bench::CacheBenchmark::run() {
	report("CacheBenchmark: Running cache benchmark ...");

	m_devicePtrSize = _findDevicePointerSize();

	bool warmup = m_options.contains("--cache-warmup");
	bool logStrides = m_options.contains("--log-stride");
	bool checkLocalMemAddressingHWStride = m_options.contains("--check-hw-stride");
	
	if (checkLocalMemAddressingHWStride) {
		_performLocalMemHWStrideCheck(logStrides);
	} else {
		_performPointerChase(logStrides, warmup);
	}

	if (m_autoDetectMemoryLatencies) {
		util::Clustering latencyClustering(m_latencies, m_maxHierarchyLevels);
		const util::Clustering::ClusterResult& result = latencyClustering.getResult();
		for (auto it = result.begin(); it != result.end(); ++it) {
			report("CacheBenchmark: Detected memory hierarchy level @ " << (*it).position << " clocks");
		}
	}
	
	m_done = true;
	return true;
}

//! \brief determines the device pointer size on the current machine
unsigned bench::CacheBenchmark::_findDevicePointerSize() {
	report("CacheBenchmark: Determining pointer size ...");

	unsigned* d_ptrSize;
	unsigned h_ptrSize;
	check( cudaMalloc((void**)&d_ptrSize, sizeof(unsigned)) );

	dim3 grid(1,1,1), block(1,1,1);
	cudaDeterminePointerSizeWrapper(d_ptrSize, grid, block);
	check( cudaDeviceSynchronize() );

	check( cudaMemcpy(&h_ptrSize, d_ptrSize, sizeof(unsigned), cudaMemcpyDeviceToHost) );
	check( cudaFree(d_ptrSize) );

	report(util::Indents(2) << "pointer size is " << h_ptrSize << " bytes");

	return h_ptrSize;
}

static std::string toString(cudaFuncCache func) {
	if (func == cudaFuncCachePreferL1) {
		return "PreferL1";
	} else if (func == cudaFuncCachePreferEqual) {
		return "PreferEqual";
	} else if (func == cudaFuncCachePreferShared) {
		return "PreferShared";
	} else if (func == cudaFuncCachePreferNone) {
		return "None";
	} else {
		return "unknown";
	}
}

//! @brief Generalized strided pointer chase benchmark, encapsulates both GT200 and C2070 microbenchmarking codes
void bench::CacheBenchmark::_performPointerChase(bool logStrides, bool warmup) {
	report("CacheBenchmark: Strided memory access " << (warmup ? "with" : "without") << " cache warmup and various array sizes");
	
	const bool useGlobalMemory = !m_options.contains("--local");
	const bool useAddressedArray = m_options.contains("--addressed");
	report("CacheBenchmark: Using " << (useGlobalMemory ? "global" : "local") << " memory space");
	report("CacheBenchmark: Using " << (useAddressedArray ? "address" : "index") << " stride array");
	if (!useGlobalMemory && useAddressedArray) {
		report("CacheBenchmark: Warning: Addressed array impossible in local memory, selecting indexed array");
	}

	report("CacheBenchmark: Setting cache config to " << toString(m_cacheConfig));
	cudaSetCacheFuncDummyWrapper(m_cacheConfig, dim3(1,1,1), dim3(1,1,1));
	
	typedef std::vector< unsigned > SizeVector;
	typedef std::vector< unsigned > StrideVector;

	// Fermi caches: L1 16, 48K, L2 256K, Kepler caches: L1 16, 32, 48K, L2 512K
 	const unsigned globalArraySizes[] = {15*1024, 16384, 17*1024,
 			28*1024, 33*1024,
 			42*1024, 49*1024,
 			230*1024, 260*1024,
 			500*1024, 514*1024,
 			1024*1024, 2048*1024, 8192*1024,
			16*1024*1024, 32*1024*1024};
	const unsigned localArraySizes[] = {16, 64, 512, 2048,
			15*1024, 16384, 17*1024,
			28*1024, 32*1024, 33*1024,
			42*1024, 49*1024, 64*1024};

	// fill the array size vector
	SizeVector sizes;
	if (useGlobalMemory) {
		sizes.insert(sizes.begin(), globalArraySizes, globalArraySizes + sizeof(globalArraySizes)/sizeof(globalArraySizes[0]));
	} else {
		sizes.insert(sizes.begin(), localArraySizes, localArraySizes + sizeof(localArraySizes)/sizeof(localArraySizes[0]));
	}
		
	// fill the stride length vector
	const unsigned minStride = 1, maxStride = 26;
	StrideVector strides;
	if (logStrides) {
		strides.push_back(1);
		for (unsigned stride = minStride; stride <= maxStride; stride++) {
			strides.push_back(pow(2, stride));
		}
	} else {
		for (unsigned stride = minStride; stride <= maxStride; stride++) {
			strides.push_back(stride);
		}
	}
	
	const dim3 grid(1,1,1), block(1,1,1);
	
	const unsigned repeats = CACHE_BENCH_STABILIZE_REPEATS;
		
	 /* perform strided accesses through memory */
	for (auto arr_it = sizes.begin(); arr_it != sizes.end(); ++arr_it) {
		for (auto stride_it = strides.begin(); stride_it != strides.end(); ++stride_it) {
			long long int averageLatency = 0;
			unsigned size = *arr_it, stride = *stride_it;

			// if using local memory, can only have a limited number of memory per thread
			if (!useGlobalMemory && size > THREADLOCAL_MEM_MAX_BYTES) {
					continue;
			}

			for (unsigned iteration = 0; iteration < repeats; iteration++) {
				util::StridedHostArray hostArray;
				void* deviceArray;

				long long int latency;
				long long int* deviceLatency;

				// allocate memory on the device
				check( cudaMalloc((void**)&deviceArray, size) );
				check( cudaMalloc((void**)&deviceLatency, sizeof(long long int)) );

				// allocate strided array on the host, either 4B or 8B
				if (useAddressedArray) {
					hostArray.create(size, stride, (unsigned long long)(void*)deviceArray);
				} else {
					hostArray.create(size, stride);
				}

				check( cudaMemcpy(deviceArray, hostArray.getDataPtr(), size, cudaMemcpyHostToDevice) );
				
				// execute either index or pointer chase in gmem or index chase in lmem
				if (useGlobalMemory) {
					if (useAddressedArray) {
						cuda8BPointerChaseCudaBmkWrapper((unsigned long long**)deviceArray, deviceLatency, warmup, grid, block);
					} else {
						cudaGlobalMemStrideWrapper((int*)deviceArray, warmup, deviceLatency, grid, block);
					}
				} else {
					cudaLocalMemStrideWrapper((int*)deviceArray, size/sizeof(int), deviceLatency, grid, block);
				}
				check( cudaDeviceSynchronize() );
				
				check( cudaMemcpy(&latency, deviceLatency, sizeof(long long int), cudaMemcpyDeviceToHost) );

				check( cudaFree(deviceArray) );
				check( cudaFree(deviceLatency) );

				averageLatency += latency;
			}
			
			stride *= (useAddressedArray ? sizeof(unsigned long long) : sizeof(int));
			m_out << size << "," << stride << "," << 1 << "," << 1 << "," << (double)averageLatency/(double)repeats << "," << UNROLL_REPEATS*LOOP_REPEATS << std::endl;
			
			if (m_autoDetectMemoryLatencies) {
				m_latencies.push_back((double)averageLatency / (double)repeats / UNROLL_REPEATS / LOOP_REPEATS);
			}
		}
	}
}

void bench::CacheBenchmark::_performLocalMemHWStrideCheck(bool logStrides) {
	report("CacheBenchmark: Strided memory access through local memory to determine hardware striding");

	report("CacheBenchmark: Setting cache config to " << toString(m_cacheConfig));
	cudaSetCacheFuncDummyWrapper(m_cacheConfig, dim3(1,1,1), dim3(1,1,1));

	typedef std::vector< unsigned > SizeVector;
	typedef std::vector< unsigned > StrideVector;

	const unsigned localArraySizes[] = {16, 64, 512, 2048,
			15*1024, 16384, 17*1024,
			28*1024, 32*1024, 33*1024,
			42*1024, 49*1024};

	// fill the array size vector
	SizeVector sizes;
	sizes.insert(sizes.begin(), localArraySizes, localArraySizes + sizeof(localArraySizes)/sizeof(localArraySizes[0]));

	// fill the stride length vector
	const unsigned minStride = 1, maxStride = 26;
	StrideVector strides;
	if (logStrides) {
		strides.push_back(1);
		for (unsigned stride = minStride; stride <= maxStride; stride++) {
			strides.push_back(pow(2, stride));
		}
	} else {
		for (unsigned stride = minStride; stride <= maxStride; stride++) {
			strides.push_back(stride);
		}
	}

	const unsigned repeats = CACHE_BENCH_STABILIZE_REPEATS;
	const unsigned suspectedHWStride = util::lexical_cast<unsigned>(m_options.at("--check-hw-stride"));

	const dim3 grid(1,1,1), block(suspectedHWStride, 1, 1);	
	
	 /* perform strided accesses through memory */
	for (auto arr_it = sizes.begin(); arr_it != sizes.end(); ++arr_it) {
		for (auto stride_it = strides.begin(); stride_it != strides.end(); ++stride_it) {
			long long int averageLatency = 0;
			unsigned size = *arr_it, stride = *stride_it;

			for (unsigned iteration = 0; iteration < repeats; iteration++) {
				long long int latency;
				long long int* deviceLatency;

				// allocate memory on the device
				check( cudaMalloc((void**)&deviceLatency, sizeof(long long int)) );

				// execute either index or pointer chase in gmem or index chase in lmem
				cudaLocalMemStrideSharedWrapper(stride, size/sizeof(int), deviceLatency, suspectedHWStride, grid, block);
				check( cudaDeviceSynchronize() );

				check( cudaMemcpy(&latency, deviceLatency, sizeof(long long int), cudaMemcpyDeviceToHost) );

				check( cudaFree(deviceLatency) );

				averageLatency += latency;
			}

			m_out << size << "," << stride * sizeof(int) << "," << 1 << "," << 1 << "," << (double)averageLatency/(double)repeats << "," << UNROLL_REPEATS*LOOP_REPEATS << std::endl;
			
			if (m_autoDetectMemoryLatencies) {
				m_latencies.push_back((double)averageLatency / (double)repeats / UNROLL_REPEATS / LOOP_REPEATS);
			}
		}
	}
}

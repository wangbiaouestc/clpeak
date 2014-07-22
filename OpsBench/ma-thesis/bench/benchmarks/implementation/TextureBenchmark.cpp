/*
 * TextureBenchmark.cpp
 *
 *  Created on: Nov 27, 2013
 *      Author: michael
 */

// C++ includes
#include <vector>
#include <cmath>

// bench includes
#include <benchmarks/interface/TextureBenchmark.hpp>
#include <util/interface/Util.hpp>
#include <util/interface/Repeat.hpp>
#include <cudatools/interface/ErrorHandler.hpp>

#ifdef REPORT_LEVEL
#undef REPORT_LEVEL
#endif

#define REPORT_LEVEL 1
#include "util/interface/Debug.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

bench::TextureBenchmark::TextureBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out):
	Benchmark(_deviceConfig, _options, _out), m_outputIndents(2) {
	
}

bench::TextureBenchmark::~TextureBenchmark() {

}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool bench::TextureBenchmark::run() {
	report("TextureBenchmark: Running texture memory benchmark ...");

	bool logStride = m_options.contains("--log-stride");
	_performTextureMemoryStride(logStride);
	
	m_done = true;
	return true;
}

//! @brief performs strided memory accesses through 1-dimensional read-only cached texture memory
void bench::TextureBenchmark::_performTextureMemoryStride(bool logStride) {
	report("TextureBenchmark: Texture memory accesses with varying strides and array sizes");
	
	typedef std::vector< unsigned > SizeVector;
 	const unsigned arraySizes[] = {15*1024, 16384, 17*1024,
 			28*1024, 33*1024,
 			42*1024, 49*1024,
 			230*1024, 260*1024,
 			500*1024, 514*1024,
 			1024*1024, 2048*1024, 8192*1024,
			16*1024*1024, 32*1024*1024};
	SizeVector sizes(arraySizes, arraySizes + sizeof(arraySizes)/sizeof(arraySizes[0]));
		
	typedef std::vector< unsigned > StrideVector;
	const unsigned minStride = 1, maxStride = 26;
	StrideVector strides;
	
	if (logStride) {
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
	
	const unsigned repeats = 4;
		
	for (auto arr_it = sizes.begin(); arr_it != sizes.end(); ++arr_it) {
		for (auto stride_it = strides.begin(); stride_it != strides.end(); ++stride_it) {
			long long int averageLatency = 0;
			unsigned size = *arr_it, stride = *stride_it;

			for (unsigned iteration = 0; iteration < repeats; iteration++) {
				int* hostArray;
				int* deviceArray;
				long long int latency;
				long long int* deviceLatency;
				
				hostArray = util::StridedHostArray::createIndexed(size, stride);
				
				check( cudaMalloc((void**)&deviceArray, size) );
				check( cudaMalloc((void**)&deviceLatency, sizeof(long long int)) );

				check( cudaMemcpy(deviceArray, hostArray, size, cudaMemcpyHostToDevice) );
				
				cudaTextureMemoryStrideWrapper(deviceArray, size/sizeof(int), deviceLatency, grid, block);
				check( cudaDeviceSynchronize() );
				
				check( cudaMemcpy(&latency, deviceLatency, sizeof(long long int), cudaMemcpyDeviceToHost) );

				check( cudaFree(deviceArray) );
				check( cudaFree(deviceLatency) );
				delete [] hostArray;
				
				averageLatency += latency;
			}
			
			m_out << size << "," << stride * sizeof(int) << "," << 1 << "," << 1 << "," << (double)averageLatency/(double)repeats << "," << UNROLL_REPEATS*LOOP_REPEATS << std::endl;
		}
	}
}


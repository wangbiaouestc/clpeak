/*
 * TextureBenchmark.h
 *
 *  Created on: Nov 27, 2013
 *      Author: michael
 */

#ifndef TEXTUREBENCHMARK_H_
#define TEXTUREBENCHMARK_H_

// bench includes
#include "benchmarks/interface/Benchmark.hpp"
#include "cudatools/interface/ErrorHandler.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////

void cudaTextureMemoryStrideWrapper(int* deviceStrides, unsigned elems, long long int* deviceLatency, const dim3& grid, const dim3& block);

/////////////////////////////////////////////////////////////////////////////////////////////

namespace bench {

	class TextureBenchmark: public Benchmark {
	public:
		TextureBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out);
		virtual ~TextureBenchmark();
	
	public:
		bool run();
	
	private:
		void _performTextureMemoryStride(bool logStride);
		
	private:
		const unsigned m_outputIndents;
	};

}

#endif /* CACHEBENCHMARK_H_ */

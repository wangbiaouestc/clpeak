/*
 * SharedMemoryBenchmark.h
 *
 *  Created on: Nov 14, 2013
 *      Author: michael
 */

#ifndef SHAREDMEMORYBENCHMARK_H_
#define SHAREDMEMORYBENCHMARK_H_

#include "Benchmark.hpp"

void cudaSharedMemStrideWrapper(int* array, unsigned elems, long long int* latency, const dim3& grid, const dim3& block);

namespace bench {

	class SharedMemoryBenchmark: public Benchmark {
	public:
		SharedMemoryBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out);
		virtual ~SharedMemoryBenchmark();

	public:
		bool run();

	protected:
		void _performPointerChase(bool logStride);
	};

} /* namespace bench */
#endif /* SHAREDMEMORYBENCHMARK_H_ */

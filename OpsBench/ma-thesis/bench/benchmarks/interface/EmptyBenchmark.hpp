/*
 * EmptyBenchmark.h
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#ifndef EMPTYBENCHMARK_H_
#define EMPTYBENCHMARK_H_

#include "benchmarks/interface/Benchmark.hpp"

namespace bench {

	class EmptyBenchmark: public Benchmark {
	public:
		EmptyBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out);
		virtual ~EmptyBenchmark();

	public:
		bool run();
	};

}

#endif /* EMPTYBENCHMARK_H_ */

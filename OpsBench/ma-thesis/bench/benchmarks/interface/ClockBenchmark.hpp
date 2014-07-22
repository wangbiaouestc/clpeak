/*
 * ClockBenchmark.hpp
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#ifndef CLOCKBENCHMARK_HPP_
#define CLOCKBENCHMARK_HPP_

#include "benchmarks/interface/Benchmark.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////

void cudaClockBenchmarkNaiveWrapper(unsigned* deviceMem, dim3 grid, dim3 block);
void cudaClockBenchmarkAdvancedWrapper(
		unsigned* d_timing,
		unsigned* d_dummy,
		unsigned a,
		unsigned b,
		unsigned iterations,
		dim3 grid, dim3 block);
void cudaDetermineCycleTimeWrapper(
	unsigned long long* elapsedCycles,
	long long int* dummyResult,
	const dim3& grid,
	const dim3& block);

/////////////////////////////////////////////////////////////////////////////////////////////

namespace bench {

	class ClockBenchmark : public Benchmark {
	public:
		ClockBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out);
		virtual ~ClockBenchmark();

	public:
		bool run();
		float getCycleTime();
		
	private:
		float _determineCycleTime();
		void _singleThreadClockRegisterTest();
		void _multiThreadClockRegisterTest();

	private:
		const unsigned m_memSize;
		unsigned* m_hostMemory;
		unsigned* m_deviceMemory;
		
		float m_nsPerClockCycle;
	};

}

#endif /* CLOCKBENCHMARK_HPP_ */

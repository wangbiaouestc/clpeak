/*
 * ClockBenchmark.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#include "benchmarks/interface/ClockBenchmark.hpp"
#include "util/interface/Util.hpp"
#include "cudatools/interface/ErrorHandler.hpp"

#ifdef REPORT_LEVEL
#undef REPORT_LEVEL
#endif

#define REPORT_LEVEL 1
#include "util/interface/Debug.hpp"

bench::ClockBenchmark::ClockBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out):
	Benchmark(_deviceConfig, _options, _out), m_memSize(6 * sizeof(unsigned)) {

	m_hostMemory = new unsigned[6];
	check( cudaMalloc((void**)&m_deviceMemory, m_memSize) );
	check( cudaMemset(m_deviceMemory, 0, m_memSize) );
}

bench::ClockBenchmark::~ClockBenchmark() {
	delete [] m_hostMemory;
	check( cudaFree(m_deviceMemory) );
}

bool bench::ClockBenchmark::run() {
	report("Running naive clock benchmark ...");

	_singleThreadClockRegisterTest();

	report("Running cudabmk multi-block clock benchmark ...");

	_multiThreadClockRegisterTest();
	
	report("Determining hardware cycle time ...");
	
	m_nsPerClockCycle = _determineCycleTime();

	report(util::Indents(2) << "detected " << m_nsPerClockCycle << "ns cycle time");
	report(util::Indents(2) << "detected " << 1.0/m_nsPerClockCycle << "GHz clock rate");
	
	m_done = true;
	
	return true;
}

float bench::ClockBenchmark::getCycleTime() {
	if (m_done)
		return m_nsPerClockCycle;
	return 0.0;
}

float bench::ClockBenchmark::_determineCycleTime() {
	cudaEvent_t start, end;
	
	check( cudaEventCreate(&start) );
	check( cudaEventCreate(&end) );
	
	unsigned long long elapsedCycles;
	unsigned long long* deviceElapsedCycles;
	long long int* deviceDummyMem;
	const dim3 grid(1,1,1), block(1,1,1);
	
	check( cudaMalloc((void**)&deviceElapsedCycles, sizeof(unsigned long long)) );
	check( cudaMalloc((void**)&deviceDummyMem, sizeof(long long int)) );
	
	check( cudaEventRecord(start) );
	cudaDetermineCycleTimeWrapper(deviceElapsedCycles, deviceDummyMem, grid, block);
	check( cudaEventRecord(end) );
	
	check( cudaDeviceSynchronize() );
	
	check( cudaMemcpy(&elapsedCycles, deviceElapsedCycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
	
	float elapsedTime = 0;
	check( cudaEventElapsedTime(&elapsedTime, start, end) );
	
	report(util::Indents(2) << "elapsed time: " << elapsedTime << "ms");
	report(util::Indents(2) << "elapsed cycles: " << elapsedCycles);
	
	return elapsedTime * 1000000.0 / (float)elapsedCycles;
}

void bench::ClockBenchmark::_singleThreadClockRegisterTest() {
	dim3 grid(1,1,1);
	dim3 block(1,1,1);

	cudaClockBenchmarkNaiveWrapper(m_deviceMemory, grid, block);

	check( cudaDeviceSynchronize() );
	check( cudaMemcpy(m_hostMemory, m_deviceMemory, m_memSize, cudaMemcpyDeviceToHost) );

	report(util::Indents(2) << "Start time : " << m_hostMemory[0]);
	report(util::Indents(2) << "End time   : " << m_hostMemory[1]);
}

void bench::ClockBenchmark::_multiThreadClockRegisterTest() {
	unsigned int ts[1024];
	unsigned int *d_ts;
	unsigned int *d_out;
	
	dim3 grid(1,1,1), block(1,1,1);

	check( cudaMalloc((void**)&d_ts, sizeof(ts)) );
	check( cudaMalloc((void**)&d_out, 4) );

	grid.x = m_deviceConfig.m_deviceProps.multiProcessorCount / 3 + 1;

	std::cout << std::endl << "advanced clock test: [" << grid.x << " blocks, " << block.x << " thread(s)/block]";
	cudaClockBenchmarkAdvancedWrapper(d_ts, d_out, 4, 6, 2, grid, block);
	std::cout << std::endl;

	check( cudaDeviceSynchronize() );
	check( cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost) );
	for (unsigned i = 0; i < grid.x; i++) {
		std::cout << "  Block " << i << ": start: " << ts[i*2] << ", stop: " << ts[i*2+1] << std::endl;
	}

	grid.x = m_deviceConfig.m_deviceProps.multiProcessorCount;
	check( cudaMemset(d_ts, 0, sizeof(ts)) );

	std::cout << std::endl << "advanced clock test: [" << grid.x << " blocks, " << block.x << " thread(s)/block]";
	cudaClockBenchmarkAdvancedWrapper(d_ts, d_out, 4, 6, 2, grid, block);
	std::cout << std::endl;

	check( cudaDeviceSynchronize() );
	check( cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost) );
	for (unsigned i = 0; i < grid.x; i++) {
		std::cout << "  Block " << i << ": start: " << ts[i*2] << ", stop: " << ts[i*2+1] << std::endl;
	}
	std::cout << std::endl;

	check( cudaFree(d_ts) );
	check( cudaFree(d_out) );
}

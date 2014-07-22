/*!
 * \file: bench.cpp
 *
 * \author: Michael Andersch
 *
 * \date: Nov 05, 2013
 *
 * \description Main file for the bench benchmarking infrastructure
 *
 */

// C++ includes
#include <iostream>
#include <fstream>

// bench includes
#include <benchmarks/interface/EmptyBenchmark.hpp>
#include <benchmarks/interface/ClockBenchmark.hpp>
#include <benchmarks/interface/CacheBenchmark.hpp>
#include <benchmarks/interface/SharedMemoryBenchmark.hpp>
#include <benchmarks/interface/InstructionBenchmark.hpp>
#include <benchmarks/interface/TextureBenchmark.hpp>

#include <util/interface/Util.hpp>
#include <util/interface/Options.hpp>

#include <cudatools/interface/DeviceConfigurations.hpp>

#ifdef REPORT_LEVEL
#undef REPORT_LEVEL
#endif

#define REPORT_LEVEL 1
#include <util/interface/Debug.hpp>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

class Bench {
public:
	Bench(int argc, char** argv) {
		_parseArgs(argc, argv);
		_determineDeviceProps();
	}

	~Bench() { }

public:
	int run() {
		
		if (m_options.contains("--help") || m_options.contains("-h")) {
			_usage();
			return 0;
		}
		
		std::ofstream out;
		bool fileOutput = false;
		if (m_options.contains("-f")) {
			out.open(m_options["-f"]);
			if (!out.good())
				error("Cannot open output file");
			fileOutput = true;
		}
		out << std::fixed;
		
		if (m_options.contains("--clock")) {
			bench::ClockBenchmark benchmark(m_deviceConfig, m_options, std::cout);
			if (!benchmark.run()) 
				std::cout << "Clock benchmark run failed" << std::endl;
			m_deviceConfig.m_cycleTime = benchmark.getCycleTime();
			m_deviceConfig.m_clockRate = 1.0 / m_deviceConfig.m_cycleTime;
		}
		
		if (m_options.contains("--cache")) {
			bench::CacheBenchmark benchmark(m_deviceConfig, m_options, fileOutput ? out : std::cout);
			if (!benchmark.run())
				std::cout << "Cache benchmark run failed" << std::endl;
		}

		if (m_options.contains("--shared")) {
			bench::SharedMemoryBenchmark benchmark(m_deviceConfig, m_options, fileOutput ? out : std::cout);
			if (!benchmark.run())
				std::cout << "Shared memory benchmark run failed" << std::endl;
		}

		if (m_options.contains("--instruction")) {
			bench::InstructionBenchmark benchmark(m_deviceConfig, m_options, fileOutput ? out : std::cout);
			if (!benchmark.run())
				std::cout << "Instruction latency benchmark run failed" << std::endl;
		}
		
		if (m_options.contains("--texture")) {
			bench::TextureBenchmark benchmark(m_deviceConfig, m_options, fileOutput ? out : std::cout);
			if (!benchmark.run())
				std::cout << "Texture memory benchmark run failed" << std::endl;
		}
		
		return 0;
	}

protected:
	void _usage() {
		std::cout << "Usage: ./bench <options>\n"
			<< "Options:\n"
			<< "-h, --help                                 print this help text\n"
			<< "--clock                                    benchmark clock and clock register\n"
			<< "--cache                                    benchmark cache hierarchy\n"
			<< "--shared                                   benchmark shared memory\n"
			<< "--instruction                              benchmark instruction latencies\n"
			<< "\n"
			<< "-f <filename>                              print output to file instead of console\n"
			<< "--cache-warmup                             cache benchmark: perform warmup iteration\n"
			<< "--log-stride                               cache benchmark: stride logarithmically, not linearly\n"
			<< "--cache-prefer <l1, shared, equal>         cache benchmark: set cudaPreferCacheFunc to given value\n"
			<< "--local                                    cache benchmark: stride through local memory, not global memory\n"
			<< "--addressed                                cache benchmark: create an array with addresses, not indices\n"
			<< "--check-hw-stride <hwstride>               cache benchmark: stride through local memory with given active threads\n"
			<< "--auto <levels>                            cache, shared benchmarks: automatically find the latency levels in the result\n"
			<< "\n";
	}
	
	void _parseArgs(int argc, char** argv) {		
		m_options.parse(argc, argv);
		
		std::cout << "options:" << std::endl;
		for (auto it = m_options.begin(); it != m_options.end(); ++it) {
			std::cout << util::Indents(2) << it->first << ": " << it->second << std::endl;
		}
		std::cout << std::endl;
	}

	void _determineDeviceProps() {
		cudaGetDevice(&m_deviceConfig.m_deviceId);
		cudaGetDeviceProperties(&m_deviceConfig.m_deviceProps, m_deviceConfig.m_deviceId);

		report("Running on device \"" << m_deviceConfig.m_deviceProps.name << "\"");
		report("Compute       : " << m_deviceConfig.m_deviceProps.major << "." << m_deviceConfig.m_deviceProps.minor);
		report("Core clock    : " << (float)m_deviceConfig.m_deviceProps.clockRate / 1e3 << " MHz");
		report("Mem clock     : " << (float)m_deviceConfig.m_deviceProps.memoryClockRate / 1e3 << " MHz");
		report("SM count      : " << m_deviceConfig.m_deviceProps.multiProcessorCount);
		report("L2$ size      : " << (float)m_deviceConfig.m_deviceProps.l2CacheSize / 1024 << " KByte");
		report("Max grid dim  : " << "(" << m_deviceConfig.m_deviceProps.maxGridSize[0] << "," << m_deviceConfig.m_deviceProps.maxGridSize[1] << "," << m_deviceConfig.m_deviceProps.maxGridSize[2] << ")");
		report("Max block dim : " << "(" << m_deviceConfig.m_deviceProps.maxThreadsDim[0] << "," << m_deviceConfig.m_deviceProps.maxThreadsDim[1] << "," << m_deviceConfig.m_deviceProps.maxThreadsDim[2] << ")");
		report("Max tpb       : " << m_deviceConfig.m_deviceProps.maxThreadsPerBlock << std::endl);
	}

protected:
	cuda::DeviceConfiguration m_deviceConfig;
	
	ProgramOptions m_options;
};

int main(int argc, char** argv) {
	Bench bench(argc, argv);
	return bench.run();
}

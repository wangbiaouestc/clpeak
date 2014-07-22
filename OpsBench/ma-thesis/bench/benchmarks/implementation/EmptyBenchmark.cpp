/*
 * EmptyBenchmark.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#include "benchmarks/interface/EmptyBenchmark.hpp"

bench::EmptyBenchmark::EmptyBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out):
Benchmark(_deviceConfig, _options, _out) {

}

bench::EmptyBenchmark::~EmptyBenchmark() {
	
}

bool bench::EmptyBenchmark::run() {
	return true;
}

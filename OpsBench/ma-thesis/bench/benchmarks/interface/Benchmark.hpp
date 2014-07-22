/*
 * Benchmark.hpp
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_

// C++ includes
#include <iostream>

// bench includes
#include <util/interface/Util.hpp>
#include <util/interface/Options.hpp>
#include <util/interface/Clustering.hpp>

#include <cudatools/interface/DeviceConfigurations.hpp>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace bench {

	class Benchmark {
	public:
		class Shmoo {
		public:
			unsigned start, step, end;
		};
	
	public:
		Benchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out):
			m_deviceConfig(_deviceConfig), m_options(_options), m_out(_out), m_done(false) { }
		virtual ~Benchmark();
	
	public:
		virtual bool run() = 0;
	
	protected:
		bool m_done;
		
		Shmoo m_shmoo;

		const cuda::DeviceConfiguration& m_deviceConfig;

		const ProgramOptions& m_options;

		std::ostream& m_out;
	};

}

#endif /* BENCHMARK_HPP_ */

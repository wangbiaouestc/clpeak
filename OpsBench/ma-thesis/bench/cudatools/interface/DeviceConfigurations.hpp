/*
 * DeviceConfigurations.hpp
 *
 *  Created on: Nov 7, 2013
 *      Author: michael
 */

#ifndef DEVICECONFIGURATIONS_HPP_
#define DEVICECONFIGURATIONS_HPP_

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace cuda {
	class DeviceConfiguration {
	public:
		DeviceConfiguration() { }
		~DeviceConfiguration() { }
		
	public:
		int m_deviceId;
		cudaDeviceProp m_deviceProps;
		float m_clockRate;
		float m_cycleTime;
	};
}

#endif /* DEVICECONFIGURATIONS_HPP_ */

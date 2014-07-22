/*
 * ErrorHandler.hpp
 *
 *  Created on: Nov 7, 2013
 *      Author: michael
 */

#ifndef ERRORHANDLER_HPP_
#define ERRORHANDLER_HPP_

#include <iostream>

#define check(expr) { cudaError_t err = expr; \
	if (err != cudaSuccess) { \
		std::cerr << "Error at " << __FILE__ << ", " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
	} \
	}

#endif /* ERRORHANDLER_HPP_ */

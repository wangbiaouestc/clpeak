/*
 * Util.hpp
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <iostream>
#include <sstream>

namespace util {

	class Indents {
	public:
		Indents(unsigned char _indents) : indents(_indents) { }

	public:
		unsigned char indents;
	};

	class StridedHostArray {
	public:
		StridedHostArray();
		~StridedHostArray();

	public:
		void create(unsigned bytes, unsigned stride);
		void create(unsigned bytes, unsigned stride, unsigned long long base);

		void* getDataPtr();

	public:
		static int* createIndexed(unsigned bytes, unsigned stride);
		static unsigned long long* createAddressed(unsigned bytes, unsigned stride, unsigned long long base);

	private:
		void _create(unsigned bytes, unsigned stride, unsigned long long base);

	private:
		int* data4B;
		unsigned long long* data8B;
	};

	template<typename T, typename U>
	static T lexical_cast(U in) {
		std::stringstream ss;
		ss << in;
		T out;
		ss >> out;
		return out;
	}

}

std::ostream& operator<<(std::ostream& out, const util::Indents& ind);

#endif /* UTIL_HPP_ */

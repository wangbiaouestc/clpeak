/*
 * Util.cpp
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#include "util/interface/Util.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& out, const util::Indents& ind) {
	for (unsigned i = 0; i < ind.indents; i++) {
		out << " ";
	}
	return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

util::StridedHostArray::StridedHostArray() : data4B(NULL), data8B(NULL) {

}

util::StridedHostArray::~StridedHostArray() {
	if (data4B) {
		delete [] data4B;
		data4B = NULL;
	}

	if (data8B) {
		delete [] data8B;
		data8B = NULL;
	}
}

void util::StridedHostArray::create(unsigned bytes, unsigned stride) {
	_create(bytes, stride, 0);
}

void util::StridedHostArray::create(unsigned bytes, unsigned stride, unsigned long long base) {
	_create(bytes, stride, base);
}

void util::StridedHostArray::_create(unsigned bytes, unsigned stride, unsigned long long base) {
	if (base) {
		unsigned elems = bytes / sizeof(unsigned long long);
		data8B = new unsigned long long[elems];
		for (int i = 0; i < elems; i++) {
			data8B[i] = base + ((i + stride) % elems) * sizeof(unsigned long long);
		}
	} else {
		unsigned elems = bytes / sizeof(int);
		data4B = new int[elems];
		for (int i = 0; i < elems; i++) {
			data4B[i] = (i + stride) % elems;
		}
	}
}

void* util::StridedHostArray::getDataPtr() {
	if (data4B)
		return data4B;
	if (data8B)
		return data8B;
	return NULL;
}

int* util::StridedHostArray::createIndexed(unsigned bytes, unsigned stride) {
	unsigned elems = bytes / sizeof(int);

	int * hostArray = new int[elems];
	for (int i = 0; i < elems; i++) {
		hostArray[i] = (i + stride) % elems;
	}

	return hostArray;
}

unsigned long long* util::StridedHostArray::createAddressed(unsigned bytes, unsigned stride, unsigned long long base) {
	unsigned elems = bytes / sizeof(unsigned long long);

	unsigned long long* hostArray = new unsigned long long[elems];
	for (int i = 0; i < elems; i++) {
		hostArray[i] = base + ((i + stride) % elems) * sizeof(unsigned long long);
	}

	return hostArray;
}


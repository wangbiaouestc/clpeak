/*
 * ClusterTest.cpp
 * 
 *  Created on: Dec 1, 2013
 *      Author: michael
 */

#include <iostream>

#include <util/interface/Clustering.hpp>

using namespace util;

int main(int argc, char** argv) {
	
	Clustering::DataVector data;
	
	std::cout << "TEST1: Expect 1 cluster at 3.4" << std::endl;
	{
	data.clear();
	data.push_back(3.3);
	data.push_back(3.4);
	data.push_back(3.5);
	
	Clustering clustering1(data, 1);
	Clustering clustering2(data, 3);
	}
	
	std::cout << "TEST2: Expect 2 clusters at 3.4 and 750.0" << std::endl;
	{
	data.clear();
	data.push_back(3.3);
	data.push_back(3.4);
	data.push_back(3.5);
	data.push_back(725.0);
	data.push_back(750.0);
	data.push_back(775.0);
	
	Clustering clustering1(data, 2);
	Clustering clustering2(data, 1);
	}
	
	std::cout << "TEST3: Expect 4 clusters (3.4, 50, 300, 750)" << std::endl;
	{
	data.clear();
	data.push_back(3.3);
	data.push_back(3.4);
	data.push_back(3.5);
	data.push_back(48);
	data.push_back(48);
	data.push_back(50);
	data.push_back(54);
	data.push_back(296);
	data.push_back(304);
	data.push_back(725.0);
	data.push_back(750.0);
	data.push_back(775.0);
	
	Clustering clustering2(data, 5);
	}
	
	
	return 0;
} 

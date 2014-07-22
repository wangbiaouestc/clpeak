#include <limits>
#include <iostream>

int main(int argc, char** argv) {
	std::cout << "limits for various data types" << std::endl << std::endl;

	std::cout << "char / unsigned char" << std::endl;
	std::cout << "mins: " << (int)std::numeric_limits<char>::min();
	std::cout << ", " << (int)std::numeric_limits<unsigned char>::min();
	std::cout << " - maxs: " << (int)std::numeric_limits<char>::max();
	std::cout << ", " << (int)std::numeric_limits<unsigned char>::max() << std::endl << std::endl;

	std::cout << "short / unsigned short" << std::endl;
	std::cout << "mins: " << std::numeric_limits<short>::min();
	std::cout << ", " << std::numeric_limits<unsigned short>::min();
	std::cout << " - maxs: " << std::numeric_limits<short>::max();
	std::cout << ", " << std::numeric_limits<unsigned short>::max() << std::endl << std::endl;

	std::cout << "int / unsigned" << std::endl;
	std::cout << "mins: " << std::numeric_limits<int>::min();
	std::cout << ", " << std::numeric_limits<unsigned>::min();
	std::cout << " - maxs: " << std::numeric_limits<int>::max();
	std::cout << ", " << std::numeric_limits<unsigned>::max() << std::endl << std::endl;

	std::cout << "long long int / unsigned long long" << std::endl;
	std::cout << "mins: " << std::numeric_limits<long long int>::min();
	std::cout << ", " << std::numeric_limits<unsigned long long>::min();
	std::cout << " - maxs: " << std::numeric_limits<long long int>::max();
	std::cout << ", " << std::numeric_limits<unsigned long long>::max() << std::endl << std::endl;

	return 0;
}

/*
 * Options.hpp
 *
 *  Created on: Nov 13, 2013
 *      Author: michael
 */

#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <map>

class ProgramOptions {
	public:
		ProgramOptions() { }
		~ProgramOptions() { }
		
	public:
		typedef std::map< std::string, std::string > OptionMap;
	
	public:
		void parse(int argc, char** argv) {
			std::string booleanOptions[] = {
				"-h",
				"--help",
				"--cache-warmup",
				"--log-stride",
				"--addressed",
				"--local",
				"--clock",
				"--cache",
				"--shared",
				"--instruction",
				"--texture"
			};
			const unsigned numBooleans = sizeof(booleanOptions) / sizeof(booleanOptions[0]);
			
			for (unsigned i = 1; i < argc; i++) {
				bool boolean = false;
				for (unsigned j = 0; j < numBooleans; j++) {
					if (std::string(argv[i]) == booleanOptions[j]) {
						boolean = true;
						break;
					}
				}
				
				if (boolean) {
					options[argv[i]] = "enabled";
					continue;
				} else {
					if (i+1 < argc) {
						options[argv[i]] = argv[i+1];
						i++;
						continue;
					}
				}
			}
		}
		
		const OptionMap& getOptions() const {
			return options;
		}
		
		bool contains(const std::string& option) const {
			if (options.find(option) != options.end()) {
				return true;
			}
			return false;
		}
		
		std::string at(const std::string& option) const {
			return options.at(option);
		}
		
		std::string& operator[](const std::string& option) {
			return options[option];
		}
		
		OptionMap::iterator begin() {
			return options.begin();
		}
		
		OptionMap::const_iterator begin() const {
			return options.begin();
		}
		
		OptionMap::iterator end() {
			return options.end();
		}
		
		OptionMap::const_iterator end() const {
			return options.end();
		}
		
	protected:
		OptionMap options;
	};

#endif /* OPTIONS_HPP_ */

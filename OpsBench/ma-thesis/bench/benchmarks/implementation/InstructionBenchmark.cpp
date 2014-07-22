/*
 * InstructionBenchmark.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: michael
 */

// bench includes
#include "benchmarks/interface/InstructionBenchmark.hpp"
#include "util/interface/Repeat.hpp"
#include "cudatools/interface/ErrorHandler.hpp"

#ifdef REPORT_LEVEL
#undef REPORT_LEVEL
#endif

#define REPORT_LEVEL 1
#include "util/interface/Debug.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////

#define MEASURE_LATENCY(FUNC)													\
do {																			\
	FUNC##Wrapper (d_ts, d_out, 4, 6, LOOP_REPEATS, grid, block);							\
	check( cudaDeviceSynchronize() );											\
	m_out << util::Indents(2) << #FUNC << " latency:" << util::Indents(2);		\
	check( cudaMemcpy(ts, d_ts, sizeof(ts), cudaMemcpyDeviceToHost) );			\
	m_out << ts[1] - ts[0] << " clocks (" << ((double)(ts[1] - ts[0])/ UNROLL_REPEATS / LOOP_REPEATS ) << " clk/instruction)" << std::endl; \
} while(0)

/////////////////////////////////////////////////////////////////////////////////////////////

bench::InstructionBenchmark::InstructionBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out):
	Benchmark(_deviceConfig, _options, _out), m_outputIndents(2) {
		
}

bench::InstructionBenchmark::~InstructionBenchmark() {
	
}

/////////////////////////////////////////////////////////////////////////////////////////////

bool bench::InstructionBenchmark::run() {
	report("InstructionBenchmark: Running instruction latency benchmark ...");

	_performLatencyTests();

	m_done = true;
	return true;
}

void bench::InstructionBenchmark::_performLatencyTests() {
	unsigned int ts[2];
	unsigned int *d_ts;
	unsigned int *d_out;

	const dim3 grid(1,1,1), block(1,1,1);

	check( cudaMalloc((void**)&d_ts, sizeof(ts)) );
	check( cudaMalloc((void**)&d_out, sizeof(unsigned int)) );

	report("Running pipeline tests ...");

	/* Pipeline latency for main FUs */
	report("Running functional unit tests ...");
	MEASURE_LATENCY(K_ADD_UINT_DEP128);
	MEASURE_LATENCY(K_RSQRT_FLOAT_DEP128);
	MEASURE_LATENCY(K_ADD_DOUBLE_DEP128);
	m_out << std::endl;

	/* ARITHMETIC INSTRUCTIONS: UINT */
	report("Running uint arithmetic tests ...");
	MEASURE_LATENCY(K_ADD_UINT_DEP128);
	MEASURE_LATENCY(K_SUB_UINT_DEP128);
	MEASURE_LATENCY(K_MAD_UINT_DEP128);
	MEASURE_LATENCY(K_MUL_UINT_DEP128);
	MEASURE_LATENCY(K_DIV_UINT_DEP128);
	MEASURE_LATENCY(K_REM_UINT_DEP128);
	MEASURE_LATENCY(K_MIN_UINT_DEP128);
	MEASURE_LATENCY(K_MAX_UINT_DEP128);
	m_out << std::endl;

	/* ARITHMETIC INSTRUCTIONS: INT */
	report("Running int arithmetic tests ...");
	MEASURE_LATENCY(K_ADD_INT_DEP128);
	MEASURE_LATENCY(K_SUB_INT_DEP128);
	MEASURE_LATENCY(K_MAD_INT_DEP128);
	MEASURE_LATENCY(K_MUL_INT_DEP128);
	MEASURE_LATENCY(K_DIV_INT_DEP128);
	MEASURE_LATENCY(K_REM_INT_DEP128);
	MEASURE_LATENCY(K_MIN_INT_DEP128);
	MEASURE_LATENCY(K_MAX_INT_DEP128);
	MEASURE_LATENCY(K_ABS_INT_DEP128);
	m_out << std::endl;

	/* ARITHMETIC INSTRUCTIONS: FLOAT */
	report("Running float arithmetic tests ...");
	MEASURE_LATENCY(K_ADD_FLOAT_DEP128);
	MEASURE_LATENCY(K_SUB_FLOAT_DEP128);
	MEASURE_LATENCY(K_MAD_FLOAT_DEP128);
	MEASURE_LATENCY(K_MUL_FLOAT_DEP128);
	MEASURE_LATENCY(K_DIV_FLOAT_DEP128);
	MEASURE_LATENCY(K_MIN_FLOAT_DEP128);
	MEASURE_LATENCY(K_MAX_FLOAT_DEP128);
	m_out << std::endl;

	/* ARITHMETIC INSTRUCTIONS: DOUBLE */
	report("Running double arithmetic tests ...");
	MEASURE_LATENCY(K_ADD_DOUBLE_DEP128);
	MEASURE_LATENCY(K_SUB_DOUBLE_DEP128);
	MEASURE_LATENCY(K_MAD_DOUBLE_DEP128);
	MEASURE_LATENCY(K_MUL_DOUBLE_DEP128);
	MEASURE_LATENCY(K_DIV_DOUBLE_DEP128);
	MEASURE_LATENCY(K_MIN_DOUBLE_DEP128);
	MEASURE_LATENCY(K_MAX_DOUBLE_DEP128);
	m_out << std::endl;

	/* LOGIC */
	report("Running logic tests ...");
	MEASURE_LATENCY(K_AND_UINT_DEP128);
	MEASURE_LATENCY(K_OR_UINT_DEP128);
	MEASURE_LATENCY(K_XOR_UINT_DEP128);
	MEASURE_LATENCY(K_SHL_UINT_DEP128);
	MEASURE_LATENCY(K_SHR_UINT_DEP128);
	m_out << std::endl;

	/* INTRINSICS */
	/* ARITHMETIC INTRINSICS: INTEGER */
	report("Running intrinsic int arithmetic tests ...");
	MEASURE_LATENCY(K_UMUL24_UINT_DEP128);
	MEASURE_LATENCY(K_MUL24_INT_DEP128);
	MEASURE_LATENCY(K_UMULHI_UINT_DEP128);
	MEASURE_LATENCY(K_MULHI_INT_DEP128);
	MEASURE_LATENCY(K_USAD_UINT_DEP128);
	MEASURE_LATENCY(K_SAD_INT_DEP128);
	m_out << std::endl;

	/* ARITHMETIC INTRINSICS: FLOAT */
	report("Running intrinsic float arithmetic tests ...");
	MEASURE_LATENCY(K_FADD_RN_FLOAT_DEP128);
	MEASURE_LATENCY(K_FADD_RZ_FLOAT_DEP128);
	MEASURE_LATENCY(K_FMUL_RN_FLOAT_DEP128);
	MEASURE_LATENCY(K_FMUL_RZ_FLOAT_DEP128);
	MEASURE_LATENCY(K_FDIVIDEF_FLOAT_DEP128);
	m_out << std::endl;

	/* ARITHMETIC INTRINSICS: DOUBLE */
	report("Running intrinsic double arithmetic tests ...");
	MEASURE_LATENCY(K_DADD_RN_DOUBLE_DEP128);
	m_out << std::endl;

	/* MATH INSTRUCTIONS: FLOAT */
	report("Running special float arithmetic tests ...");
	MEASURE_LATENCY(K_RCP_FLOAT_DEP128);
	MEASURE_LATENCY(K_SQRT_FLOAT_DEP128);
	MEASURE_LATENCY(K_RSQRT_FLOAT_DEP128);
	MEASURE_LATENCY(K_SIN_FLOAT_DEP128);
	MEASURE_LATENCY(K_COS_FLOAT_DEP128);
	MEASURE_LATENCY(K_TAN_FLOAT_DEP128);
	MEASURE_LATENCY(K_EXP_FLOAT_DEP128);
	MEASURE_LATENCY(K_EXP10_FLOAT_DEP128);
	MEASURE_LATENCY(K_LOG_FLOAT_DEP128);
	MEASURE_LATENCY(K_LOG2_FLOAT_DEP128);
	MEASURE_LATENCY(K_LOG10_FLOAT_DEP128);
	MEASURE_LATENCY(K_POW_FLOAT_DEP128);
	m_out << std::endl;

	/* MATHEMATICAL INTRINSICS: FLOAT */
	report("Running intrinsic special float arithmetic tests ...");
	MEASURE_LATENCY(K_SINF_FLOAT_DEP128);
	MEASURE_LATENCY(K_COSF_FLOAT_DEP128);
	MEASURE_LATENCY(K_TANF_FLOAT_DEP128);
	MEASURE_LATENCY(K_EXPF_FLOAT_DEP128);
	MEASURE_LATENCY(K_EXP2F_FLOAT_DEP128);
	MEASURE_LATENCY(K_EXP10F_FLOAT_DEP128);
	MEASURE_LATENCY(K_LOGF_FLOAT_DEP128);
	MEASURE_LATENCY(K_LOG2F_FLOAT_DEP128);
	MEASURE_LATENCY(K_LOG10F_FLOAT_DEP128);
	MEASURE_LATENCY(K_POWF_FLOAT_DEP128);
	m_out << std::endl;

	/* CONVERSION INTRINSICS: INT/FLOAT */
	report("Running x2x tests ...");
	MEASURE_LATENCY(K_INTASFLOAT_UINT_DEP128);
	MEASURE_LATENCY(K_FLOATASINT_FLOAT_DEP128);
	m_out << std::endl;

	/* MISC INTRINSICS: INTEGER */
	report("Running misc intrinsic tests ...");
	MEASURE_LATENCY(K_POPC_UINT_DEP128);
	MEASURE_LATENCY(K_CLZ_UINT_DEP128);
	m_out << std::endl;

	/* WARP INTRINSICS */
	report("Running warp arithmetic tests ...");
	MEASURE_LATENCY(K_ALL_UINT_DEP128);
	MEASURE_LATENCY(K_ANY_UINT_DEP128);
	m_out << std::endl;

	check( cudaFree(d_ts) );
	check( cudaFree(d_out) );
}

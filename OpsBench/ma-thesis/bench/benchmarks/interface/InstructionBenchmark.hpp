/*
 * InstructionBenchmark.hpp
 *
 *  Created on: Nov 18, 2013
 *      Author: michael
 */

#ifndef INSTRUCTIONBENCHMARK_H_
#define INSTRUCTIONBENCHMARK_H_

// bench includes
#include "benchmarks/interface/Benchmark.hpp"
#include "cudatools/interface/ErrorHandler.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////

typedef short			SHORT;
typedef	unsigned int 	UINT;
typedef int				INT;
typedef float			FLOAT;
typedef double 			DOUBLE;

/* WARP INTRINSICS */
#define ALL(a, b)	a=__all(a==b)
#define ANY(a, b)	a=__any(a==b)

/* ARITHMETIC INSTRUCTIONS */
#define ADD(a, b) 	a+=b
#define SUB(a, b)	a-=b
#define MUL(a, b)	a*=b
#define MAD(a, b)	a=a*b+a
#define DIV(a, b)	a/=b
#define REM(a, b)	a%=b
#define ABS(a, b)	a+=abs(b)
#define NEG(a, b)	a^=-b
#define MIN(a, b)	a=min(a+b,b)
#define MAX(a, b)	a=max(a+b,b)

/* LOGIC INSTRUCTIONS */
#define AND(a, b)	a&=b
#define OR(a, b)	a|=b
#define XOR(a, b)	a^=b
#define SHL(a, b)	a<<=b
#define SHR(a, b)	a>>=b
#define NOT(a, b)	a=~b
#define NOT2(a, b)	if (a>=b) a=~b
#define CNOT(a, b)	a^=(b==0)?1:0
#define ANDNOT(a, b)	a&=~b
#define ORNOT(a, b)	a|=~b
#define XORNOT(a, b)	a^=~b
#define ADDNOT(a, b) 	a+=~b
#define ANDNOTNOT(a, b)	a=~a&~b

/* ARITHMETIC INSTRINSICS: INTEGER */
#define MUL24(a, b)		a=__mul24(a, b)
#define UMUL24(a, b)	a=__umul24(a, b)
#define MULHI(a, b)		a=__mulhi(a, b)
#define UMULHI(a, b)	a=__umulhi(a, b)
#define SAD(a, b)		a=__sad(a, b, a)
#define USAD(a, b)		a=__usad(a, b, a)

/* ARITHMETIC INTRINSICS: FLOAT */
#define FADD_RN(a, b)	a=__fadd_rn(a, b)
#define FADD_RZ(a, b)	a=__fadd_rz(a, b)
#define FMUL_RN(a, b)	a=__fmul_rn(a, b)
#define FMUL_RZ(a, b)	a=__fmul_rz(a, b)
#define FDIVIDEF(a, b)	a=__fdividef(a, b)

/* ARITHMETIC INTRINSICS: DOUBLE. Requires SM1.3 */
#define DADD_RN(a, b)	a=__dadd_rn(a, b)

/* MATH INSTRUCTIONS: FLOAT */
#define RCP(a, b)	a+=1/b
#define SQRT(a, b)	a=sqrt(b)
#define RSQRT(a, b)	a=rsqrt(b)
#define SIN(a, b)	a=sinf(b)
#define COS(a, b)	a=cosf(b)
#define TAN(a, b)	a=tanf(b)
#define EXP(a, b)	a=expf(b)
#define EXP10(a, b)	a=exp10f(b)
#define LOG(a, b)	a=logf(b)
#define LOG2(a, b)	a=log2f(b)
#define LOG10(a, b)	a=log10f(b)
#define POW(a, b)	a=powf(a, b)

/* MATH INTRINSICS: FLOAT */
#define SINF(a, b)	a=__sinf(b)
#define COSF(a, b)	a=__cosf(b)
//#define SINCOSF
#define TANF(a, b)		a=__tanf(b)
#define EXPF(a, b)		a=__expf(b)
#define EXP2F(a, b)		a=exp2f(b)
#define EXP10F(a, b)	a=__exp10f(b)
#define LOGF(a, b)		a=__logf(b)
#define LOG2F(a, b)		a=__log2f(b)
#define LOG10F(a, b)	a=__log10f(b)
#define POWF(a, b)		a=__powf(a, b)

/* CONVERSION INTRINSICS */
#define INTASFLOAT(a, b)		a=__int_as_float(b)
#define FLOATASINT(a, b)		a=__float_as_int(b)

/* MISC INTRINSICS */
#define POPC(a, b)	a=__popc(b)
#define SATURATE(a, b)	a=saturate(b)
#define CLZ(a, b)	a=__clz(b)  //count leading zeros
#define CLZLL(a, b)	a=__clzll(b)  //count leading zeros
#define FFS(a, b)	a=__ffs(b)
#define FFSLL(a, b)	a=__ffsll(b)

/* DATA MOVEMENT AND CONVERSION INSTRUCTIONS */
#define MOV(a, b)	a+=b; b=a
#define MOV4(a, b, c)	a=b^c; b=a

/////////////////////////////////////////////////////////////////////////////////////////////

#define K_OP_DEP(OP, DEP, TYPE)\
__global__ void K_##OP##_##TYPE##_DEP##DEP (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its) 	\
{														\
	TYPE t1 = p1; \
	TYPE t2 = p2; \
	unsigned int start_time=0, stop_time=1; \
	do { \
		start_time = clock(); \
		for (int i = 0; i < its; i++) { \
			repeat(DEP, OP(t1, t2); OP(t2, t1);) \
		} \
		stop_time = clock(); \
	} while(stop_time < start_time); \
	out[0] = (unsigned int)(t1 + t2); \
	ts[0] = start_time; \
	ts[1] = stop_time; \
} \
void K_##OP##_##TYPE##_DEP##DEP##Wrapper (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its, const dim3& grid, const dim3& block) { \
	K_##OP##_##TYPE##_DEP##DEP <<<grid, block>>> (ts, out, p1, p2, its); \
}

#define K_OP_DEP_HEADER(OP, DEP, TYPE)\
void K_##OP##_##TYPE##_DEP##DEP##Wrapper (unsigned int *ts, unsigned int* out, TYPE p1, TYPE p2, int its, const dim3& grid, const dim3& block);

/////////////////////////////////////////////////////////////////////////////////////////////

/* WARP VOTE INTRINSICS */
K_OP_DEP_HEADER(ALL, 128, UINT)
K_OP_DEP_HEADER(ANY, 128, UINT)

/* ARITHMETIC INSTRUCTIONS: UINT*/
K_OP_DEP_HEADER(ADD, 128, UINT)
K_OP_DEP_HEADER(SUB, 128, UINT)
K_OP_DEP_HEADER(MUL, 128, UINT)
K_OP_DEP_HEADER(DIV, 128, UINT)
K_OP_DEP_HEADER(REM, 128, UINT)
K_OP_DEP_HEADER(MAD, 128, UINT)
K_OP_DEP_HEADER(MIN, 128, UINT)
K_OP_DEP_HEADER(MAX, 128, UINT)

/* ARITHMETIC INSTRUCTIONS: INT */
K_OP_DEP_HEADER(ADD, 128, INT)
K_OP_DEP_HEADER(SUB, 128, INT)
K_OP_DEP_HEADER(MUL, 128, INT)
K_OP_DEP_HEADER(DIV, 128, INT)
K_OP_DEP_HEADER(REM, 128, INT)
K_OP_DEP_HEADER(MAD, 128, INT)
K_OP_DEP_HEADER(ABS, 128, INT)
K_OP_DEP_HEADER(NEG, 128, INT)
K_OP_DEP_HEADER(MIN, 128, INT)
K_OP_DEP_HEADER(MAX, 128, INT)

/* ARITHMETIC INSTRUCTIONS: FLOAT */
K_OP_DEP_HEADER(ADD, 128, FLOAT)
K_OP_DEP_HEADER(SUB, 128, FLOAT)
K_OP_DEP_HEADER(MUL, 128, FLOAT)
K_OP_DEP_HEADER(DIV, 128, FLOAT)
K_OP_DEP_HEADER(MAD, 128, FLOAT)
K_OP_DEP_HEADER(ABS, 128, FLOAT)
K_OP_DEP_HEADER(MIN, 128, FLOAT)
K_OP_DEP_HEADER(MAX, 128, FLOAT)

/* ARITHMETIC INSTRUCTIONS: DOUBLE */
K_OP_DEP_HEADER(ADD, 128, DOUBLE)
K_OP_DEP_HEADER(SUB, 128, DOUBLE)
K_OP_DEP_HEADER(MUL, 128, DOUBLE)
K_OP_DEP_HEADER(DIV, 128, DOUBLE)
K_OP_DEP_HEADER(MAD, 128, DOUBLE)
K_OP_DEP_HEADER(ABS, 128, DOUBLE)
K_OP_DEP_HEADER(MIN, 128, DOUBLE)
K_OP_DEP_HEADER(MAX, 128, DOUBLE)

/* LOGIC INSTRUCTIONS */
K_OP_DEP_HEADER(AND,  128, UINT)
K_OP_DEP_HEADER(OR,   128, UINT)
K_OP_DEP_HEADER(XOR,  128, UINT)
K_OP_DEP_HEADER(SHL,  128, UINT)
K_OP_DEP_HEADER(SHR,  128, UINT)
K_OP_DEP_HEADER(NOT,  128, UINT)
K_OP_DEP_HEADER(NOT2,  128, INT)
K_OP_DEP_HEADER(CNOT, 128, UINT)
K_OP_DEP_HEADER(ANDNOT,  128, UINT)
K_OP_DEP_HEADER(ORNOT,   128, UINT)
K_OP_DEP_HEADER(XORNOT,  128, UINT)
K_OP_DEP_HEADER(ADDNOT,  128, UINT)
K_OP_DEP_HEADER(ANDNOTNOT,  128, UINT)

/* ARITHMETIC INSTRINSICS: UINT/INT */
K_OP_DEP_HEADER(UMUL24, 128, UINT)
K_OP_DEP_HEADER(MUL24, 128, INT)
K_OP_DEP_HEADER(UMULHI, 128, UINT)
K_OP_DEP_HEADER(MULHI, 128, INT)
K_OP_DEP_HEADER(USAD, 128, UINT)
K_OP_DEP_HEADER(SAD, 128, INT)

/* ARITHMETIC INSTRINSICS: FLOAT */
K_OP_DEP_HEADER(FADD_RN, 128, FLOAT)
K_OP_DEP_HEADER(FADD_RZ, 128, FLOAT)
K_OP_DEP_HEADER(FMUL_RN, 128, FLOAT)
K_OP_DEP_HEADER(FMUL_RZ, 128, FLOAT)
K_OP_DEP_HEADER(FDIVIDEF, 128, FLOAT)

/* INSTRINSICS: DOUBLE */
K_OP_DEP_HEADER(DADD_RN, 128, DOUBLE)

/* MATH INSTRUCTIONS: FLOAT */
K_OP_DEP_HEADER(RCP, 128, FLOAT)
K_OP_DEP_HEADER(SQRT, 128, FLOAT)
K_OP_DEP_HEADER(RSQRT, 128, FLOAT)
K_OP_DEP_HEADER(SIN, 128, FLOAT)
K_OP_DEP_HEADER(COS, 128, FLOAT)
K_OP_DEP_HEADER(TAN, 128, FLOAT)
K_OP_DEP_HEADER(EXP, 128, FLOAT)
K_OP_DEP_HEADER(EXP10, 128, FLOAT)
K_OP_DEP_HEADER(LOG, 128, FLOAT)
K_OP_DEP_HEADER(LOG2, 128, FLOAT)
K_OP_DEP_HEADER(LOG10, 128, FLOAT)
K_OP_DEP_HEADER(POW, 128, FLOAT)

/* MATH INTRINSICS: FLOAT */
K_OP_DEP_HEADER(SINF, 128, FLOAT)
K_OP_DEP_HEADER(COSF, 128, FLOAT)
K_OP_DEP_HEADER(TANF, 128, FLOAT)
K_OP_DEP_HEADER(EXPF, 128, FLOAT)
K_OP_DEP_HEADER(EXP2F, 128, FLOAT)
K_OP_DEP_HEADER(EXP10F, 128, FLOAT)
K_OP_DEP_HEADER(LOGF, 128, FLOAT)
K_OP_DEP_HEADER(LOG2F, 128, FLOAT)
K_OP_DEP_HEADER(LOG10F, 128, FLOAT)
K_OP_DEP_HEADER(POWF, 128, FLOAT)

/* CONVERSION */
K_OP_DEP_HEADER(INTASFLOAT, 128, UINT)
K_OP_DEP_HEADER(FLOATASINT, 128, FLOAT)

/* MISC */
K_OP_DEP_HEADER(POPC, 128, UINT)
K_OP_DEP_HEADER(CLZ, 128, UINT)
K_OP_DEP_HEADER(CLZLL, 128, UINT)
K_OP_DEP_HEADER(FFS, 128, UINT)
K_OP_DEP_HEADER(FFSLL, 128, UINT)
K_OP_DEP_HEADER(SATURATE, 128, FLOAT)

/////////////////////////////////////////////////////////////////////////////////////////////

namespace bench {

	class InstructionBenchmark: public Benchmark {
	public:
		InstructionBenchmark(const cuda::DeviceConfiguration& _deviceConfig, const ProgramOptions& _options, std::ostream& _out);
		virtual ~InstructionBenchmark();
	
	public:
		bool run();
	
	private:
		void _performLatencyTests();

	private:
		const unsigned m_outputIndents;
	};

}

#endif /* CACHEBENCHMARK_H_ */

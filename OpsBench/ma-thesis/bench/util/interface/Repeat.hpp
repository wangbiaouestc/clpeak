/*
 * Repeat.hpp
 *
 *  Created on: Nov 5, 2013
 *      Author: michael
 */

#ifndef REPEAT_HPP_
#define REPEAT_HPP_

#define UNROLL_REPEATS 256
#define LOOP_REPEATS 128

#define PASTER(x, y) x ## y
#define EVALUATOR(x,y) PASTER(x,y)

#define repeat(n, expr) EVALUATOR(repeat,n) (expr)

#define repeat2(expr) expr expr
#define repeat4(expr) expr expr expr expr
#define repeat8(expr) expr expr expr expr expr expr expr expr
#define repeat16(expr) expr expr expr expr expr expr expr expr expr expr expr expr expr expr expr expr
#define repeat32(expr) repeat16(expr) repeat16(expr)
#define repeat64(expr) repeat32(expr) repeat32(expr)
#define repeat128(expr) repeat64(expr) repeat64(expr)
#define repeat256(expr) repeat128(expr) repeat128(expr)
#define repeat512(expr) repeat256(expr) repeat256(expr)
#define repeat1024(expr) repeat512(expr) repeat512(expr)
#define repeat2048(expr) repeat1024(expr) repeat1024(expr)

#endif /* REPEAT_HPP_ */

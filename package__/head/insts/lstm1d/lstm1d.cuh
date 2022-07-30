#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"

#include "pkg_head/insts/lstm1d/lstm1d_th11.cuh"

/*
//	Y = F(X@.W + .B)

//Arguments = [X, Y, istart,ystart,wstart,locdstart, drop_rate]

	_th11 : each thread compute Y 1 pixel
*/

/*			lstm1d : [X] -> [Y]
x:[X]

f0 = f(x@W + h[-1]@U + b)
f1 = f(x@W + h[-1]@U + b)
f2 = f(x@W + h[-1]@U + b)
g0 = g(x@W + h[-1]@U + b)
e = f0 * e[-1] + f1 * g0
h = f2 * e 							#here lstm propose f2*j(x) where j can be f(x)=x or f(x)=tanh(x). It's propose by wiki that f(x)=x is better, so I juste use it, and make sens

h:[Y]

|||||||| f(x) = 1 / (1 + exp(-x))	= logistic
|||||||| g(x) = tanh(x)				= tanh

Input = X
Output = Y*2 (store e and h)

W:[X,Y]
U:[X,X]
B:[Y]

===== Var struct
e:[Y]
h:[Y]

===== Weight struct
f0W:[X,Y]
f0U:[Y,Y]
f0B:[Y]
f1W:[X,Y]
f1U:[Y,Y]
f1B:[Y]
f2W:[X,Y]
f2U:[Y,Y]
f2B:[Y]
f3W:[X,Y]
f3U:[Y,Y]
f3B:[Y]

===== Locd struct
f0:[Y]					#
f1:[Y]					#	Just store f0,f1,f2,g0
f2:[Y]					#
g0:[Y]					#
*/

/* ======== Gpu computation MODS ============
th11:
	each kernel compute completely one output pixel. [No shared] [No consts] [No texture]
*/

//=========================== Sizes ===============================

/*
	inputs = X
	vars = Y
	weights = 4*(X*Y + Y*Y + Y)
	locds = 4*Y
*/

void lstm1d_check(uint * param);

//======================= Cpu_t forward ===========================

void lstm1d_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void lstm1d_use_call_mode_th11(Use_t * use, uint inst, uint time);

void lstm1d_use(Use_t * use, uint inst, uint time);

//========================== Train_t =======================

//-------------------------- forward ---------------------

void lstm1d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void lstm1d_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void lstm1d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void lstm1d_backward(Train_t * train, uint inst, uint time, uint start_seed);
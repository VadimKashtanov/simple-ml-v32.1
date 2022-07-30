#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"

#include "pkg_head/insts/dotgaussfiltre2d/dotgaussfiltre2d_th11.cuh"

/*
//	Y = F(X@.W + .B)

//Arguments = [X, Y, activ, input_start,ystart,wstart,locdstart, drop_rate]

//		activ : logistic, tanh, gauss, relu
	_th11 : each thread compute Y 1 pixel
*/

/*
		      [p0,p1,p2]	
		      [p3,p4,p5]
		      [p6,p7,p8]
[a0,a1,a2] -> [c0,c1,c2]
[b0,b1,b2] -> [d0,d1,d2]

c0 = exp(-a0^2 + p0) + exp(-a1^2 + p3) + exp(-a2^2 + p6)
d1 = exp(-b0^2 + p1) + exp(-b1^2 + p4) + exp(-b2^2 + p7)
*/

/* ======== Gpu computation MODS ============
th11:
	each kernel compute completely one output pixel. [No shared] [No consts] [No texture]
*/

//=========================== Sizes ===============================

/*
	inputs = X
	vars = Y
	weights = X*Y + X*X + Y
	locds = Y
*/

void dotgaussfiltre2d_check(uint * param);

//======================= Cpu_t forward ===========================

void dotgaussfiltre2d_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void dotgaussfiltre2d_use_call_mode_th11(Use_t * use, uint inst, uint time);

void dotgaussfiltre2d_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dotgaussfiltre2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void dotgaussfiltre2d_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void dotgaussfiltre2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void dotgaussfiltre2d_backward(Train_t * train, uint inst, uint time, uint start_seed);
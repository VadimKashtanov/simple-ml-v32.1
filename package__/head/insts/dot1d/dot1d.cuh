#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"
#include "pkg_head/insts/dot1d/dot1d_th11.cuh"

/*
//	Y = F(X@.W + .B)

//Arguments = [Ax, Yx, activ, input_start, ystart, wstart, locdstart, drop_rate]

//		activ : logistic, tanh, gauss, relu
	_th11 : each thread compute Y 1 pixel
*/

/*
	Weight storage is not the same

X = [a,b,c,d]
W = [
		[0,1,2],
		[3,4,5],
		[6,7,8],
		[9,10,11]
	]
Y = [0*a+b*3+c*6+d*9, 1*a+4*b+7*c+10*d, 2*a+5*b+8*c+d*11]
*/

/* ======== Gpu computation MODS ============
th11:
	each kernel compute completely one output pixel. [No shared] [No consts] [No texture]
*/

//=========================== Sizes ===============================

/*
	inputs = Ax
	vars = Yx
	weights = Ax*Yx + Yx
	locds = Yx
*/

void dot1d_check(uint * param);

//======================= Cpu_t forward ===========================

void dot1d_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void dot1d_use_call_mode_th11(Use_t * use, uint inst, uint time);

void dot1d_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dot1d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void dot1d_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void dot1d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void dot1d_backward(Train_t * train, uint inst, uint time, uint start_seed);
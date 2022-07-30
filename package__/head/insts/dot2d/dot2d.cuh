#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"

#include "pkg_head/insts/dot2d/dot2d_th11.cuh"

/*
//	Y = F(X@.W + .B)

//Arguments = [Ax, Ay, Bx, activ, input_start, ystart, wstart, locdstart, drop_rate]
//		activ : logistic, tanh, gauss, relu

	_th1x1 : each thread compute 1 Y pixel
*/

//=========================== Sizes ===============================

/*
	inputs = Ax*Ay
	vars = Ay*Bx
	weights = Ax*Bx + Ay*Bx
	locds = Ay*Ax
*/

void dot2d_check(uint * param);

//======================= Cpu_t Forward ===========================

void dot2d_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void dot2d_use_call_mode_th11(Use_t * use, uint inst, uint time);

void dot2d_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dot2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void dot2d_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void dot2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void dot2d_backward(Train_t * train, uint inst, uint time, uint start_seed);
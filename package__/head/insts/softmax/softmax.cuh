#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"

#include "pkg_head/insts/softmax/softmax_th32.cuh"

//	Y = exp(x)/sum(exp(x))

//Arguments = [len, input_start, ystart]

//implementation : f(x) = exp(x)/sum(exp(x))
//with max : f(x) = exp(x - max)/sum(exp(x - max))     but we don't implement it

/* ======== Gpu computation MODS ============
th32:
	maximum 32 pixels. [No shared] [No consts] [No texture]
*/

//=========================== Sizes ===============================

/*
	inputs = len
	vars = len
	weights = 0
	locds = 0
*/

void softmax_check(uint * param);

//======================= Cpu_t forward ===========================

void softmax_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Model Forward ===========================

void softmax_use_call_mode_th32(Use_t * use, uint inst, uint time);

void softmax_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void softmax_forward_call_mode_th32(Train_t * train, uint inst, uint time, uint start_seed);

void softmax_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void softmax_backward_call_mode_th32(Train_t * train, uint inst, uint time, uint start_seed);

void softmax_backward(Train_t * train, uint inst, uint time, uint start_seed);
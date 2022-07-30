#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"

#include "pkg_head/insts/gaussfiltre2d/gaussfiltre2d_th11.cuh"

/* ======== Gpu computation MODS ============
th11:
	each kernel compute completely one output pixel. [No shared] [No consts] [No texture]
*/

//=========================== Sizes ===============================

/*
	inputs = 
	vars = 
	weights = 
	locds = 
*/

void gaussfiltre2d_check(uint * param);

//======================= Cpu_t forward ===========================

void gaussfiltre2d_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void gaussfiltre2d_use_call_mode_th11(Use_t * use, uint inst, uint time);

void gaussfiltre2d_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void gaussfiltre2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void gaussfiltre2d_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void gaussfiltre2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void gaussfiltre2d_backward(Train_t * train, uint inst, uint time, uint start_seed);
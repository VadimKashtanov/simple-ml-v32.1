#pragma once

#include "score.cuh"

#include "meta_package_definitions.cuh"

#include "pkg_head/insts/dot2drecurent/dot2drecurent_th11.cuh"


//	Params : [Ax,Ay,At, Bx, activ, ist,yst,wst,lst, drate]
//	At - de combien de lignes on va en arriere. Si At=1 =>  A=A[t-1]


/* ======== Gpu computation MODS ============
th11:
	each kernel compute completely one output pixel. [No shared] [No consts] [No texture]
*/

//=========================== Sizes ===============================

/*
	inputs = Ax*Ay
	vars = Ax*Bx
	weights = Ax*Bx + Bx*Ay
	locds = Bx*Ay
*/

void dot2drecurent_check(uint * param);

//======================= Cpu_t forward ===========================

void dot2drecurent_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void dot2drecurent_use_call_mode_th11(Use_t * use, uint inst, uint time);

void dot2drecurent_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dot2drecurent_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void dot2drecurent_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void dot2drecurent_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void dot2drecurent_backward(Train_t * train, uint inst, uint time, uint start_seed);
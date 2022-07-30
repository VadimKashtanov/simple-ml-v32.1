#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"


#include "pkg_head/insts/lstm2d/lstm2d_th11.cuh"

/*
	[Ax, Ay, Bx, input_start, ystart, wstart, locdstart, drop_rate]

x:[Ax,Ay] -> h[Bx,Ay]

f0 = f(x@W + h[-1]@U + b)
f1 = f(x@W + h[-1]@U + b)
f2 = f(x@W + h[-1]@U + b)
g0 = g(x@W + h[-1]@U + b)
e = f0 * e[-1] + f1 * g0
h = f2 * e

*/

/* ======== Gpu computation MODS ============
th11:
	each kernel compute completely one output pixel. [No shared] [No consts] [No texture]
*/

//=========================== Sizes ===============================

/*
	inputs = Ax*Ay
	vars = 2 * Bx*Ay
	weights = 4 * (Bx*Ax + Bx*Bx + Bx*Ay)
	locds = 4 * Bx*Ay
*/

void lstm2d_check(uint * param);

//======================= Cpu_t forward ===========================

void lstm2d_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void lstm2d_use_call_mode_th11(Use_t * use, uint inst, uint time);

void lstm2d_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void lstm2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void lstm2d_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void lstm2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void lstm2d_backward(Train_t * train, uint inst, uint time, uint start_seed);
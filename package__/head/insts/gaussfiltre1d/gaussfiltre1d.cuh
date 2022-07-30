#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"

#include "pkg_head/insts/gaussfiltre1d/gaussfiltre1d_th11.cuh"

/*	n = 3
gauss_filtre = 1/n * [ exp(-(w*x0 + p0)^2) + exp(-(w*x0 + p1)^2) + exp(-(w*x0 + p2)^2)] + b
w=1 is good
	    [ 0  1  3 ]
	n=3 [ -1 0 -2 ]   p's
	    [ 3 -1 .5 ]

[ 1 2 ] [ .	 .  . ]    '.' (0,0) = 1/n*[e^(-(1+0)^2) + ...] + b[0,0]
[ 3 4 ] [ .  .  . ]

	+b  [ .  .  . ]
	    [ .  .  . ]
*/

/*	Maths
f = 1/n * [ exp(-(w*x + p)^2 - k^2) ]
df/dw = - 1/n * x * 2*(w*x + p) * exp(-(w*x + p)^2)
df/dx = - 1/n * w * 2*(w*x + p) * exp(-(w*x + p)^2)
df/dp = - 1/n * 1 * 2*(w*x + p) * exp(-(w*x + p)^2)
df/dk = - 1/n * 2*k * exp(-(w*x + p)^2 - k^2)

locd = - 1/n * 2*(w*x + p) * exp(-(w*x + p)^2)
df/dw = locd * x
df/dx = locd * w
df/dp = locd

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

void gaussfiltre1d_check(uint * param);

//======================= Cpu_t forward ===========================

void gaussfiltre1d_cpu(Cpu_t * cpu, uint inst, uint time);

//======================= Use_t Forward ===========================

void gaussfiltre1d_use_call_mode_th11(Use_t * use, uint inst, uint time);

void gaussfiltre1d_use(Use_t * use, uint inst, uint time);

//======================== Train_t =======================

//-------------------------- forward ---------------------

void gaussfiltre1d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void gaussfiltre1d_forward(Train_t * train, uint inst, uint time, uint start_seed);

//-------------------------- backward ---------------------

void gaussfiltre1d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed);

void gaussfiltre1d_backward(Train_t * train, uint inst, uint time, uint start_seed);
#pragma once

#include "score.cuh"

#include "meta_package_definitions.cuh"


__global__
void dot2d_use_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight);

//===================================

__global__
void dot2d_forward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets);

// =================================

__global__
void dot2d_backward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);

/*		Y = f(A@W + B)
	dLdA = (f'(A@W+B) * dLdY) @ W.t
	dLdW = A.t @ (f'(A@W+B) * dLdY).T
	dLdB = (f'(A@W+B) * dLdY)
where in code
	(f'(A@W+B) * dLdY) = _locd * grad = dlds	//because dlds == dL/d(A@W + B) == dL/d(A@W)
*/

__global__
void dot2d_backward_th1x1_bias(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);

__global__
void dot2d_backward_th1x1_input(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);

__global__
void dot2d_backward_th1x1_weight(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);


/*	Do it with one function, that just go backward the forward function with atomicAdd
	(pixels will be `+=` y multiple threads, because it's how dot function work)
	at end we will have the same amount
*/
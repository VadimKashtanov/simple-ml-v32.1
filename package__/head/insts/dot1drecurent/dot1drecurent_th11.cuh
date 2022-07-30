#pragma once

#include "score.cuh"

#include "meta_package_definitions.cuh"

//			  0  1   2    3      4  5   6    7     8
//	Params : [Ax,At, Yx, activ, ist,yst,wst,lst, drate]
//	At - de combien de lignes on va en arriere. Si At=1 =>  A=A[t-1]

__global__
void dot1drecurent_use_th1x1(
	uint Ax, uint At, uint Yx,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight);

__global__
void dot1drecurent_forward_th1x1(
	uint Ax, uint At, uint Yx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets);

__global__
void dot1drecurent_backward_th1x1(
	uint Ax, uint At, uint Yx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);
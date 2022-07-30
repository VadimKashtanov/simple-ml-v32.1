#pragma once

#include "score.cuh"

#include "meta_package_definitions.cuh"

//======================= Use_t Forward ===========================

__global__
void dotgaussfiltre2d_use_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint total,
	uint input_start, uint ystart, uint wstart,
	float * var, float * weight);

//-------------------------- forward ---------------------

__global__
void dotgaussfiltre2d_forward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets);

//-------------------------- backward ---------------------

__global__
void dotgaussfiltre2d_backward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);
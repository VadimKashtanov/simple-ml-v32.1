#pragma once

#include "score.cuh"

#include "meta_package_definitions.cuh"


//======================= Use_t Forward ===========================

__global__
void lstm2d_use_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight);

//======================== Train_t =======================

//-------------------------- forward ---------------------

__global__
void lstm2d_forward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets);

//-------------------------- backward ---------------------

__global__
void lstm2d_backward_INPUT_th1x1(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drate,
	uint sets);

__global__
void lstm2d_backward_H1_BIAS_th1x1(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drate,
	uint sets);

__global__
void lstm2d_backward_BIAS_ONLY_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint time,
	uint istart, uint ystart, uint wstart, uint lsize,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);
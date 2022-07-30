#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"

//	----- USE -----

__global__
void gaussfiltre2d_use_th1x1(
	uint X, uint Y,
	uint time,
	uint vars,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight);

//  --------- Forward -------- 

__global__
void gaussfiltre2d_forward_th1x1(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets);

//  --------- Backward -------- 

__global__
void gaussfiltre2d_backward_th1x1(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets);
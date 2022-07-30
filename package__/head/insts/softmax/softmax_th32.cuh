#pragma once

#include "score.cuh"

#include "meta_package_definitions.cuh"

//======================= Model Forward ===========================

__global__
void softmax_use_th32(
	uint len,
	uint time,
	uint vsize,
	uint istart, uint ystart,
	float * var);

//======================== Train_t =======================

//-------------------------- forward ---------------------

__global__
void softmax_forward_th32(
	uint len,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart, uint lstart,
	uint sets,
	float * var);

//-------------------------- backward ---------------------

__global__
void softmax_backward_th32(
	uint len,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart, uint lstart,
	uint sets,
	float * var, float * grad);
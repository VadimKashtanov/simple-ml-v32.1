#pragma once

#include "score.cuh"
#include "meta_package_definitions.cuh"


void kconvl33samepool22max_use_const_MemCpyToSymbol(float * arr, uint len);

//	================== Use ==================

__global__
void kconvl33samepool22max_use_const_th1x1(
	uint n0, uint n1, uint Ax, uint Ay,
	uint activ,							
	uint time,
	uint total, uint wsize,
	uint istart, uint wstart, uint ystart,
	float * var, float * weight);

//========================		Train_t	  =========================

void kconvl33samepool22max_train_const_MemCpyToSymbol(float * arr, uint len);

//----------------------------- forward ---------------------------

__global__
void kconvl33samepool22max_forward_const_th1x1(
	uint n0, uint n1, uint Ax, uint Ay,
	uint activ,							
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint wstart, uint ystart, uint lstart,
	uint seed, float drop_rate,
	uint set, uint sets,
	float * var, float * weight, float * locd);

//----------------------------- backward ---------------------------

__global__
void kconvl33samepool22max_backward_const_th1x1(
	uint n0, uint n1, uint Xxlen, uint Xylen,
	uint activ,
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint wstart, uint ystart, uint lstart,
	float * var, float * weight, float * locd,
	float * grad, float * meand,
	uint seed, float drop_rate,
	uint set, uint sets);
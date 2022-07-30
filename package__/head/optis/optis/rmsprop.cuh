#pragma once

#include "kernel/head/optis.cuh"

float opti_rmsprop_alpha = 1e-5;
float opti_rmsprop_beta = 1e-4;

/*
Root Mean Squared Propagation

	v = beta * v + (1-beta) * grad(w)^2
	w -= alpha * grad(w) / sqrt(v)
*/
	
void * RMSPROP_space_mk(Opti_t * opti);
void * RMSPROP_free(Opti_t * opti);

__global__
void RMSPROP_kernel_th11(
	float alpha, float beta,
	uint weights,
	float * v, float * weight, float * meand);

void RMSPROP_optimize(Opti_t * opti);
#pragma once

#include "kernel/head/optis.cuh"

float opti_momentum_alpha = 1e-5;
float opti_momentum_moment = 1 - 1e-5;

/*
Stocastic Gradient Descent with momentum

Vannila or classic grandient descent. Only with a gradient step and momentum

	v = moment * v - alpha * grad(w)
	w += v
*/

void * MOMENTUM_space_mk(Opti_t * opti);
void MOMENTUM_free(Opti_t * opti);

__global__
void momentum_kernel_th11(
	float alpha, float moment,
	uint weights,
	float * v, float * weight, float * meand);

void MOMENTUM_optimize(Opti_t * opti);
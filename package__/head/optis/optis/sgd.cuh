#pragma once

#include "kernel/head/optis.cuh"

float opti_sgd_alpha = 1e-5;

/*
Stocastic Gradient Descent.

Vannila or classic grandient descent. Only with a gradient step.

	w -= alpha * grad(w)
*/

void * SGD_space_mk(Opti_t * opti);
void * SGD_free(Opti_t * opti);

__global__
void sgd_kernel_th11(
	float alpha,
	uint weights,
	float * weight, float * meand);

void SGD_optimize(Opti_t * opti);
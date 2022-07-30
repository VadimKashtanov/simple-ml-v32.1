#pragma once

#include "kernel/head/optis.cuh"

float opti_adamax_alpha = 0.002;
float opti_adamax_beta0 = 0.9;
float opti_adamax_beta1 = 0.999;

/*
AdaMax:

	m = beta0*m + (1 - beta0)*grad(w)
	u = max(beta1 * u, abs(grad(w)))
	
	w -= alpha * m / (u * (1 - beta1^t))
*/

typedef struct {
	float * m_d, * u_d;

	uint echopes;
} AdamaxData_t;

void * ADAMAX_space_mk(Opti_t * opti);
void * ADAMAX_free(Opti_t * opti);

__global__
void adamax_kernel_th11(
	float alpha, float beta0, float beta1,
	uint weights,
	float * m, float *u, float * weight, float * meand);

void ADAMAX_optimize(Opti_t * opti);
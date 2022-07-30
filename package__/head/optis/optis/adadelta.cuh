#pragma once

#include "kernel/head/optis.cuh"

float opti_adadelta_beta0 = 1e-5;
float opti_adadelta_beta1 = 1e-5;

/*
Adaptive Learning Rate Methode

	m = beta0*m + (1 - beta0)*grad(w)^2	
	delta_w = - sqrt(v + 1e-8) / sqrt(m + 1e-8)
	v = beta1*v + (1 - beta1)*delta_w^2

	w -= delta_w
*/

typedef struct {
	float * m_d, * v_d;
} AdadeltaData_t;

void * ADADELTA_space_mk(Opti_t * opti);
void * ADADELTA_free(Opti_t * opti);

__global__
void adadelta_kernel_th11(
	float beta0, float beta1,
	uint weights,
	float * m, float *v, float * weight, float * meand);

void ADADELTA_optimize(Opti_t * opti);
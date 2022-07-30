#include "package/head/optis/optis/adam.cuh"

/*
Maths:
	m = beta0*m + (1 - beta0)*grad(w)
	v = beta1*m + (1 - beta1)*grad(w)^2

	_m = m / ( 1 - beta0^t )
	_v = v / ( 1 - beta1^t )		t is echope
	
	w -= alpha * _m / sqrt(_v + eta)

Optimized:
	m = beta0*m + (1 - beta0)*grad(w)
	v = beta1*m + (1 - beta1)*grad(w)^2

	w -= alpha * m / ((1 - beta0^t) * sqrt(v/(1 - beta1^t) + 1e-8))
*/

__global__
void adam_kernel_th11(
	float beta0, float beta1, float alpha,
	uint echope,
	uint weights,
	float * m, float *v, float * weight, float * meand)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	float m_tmpt, v_tmpt, dw;

	if (x < weights) {
		dw = meand[set*weights + x];
		
		m_tmpt = m[set*weights + x];
		v_tmpt = v[set*weights + x];

		m_tmpt = beta0*m_tmpt + (1 - beta0)*dw
		v_tmpt = beta1*v_tmpt + (1 - beta1)*dw*dw;

		m[set*weights + x] = m_tmpt;
		v[set*weights + x] = v_tmpt;

		weight[set*weights + x] -= alpha * m_tmpt / ((1 - pow(beta0,t)) * sqrt(v_tmpt/(1 - pow(beta1,t)) + 1e-8));
	}
};

void ADAM_optimize(Opti_t * opti)
{
	AdamData_t * adamdata = (AdamData_t*)opti->opti_space;

	adam_kernel_th11<<<dim3(KERN_DIV(opti->train->mdl->weights, 16), opti->train->sets),dim3(16,1)>>>(
		opti_adam_beta0, opti_adam_beta1, opti_adam_alpha
		adamdata->echope,
		opti->train->mdl->weights,
		adamdata->m_d, adamdata->v_d, opti->train->_weight, opti->train->_meand
	);

	adamdata->echope++;
};
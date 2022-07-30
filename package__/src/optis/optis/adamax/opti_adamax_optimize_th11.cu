#include "package/head/optis/optis/adamax.cuh"

/*
	m = beta0*m + (1 - beta0)*grad(w)
	u = max(beta1 * u, abs(grad(w)))
	
	w -= (alpha/(1 - beta1^t)) * m / u
*/

__global__
void adamax_kernel_th11(
	float alpha, float beta0, float beta1,
	uint weights,
	float * m, float *u, float * weight, float * meand)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	float m_tmpt, u_tmpt, dw;

	if (x < weights) {
		dw = meand[set*weights + x];
		
		m_tmpt = m[set*weights + x];
		u_tmpt = u[set*weights + x];

		m_tmpt = beta0*m_tmpt + (1 - beta0)*dw;
		u_tmpt = max(beta1 * u_tmpt, abs(dw));

		m[set*weights + x] = m_tmpt;
		u[set*weights + x] = u_tmpt;

		weight[set*weights + x] -= (alpha / (1 - pow(beta0, t))) * m_tmpt/u_tmpt;
	}
};

void ADAMAX_optimize(Opti_t * opti)
{
	AdamaxData_t * adamaxdata = (AdamaxData_t*)opti->opti_space;

	adamax_kernel_th11<<<dim3(KERN_DIV(opti->train->mdl->weights, 16), opti->train->sets),dim3(16,1)>>>(
		opti_adamax_alpha, opti_adamax_beta0, opti_adamax_beta1,
		adamaxdata->echopes,
		opti->train->mdl->weights,
		adamaxdata->m_d, adamaxdata->u_d, opti->train->_weight, opti->train->_meand
	);

	adamaxdata->echopes++;
};
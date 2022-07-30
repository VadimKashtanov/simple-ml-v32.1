#include "package/head/optis/optis/adadelta.cuh"

/*
	m = beta0*m + (1 - beta0)*grad(w)^2
	delta_w = - sqrt(v + 1e-8) / sqrt(m + 1e-8)
	v = beta1*v + (1 - beta1)*delta_w^2

	w = delta_w
*/

__global__
void adadelta_kernel_th11(
	float beta0, float beta1,
	uint weights,
	float * m, float *v, float * weight, float * meand)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	float m_tmpt, old_v_tmpt, delta_w, dw;

	//	dw - derivative of w
	//	delta_w - change in w

	if (x < weights) {
		dw = meand[set*weights + x];
		
		m_tmpt = m[set*weights + x];
		old_v_tmpt = v[set*weights + x];

		m_tmpt = beta0*m_tmpt + (1 - beta0)*dw;
		m[set*weights + x] = m_tmpt;

		delta_w = - sqrt(old_v_tmpt + 1e-8) / sqrt(m_tmpt + 1e-8);

		v[set*weights + x] = beta1*old_v_tmpt + (1 - beta1)*delta_w*delta_w;

		weight[set*weights + x] += delta_w;	//le `-` est deja dans le delta_w
	}
};

void ADADELTA_optimize(Opti_t * opti)
{
	AdadeltaData_t * adadeltadata = (AdadeltaData_t*)opti->opti_space;

	adadelta_kernel_th11<<<dim3(KERN_DIV(opti->train->mdl->weights, 16), opti->train->sets),dim3(16,1)>>>(
		opti_adadelta_beta0, opti_adadelta_beta1,
		opti->train->mdl->weights,
		adadeltadata->m_d, adadeltadata->v_d, opti->train->_weight, opti->train->_meand
	);
};
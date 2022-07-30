#include "package/head/optis/optis/rmsprop.cuh"

__global__
void RMSPROP_kernel_th11(
	float alpha, float beta,
	uint weights,
	float * v, float * weight, float * meand)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	float _v, dw;

	if (x < weights) {
		dw = meand[set*weights + x];
		_v = beta * v[set*weights + x] + (1-beta) * pow(dw,2);
		v[set*weights + x] = _v;
		weight[set*weights + x] -= alpha * dw * pow(_v + 1e-8, -0.5);	//eta = 1e-8
	}
};

void RMSPROP_optimize(Opti_t * opti)
{
	rmsprop_kernel_th11<<<dim3(KERN_DIV(opti->train->mdl->weights, 16), opti->train->sets),dim3(16,1)>>>(
		opti_momentum_alpha, opti_momentum_moment,
		opti->train->mdl->weights,
		(float*)opti->opti_space, opti->train->_weight, opti->train->_meand
	);
};
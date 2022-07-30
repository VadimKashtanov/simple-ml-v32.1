#include "package/head/optis/optis/momentum.cuh"

__global__
void momentum_kernel_th11(
	float alpha, float moment,
	uint weights,
	float * v, float * weight, float * meand)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	float _v;

	if (x < weights) {
		_v = moment * v[set*weights + x] - alpha * meand[set*weights + x];
		v[set*weights + x] = _v;
		weight[set*weights + x] += _v;
	}
};

void MOMENTUM_optimize(Opti_t * opti)
{
	momentum_kernel_th11<<<dim3(KERN_DIV(opti->train->mdl->weights, 16), opti->train->sets),dim3(16,1)>>>(
		opti_momentum_alpha, opti_momentum_moment,
		opti->train->mdl->weights,
		(float*)opti->opti_space, opti->train->_weight, opti->train->_meand
	);
};
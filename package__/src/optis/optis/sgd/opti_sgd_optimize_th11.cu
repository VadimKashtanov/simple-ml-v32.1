#include "package/head/optis/optis/sgd/sgd.cuh"

__global__
void sgd_kernel_th11(
	float sgd_alpha,
	uint weights,
	float * weight, float * meand)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	if (x < weights)
		weight[set*weights + x] -= sgd_alpha * meand[set*weights + x];
};

void SGD_optimize(Opti_t * opti)
{
	sgd_kernel_th11<<<dim3(KERN_DIV(opti->train->mdl->weights, 16), opti->train->sets),dim3(16,1)>>>(
		opti_sgd_alpha,
		opti->train->mdl->weights,
		opti->train->_weight, opti->train->_meand
	);
};
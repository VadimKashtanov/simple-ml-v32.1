#include "package/head/optis/optis/momentum.cuh"

void * MOMENTUM_space_mk(Opti_t * opti) {
	float * ret_d;

	SAFE_CUDA(cudaMalloc((void**)&ret_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	SAFE_CUDA(cudaMemset(ret_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	return (void*)ret_d;
};

void MOMENTUM_free(Opti_t * opti) {
	SAFE_CUDA(cudaFree((float*)opti->opti_space))
};
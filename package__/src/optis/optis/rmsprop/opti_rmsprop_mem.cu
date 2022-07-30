#include "package/head/optis/optis/rmsprop.cuh"

void * RMSPROP_space_mk(Opti_t * opti) {
	float * v0_d;//, * v1_d;

	SAFE_CUDA(cudaMalloc((void**)&v0_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	//SAFE_CUDA(cudaMalloc((void**)&v1_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	SAFE_CUDA(cudaMemset(v0_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	//SAFE_CUDA(cudaMemset(v1_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	//RMSprop_data_t * ret = (RMSprop_data_t*)malloc(sizeof(RMSprop_data_t));

	//ret->v0_d = v0_d;
	//ret->v1_d = v1_d;

	//return (void*)ret;
	return (void*)v0_d;
};

void RMSPROP_free(Opti_t * opti) {
	//SAFE_CUDA(cudaFree((RMSprop_data_t*)opti->opti_space->v0_d))
	//SAFE_CUDA(cudaFree((RMSprop_data_t*)opti->opti_space->v1_d))
	//free((RMSprop_data_t*)opti->opti_space);
	SAFE_CUDA(cudaFree((float*)opti->opti_space))
};
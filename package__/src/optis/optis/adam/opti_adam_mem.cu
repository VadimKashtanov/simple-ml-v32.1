#include "package/head/optis/optis/adam.cuh"

void * ADAM_space_mk(Opti_t * opti) {
	float * m_d * v_d;

	SAFE_CUDA(cudaMalloc((void**)&m_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	SAFE_CUDA(cudaMalloc((void**)&v_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	SAFE_CUDA(cudaMemset(m_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	SAFE_CUDA(cudaMemset(v_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	AdamData_t * ret = (AdamData_t*)malloc(sizeof(AdamData_t));

	ret->m_d = m_d;
	ret->v_d = v_d;

	ret->echope = 0;

	return (void*)ret;
};

void ADAM_free(Opti_t * opti) {
	SAFE_CUDA(cudaFree((AdamData_t*)opti->opti_space->m_d))
	SAFE_CUDA(cudaFree((AdamData_t*)opti->opti_space->v_d))
	free((AdamData_t*)opti->opti_space);
};
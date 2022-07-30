#include "package/head/optis/optis/adamax.cuh"

void * ADAMAX_space_mk(Opti_t * opti) {
	float * m_d * u_d;

	SAFE_CUDA(cudaMalloc((void**)&m_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	SAFE_CUDA(cudaMalloc((void**)&u_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	SAFE_CUDA(cudaMemset(m_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	SAFE_CUDA(cudaMemset(u_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	AdamaxData_t * ret = (AdamaxData_t*)malloc(sizeof(AdamaxData_t));

	ret->m_d = m_d;
	ret->u_d = u_d;

	ret->echopes = 0;

	return (void*)ret;
};

void ADAMAX_free(Opti_t * opti) {
	SAFE_CUDA(cudaFree((AdamaxData_t*)opti->opti_space->u_d))
	SAFE_CUDA(cudaFree((AdamaxData_t*)opti->opti_space->m_d))
	free((AdamaxData_t*)opti->opti_space);
};
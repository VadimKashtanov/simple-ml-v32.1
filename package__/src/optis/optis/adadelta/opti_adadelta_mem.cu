#include "package/head/optis/optis/adadelta.cuh"

void * ADADELTA_space_mk(Opti_t * opti) {
	float * m_d * v_d;

	SAFE_CUDA(cudaMalloc((void**)&m_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	SAFE_CUDA(cudaMalloc((void**)&v_d, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	SAFE_CUDA(cudaMemset(m_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))
	SAFE_CUDA(cudaMemset(v_d, 0, sizeof(float) * opti->train->sets * opti->train->mdl->weights))

	AdadeltaData_t * ret = (AdadeltaData_t*)malloc(sizeof(AdadeltaData_t));

	ret->m_d = m_d;
	ret->v_d = v_d;

	return (void*)ret;
};

void ADADELTA_free(Opti_t * opti) {
	SAFE_CUDA(cudaFree((AdadeltaData_t*)opti->opti_space->v_d))
	SAFE_CUDA(cudaFree((AdadeltaData_t*)opti->opti_space->m_d))
	free((AdadeltaData_t*)opti->opti_space);
};
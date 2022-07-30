#include "kernel/head/use.cuh"

Use_t* use_mk(Mdl_t * mdl, Data_t * data) {
	Use_t * ret = (Use_t*)malloc(sizeof(Use_t));

	//	Dependances
	ret->mdl = mdl;
	ret->data = data;

	//	Weights
	SAFE_CUDA(cudaMalloc((void**)&ret->weight_d, sizeof(float) * mdl->weights));
	SAFE_CUDA(cudaMemcpy(ret->weight, mdl->weight, sizeof(float) * mdl->weights, cudaMemcpyHostToDevice));

	//	Vars
	SAFE_CUDA(cudaMalloc((void**)&ret->var_d, sizeof(float) * data->lines * mdl->total));
	//SAFE_CUDA(cudaMemset(ret->var_d, 0, sizeof(float) * data->lines * mdl->total));	//vars have to be set and start from input

	return ret;
};

void use_set_input(Use_t * use) {
	for (uint t=0; t < use->data->lines; t++) {
		SAFE_CUDA(
			cudaMemcpy(
				use->var_d + t*use->mdl->total,
				data->input_d + t*use->mdl->inputs,
				sizeof(float) * use->mdl->inputs,
				cudaMemcpyHostToDevice
			)
		)
	}
};

void use_forward(Use_t * use) {
	for (uint t=0; t < use->data->lines; t++)
		for (uint i=0; i < use->mdl->insts; i++)
			INST_USE[use->mdl->id[i]](use, i, t);
};

void use_free(Use_t * use) {
	SAFE_CUDA(cudaFree(use->var_d));
	SAFE_CUDA(cudaFree(use->weight_d));
	free(use);
};
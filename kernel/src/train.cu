#include "kernel/head/train.cuh"

Train_t* mk_train(Mdl_t * mdl, Data_t * data, uint sets)
{
	Train_t * ret = (Train_t*)malloc(sizeof(Train_t));

	ret->mdl = mdl;
	ret->sets = sets;

	uint lines = data->lines;

	SAFE_CUDA(cudaMalloc((void**)&ret->_weight, sizeof(float) * (mdl->weights*sets)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_var, sizeof(float) * (mdl->total*sets*lines)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_locd, sizeof(float) * (mdl->locds*sets*lines)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_grad, sizeof(float) * (mdl->total*sets*lines)));
	SAFE_CUDA(cudaMalloc((void**)&ret->_meand, sizeof(float) * (mdl->weights*sets)));

	return ret;
};

void train_free(Train_t * train) {
	SAFE_CUDA(cudaFree(train->_weight));
	SAFE_CUDA(cudaFree(train->_var));
	SAFE_CUDA(cudaFree(train->_locd));
	SAFE_CUDA(cudaFree(train->_grad));
	SAFE_CUDA(cudaFree(train->_meand));

	free(train);
};

//	---------------------------------------------------------------------------------------------

static __global__ void kernel_random_weights(uint rnd_seed, uint weights, float * _weight) {
	uint wid = threadIdx.x + blockIdx.x*blockDim.x;
	uint set = threadIdx.y + blockIdx.y*blockDim.y;
	uint pos = set*weights + wid;

	if (wid < weights)
		_weight[pos] = pseudo_randomf(rnd_seed + pos);
};

void train_random_weights(Train_t * train) {
	uint rnd_seed = rand() % 10000;

	kernel_random_weights<<<dim3(KERN_DIV(train->mdl->weights,32), train->sets), dim3(32,1)>>>(
		rnd_seed, train->mdl->weights, train->_weight);
};

//	---------------------------------------------------------------------------------------------

static __global__ void kernel_random_weights_from_mdl(uint rnd_seed, uint weights, float * _weight, float * mdl_weight_d) {
	uint wid = threadIdx.x + blockIdx.x*blockDim.x;
	uint set = threadIdx.y + blockIdx.y*blockDim.y;
	uint pos = set*weights + wid;

	if (wid < weights)
		_weight[pos] = mdl_weight_d[wid] + 0.02*(pseudo_randomf(start_seed + pos) - 0.5);//0.01*(2*(rnd()-0.5)))
};

void train_random_weights_from_mdl(Train_t * train) {
	uint rnd_seed = rand() % 10000;

	float * mdl_weights_d;
	SAFE_CUDA(cudaMalloc((void**)&mdl_weights_d, sizeof(float)*train->mdl->weights));
	SAFE_CUDA(cudaMemcpy(mdl_weights_d, train->mdl->weight, sizeof(float)*train->mdl->weights, cudaMemcpyHostToDevice));

	kernel_random_weights_from_mdl<<<dim3(KERN_DIV(train->mdl->weights, 32), train->sets),dim3(32,1)>>>(
		start_seed, train->mdl->weights, train->_weight, mdl_weights_d);

	SAFE_CUDA(cudaFree(mdl_weights_d));
};

//	---------------------------------------------------------------------------------------------

void train_cpy_ws_to_mdl(Train_t * train, uint set) {
	SAFE_CUDA(cudaMemcpy(
		train->mdl->weight, train->_weight + set*train->mdl->weights,
		sizeof(float)*train->mdl->weights, cudaMemcpyDeviceToHost));
};

//	---------------------------------------------------------------------------------------------

Train_t * extract_to_new_train(Train_t * old, uint amount, uint * set_id) {
	Train_t * new_train = mk_train(old->mdl, old->data, amount);
	
	uint ws = old->mdl->weights;

	for (uint s=0; s < amount; s++)
		SAFE_CUDA(cudaMemcpy(new_train->_weight + s*ws, old->_weight + set_id[s]*ws, sizeof(float)*ws, cudaMemcpyDeviceToDevice))

	return new_train;
};

//	---------------------------------------------------------------------------------------------

static __global__ void kernel_set_input(float * _var, float * input, uint vsize, uint sets, uint inputs) {
	/*uint kid = blockIdx.x,	\
		 time = blockIdx.y,	\	//time == line
		 set = blockIdx.z;
	_var[time*sets*vsize + set*vsize + kid] = input[time*inputs + kid];*/

	uint kid = blockIdx.x,	\
		 time = blockIdx.y;	//time == line
	for (uint set=0; set < sets; set++)
		_var[time*sets*vsize + set*vsize + kid] = input[time*inputs + kid];
};

void train_set_input(Train_t * train) {
	//kernel_set_input<<<dim3(train->mdl->inputs, train->times, train->sets),dim3(1,1,1)>>>(
	//	train->_var, train->data->input_d, train->mdl->total, train->sets, train->mdl->inputs);
	kernel_set_input<<<dim3(train->mdl->inputs, train->times),dim3(1,1)>>>(
		train->_var, train->data->input_d, train->mdl->total, train->sets, train->mdl->inputs);
};

//	---------------------------------------------------------------------------------------------

void train_null_grad_meand(Train_t * train) {
	SAFE_CUDA(cudaMemset(train->_meand, 0, sizeof(float) * train->sets * train->mdl->weights))
	SAFE_CUDA(cudaMemset(train->_grad, 0, sizeof(float) * train->sets * train->data->lines * train->mdl->total))
};

void train_forward(Train_t * train, uint start_seed) {
	for (uint t=0; t < train->data->lines; t++) {
		for (uint i=0; i < train->mdl->insts; i++) {
			INST_FORWARD[train->mdl->id[i]](train, i, t, start_seed);
		}
	}
};

void train_backward(Train_t * train, uint start_seed) {
	for (int t=train->data->lines-1; t >= 0; t--) {
		for (int i=train->mdl->insts-1; i >= 0; i--) {
			INST_BACKWARD[train->mdl->id[i]](train, i, t, start_seed);
		}
	}
};
#include "package/head/gtics/elite/elite.cuh"

static __global__
void kernel_elite_select(
	uint sets, uint ws,
	uint elites, uint portion,
	uint seed,
	float * _old, float * _new)
{
	uint th_x = threadIdx.x + blockIdx.x * blockDim.x;
	uint th_elite = blockIdx.y;

	uint elite_pos, clone_pos;

	if (thx < ws) {
	
		for (uint i=0; i < portion; i++)
		{
			//	Make a new clone and add weights

#define PODIUM ((uint*)const_mem)	/*const_mem is `float*`, so we have to interpret it as `uint*` */

			elite_pos = PODIUM[th_elite]*ws + th_x;
			clone_pos = PODIUM[th_elite + th_elite*portion + i]*ws + th_x;

#undef PODIUM

			_new[clone_pos] = _old[elite_pos] + pseudo_randomf(seed + clone_pos) * 0.1;
		}
	}
}

void gtic_select_elite(Gtic_t * gtic) {
	float * new_weights_d;

	uint sets = gtic->opti->sets;
	uint ws = gtic->opti->mdl->weights;

	SAFE_CUDA(cudaMalloc((void**)&new_weights_d, sizeof(float) * ws * sets));

	//	########### Build podium [best-set-id, 2nd-best-set-id, ... worst-set-id]
	SAFE_CUDA(cudaMemcpyToSymbol(const_mem, gtic->opti->podium, len * sizeof(uint)));	//const_mem is float* type, but it's not a probleme because type doesn't matters, you can (uint*)const_meme, and it compute as an `uint` and not `float`. Float is juste a size and a data structure.
	
	//	########## Launch Cloning
	kernel_elite_select<<<dim3(KERN_DIV(ws,16), elites),dim3(16,1)>>>(
		sets, ws,
		elites, portion,
		rand() % 100000,
		gtic->opti->train->_weight, new_weights_d,
	);

	//	############ Copy new to old in `Train_t` and free the tempt `new_weights_d`
	SAFE_CUDA(cudaMemcpy(gtic->opti->train->_weight, new_weights_d, sizeof(float) * ws * sets, cudaMemcpyDeviceToDevice));
	SAFE_CUDA(cudaFree(new_weights_d));
};
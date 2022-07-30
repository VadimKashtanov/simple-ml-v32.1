#include "pkg_head/insts/gaussfiltre1d/gaussfiltre1d_th11.cuh"

__global__
void gaussfiltre1d_forward_th1x1(
	uint len,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd,
	uint seed,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	float _tmp;

	if (x < len) {
		_tmp = var[time*sets*total + istart + x] + weight[ws*set + wstart + x];
		var[time*sets*total + set*total + ystart + x] = exp(-pow(_tmp,2));
		locd[time*sets*lsize + set*lsize + lstart + x] = -2*_tmp*exp(-pow(_tmp,2));
	}
};
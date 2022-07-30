#include "pkg_head/insts/gaussfiltre1d/gaussfiltre1d_th11.cuh"

__global__
void gaussfiltre1d_backward_th1x1(
	uint len,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;

	float dlds;

	if (x < len) {
		dlds = grad[time*sets*total + set*total + ystart + x] * locd[time*sets*lsize + set*lsize + lstart + x];

		grad[time*sets*total + istart + x] += dlds;
		meand[ws*set + wstart + x] += dlds;
	}
};
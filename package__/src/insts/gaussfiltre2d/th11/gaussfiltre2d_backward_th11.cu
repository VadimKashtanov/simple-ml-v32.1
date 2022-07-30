#include "pkg_head/insts/gaussfiltre2d/gaussfiltre2d_th11.cuh"

__global__
void gaussfiltre2d_backward_th1x1(
	uint X, uint Y,
	uint time,
	uint istart, uint ystart, uint wstart, uint lstart,
	uint total, uint wsize, uint lsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.z;

	float dlds;

	if (x < len) {
		dlds = grad[time*sets*total + set*total + ystart + (y*X+x)] * locd[time*sets*lsize + set*lsize + lstart + (y*X+x)];

		grad[time*sets*total + istart + (y*X+x)] += dlds;
		atomicAdd(meand + ws*set + wstart + x, dlds);
	}
};
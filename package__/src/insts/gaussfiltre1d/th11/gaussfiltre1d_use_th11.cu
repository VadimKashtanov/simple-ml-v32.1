#include "pkg_head/insts/gaussfiltre1d/gaussfiltre1d_th11.cuh"

__global__
void gaussfiltre1d_use_th1x1(
	uint len,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x < len) {
		var[time*total + ystart + x] = exp(-pow(var[time*total + istart + x] + weight[wstart + x],2));
	}
};
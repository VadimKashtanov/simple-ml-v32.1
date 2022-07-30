#include "pkg_head/insts/softmax.cuh"

__global__
void softmax_use_th32(
	uint len,
	uint time,
	uint total,
	uint istart, uint ystart,
	float * var)
{
	uint pos = threadIdx.x;

	if (pos < len) {
		float exped = exp(-var[time*total + istart + pos]);
		__shared__ float sum;
		if (pos == 0) sum = 0;
		__syncthreads();
		atomicAdd(&sum, exped);
		__syncthreads();
		var[time*total + ystart + pos] = exped / sum;
	}
}
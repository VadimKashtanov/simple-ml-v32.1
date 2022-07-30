#include "pkg_head/insts/softmax.cuh"

__global__
void softmax_backward_th32(
	uint len,
	uint time,
	uint total, uint lsize,
	uint istart, uint ystart, uint lstart,
	uint sets,
	float * var, float * grad)
{
	uint pos = threadIdx.x;
	uint set = blockIdx.x;

	if (pos < len) {
		__shared__ float grads[32];
		__shared__ float ys[32];
		__shared__ float this_x_grad[32];

		uint start = time*sets*total + set*total + ystart + pos;

		grads[pos] = grad[start];	//the error
		ys[pos] = var[start];
		this_x_grad[pos] = 0;

		for (uint i=0; i < len; i++)
			atomicAdd(&grads[i], grads[i]*ys[i]*ys[pos]);
		atomicAdd(&this_x_grad[pos], -grads[pos]*pow(ys[pos],2));	//to avoid (if/else in for loop). We just exclude case
		atomicAdd(&this_x_grad[pos], grad[pos]*ys[pos]*(1 - ys[pos]));
		//
		start -= (ystart + pos);
		atomicAdd(&grad[start + istart + pos], this_x_grad[pos]);
	}
}
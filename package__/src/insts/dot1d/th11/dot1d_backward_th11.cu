#include "pkg_head/insts/dot1d.cuh"

__global__
void dot1d_backward_th1x1(
	uint Ax, uint Yx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	/*	Kernel coordinates	*/
	uint _Yx = threadIdx.x + blockIdx.x*blockDim.x, \
		 set = blockIdx.y;

	if (_Yx < Yx) {
		uint Apos = time*sets*total + set*total + input_start;
		uint weight_start = set*wsize + wstart;
		uint Bpos = weight_start + _Yx*Ax;

		float dlds = locd[time*sets*locdsize + set*locdsize + locdstart + _Yx] * grad[time*sets*total + set*total + ystart + _Yx];

		meand[weight_start + Yx*Ax + _Yx] += dlds;

		for (uint i=0; i < Ax; i++) {
			if (pseudo_randomf(Apos*seed) >= drop_rate) {
				atomicAdd(&grad[Apos], dlds * weight[Bpos]);
				atomicAdd(&meand[Bpos], dlds * var[Apos]);
			}
			Apos++;
			Bpos += Yx;
		}
	}
};
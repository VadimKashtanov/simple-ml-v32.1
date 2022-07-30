#include "pkg_head/insts/dot1drecurent/dot1drecurent_th11.cuh"

__global__
void dot1drecurent_backward_th1x1(
	uint Ax, uint At, uint Yx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;
	float dlds;
	uint Apos, Wpos;

	if (y < Yx) {
		dlds = grad[time*sets*total + set*total + ystart + y] * locd[time*locdsize*sets + set*locdsize + locdstart + y];

		Apos = (time-At)*total*sets + set*total + istart;
		Wpos = ws*set + wstart + y;

		for (uint i=0; i < Ax; i++) {
			if (pseudo_randomf(Apos*seed) >= drop_rate) {
				atomicAdd(grad + Apos, weight[Wpos] * dlds);
				atomicAdd(meand + Wpos, var[Apos] * dlds);
			}

			Apos++;
			Wpos += Yx;
		}

		meand[Wpos + Yx] += dlds;
	}
}
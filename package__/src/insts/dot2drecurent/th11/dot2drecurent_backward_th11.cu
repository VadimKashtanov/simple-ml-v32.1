#include "pkg_head/insts/dot2drecurent/dot2drecurent_th11.cuh"

__global__
void dot2drecurent_backward_th1x1(
	uint Ax, uint Ay, uint At, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	uint set = blockIdx.y;

	float dlds;
	uint Apos, Wpos;

	if (y < Yx) {
		dlds = grad[time*sets*total + set*total + ystart + (y*Bx + x)] * locd[time*locdsize*sets + set*locdsize + locdstart + (y*Bx + x)];

		Apos = (time-At)*total*sets + set*total + istart + y*Ax;
		Wpos = ws*set + wstart + y;

		for (uint i=0; i < Ax; i++) {
			if (pseudo_randomf(Apos*seed) >= drop_rate) {
				atomicAdd(grad + Apos, weight[Wpos] * dlds);
				atomicAdd(meand + Wpos, var[Apos] * dlds);
			}

			Apos++;
			Wpos += Bx;
		}

		meand[ws*set + wstart + Bx*Ax + (y*Bx + x)] += dlds;
	}
}
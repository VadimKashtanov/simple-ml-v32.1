#include "pkg_head/insts/dotgaussfiltre2d/dotgaussfiltre2d_th11.cuh"

__global__
void dotgaussfiltre2d_backward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
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

	float _grad, dlds;

	if (x < Bx && y < Ay) {

		_grad = grad[time*sets*total + set*total + ystart + (y*Bx+x)];

		for (uint i=0; i < Ax; i++) {
			apos = time*sets*total + set*total + istart + y*Ax + i;

			if (pseudo_randomf(apos*seed) >= drop_rate) {
				ppos = ws*set + wstart + i*Bx + x;

				dlds = locd[time*sets*lsize + set*lsize + lstart + Ax*(y*Bx+x) + i] * _grad;

				atomicAdd(grad + apos, dlds);
				atomicAdd(meand + ppos, dlds);
			}
		}
	}
}
	/*uint x = threadIdx.x + blockIdx.x * blockDim.x;		//	Ax
	uint y = threadIdx.y + blockIdx.y * blockDim.y;		//	Ay
	uint set = blockIdx.z;

	uint apos = time*sets*total + set*total + istart + y*Ax + x;

	if (x < Ax && y < Ay && pseudo_randomf(apos)) {
		float input_value = var[apos];
		float __grad_input = 0;
		float wpos;

		for (uint i=0; i < Bx; i++) {
			wpos = ws*set + wstart + x*Bx + x;

			grad[ws*set + wstart + ] += input_value * 
			__grad_input += weight[] * ;
		};
	};
};
#include "pkg_head/insts/dotgaussfiltre2d/dotgaussfiltre2d_th11.cuh"

__global__
void dotgaussfiltre2d_forward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint istart, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockIdx.y;
	uint set = blockIdx.z;

	float _sum, _tmp;

	if (x < Bx && y < Ay) {

		_sum = 0;

		for (uint i=0; i < Ax; i++) {
			apos = time*sets*total + set*total + istart + y*Ax + i;

			if (pseudo_randomf(apos*seed) >= drop_rate) {
				ppos = ws*set + wstart + i*Bx + x;

				_tmp = var[apos] + weight[ppos];
						
				_sum += exp(-pow(_tmp, 2));

				locd[time*sets*lsize + set*lsize + lstart + Ax*(y*Bx+x) + i] = -2*(_tmp)*exp(-(_tmp)**2);
			} /*else {
			var[apos] == 0;
			}*/
		}
		var[time*sets*total + set*total + ystart + (y*Bx+x)] = _sum;
	}
};
#include "pkg_head/insts/dotgaussfiltre2d/dotgaussfiltre2d_th11.cuh"

__global__
void dotgaussfiltre2d_use_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockIdx.y;

	float _tmp;

	if (x < Bx && y < Ay) {

		for (uint y=0; y < Ay; y++) {
			for (uint x=0; x < Bx; x++) {
				_tmp = 0;

				for (uint i=0; i < Ax; i++) {

					apos = time*total + istart + y*Ax + i;
					ppos = wstart + i*Bx + x;
					
					_tmp += exp(-pow(var[apos] + weight[ppos],2));
				}

				var[time*total + ystart + (y*Bx+x)] = _tmp;
			}
		}

	}
};
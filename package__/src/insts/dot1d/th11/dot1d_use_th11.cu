#include "pkg_head/insts/dot1d.cuh"

__global__
void dot1d_use_th1x1(
	uint Ax, uint Yx,
	uint activ,
	uint time,
	uint total,
	uint input_start, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint _Yx = threadIdx.x + blockIdx.x*blockDim.x;

	if (_Yx < Yx) {
		uint Apos = time*total + input_start;
		uint Bpos = wstart + _Yx*Ax;	//	Dot1d does not store W as Dot2d       in fact Dot2D.T = Dot1d  (it would be better to change it)

		float sum = 0;
		for (uint i=0; i < Ax; i++) {
			sum += var[Apos] * weight[Bpos];
			Apos++;
			Bpos += Yx;
		}
		sum += weight[wstart + Yx*Ax + _Yx];

		if (activ == 0) sum = 1 / (1 + exp(-sum));
		else if (activ == 1) sum = tanh(sum);
		else if (activ == 2) sum = exp(-pow(sum,2));
		else sum = sum*(sum > 0);

		var[time*total + ystart + _Yx] = sum;
	}
};
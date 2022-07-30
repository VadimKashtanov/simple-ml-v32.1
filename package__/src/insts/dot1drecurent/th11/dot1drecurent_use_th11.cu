#include "pkg_head/insts/dot1drecurent/dot1drecurent_th11.cuh"

__global__
void dot1drecurent_use_th1x1(
	uint Ax, uint At, uint Yx,
	uint activ,
	uint time,
	uint total,
	uint input_start, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	float _tmp;

	uint Apos, Wpos;

	if (y < Yx) {

		_tmp = 0;

		Apos = (time-At)*total + istart;
		Wpos = wstart + y;

		for (uint i=0; i < Ax; i++) {
			_tmp += var[Apos] * weight[Wpos];
			Apos++;
			Wpos += Yx;
		}

		_tmp += weight[Wpos + Yx];	//==wstart + Ax*Yx + y      car on a deja +y, et on a += Ax*Yx (for i<Ax) {+=Yx}

		if (activ == 0)	_tmp = 1 / (1 + exp(-_tmp));
		else if (activ == 1) _tmp = tanh(_tmp);
		else if (activ == 2) _tmp = exp(-_tmp*_tmp);
		else _tmp *= (tmp > 0);

		var[time*total + ystart + y] = _tmp;
	}
};
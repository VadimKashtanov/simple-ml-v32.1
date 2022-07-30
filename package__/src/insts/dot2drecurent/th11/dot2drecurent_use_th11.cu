#include "pkg_head/insts/dot2drecurent/dot2drecurent_th11.cuh"

__global__
void dot2drecurent_use_th1x1(
	uint Ax, uint Ay, uint At, uint Bx,
	uint activ,
	uint time,
	uint total,
	uint istart, uint ystart, uint wstart,
	float * var, float * weight)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	float _tmp;

	uint Apos, Wpos;

	if (y < Ay && x < Bx) {

		_tmp = 0;

		Apos = (time-At)*total + istart + y*Ax;
		Wpos = wstart + y;

		for (uint i=0; i < Ax; i++) {
			_tmp += var[Apos] * weight[Wpos];
			Apos++;
			Wpos += Bx;
		}

		_tmp += weight[wstart + Bx*Ax + (y*Bx + x)];	//==wstart + Ax*Yx + y      car on a deja +y, et on a += Ax*Yx (for i<Ax) {+=Yx}

		if (activ == 0)	_tmp = 1 / (1 + exp(-_tmp));
		else if (activ == 1) _tmp = tanh(_tmp);
		else if (activ == 2) _tmp = exp(-_tmp*_tmp);
		else _tmp *= (tmp > 0);

		var[time*total + ystart + (y*Bx + x)] = _tmp;
	}
};
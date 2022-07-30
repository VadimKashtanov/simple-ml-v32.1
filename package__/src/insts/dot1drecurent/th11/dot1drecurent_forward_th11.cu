#include "pkg_head/insts/dot1drecurent/dot1drecurent_th11.cuh"

__global__
void dot1drecurent_forward_th1x1(
	uint Ax, uint At, uint Yx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets)
{
	uint y = threadIdx.x + blockIdx.x * blockDim.x;
	uint set = blockIdx.y;
	float _tmp, _locd;

	uint Apos, Wpos;

	if (y < Yx) {
		_tmp = 0;

		Apos = (time-At)*total*sets + set*total + istart;
		Wpos = ws*set + wstart + y;

		for (uint i=0; i < Ax; i++) {
			if (pseudo_randomf(Apos*seed) >= drop_rate)
				_tmp += var[Apos] * weight[Wpos];
			Apos++;
			Wpos += Yx;
		}

		_tmp += weight[Wpos + Yx];	//==wstart + Ax*Yx + y      car on a deja +y, et on a += Ax*Yx (for i<Ax) {+=Yx}

		if (activ == 0)	{
			_tmp = 1 / (1 + exp(-_tmp));
			_locd = _tmp * (1 - _tmp);

		} else if (activ == 1) {
			_tmp = tanh(_tmp);
			_locd = 1 - _tmp*_tmp;

		} else if (activ == 2) {
			_locd = -2*_tmp;
			_tmp = exp(-_tmp*_tmp);
			_locd = _tmp * _locd;
		} else {
			_locd = (tmp > 0);
			_tmp = _tmp * _locd;
		}

		var[time*sets*total + set*total + ystart + y] = _tmp;
		locd[time*locdsize*sets + set*locdsize + locdstart + y] = _locd;
	}
}
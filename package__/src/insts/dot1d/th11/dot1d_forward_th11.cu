#include "pkg_head/insts/dot1d.cuh"

__global__
void dot1d_forward_th1x1(
	uint Ax, uint Yx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets)
{
	uint _Yx = threadIdx.x + blockIdx.x*blockDim.x, \
		 set = blockIdx.y;

	if (_Yx < Yx) {

		uint Apos = time*sets*total + set* + input_start;
		uint weight_start = set*wsize + wstart;
		uint Bpos = weight_start + _Yx*Ax;

		float sum = 0;
		for (uint i=0; i < Ax; i++) {
			if (pseudo_randomf(Apos*seed) >= drop_rate)
				sum += var[Apos] * weight[Bpos];
			Apos++;
			Bpos += Yx;
		}
		sum += weight[weight_start + Yx*Ax + _Yx];
		
		float __locd;

		if (activ == 0) {
			sum = 1 / (1 + exp(-sum));
			__locd = sum*(1 - sum);	//f'(x) = f(x)(1 - f(x))
		} else if (activ == 1) {
			sum = tanh(sum);
			__locd = 1 - sum*sum;	//f'(x) = 1 - tanh(x)^2
		} else if (activ == 2) {
			__locd = sum;
			sum = exp(-pow(sum,2));
			__locd = -2*__locd*sum;	//f'(x) = -2x*e^(-x^2)
		} else  if (activ == 3) {
			__locd = (sum > 0);
			sum = sum*__locd;
		}

		var[time*sets*total + set*total + ystart + _Yx] = sum;		//same assembler than putting it in if/else structure
		locd[time*sets*locdsize + set*locdsize + locdstart + _Yx] = __locd;
	}
};

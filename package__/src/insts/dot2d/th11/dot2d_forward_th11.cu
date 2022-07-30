#include "pkg_head/insts/dot2d.cuh"

__global__
void dot2d_forward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,
	uint linesize, uint wsize, uint locdsize,
	float * var, float * weight, float * locd,
	uint seed, float drop_rate,
	uint sets)
{
	uint Yx = threadIdx.x + blockIdx.x*blockDim.x,	\
		 Yy = threadIdx.y + blockIdx.y*blockDim.y,	\
		 set = threadIdx.z + blockIdx.z*blockDim.z;

	uint Apos = time*sets*linesize + set*linesize + input_start + Yy*Ax;
	uint weight_start = set*wsize + wstart;
	uint Bpos = weight_start + Yx;

	float sum = 0;
	for (uint i=0; i < Ax; i++) {
		if (pseudo_randomf(Apos + seed) >= drop_rate)
			sum += var[Apos] * weight[Bpos];
		Apos++;
		Bpos += Bx;
	}
	sum += weight[weight_start + Bx*Ax + Yy*Bx + Yx];
	
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
	} else {
		__locd = (sum >= 0);
		sum = sum*__locd;
	}

	var[time*sets*linesize + set*linesize + ystart + Yy*Bx + Yx] = sum;		//same assembler than putting it in if/else structure
	locd[time*sets*locdsize + set*locdsize + locdstart + Yy*Bx + Yx] = __locd;
};
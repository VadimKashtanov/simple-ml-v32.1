#include "pkg_head/insts/dot2d.cuh"

__global__
void dot2d_backward_th1x1(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	uint Yx = threadIdx.x + blockIdx.x*blockDim.x,	\
		 Yy = threadIdx.y + blockIdx.y*blockDim.y,	\
		 set = threadIdx.z + blockIdx.z*blockDim.z;

	uint Apos = time*sets*total + set*total + input_start + Yy*Ax;
	uint weight_start = set*wsize + wstart;
	uint Bpos = weight_start + Yx;

	uint Y_pos = Yy*Bx + Yx;

	float dlds = grad[time*sets*total + set*total + ystart + Y_pos] * locd[time*sets*locdsize + set*locdsize + locdstart + Y_pos];

	meand[weight_start + Bx*Ax + Yy*Bx + Yx] += dlds;

	for (uint i=0; i < Ax; i++) {
		if (pseudo_randomf(Apos + seed) >= drop_rate) {
			atomicAdd(&grad[Apos], dlds * weight[Bpos]);
			atomicAdd(&meand[Bpos], dlds * var[Apos]);
		}
		Apos++;
		Bpos += Bx;
	}
};

//=============================================================================================

__global__
void dot2d_backward_th1x1_bias(
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	/*	Kernel coordinates	*/
	uint _Yx = blockIdx.x + blockIdx.x*blockDim.x,	\
		 _Yy = blockIdx.y + blockIdx.y*blockDim.y,	\
		 set = blockIdx.z + blockIdx.z*blockDim.z;

	/*	Train_t starts */
	uint time_sets = time*sets + set;
	uint y_pos = _Yy*Bx + _Yx;
	uint trt_out = time_sets*total + ystart + y_pos;
	uint trt_locd = time_sets*locdsize + locdstart + y_pos;

	meand[set*wsize + wstart + Ax*Bx + _Yy*Bx + _Yx] += locd[trt_locd] * grad[trt_out];
};

__global__
void dot2d_backward_th1x1_input(		//grad input = dLdS @ W.T
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,	//in a mdl line
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	/*	Kernel coordinates	*/
	uint _Ax = blockIdx.x + blockIdx.x*blockDim.x,	\
		 _Ay = blockIdx.y + blockIdx.y*blockDim.y,	\
		 set = blockIdx.z + blockIdx.z*blockDim.z;

	/*	Train_t starts */
	uint time_sets = time*sets + set;
	uint trt_out = time_sets*total + ystart + _Ay*Bx;
	uint trt_locd = time_sets*locdsize + locdstart + _Ay*Bx;
	uint trt_A = time_sets*total + input_start + _Ay*Ax + _Ax;
	uint trt_B = set*wsize + wstart + _Ax*Bx;

	if (pseudo_randomf(trt_A + seed) >= drop_rate)
	{
		float tmp = 0;

		for (uint i=0; i < Bx; i++)
			tmp += weight[trt_B + i] * (locd[trt_locd + i] * grad[trt_out + i]);

		grad[trt_A] += tmp;
	}
}

__global__
void dot2d_backward_th1x1_weight(		//grad weight = input.T @ dLdS
	uint Ax, uint Ay, uint Bx,
	uint activ,
	uint time,
	uint input_start, uint ystart, uint wstart, uint locdstart,
	uint total, uint wsize, uint locdsize,
	float * var, float * weight, float * locd, float * grad, float * meand,
	uint seed, float drop_rate,
	uint sets)
{
	/*	Kernel coordinates	*/
	uint _Bx = blockIdx.x + blockIdx.x*blockDim.x,	\
		 _By = blockIdx.y + blockIdx.y*blockDim.y,	\
		 set = blockIdx.z + blockIdx.z*blockDim.z;

	/*	Train_t starts */
	uint time_sets = time*sets + set;
	uint trt_out = time_sets*total + ystart + _Bx;
	uint trt_locd = time_sets*locdsize + locdstart + _Bx;
	uint trt_A = time_sets*total + input_start + _By;
	uint trt_B = set*wsize + wstart + _By*Bx + _Bx;

	float tmp = 0;

	for (uint i=0; i < Ay; i++) {
		if (pseudo_randomf((trt_A + i*Ax) + seed) >= drop_rate)
			tmp += var[trt_A + i*Ax] * (locd[trt_locd + i*Bx] * grad[trt_out + i*Bx]);
	}

	meand[trt_B] += tmp;
};
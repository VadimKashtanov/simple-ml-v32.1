#include "pkg_head/insts/kconvl33samepool22max.cuh"

void kconvl33samepool22max_use_const_MemCpyToSymbol(float * arr, uint len) {
	SAFE_CUDA(cudaMemcpyToSymbol(
		const_mem, 
		arr,
		len * sizeof(float)
	))
};

__device__ static
inline float max_of_4(float a, float b, float c, float d) {
	float max = a;
	if (b > a) max = b;
	if (c > max) max = c;
	if (d > max) max = d;
	return max;
};

__global__
void kconvl33samepool22max_use_const_th1x1(
	uint n0, uint n1, uint Ax, uint Ay,
	uint activ,							
	uint time,
	uint total, uint wsize,
	uint istart, uint wstart, uint ystart,
	float * var, float * weight)
{
	uint out_x = threadIdx.x + blockIdx.x*blockDim.x,	\	//+1 because we don't compute border of output
		 out_y = threadIdx.y + blockIdx.y*blockDim.y;		//+1 it's an usefull approximation
	uint _n1   = threadIdx.z + blockIdx.z*blockDim.z;

	//ou out_x <= Ax/2-1
	if (out_x < Ax/2 && out_y < Ay/2) {	//car il y aura des truc en trop (normalement c'est divisible par 2)
		uint _Ax = out_x * 2;
		uint _Ay = out_y * 2;

		uint kstart = _n1*9*n0;
		uint istart = time*total + istart;

		float _00=0, _10=0, _01=0, _11=0;
		uint cond;

		float _image_value;

		int ximg, yimg;
		uint kpos;

		for (uint _n0=0; _n0 < n0; _n0++) {
			//Iterate thought Image
			for (uint y=0; y < 4; y++) {
				for (uint x=0; x < 4; x++) {
					yimg = _Ay+y-1;
					ximg = _Ax-1+x;

					if (ximg>=0 && ximg<Ax && yimg>=0 && yimg<Ay) {
						kpos = kstart + _n0*9 + y*3 + x;

						_image_value = var[istart + _n0*Ax*Ay + yimg*Ax + ximg];

						if (_image_value != 0.0) {
							//
							cond = x<3 && y<3;
							//_00 += _image_value * const_mem[cond*(kpos)]*cond;
							if (cond) {
								_00 += _image_value * const_mem[kpos];
							}
							
							cond = x>0 && y<3;
							//_10 += _image_value * const_mem[cond*(kpos-1)]*cond;
							if (cond) {
								_10 += _image_value * const_mem[kpos-1];
							}
							
							cond = x<3 && y>0;
							//_01 += _image_value * const_mem[cond*(kpos-3)]*cond;
							if (cond) {
								_01 += _image_value * const_mem[kpos-3];
							}
							
							cond = x>0 && y>0;
							//_11 += _image_value * const_mem[cond*(kpos-4)]*cond;
							if (cond) {
								_11 += _image_value * const_mem[kpos-4];
							}
						}
					}
				}
			}

			//
			//istart = _n0*Ax*Ay;
			//istart += Ax*Ay;
			//kstart += 9;
		}

		//	Bias
		uint bias = wstart + n0*n1*9 + _n1*Ax*Ay + _Ay*Ax + _Ax;

		_00 += weight[bias   	   ];
		_10 += weight[bias + 1 	   ];
		_01 += weight[bias + Ax	   ];
		_11 += weight[bias + Ax + 1];

		if (activ == 0) {
			_00 = 1 / (1 + exp(-_00));
			_10 = 1 / (1 + exp(-_10));
			_01 = 1 / (1 + exp(-_01));
			_11 = 1 / (1 + exp(-_11));
		} else if (activ == 1) {
			_00 = tanh(_00);
			_10 = tanh(_10);
			_01 = tanh(_01);
			_11 = tanh(_11);
		} else if (activ == 2) {
			_00 = exp(-_00*_00);
			_10 = exp(-_10*_10);
			_01 = exp(-_01*_01);
			_11 = exp(-_11*_11);
		} else {
			_00 = _00*(_00 >= 0);
			_10 = _10*(_10 >= 0);
			_01 = _01*(_01 >= 0);
			_11 = _11*(_11 >= 0);
		}

		var[time*total + ystart + _n1*(Ax*Ay/4) + out_y*(Ax/2) + out_x] = max_of_4(_00, _10, _01, _11);
	}
};
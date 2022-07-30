#include "pkg_head/insts/kconvl33samepool22max.cuh"

void kconvl33samepool22max_train_const_MemCpyToSymbol(float * arr, uint len) {
	SAFE_CUDA(cudaMemcpyToSymbol(
		const_mem,
		arr,
		len * sizeof(float)
	))
};			//cudaMemcpyToSymbol have to be in same file, as kernels that will use this const_mem
			//__constant__ have to be declared in one .cuh
			// pourquoi ?
			// nvcc pas parfait

static __device__
float activate(float x, uint activ) {
	if (activ == 0) return 1 / (1 + exp(-x));
	else if (activ == 1) return tanh(x);
	else if (activ == 2) return exp(-x*x);
	else return x * (x >= 0);
};

static __device__
float max_of_4(float a, float b, float c, float d, float * max_id) {
	uint _max_id = 0;
	float max = a;
	if (b > a) {
		max = b;
		_max_id = 1; 
	}
	if (c > max) {
		max = c;
		_max_id = 2;
	}
	if (d > max) {
		max = d;
		_max_id = 3;
	}
	*max_id = _max_id;
	return max;
};

static __device__
float compute_locd(float a, float x, uint activ) {
	if (activ == 0) return a * (1 - a);
	else if (activ == 1) return 1 - a*a;
	else if (activ == 2) return -2*x*a;
	else return x >= 0;
};

__global__
void kconvl33samepool22max_forward_const_th1x1(
	uint n0, uint n1, uint Ax, uint Ay,
	uint activ,							
	uint time,
	uint total, uint wsize, uint lsize,
	uint istart, uint wstart, uint ystart, uint lstart,
	uint seed, float drop_rate,
	uint set, uint sets,
	float * var, float * weight, float * locd)
{
	uint out_x = threadIdx.x + blockIdx.x*blockDim.x,	\	//+1 because we don't compute border of output
		 out_y = threadIdx.y + blockIdx.y*blockDim.y;		//+1 it's an usefull approximation
	uint _n1   = threadIdx.z + blockIdx.z*blockDim.z;

	if (out_x < Ax/2 && out_y < Ay/2 && _n1 < n1) {

		int _Ax = out_x * 2;
		int _Ay = out_y * 2;

		uint istart = time*sets*total + set*total + istart;

		float _00=0, _10=0, _01=0, _11=0;

		float _image_value;

		int ximg, yimg, imgpos;
		uint kpos;

		for (uint _n0=0; _n0 < n0; _n0++) {
			//Iterate thought All accesible pixels from the _00,_10,_01,_11
			for (uint y=0; y < 4; y++) {
				for (uint x=0; x < 4; x++) {
					yimg = _Ay+y-1;
					ximg = _Ax-1+x;
					imgpos = istart + _n0*Ax*Ay + yimg*Ax + ximg;

					if (ximg>=0 && ximg<Ax && yimg>=0 && yimg<Ay && pseudo_randomf(imgpos + seed) >= drop_rate) {
						kpos = _n1*9*n0 + _n0*9 + y*3 + x;

						_image_value = var[imgpos];
						
						if (_image_value != 0.0) {
							if (x<3 && y<3)
								_00 += _image_value * const_mem[kpos-0-0];
							
							if (x>0 && y<3)
								_10 += _image_value * const_mem[kpos-1-0];
							
							if (x<3 && y>0)
								_01 += _image_value * const_mem[kpos-0-3];
							
							if (x>0 && y>0)
								_11 += _image_value * const_mem[kpos-1-3];
						}
					}
				}
			}
			/*for (int i=-1; i < 2; i++) {
				for (int j=-1; j < 2; j++) {
					_image_value = const_mem[_n1*9*n0 + _n0*9 + (i+1)*3 + (j+1)];
					if (_Ax + j >= 0 && _Ay + i >= 0)
						_00 += _image_value * var[istart + _n0*Ax*Ay + (_Ay + i)*Ax + (_Ax + j)];

					if (_Ax + j + 1 < Ax && _Ay + i >= 0 )
						_10 += _image_value * var[istart + _n0*Ax*Ay + (_Ay + i)*Ax + (_Ax + j + 1)];

					if (_Ax + j >=0 && _Ay + i + 1 < Ay)
						_01 += _image_value * var[istart + _n0*Ax*Ay + (_Ay + i + 1)*Ax + (_Ax + j)];

					if (_Ax + j + 1 < Ax && _Ay + i + 1 < Ay)
						_11 += _image_value * var[istart + _n0*Ax*Ay + (_Ay + i + 1)*Ax + (_Ax + j + 1)];
				}
			}*/
		}

		//	Bias
		uint bias = set*wsize + wstart + n0*n1*9 + _n1*Ax*Ay + _Ay*Ax + _Ax;

		_00 += weight[bias   	   ];
		_10 += weight[bias + 1 	   ];
		_01 += weight[bias + Ax	   ];
		_11 += weight[bias + Ax + 1];

		float __locd, max, max_id;
		float a_00, a_10, a_01, a_11;

		a_00 = activate(_00, activ);
		a_10 = activate(_10, activ);
		a_01 = activate(_01, activ);
		a_11 = activate(_11, activ);

		max = max_of_4(a_00, a_10, a_01, a_11, &max_id);

		if (max_id == 0) __locd = compute_locd(a_00, _00, activ);
		else if (max_id == 1) __locd = compute_locd(a_10, _10, activ);
		else if (max_id == 2) __locd = compute_locd(a_01, _01, activ);
		else if (max_id == 3) __locd = compute_locd(a_11, _11, activ);

		//printf("%f, %f, %f, %f = %f\n", a_00, a_10, a_01, a_11, max_id);

		//									*2 is because size is output*2 (__locd, max_id)
		uint this_y_pixel_locd = time*sets*lsize + set*lsize + lstart + _n1*2*(Ax*Ay/4) + out_y*2*(Ax/2) + 2*out_x;

		locd[this_y_pixel_locd	  ] = __locd;
		locd[this_y_pixel_locd + 1] = max_id;

		var[time*sets*total + set*total + ystart + _n1*(Ax*Ay/4) + out_y*(Ax/2) + out_x] = max;
	}
};
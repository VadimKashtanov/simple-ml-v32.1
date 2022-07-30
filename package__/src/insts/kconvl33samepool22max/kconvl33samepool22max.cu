#include "pkg_head/insts/kconvl33samepool22max/kconvl33samepool22max.cuh"

void kconvl33samepool22max_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
	if (param[0]%2!=0) raise(SIGINT);
	if (param[1]%2!=0) raise(SIGINT);
	if (param[2] == 0) raise(SIGINT);
	if (param[3] == 0) raise(SIGINT);
	if (param[4] >= 4) raise(SIGINT);
	if (param[9] >100) raise(SIGINT);
}

static float max_4(float _00, float _10, float _01, float _11) {
	float max = _00;
	if (max < _10) max = _10;
	if (max < _01) max = _01;
	if (max < _11) max = _11;
	return max;
};

static float activate(float x, uint activ) {
	if (activ == 0) return 1 / (1 + exp(-x));
	else if (activ == 1) return tanh(x);
	else if (activ == 2) return exp(-x*x);
	else return x * (x >= 0);
};

void kconvl33samepool22max_cpu_call(
	Cpu_t * cpu, uint inst, uint time)
{
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 n0=mdl->param[inst][2],			\
		 n1=mdl->param[inst][3],			\
		 activ=mdl->param[inst][4],		\
		 istart=mdl->param[inst][5],	\
		 ystart=mdl->param[inst][6],		\
		 wstart=mdl->param[inst][7];

	float * var = cpu->var;
	float * weight = mdl->weight;

	uint Yx=Ax/2, Yy=Ay/2;

	float _00, _01, _10, _11;
	float __w;
	uint bias;

	int __y, __x;

	for (uint _n1=0; _n1 < n1; _n1++) {
		for (uint y=0; y < Yy; y++) {
			for (uint x=0; x < Yx; x++) {
				_00 = 0;
				_10 = 0;
				_01 = 0;
				_11 = 0;
				for (uint _n0=0; _n0 < n0; _n0++) {
					for (uint _x=0; _x < 3; _x++) {
						for (uint _y=0; _y < 3; _y++) {
							__w = weight[wstart + _n1*9*n0 + _n0*9 + _y*3 + _x];

							//(0,0)
							__y = y*2+_y-1;
							__x = x*2+_x-1;
							if (__y >=0 && __x >= 0)
								_00 += var[time*mdl->total + istart + __y*Ax + __x] * __w;

							//(1,0)
							__y = y*2+_y-1;
							__x = x*2+_x;
							if (__y >= 0 && __x < Ax)
								_10 += var[time*mdl->total + istart + __y*Ax + __x] * __w;

							//(0,1)
							__y = y*2+_y;
							__x = x*2+_x-1;
							if (__y < Ay && __x >= 0)
								_01 += var[time*mdl->total + istart + __y*Ax + __x] * __w;

							//(1,1)
							__y = y*2+_y;
							__x = x*2+_x;
							if (__y < Ay && __x < Ax)
								_11 += var[time*mdl->total + istart + __y*Ax + __x] * __w;
						}
					}
				}
				
				bias = wstart + 9*n1*n0 + _n1*Ax*Ay;
				_00 = activate(_00 + weight[bias + y*2*Ax + x*2], activ);			//bias tensor is same size as X, because kconvl is same size as X
				_10 = activate(_10 + weight[bias + y*2*Ax + x*2 + 1], activ);		//bias is added to .k
				_01 = activate(_01 + weight[bias + (y*2+1)*Ax + x*2], activ);		//and after that polled
				_11 = activate(_11 + weight[bias + (y*2+1)*Ax + x*2 + 1], activ);	//

				//
				var[time*mdl->total + ystart + _n1*Yx*Yy + y*Yx + x] = max_4(_00, _10, _01, _11);
			}
		}
	}
}

void kconvl33samepool22max_use_call(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;
	
	uint n0=mdl->param[inst][2];
	uint n1=mdl->param[inst][3];

	if (n0*n1*9 < MAX_CONST_FLOATS) {
		kconvl33samepool22max_use_call_mode_th11(use, inst, time);
	} else {
		ERR("n0*n1*9 is more than const meme.");
	}
}

void kconvl33samepool22max_forward_call(Train_t * train, uint inst, uint time, uint start_seed)
{
	Mdl_t * mdl = train->mdl;

		//Xxlen=mdl->param[inst][0],			
		//Xylen=mdl->param[inst][1],			
	uint n0=mdl->param[inst][2];
	uint n1=mdl->param[inst][3];				
		// activ=mdl->param[inst][4],			
		// input_start=mdl->param[inst][5];
		// ystart=mdl->param[inst][6];		
		// wstart=mdl->param[inst][7];			
		// locdstart=mdl->param[inst][8];		
		// drop_rate_int=mdl->param[inst][9];

	//uint Yxlen = Xxlen/2,	\
	//	 Yylen = Xylen/2;

	if (n0*n1*9 < MAX_CONST_FLOATS) {
		kconvl33samepool22max_forward_call_mode_th11(train, inst, time, start_seed);
		//forward_const_th11(train, inst, time, start_seed);
	} else {
		ERR("n0*n1*9 is more than const meme.");
	}
};

void kconvl33samepool22max_backward_call(Train_t * train, uint inst, uint time, uint start_seed)
{
	Mdl_t * mdl = train->mdl;

		//Xxlen=mdl->param[inst][0],			
		//Xylen=mdl->param[inst][1],			
	uint n0=mdl->param[inst][2];
	uint n1=mdl->param[inst][3];				
		// activ=mdl->param[inst][4],			
		// input_start=mdl->param[inst][5];
		// ystart=mdl->param[inst][6];		
		// wstart=mdl->param[inst][7];			
		// locdstart=mdl->param[inst][8];		
		// drop_rate_int=mdl->param[inst][9];

	//uint Yxlen = Xxlen/2,	\
	//	 Yylen = Xylen/2;

	if (n0*n1*9 < MAX_CONST_FLOATS) {
		kconvl33samepool22max_backward_call_mode_th11(train, inst, time, start_seed);
	} else {
		ERR("n0*n1*9 is more than const meme.");
	}
};
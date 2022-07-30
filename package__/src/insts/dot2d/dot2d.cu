#include "pkg_head/insts/dot2d/dot2d.cuh"

void dot2d_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
	if (param[2] == 0) raise(SIGINT);
	if (param[8] >100) raise(SIGINT);
	if (param[3] >= 4) raise(SIGINT);
};

void dot2d_cpu_call(
	Cpu_t * cpu, uint inst, uint time)
{
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],			\
		 Ay=mdl->param[inst][1],			\
		 Bx=mdl->param[inst][2],			\
		 activ=mdl->param[inst][3],			\
		 input_start=mdl->param[inst][4],	\
		 ystart=mdl->param[inst][5],		\
		 wstart=mdl->param[inst][6];

	float * var = cpu->var;
	float * weight = mdl->weight;

	float _tmp;
	uint _inp, _w;
	
	for (uint x=0; x < Bx; x++) {
		for (uint y=0; y < Ay; y++) {
			_tmp = 0;

			//Scalar product of 2 vectors in input (A) and weight (B)
			_inp = time*mdl->total + input_start + Ax*y;
			_w = wstart + x;
			for (uint i=0; i < Ax; i++) {
				_tmp += var[_inp + i] * weight[_w + i*Bx];
			}

			//Adding bias
			_tmp = _tmp + weight[wstart + Bx*Ax + y*Bx + x];

			//Activation
			if (activ == 0) _tmp = 1 / (1 + exp(-_tmp));
			else if (activ == 1) _tmp = tanh(_tmp);
			else if (activ == 2) _tmp = exp(-_tmp*_tmp);
			else _tmp = _tmp * (_tmp >= 0);

			//Write it to Y
			var[time*mdl->total + ystart + y*Bx + x] = _tmp;
		}
	}
};

void dot2d_use_call(Use_t * use, uint inst, uint time) {
	dot2d_use_call_mode_th11(use, inst, time);
};

void dot2d_forward_call(Train_t * train, uint inst, uint time, uint start_seed) {
	dot2d_forward_call_mode_th11(train, inst, time, start_seed);
};

void dot2d_backward_call(Train_t * train, uint inst, uint time, uint start_seed) {
	dot2d_backward_call_mode_th11(train, inst, time, start_seed);
};
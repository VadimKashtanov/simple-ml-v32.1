#include "pkg_head/insts/dot1d/dot1d.cuh"

void dot1d_check(uint * param) {
	//>0 <==> >= 1
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
	if (param[7] >100) raise(SIGINT);
	if (param[2] >= 4) raise(SIGINT);
};

void dot1d_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],				\
		 Yx=mdl->param[inst][1],				\
		 activ=mdl->param[inst][2],			\
		 input_start=mdl->param[inst][3],		\
		 ystart=mdl->param[inst][4],			\
		 wstart=mdl->param[inst][5];

	float * var = cpu->var;
	float * weight = mdl->weight;

	float _tmp;
	uint _inp=time*mdl->total + input_start,	\
		 _w=wstart;
	
	for (uint y=0; y < Yx; y++) {
		_tmp = 0;
		//Scalar product of 2 vectors in input (A) and weight (B)
		for (uint i=0; i < Ax; i++)
			_tmp += var[_inp + i] * weight[_w + y*Yx + i];//weight[_w + i*Yx];

		//Adding bias
		_tmp += weight[wstart + Ax*Yx + x];
		
		//Activation
		if (activ == 0) _tmp = 1 / (1 + exp(-_tmp));
		else if (activ == 1) _tmp = tanh(_tmp);
		else if (activ == 2) _tmp = exp(-_tmp*_tmp);
		else _tmp = _tmp * (_tmp > 0);
		
		//Write it to Y
		var[time*mdl->total + ystart + y] = _tmp;

		//Next colon of weights
		//_w++;
	}
};

void dot1d_use(Use_t * use, uint inst, uint time) {
	dot1d_use_call_mode_th11(use, inst, time);
};

void dot1d_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	dot1d_forward_call_mode_th11(train, inst, time, start_seed);
};

void dot1d_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	dot1d_backward_call_mode_th11(train, inst, time, start_seed);
};
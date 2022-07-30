#include "pkg_head/insts/softmax.cuh"

void softmax_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
};

void softmax_cpu_call(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint 		len = mdl->param[inst][0],	\
		input_start	= mdl->param[inst][1],	\
			 ystart = mdl->param[inst][2];

	float __sum = 0;
	float tmp;

	uint vstart = time*mdl->total;
	float * var = cpu->var;

	for (uint i=0; i < len; i++) {
		tmp = exp(-var[vstart + input_start + i]);
		var[vstart + ystart + i] = exp(-var[vstart + input_start + i]);
		__sum += tmp;
	}

	for (uint i=0; i < len; i++)
		var[vstart + ystart + i] /= __sum;
};

void softmax_use_call(Use_t * use, uint inst, uint time) {
	softmax_use_call_mode_th32(use, inst, time);
};

void softmax_forward_call(Train_t * train, uint inst, uint time, uint start_seed) {
	softmax_forward_call_mode_th32(train, inst, time, start_seed);
};

void softmax_backward_call(Train_t * train, uint inst, uint time, uint start_seed) {
	softmax_backward_call_mode_th32(train, inst, time, start_seed);
};
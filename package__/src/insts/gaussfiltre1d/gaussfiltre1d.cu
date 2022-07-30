#include "pkg_head/insts/gaussfiltre1d/gaussfiltre1d.cuh"

//			   0      1      2      3      4 
//Arguments = [len, istart,ystart,wstart,lstart]

void gaussfiltre1d_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
};

void gaussfiltre1d_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint len=mdl->param[inst][0],		\
		 istart=mdl->param[inst][1],	\
		 ystart=mdl->param[inst][2],	\
		 wstart=mdl->param[inst][3],	\
		 locdstart=mdl->param[inst][4];

	uint inp = total*time + istart;
	uint out = total*time + ystart;

	uint total = mdl->total;

	float * var = cpu->var;
	float * weight = mdl->weight;

	float a,p;

	for (uint y=0; y < len; y++) {
		a = var[inp + (y)];
		p = weight[wstart + (x)];
		var[out + y] = exp(-(a+p)**2);
	}
};

void gaussfiltre1d_use(Use_t * use, uint inst, uint time) {
	gaussfiltre1d_use_call_mode_th11(use, inst, time);
};

void gaussfiltre1d_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	gaussfiltre1d_forward_call_mode_th11(train, inst, time, start_seed);	
};

void gaussfiltre1d_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	gaussfiltre1d_backward_call_mode_th11(train, inst, time, start_seed);
};
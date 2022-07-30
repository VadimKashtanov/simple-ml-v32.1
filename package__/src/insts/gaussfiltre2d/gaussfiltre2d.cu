#include "pkg_head/insts/gaussfiltre2d/gaussfiltre2d.cuh"

//			   0  1    2      3      4       5
//Arguments = [X,Y, istart,ystart,wstart,lstart]

void gaussfiltre2d_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
};

void gaussfiltre2d_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint X=mdl->param[inst][0],		\
		 Y=mdl->param[inst][1],		\
		 istart=mdl->param[inst][2],\
		 ystart=mdl->param[inst][3],\
		 wstart=mdl->param[inst][4];

	uint inp = total*time + istart;
	uint out = total*time + ystart;

	uint total = mdl->total;

	float * var = cpu->var;
	float * weight = mdl->weight;

	float a,p;

	for (uint y=0; y < Y; y++) {
		for (uint x=0; x < X; x++) {
			a = var[inp + (y*X+x)];
			p = weight[wstart + (x)];
			var[out + y*X + x] = exp(-(a+p)**2);
		}
	}
};

void gaussfiltre2d_use(Use_t * use, uint inst, uint time) {
	gaussfiltre2d_use_call_mode_th11(use, inst, time);
};

void gaussfiltre2d_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	gaussfiltre2d_forward_call_mode_th11(train, inst, time, start_seed);
};

void gaussfiltre2d_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	gaussfiltre2d_backward_call_mode_th11(train, inst, time, start_seed);
};
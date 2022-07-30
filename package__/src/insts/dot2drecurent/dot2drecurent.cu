#include "pkg_head/insts/dot2drecurent/dot2drecurent.cuh"

//			   0  1    2   3    4      5       6    7			8
//Arguments = [Ax,Ay, At, Bx, istart,ystart,wstart,locdstart, drate]

void dot2drecurent_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
	if (param[3] == 0) raise(SIGINT);
	if (param[8] >100) raise(SIGINT);
};

void dot2drecurent_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],	\
		 Ay=mdl->param[inst][1],	\
		 At=mdl->param[inst][2],	\
		 Bx=mdl->param[inst][3],	\
		 istart=mdl->param[inst][4],\
		 ystart=mdl->param[inst][5],\
		 wstart=mdl->param[inst][6],\
		 lstart=mdl->param[inst][7],\
		 drate=mdl->param[inst][8];

	uint total = mdl->total;

	float * var = cpu->var;
	float * weight = mdl->weight;

	float tmp;

	uint Apos, Wpos;

	if (time - At >= 0) {
		for (uint y=0; y < Ay; y++) {
			for (uint x=0; x < Bx; x++) {
				_tmp = 0;

				Apos = (time-At)*total + istart + y*Ax;
				Wpos = wstart + y;

				for (uint i=0; i < Ax; i++) {
					_tmp += var[Apos] * weight[Wpos];
					Apos++;
					Wpos += Bx;
				}

				_tmp += weight[wstart + Ax*Bx + (y*Bx + x)];

				if (activ == 0)	_tmp = 1 / (1 + exp(-_tmp));
				else if (activ == 1) _tmp = tanh(_tmp);
				else if (activ == 2) _tmp = exp(-_tmp*_tmp);
				else _tmp *= (tmp > 0);

				var[time*total + ystart + (y*Bx + x)] = _tmp;
			}
		}
	}
};

void dot2drecurent_use(Use_t * use, uint inst, uint time) {
	dot2drecurent_use_call_mode_th11(use, inst, time);
};

void dot2drecurent_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	dot2drecurent_forward_call_mode_th11(train, inst, time, start_seed);
};

void dot2drecurent_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	dot2drecurent_backward_call_mode_th11(train, inst, time, start_seed);
};
#include "pkg_head/insts/dot1drecurent/dot1drecurent.cuh"

//			  0  1   2    3      4  5   6    7     8
//	Params : [Ax,At, Yx, activ, ist,yst,wst,lst, drate]
//	At - de combien de lignes on va en arriere. Si At=1 =>  A=A[t-1]

void dot1drecurent_check(uint * param) {
	if (param[0] == 0) 			 raise(SIGINT);
	if (param[2] == 0) 			 raise(SIGINT);
	if (param[3] >= ACTIV_FUNCS) raise(SIGINT);
	if (param[8] >100) 			 raise(SIGINT);
};

void dot1drecurent_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],		\
		 At=mdl->param[inst][1],		\
		 Yx=mdl->param[inst][2],		\
		 acitv=mdl->param[inst][3]
		 istart=mdl->param[inst][4],	\
		 ystart=mdl->param[inst][5],	\
		 wstart=mdl->param[inst][6],	\
		 lstart=mdl->param[inst][7],	\
		 drate=mdl->param[inst][8];

	uint total = mdl->total;

	float * var = cpu->var;
	float * weight = mdl->weight;

	float tmp;

	uint Apos, Wpos;

	if (time - At >= 0) {
		for (uint y=0; y < Yx; y++) {
			_tmp = 0;

			Apos = (time-At)*total + istart;
			Wpos = wstart + y;

			for (uint i=0; i < Ax; i++) {
				_tmp += var[Apos] * weight[Wpos];
				Apos++;
				Wpos += Yx;
			}

			_tmp += weight[Wpos + Yx];	//==wstart + Ax*Yx + y      car on a deja +y, et on a += Ax*Yx (for i<Ax) {+=Yx}

			if (activ == 0)	_tmp = 1 / (1 + exp(-_tmp));
			else if (activ == 1) _tmp = tanh(_tmp);
			else if (activ == 2) _tmp = exp(-_tmp*_tmp);
			else _tmp *= (tmp > 0);

			var[time*total + ystart + y] = _tmp;
		}
	}
};

void dot1drecurent_use(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],		\
		 At=mdl->param[inst][1],		\
		 Yx=mdl->param[inst][2],		\
		 acitv=mdl->param[inst][3]
		 istart=mdl->param[inst][4],	\
		 ystart=mdl->param[inst][5],	\
		 wstart=mdl->param[inst][6],	\
		 lstart=mdl->param[inst][7],	\
		 drate=mdl->param[inst][8];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	dot1drecurent_use_call_mode_th11(use, inst, time);
};

void dot1drecurent_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],		\
		 At=mdl->param[inst][1],		\
		 Yx=mdl->param[inst][2],		\
		 acitv=mdl->param[inst][3]
		 istart=mdl->param[inst][4],	\
		 ystart=mdl->param[inst][5],	\
		 wstart=mdl->param[inst][6],	\
		 lstart=mdl->param[inst][7],	\
		 drate=mdl->param[inst][8];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	dot1drecurent_forward_call_mode_th11(train, inst, time, start_seed);
};

void dot1drecurent_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],		\
		 At=mdl->param[inst][1],		\
		 Yx=mdl->param[inst][2],		\
		 acitv=mdl->param[inst][3]
		 istart=mdl->param[inst][4],	\
		 ystart=mdl->param[inst][5],	\
		 wstart=mdl->param[inst][6],	\
		 lstart=mdl->param[inst][7],	\
		 drate=mdl->param[inst][8];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	dot1drecurent_backward_call_mode_th11(train, inst, time, start_seed);
};
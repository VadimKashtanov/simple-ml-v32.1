#include "pkg_head/insts/lstm1d/lstm1d.cuh"

//			   0  1    2      3      4       5          6 
//Arguments = [X, Y, istart,ystart,wstart,locdstart, drop_rate]

void lstm1d_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
	if (param[6] >100) raise(SIGINT);
};

void lstm1d_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint X=mdl->param[inst][0],		\
		 Y=mdl->param[inst][1],		\
		 istart=mdl->param[inst][2],\
		 ystart=mdl->param[inst][3],\
		 wstart=mdl->param[inst][4];

	uint total = mdl->total;
	
	uint inp = total*time + istart;
	uint W = wstart;
	uint out = total*time + ystart;

	uint lineW = X*Y + Y*Y + Y;	// == sizeof(W + U + B). There is 4 sets of (W,U,B) for f0,f1,f2 and g0

	float * var = cpu->var;
	float * weight = mdl->weight;

	float vpos, wpos;

	float f0,f1,f2,g0,  e_1, e, h;

	for (uint y=0; y < Y; y++) {
		f0=0; f1=0; f2=0; g0=0;

		//	x @ W
		for (uint k=0; k < X; k++) {
			vpos = inp + k;
			wpos = k*X + y;
			f0 += var[vpos]*w[W + wpos];
			f1 += var[vpos]*w[W + lineW + wpos];
			f2 += var[vpos]*w[W + 2*lineW + wpos];
			g0 += var[vpos]*w[W + 3*lineW + wpos];
		}

		//	h[-1] @ U
		if (time > 0) {
			for (uint k=0; k < X; k++) {
				vpos = (time-1)*total + istart + Y + k; 		//out - total == total*(l-1) + ystart
				wpos = X*Y + k*X + y;
				f0 += var[vpos]*w[W + wpos];
				f1 += var[vpos]*w[W + lineW + wpos];
				f2 += var[vpos]*w[W + 2*lineW + wpos];
				g0 += var[vpos]*w[W + 3*lineW + wpos];
			}
		}

		//	+ B
		wpos = X*Y + Y*Y + y;
		f0 += w[W + wpos];
		f1 += w[W + lineW + wpos];
		f2 += w[W + 2*lineW + wpos];
		g0 += w[W + 3*lineW + wpos];

		// activate(_sum)
		f0 = logistic(f0);
		f1 = logistic(f1);
		f2 = logistic(f2);
		g0 = tanh(g0);

		// e = f0 * e[-1] + f1 * g0
		// l - 1 have to be >= 0 || l > 0
		if (l > 0) e_1 = var[total*(time-1) + ystart + y];
		else e_1 = 0;
		
		e = f0*e_1 + f1*g0;
		h = f2 * (e);	//f(x)=x

		var[out + y] = e;
		var[out + Y + y] = h;
	};
};

void lstm1d_use(Use_t * use, uint inst, uint time) {
	lstm1d_use_call_mode_th11(use, inst, time);
};

void lstm1d_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	lstm1d_forward_call_mode_th11(train, inst, time, start_seed);
};

void lstm1d_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	lstm1d_backward_call_mode_th11(train, inst, time, start_seed);
};
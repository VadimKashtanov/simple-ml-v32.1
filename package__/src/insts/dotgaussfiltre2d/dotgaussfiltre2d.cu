#include "pkg_head/insts/dotgaussfiltre2d/dotgaussfiltre2d.cuh"

//			   0  1  2     3      4       5       6        7 
//Arguments = [Ax,Ay,Bx, istart,ystart,wstart,locdstart, drate]

/*

	  	      [p0,p1]
		      [p2,p3]
		      [p4,p5]
[a0,a1,a2] -> [y0,y1]
[a3,a4,a5] -> [y2,y3]

Y[y*Bx + x] = sum( exp(-(a[y*Ax + i] + p[i*Bx + x])^2) for i in range(Ax))

y0 = exp(-(a0+p0)^2) + exp(-(a1+p2)^2) + exp(-(a2+p4)^2)
y3 = exp(-(a3+p1)^2) + exp(-(a4+p3)^2) + exp(-(a5+p5)^2)

locd = -2(a+p)y

*/

void dotgaussfiltre2d_check(uint * param) {
	if (param[0] == 0) raise(SIGINT);
	if (param[1] == 0) raise(SIGINT);
	if (param[2] == 0) raise(SIGINT);
	if (param[7] >100) raise(SIGINT);
};

void dotgaussfiltre2d_cpu(Cpu_t * cpu, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 Bx=mdl->param[inst][2],		\
		 istart=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],	\
		 wstart=mdl->param[inst][5],	\
		 lstart=mdl->param[inst][6],	\
		 drate=mdl->param[inst][7];

	uint inp = total*time + istart;
	uint W = wstart;
	uint out = total*time + ystart;

	uint total = mdl->total;

	float * var = cpu->var;
	float * weight = mdl->weight;

	float _tmp;

	for (uint y=0; y < Ay; y++) {
		for (uint x=0; x < Bx; x++) {
			_tmp = 0;

			for (uint i=0; i < Ax; i++) {

				apos = time*total + istart + y*Ax + i;
				ppos = wstart + i*Bx + x;
				
				_tmp += exp(-pow(var[apos] + weight[ppos],2));
			}

			var[time*total + ystart + (y*Bx+x)] = _tmp;
		}
	}
};

void dotgaussfiltre2d_use(Use_t * use, uint inst, uint time) {
	dotgaussfiltre2d_use_call_mode_th11(use, inst, time);
};

void dotgaussfiltre2d_forward(Train_t * train, uint inst, uint time, uint start_seed) {
	dotgaussfiltre2d_forward_call_mode_th11(train, inst, time, start_seed);
};

void dotgaussfiltre2d_backward(Train_t * train, uint inst, uint time, uint start_seed) {
	dotgaussfiltre2d_backward_call_mode_th11(train, inst, time, start_seed);
};
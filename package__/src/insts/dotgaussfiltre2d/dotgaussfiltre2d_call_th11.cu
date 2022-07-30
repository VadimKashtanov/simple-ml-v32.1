#include "pkg_head/insts/dotgaussfiltre2d/dotgaussfiltre2d_th11.cuh"

void dotgaussfiltre2d_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = cpu->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 Bx=mdl->param[inst][2],		\
		 istart=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],	\
		 wstart=mdl->param[inst][5],	\
		 lstart=mdl->param[inst][6],	\
		 drate=mdl->param[inst][7];

	dotgaussfiltre2d_use_th1x1<<<dim3(KERN_DIV(Bx,32),KERN_DIV(Ay,32)),dim3(32,32)>>>(
		Ax, Ay, Bx,
		time,
		mdl->total,
		input_start, ystart, wstart,
		use->var, use->weight);
};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dotgaussfiltre2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 Bx=mdl->param[inst][2],		\
		 istart=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],	\
		 wstart=mdl->param[inst][5],	\
		 lstart=mdl->param[inst][6],	\
		 drate=mdl->param[inst][7];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;
	
	float drop_rate = drop_rate_int/100;

	dotgaussfiltre2d_forward_th1x1<<<dim3(KERN_DIV(Bx,32),KERN_DIV(Ay,32),train->sets),dim3(32,32,1)>>>(
		Ax, Ay, Bx,
		time,
		input_start, ystart, wstart, locdstart,
		train->mdl->total, train->mdl->weights, train->mdl->locds,
		train->_var, train->_weight, train->_locd,
		inst*start_seed, drop_rate,
		train->sets);
};

//-------------------------- backward ---------------------

void dotgaussfiltre2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],		\
		 Ay=mdl->param[inst][1],		\
		 Bx=mdl->param[inst][2],		\
		 istart=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],	\
		 wstart=mdl->param[inst][5],	\
		 lstart=mdl->param[inst][6],	\
		 drate=mdl->param[inst][7];

	uint total = mdl->total;
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	float drop_rate = drop_rate_int/100;

	dotgaussfiltre2d_backward_th1x1<<<dim3(KERN_DIV(Bx,32),KERN_DIV(Ay,32),train->sets),dim3(32,32,1)>>>(
		Ax, Ay, Bx,
		time,
		input_start, ystart, wstart, locdstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, drop_rate,
		train->sets);
};
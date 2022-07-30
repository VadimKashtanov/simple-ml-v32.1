#include "pkg_head/insts/dot2drecurent/dot2drecurent_th11.cuh"

void dot2drecurent_use_call_mode_th11(Use_t * use, uint inst, uint time) {
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

	if (time - At >= 0) {
		dot2drecurent_use_th1x1<<<dim3(KERN_DIV(Bx,32)),dim3(32)>>>(
			Ax, Ay, At, Bx,
			activ,
			time,
			mdl->total,
			istart, ystart, wstart,
			use->var, use->weight);
	};
};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dot2drecurent_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

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
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	float drop_rate = drate/100;

	if (time - At >= 0) {
		dot2drecurent_forward_th1x1<<<dim3(KERN_DIV(Bx,16),KERN_DIV(Ay,16),train->sets),dim3(16,16,1)>>>(
			Ax, Ay, At, Bx,
			activ,
			time,	//output time (Yx time)
			istart, ystart, wstart, lstart,
			train->mdl->total, train->mdl->weights, train->mdl->locds,
			train->_var, train->_weight, train->_locd,
			inst*start_seed, drop_rate,
			train->sets);
	}
};

//-------------------------- backward ---------------------

void dot2drecurent_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

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
	uint wsize = mdl->weights;
	uint locdsize = mdl->locds;

	float drop_rate = drate/100;

	if (time - At >= 0) {
		dot2drecurent_backward_th1x1<<<dim3(KERN_DIV(Bx,16),KERN_DIV(Ay,16),train->sets),dim3(16,16,1)>>>(
			Ax, Ay, At, Bx,
			activ,
			time,
			input_start, ystart, wstart, locdstart,
			mdl->total, mdl->weights, mdl->locds,
			train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
			inst*start_seed, drop_rate,
			train->sets);
	}
};
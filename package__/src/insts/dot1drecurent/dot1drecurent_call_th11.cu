#include "pkg_head/insts/dot1drecurent/dot1drecurent_th11.cuh"

void dot1drecurent_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint Ax=mdl->param[inst][0],		\
		 At=mdl->param[inst][1],		\
		 Yx=mdl->param[inst][2],		\
		 acitv=mdl->param[inst][3]
		 istart=mdl->param[inst][4],	\
		 ystart=mdl->param[inst][5],	\
		 wstart=mdl->param[inst][6],	\
		 lstart=mdl->param[inst][7],	\
		 drate=mdl->param[inst][8];

	if (time - At >= 0) {
		dot1drecurent_use_th1x1<<<dim3(KERN_DIV(Yx,32)),dim3(32)>>>(
			Ax, At, Yx,
			activ,
			time,
			mdl->total,
			istart, ystart, wstart,
			use->var, use->weight);
	}; /*else {
		negativ line doesn't exists
	}*/

};

void dot1drecurent_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
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

	float drop_rate = drate/100;

	if (time - At >= 0) {
		dot1drecurent_forward_th1x1<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
			Ax, At, Yx,
			activ,
			time,	//output time (Yx time)
			istart, ystart, wstart, locdstart,
			train->mdl->total, train->mdl->weights, train->mdl->locds,
			train->_var, train->_weight, train->_locd,
			inst*start_seed, drop_rate,
			train->sets);
	}
};

void dot1drecurent_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
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

	float drop_rate = drate/100;

	if (time - At >= 0) {
		dot1drecurent_backward_th1x1<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
			Ax, At, Yx,
			activ,
			time,
			input_start, ystart, wstart, locdstart,
			mdl->total, mdl->weights, mdl->locds,
			train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
			inst*start_seed, drop_rate,
			train->sets);
	}
};
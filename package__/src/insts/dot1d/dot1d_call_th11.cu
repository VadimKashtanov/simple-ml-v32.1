#include "pkg_head/insts/dot1d/dot1d_th11.cuh"

void dot1d_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint Ax=mdl->param[inst][0],				\
		 Yx=mdl->param[inst][1],				\
		 activ=mdl->param[inst][2],			\
		 input_start=mdl->param[inst][3],		\
		 ystart=mdl->param[inst][4],			\
		 wstart=mdl->param[inst][5];

	dot1d_use_th1x1<<<dim3(KERN_DIV(Yx,32)),dim3(32)>>>(
		Ax, Yx,
		activ,
		time,
		mdl->total,
		input_start, ystart, wstart,
		use->var, use->weight);
};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dot1d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],			\
		 Yx=mdl->param[inst][1],			\
		 activ=mdl->param[inst][2],		\
		 input_start=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],		\
		 wstart=mdl->param[inst][5],		\
		 locdstart=mdl->param[inst][6],	\
		 drop_rate_int=mdl->param[inst][7];

	float drop_rate = drop_rate_int/100;

	dot1d_forward_th1x1<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
		Ax, Yx,
		activ,
		time,
		input_start, ystart, wstart, locdstart,
		train->mdl->total, train->mdl->weights, train->mdl->locds,
		train->_var, train->_weight, train->_locd,
		inst*start_seed, drop_rate,
		train->sets);
};

//-------------------------- backward ---------------------

void dot1d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],			\
		 Yx=mdl->param[inst][1],			\
		 activ=mdl->param[inst][2],		\
		 input_start=mdl->param[inst][3],	\
		 ystart=mdl->param[inst][4],		\
		 wstart=mdl->param[inst][5],		\
		 locdstart=mdl->param[inst][6],	\
		 drop_rate_int=mdl->param[inst][7];

	float drop_rate = drop_rate_int/100;

	dot1d_backward_th1x1<<<dim3(KERN_DIV(Yx,16),train->sets),dim3(16,1)>>>(
		Ax, Yx,
		activ,
		time,
		input_start, ystart, wstart, locdstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, drop_rate,
		train->sets);
};
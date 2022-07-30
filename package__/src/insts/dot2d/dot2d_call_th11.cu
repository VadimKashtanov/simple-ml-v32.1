#include "pkg_head/insts/dot2d/dot2d_th11.cuh"

void dot2d_use_call_mode_th11(Use_t * use, uint inst, uint time) {
	Mdl_t * mdl = use->mdl;

	uint Ax=mdl->param[inst][0],				\
		 Ay=mdl->param[inst][1],				\
		 Bx=mdl->param[inst][2],				\
		 activ=mdl->param[inst][3],			\
		 input_start=mdl->param[inst][4],		\
		 ystart=mdl->param[inst][5],			\
		 wstart=mdl->param[inst][6];

	//	This kernel Call is th1x1
	//	donc en fait le kernel est ecrit est garantis le fonctionnement sous thread:(1,1)
	dot2d_use_th1x1<<<dim3(Bx, Ay),dim3(1,1)>>>(
		Ax, Ay, Bx,
		activ,
		time,
		mdl->total,
		input_start, ystart, wstart,
		use->var, use->weight);
};

//======================== Train_t =======================

//-------------------------- forward ---------------------

void dot2d_forward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],			\
		 Ay=mdl->param[inst][1],			\
		 Bx=mdl->param[inst][2],			\
		 activ=mdl->param[inst][3],		\
		 input_start=mdl->param[inst][4],	\
		 ystart=mdl->param[inst][5],		\
		 wstart=mdl->param[inst][6],		\
		 locdstart=mdl->param[inst][7],	\
		 drop_rate_int=mdl->param[inst][8];

	float drop_rate = drop_rate_int/100;

	//	Le seul mode est selui du threadblock(1,1) car th1x1
	dot2d_forward_th1x1<<<dim3(Bx, Ay,train->sets),dim3(1,1,1)>>>(
		Ax, Ay, Bx,
		activ,
		time,
		input_start, ystart, wstart, locdstart,
		train->mdl->total, train->mdl->weights, train->mdl->locds,
		train->_var, train->_weight, train->_locd,
		inst*start_seed, drop_rate,
		train->sets);
};

//-------------------------- backward ---------------------

void dot2d_backward_call_mode_th11(Train_t * train, uint inst, uint time, uint start_seed) {
	Mdl_t * mdl = train->mdl;

	uint Ax=mdl->param[inst][0],			\
		 Ay=mdl->param[inst][1],			\
		 Bx=mdl->param[inst][2],			\
		 activ=mdl->param[inst][3],		\
		 input_start=mdl->param[inst][4],	\
		 ystart=mdl->param[inst][5],		\
		 wstart=mdl->param[inst][6],		\
		 locdstart=mdl->param[inst][7],	\
		 drop_rate_int=mdl->param[inst][8];

	float drop_rate = drop_rate_int/100;

	dot2d_backward_th1x1<<<dim3(Bx,Ay,train->sets),dim3(1,1,1)>>>(
		Ax, Ay, Bx,
		activ,
		time,
		input_start, ystart, wstart, locdstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, drop_rate,
		train->sets);

	/*dot2d_backward_th1x1_bias<<<dim3(Bx,Ay,train->sets),dim3(1,1,1)>>>(
		Ax, Ay, Bx,
		activ,
		time,
		input_start, ystart, wstart, locdstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, drop_rate,
		train->sets);

	dot2d_backward_th1x1_input<<<dim3(Ax,Ay,train->sets),dim3(1,1,1)>>>(
		Ax, Ay, Bx,
		activ,
		time,
		input_start, ystart, wstart, locdstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, drop_rate,
		train->sets);

	dot2d_backward_th1x1_weight<<<dim3(Bx,Ax,train->sets),dim3(1,1,1)>>>(
		Ax, Ay, Bx,
		activ,
		time,
		input_start, ystart, wstart, locdstart,
		mdl->total, mdl->weights, mdl->locds,
		train->_var, train->_weight, train->_locd, train->_grad, train->_meand,
		inst*start_seed, drop_rate,
		train->sets);*/
}